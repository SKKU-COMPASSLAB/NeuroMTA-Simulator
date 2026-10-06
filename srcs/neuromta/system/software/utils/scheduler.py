from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from neuromta.system.software.utils.compiler import MeshDeviceActionType, MeshDeviceCompiledAction, MeshDeviceCompiledTensorStats
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshKernelType, MeshMemoryBankDescriptor, MeshMemoryType
from neuromta.system.software.utils.runtime_utils import create_linear_mapping_requistes, create_conv2d_mapping_requistes, create_elementwise_mapping_requistes, create_reduction_mapping_requistes, create_sdpa_mapping_requistes


__all__ = [
    "MeshFCFSScheduler",
    "MeshFRFCFSScheduler",
    "MeshKernelAIStats",
    "MeshPlanCandidate",
    "MeshSchedulingDomain",
    "MeshReadyKernelInfo",
    "MeshRoundRobinScheduler",
    "MeshDeviceScheduler",
    "MeshSchedulingContext",
    "MeshUtilizationScheduler",
    "MeshWorkloadSchedulingHint",
]


@dataclass(frozen=True)
class MeshWorkloadSchedulingHint:
    priority: int = 0
    weight: float = 1.0
    max_wait_cycles: int | None = None

    def __post_init__(self):
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("priority must be an integer.")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or self.weight <= 0
        ):
            raise ValueError("weight must be positive.")
        if self.max_wait_cycles is not None and (
            not isinstance(self.max_wait_cycles, int)
            or isinstance(self.max_wait_cycles, bool)
            or self.max_wait_cycles <= 0
        ):
            raise ValueError("max_wait_cycles must be a positive integer or None.")


@dataclass(frozen=True)
class MeshSchedulingDomain:
    domain_id: str
    ccg_tile_mesh: np.ndarray
    dma_ids: tuple[int, ...]

    def __post_init__(self):
        mesh = np.asarray(self.ccg_tile_mesh, dtype=int)
        if not self.domain_id:
            raise ValueError("domain_id must not be empty.")
        if mesh.ndim != 2 or mesh.size == 0:
            raise ValueError("ccg_tile_mesh must be a non-empty 2D array.")
        if len(set(mesh.flatten().tolist())) != mesh.size:
            raise ValueError("ccg_tile_mesh must not contain duplicate CCG IDs.")
        if not self.dma_ids:
            raise ValueError("dma_ids must not be empty.")
        mesh = mesh.copy()
        mesh.setflags(write=False)
        object.__setattr__(self, "ccg_tile_mesh", mesh)
        object.__setattr__(self, "dma_ids", tuple(int(dma_id) for dma_id in self.dma_ids))


@dataclass(frozen=True)
class MeshReadyKernelInfo:
    execution_id: str
    workload_id: str
    ready_cycle: int
    waiting_cycles: int
    kernel_index: int
    arrival_cycle: int
    submission_index: int
    kernel_id: str = ""
    domain_id: str = "device"
    priority: int = 0
    weight: float = 1.0
    max_wait_cycles: int | None = None
    dispatch_count: int = 0
    service_core_cycles: int = 0


@dataclass(frozen=True)
class MeshPlanCandidate:
    plan_id: int
    kernel_ids: tuple[str, ...]
    workload_ids: tuple[str, ...]
    makespan_cycles: int
    sequential_cycles: int
    utilization: float
    benefit: float
    peak_dma_utilization: float
    placements: tuple[Any, ...] = ()


@dataclass(frozen=True)
class MeshSchedulingContext:
    cycle: int
    ready_kernels: tuple[MeshReadyKernelInfo, ...]
    running_kernel_ids: tuple[str, ...]
    domains: tuple[MeshSchedulingDomain, ...] = ()


@dataclass(frozen=True)
class MeshKernelAIStats:
    arithmetic_intensity: float
    ccg_tile_mesh: np.ndarray
    tensor_placements: tuple[tuple[Any, dict], ...]
    allocated_banks: tuple[MeshMemoryBankDescriptor, ...]
    persistent_placements: tuple[tuple[str, MeshDeviceCompiledTensorStats], ...]


class MeshDeviceScheduler:
    def __init__(self, starvation_cycles: int | None = 200_000, candidate_window: int = 16, decision_hook=None):
        if starvation_cycles is not None and (
            not isinstance(starvation_cycles, int)
            or isinstance(starvation_cycles, bool)
            or starvation_cycles <= 0
        ):
            raise ValueError("starvation_cycles must be a positive integer or None.")
        if (
            not isinstance(candidate_window, int)
            or isinstance(candidate_window, bool)
            or candidate_window <= 0
        ):
            raise ValueError("candidate_window must be positive.")
        if decision_hook is not None and not callable(decision_hook):
            raise TypeError("decision_hook must be callable or None.")
        self.starvation_cycles = starvation_cycles
        self.candidate_window = candidate_window
        self.decision_hook = decision_hook
        self._decision_metadata: dict = {}
        self._ready_cycles: dict[tuple[str, int, str], int] = {}
        self._submission_indices: dict[tuple[str, int, str], int] = {}
        self._next_submission_index = 0
        self._noc_ops_cache = {}

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def decision_metadata(self) -> dict:
        return dict(self._decision_metadata)

    def reset(self):
        self._decision_metadata = {}
        self._ready_cycles = {}
        self._submission_indices = {}
        self._next_submission_index = 0
        self._noc_ops_cache = {}

    def schedule_actions(self, runtime_context, actions: list[tuple], persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats] | None = None, running_kernel_ids: tuple[str, ...] = ()) -> tuple[list[tuple[MeshDeviceCompiledAction, Any]], list[tuple[MeshDeviceCompiledAction, Any]]]:
        if runtime_context is None:
            raise ValueError("A runtime context must be registered before scheduling kernel actions.")

        persistent_state_placements = {} if persistent_state_placements is None else persistent_state_placements
        cycle = int(runtime_context.device.timestamp)
        action_map = {}
        ready_kernels = []
        domains = {}

        for item in actions:
            if len(item) == 2:
                action, workload_state = item
                domain = self._full_device_domain(runtime_context)
            elif len(item) == 3:
                action, workload_state, domain = item
            else:
                raise ValueError("Scheduled actions must contain an action, workload state, and optional domain.")
            if action.action_type != MeshDeviceActionType.RUN_KERNEL:
                raise ValueError(f"MeshDeviceScheduler only accepts RUN_KERNEL actions, got {action.action_type}.")
            if not isinstance(domain, MeshSchedulingDomain):
                raise TypeError(f"Expected MeshSchedulingDomain, got {type(domain).__name__}.")
            key = (workload_state.workload_id, workload_state.cursor, action.kernel_id)
            if key not in self._ready_cycles:
                self._ready_cycles[key] = cycle
                self._submission_indices[key] = self._next_submission_index
                self._next_submission_index += 1
            hint = workload_state.scheduling_hint or MeshWorkloadSchedulingHint()
            execution_id = self._execution_id(key)
            action_map[execution_id] = (action, workload_state, domain, key)
            ready_cycle = self._ready_cycles[key]
            ready_kernels.append(MeshReadyKernelInfo(execution_id=execution_id, workload_id=workload_state.workload_id, ready_cycle=ready_cycle, waiting_cycles=max(0, cycle - ready_cycle), kernel_index=workload_state.cursor, arrival_cycle=workload_state.arrival_cycle, submission_index=self._submission_indices[key], kernel_id=action.kernel_id, domain_id=domain.domain_id, priority=hint.priority, weight=hint.weight, max_wait_cycles=hint.max_wait_cycles))
            domains[domain.domain_id] = domain

        context = MeshSchedulingContext(cycle=cycle, ready_kernels=tuple(ready_kernels), running_kernel_ids=tuple(running_kernel_ids), domains=tuple(domains.values()))
        ordered_ids = self.order_ready(context)
        search_anchors = self.plan_search_anchors(context)
        if search_anchors is not None:
            anchor_ids = tuple(execution_id for execution_id in search_anchors if execution_id in action_map)
            ordered_ids = anchor_ids + tuple(execution_id for execution_id in ordered_ids if execution_id not in set(anchor_ids))
        self._active_ready_count = min(len(ordered_ids), self.candidate_window)
        dispatch_limit = self._get_dispatch_limit(context)
        if dispatch_limit is not None and dispatch_limit <= 0:
            return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        if self._use_joint_planning():
            return self._schedule_joint_actions(runtime_context, context, ordered_ids, action_map, persistent_state_placements)
        selection_candidates = tuple(MeshPlanCandidate(plan_id=index, kernel_ids=(execution_id,), workload_ids=(action_map[execution_id][1].workload_id,), makespan_cycles=0, sequential_cycles=0, utilization=0.0, benefit=0.0, peak_dma_utilization=0.0) for index, execution_id in enumerate(ordered_ids[:self.candidate_window]))
        selected_plan_id = self.select_plan(context, selection_candidates)
        if selected_plan_id is None:
            return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        selected_candidate = next((candidate for candidate in selection_candidates if candidate.plan_id == selected_plan_id), None)
        if selected_candidate is None:
            raise RuntimeError(f"Scheduler {self.name} selected unknown plan ID {selected_plan_id}.")
        selected_execution_id = selected_candidate.kernel_ids[0]
        ordered_ids = (selected_execution_id,) + tuple(execution_id for execution_id in ordered_ids if execution_id != selected_execution_id)
        scheduled = []
        suspended = []

        for index, execution_id in enumerate(ordered_ids):
            action, workload_state, domain, key = action_map[execution_id]
            if index >= self.candidate_window or dispatch_limit is not None and len(scheduled) >= dispatch_limit:
                suspended.append((action, workload_state, domain))
                continue
            kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
            if self._place_kernel(runtime_context, workload_state.compiled_workload, kernel_stats, persistent_state_placements, domain):
                scheduled.append((action, workload_state))
                self._ready_cycles.pop(key, None)
                self._submission_indices.pop(key, None)
                self._on_action_scheduled(workload_state.workload_id)
            else:
                suspended.append((action, workload_state, domain))
                if not self._continue_after_placement_failure():
                    suspended.extend((action_map[remaining_id][0], action_map[remaining_id][1], action_map[remaining_id][2]) for remaining_id in ordered_ids[index + 1:])
                    break

        if scheduled:
            selected_ids = tuple(self._execution_id((state.workload_id, state.cursor, action.kernel_id)) for action, state in scheduled)
            core_count = sum(state.compiled_workload.get_kernel_stats(action.kernel_id).ccg_tile_mesh.size for action, state in scheduled)
            total_cores = max(1, len(runtime_context.ccg_tile_ids))
            candidate = MeshPlanCandidate(plan_id=0, kernel_ids=selected_ids, workload_ids=tuple(state.workload_id for _, state in scheduled), makespan_cycles=0, sequential_cycles=0, utilization=min(1.0, core_count / total_cores), benefit=0.0, peak_dma_utilization=0.0)
            self.on_dispatch(context, candidate)
            self._decision_metadata = {**self._decision_metadata, "selected_kernel_ids": selected_ids}

        return scheduled, suspended

    def _get_dispatch_limit(self, context: MeshSchedulingContext) -> int | None:
        return None

    def _use_joint_planning(self) -> bool:
        return False

    def _schedule_joint_actions(self, runtime_context, context, ordered_ids, action_map, persistent_state_placements):
        ordered_ids = tuple(ordered_ids[:self.candidate_window])
        base_snapshot = self._snapshot_placement_state(runtime_context, action_map, persistent_state_placements)
        plans = []
        baseline_cycles = {}
        plan_id = 0
        trace = [] if self.decision_hook is not None else None

        def record(phase, reason, kernel_ids, placements=(), predicted_cycles=(), sequential_cycles=None, makespan_cycles=None, benefit=None, peak_dma_utilization=None, candidate_id=None):
            if trace is None:
                return
            trace.append({
                "phase": phase,
                "reason": reason,
                "selection_reason": self._decision_metadata.get("selection_reason"),
                "candidate_id": candidate_id,
                "kernel_ids": tuple(kernel_ids),
                "workload_ids": tuple(action_map[execution_id][1].workload_id for execution_id in kernel_ids),
                "mesh_shapes": tuple(tuple(placement.ccg_tile_mesh.shape) if placement is not None else None for placement in placements),
                "core_ids": tuple(tuple(placement.ccg_tile_mesh.flatten().tolist()) if placement is not None else None for placement in placements),
                "predicted_kernel_cycles": tuple(predicted_cycles),
                "sequential_cycles": sequential_cycles,
                "makespan_cycles": makespan_cycles,
                "benefit": benefit,
                "peak_dma_utilization": peak_dma_utilization,
            })

        def shape_options(execution_id):
            action, workload_state, domain, _ = action_map[execution_id]
            kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
            all_plans = self._get_shape_plans(runtime_context, workload_state.compiled_workload, kernel_stats, persistent_state_placements, domain)
            selected_plans = self._diverse_shape_plans(all_plans, domain.ccg_tile_mesh.size)
            if not selected_plans:
                record("shape_search", "no_feasible_shape", (execution_id,))
            elif trace is not None:
                selected_shapes = {(tuple(placement.ccg_tile_mesh.shape), tuple(placement.ccg_tile_mesh.flatten())) for placement, _ in selected_plans}
                for placement, cycles in all_plans:
                    if (tuple(placement.ccg_tile_mesh.shape), tuple(placement.ccg_tile_mesh.flatten())) not in selected_shapes:
                        record("shape_search", "shape_candidate_limit", (execution_id,), (placement,), (cycles,))
            return selected_plans

        for execution_id in ordered_ids:
            action, workload_state, domain, _ = action_map[execution_id]
            kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
            shape_plans = self._get_singleton_plans(runtime_context, workload_state.compiled_workload, kernel_stats, persistent_state_placements, domain)
            if not shape_plans:
                record("singleton", "no_feasible_shape", (execution_id,))
                continue
            placement, cycles = shape_plans[0]
            baseline_cycles[execution_id] = cycles
            memory_utilization = self._estimate_memory_utilization(runtime_context, kernel_stats.kernel_desc, placement, domain, cycles)
            plans.append(MeshPlanCandidate(plan_id, (execution_id,), (workload_state.workload_id,), cycles, cycles, placement.ccg_tile_mesh.size / runtime_context.device.get_ccg_tile_mesh().size, 0.0, memory_utilization, ((action, workload_state, domain, placement),)))
            record("singleton", "eligible", (execution_id,), (placement,), (cycles,), cycles, cycles, 0.0, memory_utilization, plan_id)
            plan_id += 1

        for first_index, first_id in enumerate(ordered_ids):
            if first_id not in baseline_cycles or not any(
                second_id in baseline_cycles for second_id in ordered_ids[first_index + 1:]
            ):
                continue
            first_action, first_state, first_domain, _ = action_map[first_id]
            first_stats = first_state.compiled_workload.get_kernel_stats(first_action.kernel_id)
            first_plans = shape_options(first_id)
            for first_placement, first_cycles in first_plans:
                self._restore_placement_state(runtime_context, action_map, persistent_state_placements, base_snapshot)
                if not self._commit_kernel_plan(runtime_context, first_stats, first_placement, persistent_state_placements):
                    record("pair", "first_placement_failed", (first_id,), (first_placement,), (first_cycles,))
                    continue
                first_snapshot = self._snapshot_placement_state(runtime_context, action_map, persistent_state_placements)
                for second_id in ordered_ids[first_index + 1:]:
                    if second_id not in baseline_cycles:
                        continue
                    second_action, second_state, second_domain, _ = action_map[second_id]
                    second_stats = second_state.compiled_workload.get_kernel_stats(second_action.kernel_id)
                    second_plans = shape_options(second_id)
                    if not second_plans:
                        record("pair", "no_second_shape_after_first_placement", (first_id, second_id), (first_placement, None), (first_cycles, None))
                    for second_placement, second_cycles in second_plans:
                        self._restore_placement_state(runtime_context, action_map, persistent_state_placements, first_snapshot)
                        if not self._commit_kernel_plan(runtime_context, second_stats, second_placement, persistent_state_placements):
                            record("pair", "second_placement_failed", (first_id, second_id), (first_placement, second_placement), (first_cycles, second_cycles))
                            continue
                        sequential_cycles = baseline_cycles[first_id] + baseline_cycles[second_id]
                        makespan_cycles = max(first_cycles, second_cycles)
                        benefit = (sequential_cycles - makespan_cycles) / sequential_cycles
                        first_memory_utilization = self._estimate_memory_utilization(runtime_context, first_stats.kernel_desc, first_placement, first_domain, first_cycles)
                        second_memory_utilization = self._estimate_memory_utilization(runtime_context, second_stats.kernel_desc, second_placement, second_domain, second_cycles)
                        shared_dma = bool(set(first_domain.dma_ids) & set(second_domain.dma_ids))
                        peak_dma_utilization = first_memory_utilization + second_memory_utilization if shared_dma else max(first_memory_utilization, second_memory_utilization)
                        if benefit <= 0 or peak_dma_utilization > 1.0 + 1e-9:
                            reason = "nonpositive_benefit" if benefit <= 0 else "dma_utilization_exceeded"
                            record("pair", reason, (first_id, second_id), (first_placement, second_placement), (first_cycles, second_cycles), sequential_cycles, makespan_cycles, benefit, peak_dma_utilization)
                            continue
                        core_count = first_placement.ccg_tile_mesh.size + second_placement.ccg_tile_mesh.size
                        plans.append(MeshPlanCandidate(plan_id, (first_id, second_id), (first_state.workload_id, second_state.workload_id), makespan_cycles, sequential_cycles, core_count / runtime_context.device.get_ccg_tile_mesh().size, benefit, peak_dma_utilization, ((first_action, first_state, first_domain, first_placement), (second_action, second_state, second_domain, second_placement))))
                        record("pair", "eligible", (first_id, second_id), (first_placement, second_placement), (first_cycles, second_cycles), sequential_cycles, makespan_cycles, benefit, peak_dma_utilization, plan_id)
                        plan_id += 1
                self._restore_placement_state(runtime_context, action_map, persistent_state_placements, base_snapshot)

        pair_plans = sorted((plan for plan in plans if len(plan.kernel_ids) == 2), key=self._candidate_key, reverse=True)[:8]
        for pair_plan in pair_plans:
            self._restore_placement_state(runtime_context, action_map, persistent_state_placements, base_snapshot)
            pair_valid = True
            pair_cycles = []
            pair_memory_utilizations = []
            for action, workload_state, domain, placement in pair_plan.placements:
                kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
                if not self._commit_kernel_plan(runtime_context, kernel_stats, placement, persistent_state_placements):
                    pair_valid = False
                    break
                cycles = self._estimate_kernel_cycles(runtime_context, kernel_stats.kernel_desc, placement, domain)
                pair_cycles.append(cycles)
                pair_memory_utilizations.append(self._estimate_memory_utilization(runtime_context, kernel_stats.kernel_desc, placement, domain, cycles))
            if not pair_valid:
                record("triple", "pair_placement_failed", pair_plan.kernel_ids, tuple(item[3] for item in pair_plan.placements), tuple(pair_cycles))
                continue
            pair_snapshot = self._snapshot_placement_state(runtime_context, action_map, persistent_state_placements)
            for third_id in ordered_ids:
                if third_id in pair_plan.kernel_ids or third_id not in baseline_cycles:
                    continue
                third_action, third_state, third_domain, _ = action_map[third_id]
                third_stats = third_state.compiled_workload.get_kernel_stats(third_action.kernel_id)
                third_plans = self._diverse_shape_plans(self._get_shape_plans(runtime_context, third_state.compiled_workload, third_stats, persistent_state_placements, third_domain), third_domain.ccg_tile_mesh.size)
                if not third_plans:
                    record("triple", "no_third_shape_after_pair_placement", pair_plan.kernel_ids + (third_id,), tuple(item[3] for item in pair_plan.placements) + (None,), tuple(pair_cycles) + (None,))
                for third_placement, third_cycles in third_plans[:4]:
                    self._restore_placement_state(runtime_context, action_map, persistent_state_placements, pair_snapshot)
                    if not self._commit_kernel_plan(runtime_context, third_stats, third_placement, persistent_state_placements):
                        record("triple", "third_placement_failed", pair_plan.kernel_ids + (third_id,), tuple(item[3] for item in pair_plan.placements) + (third_placement,), tuple(pair_cycles) + (third_cycles,))
                        continue
                    sequential_cycles = sum(baseline_cycles[execution_id] for execution_id in pair_plan.kernel_ids + (third_id,))
                    makespan_cycles = max(*pair_cycles, third_cycles)
                    benefit = (sequential_cycles - makespan_cycles) / sequential_cycles
                    third_memory_utilization = self._estimate_memory_utilization(runtime_context, third_stats.kernel_desc, third_placement, third_domain, third_cycles)
                    peak_dma_utilization = sum(pair_memory_utilizations) + third_memory_utilization
                    if benefit <= 0 or peak_dma_utilization > 1.0 + 1e-9:
                        reason = "nonpositive_benefit" if benefit <= 0 else "dma_utilization_exceeded"
                        record("triple", reason, pair_plan.kernel_ids + (third_id,), tuple(item[3] for item in pair_plan.placements) + (third_placement,), tuple(pair_cycles) + (third_cycles,), sequential_cycles, makespan_cycles, benefit, peak_dma_utilization)
                        continue
                    core_count = sum(placement.ccg_tile_mesh.size for _, _, _, placement in pair_plan.placements) + third_placement.ccg_tile_mesh.size
                    plans.append(MeshPlanCandidate(plan_id, pair_plan.kernel_ids + (third_id,), pair_plan.workload_ids + (third_state.workload_id,), makespan_cycles, sequential_cycles, core_count / runtime_context.device.get_ccg_tile_mesh().size, benefit, peak_dma_utilization, pair_plan.placements + ((third_action, third_state, third_domain, third_placement),)))
                    record("triple", "eligible", pair_plan.kernel_ids + (third_id,), tuple(item[3] for item in pair_plan.placements) + (third_placement,), tuple(pair_cycles) + (third_cycles,), sequential_cycles, makespan_cycles, benefit, peak_dma_utilization, plan_id)
                    plan_id += 1
            self._restore_placement_state(runtime_context, action_map, persistent_state_placements, base_snapshot)

        self._restore_placement_state(runtime_context, action_map, persistent_state_placements, base_snapshot)
        candidates = tuple(plans)
        selected_plan_id = self.select_plan(context, candidates)
        selected = next((candidate for candidate in candidates if candidate.plan_id == selected_plan_id), None)
        if trace is not None:
            for item in trace:
                item["selection_reason"] = self._decision_metadata.get("selection_reason")
                if item["reason"] == "eligible" and item["candidate_id"] != selected_plan_id:
                    item["reason"] = "not_selected_by_policy"
        if selected is None:
            if trace is not None:
                self.decision_hook(context, tuple(trace))
            return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        commit_snapshot = self._snapshot_placement_state(runtime_context, action_map, persistent_state_placements)
        for action, workload_state, _, placement in selected.placements:
            kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
            if not self._commit_kernel_plan(runtime_context, kernel_stats, placement, persistent_state_placements):
                self._restore_placement_state(runtime_context, action_map, persistent_state_placements, commit_snapshot)
                record("selection", "selected_placement_failed", selected.kernel_ids, tuple(item[3] for item in selected.placements), candidate_id=selected.plan_id)
                if trace is not None:
                    self.decision_hook(context, tuple(trace))
                return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        scheduled_ids = set(selected.kernel_ids)
        scheduled = []
        suspended = []
        for execution_id, (action, workload_state, domain, key) in action_map.items():
            if execution_id in scheduled_ids:
                scheduled.append((action, workload_state))
                self._ready_cycles.pop(key, None)
                self._submission_indices.pop(key, None)
                self._on_action_scheduled(workload_state.workload_id)
            else:
                suspended.append((action, workload_state, domain))
        self.on_dispatch(context, selected)
        self._decision_metadata = {**self._decision_metadata, "benefit": selected.benefit, "selected_kernel_ids": selected.kernel_ids}
        if trace is not None:
            for item in trace:
                if item["candidate_id"] == selected.plan_id and item["reason"] == "eligible":
                    item["reason"] = "selected"
            self.decision_hook(context, tuple(trace))
        return scheduled, suspended

    def _get_shape_plans(self, runtime_context, workload, kernel_stats, persistent_state_placements, domain):
        if kernel_stats.is_placed:
            ccg_tile_mesh = kernel_stats.ccg_tile_mesh
            ccg_ids = ccg_tile_mesh.flatten().tolist()
            domain_ids = set(domain.ccg_tile_mesh.flatten().tolist())
            if any(ccg_id not in runtime_context.ccg_kernel_vacancy or ccg_id not in domain_ids for ccg_id in ccg_ids):
                raise ValueError("The pre-placed kernel mesh contains CCG IDs outside its scheduling domain.")
            if not self._supports_mesh_shape(kernel_stats.kernel_desc, ccg_tile_mesh.shape, allow_idle_cores=True):
                raise ValueError(f"Kernel type {kernel_stats.kernel_desc.kernel_type.name} does not support pre-placed mesh shape {ccg_tile_mesh.shape}.")
            if any(not runtime_context.ccg_kernel_vacancy[ccg_id] for ccg_id in ccg_ids):
                return []
            placement_plan = self._plan_tensor_placements(runtime_context, workload, kernel_stats.kernel_desc, ccg_tile_mesh, persistent_state_placements, domain)
            if placement_plan is None:
                return []
            tensor_placements, allocated_banks, persistent_placements = placement_plan
            tensor_placements = list(tensor_placements)
            planned_stats = {id(tensor_stats) for tensor_stats, _ in tensor_placements}
            for tensor_desc in kernel_stats.kernel_desc.input_tensors + kernel_stats.kernel_desc.output_tensors:
                tensor_stats = workload.get_tensor_stats(tensor_desc)
                if tensor_stats.is_placed and id(tensor_stats) not in planned_stats:
                    tensor_placements.append((tensor_stats, tensor_stats.tile_placement))
                    planned_stats.add(id(tensor_stats))
            tensor_placements = tuple(tensor_placements)
            placement_map = {id(tensor_stats.tensor_desc): placement for tensor_stats, placement in tensor_placements}
            arithmetic_intensity = self.get_ai_with_shape(kernel_stats.kernel_desc, ccg_tile_mesh.shape, placement_map)
            placement = MeshKernelAIStats(arithmetic_intensity, ccg_tile_mesh.copy(), tensor_placements, allocated_banks, persistent_placements)
            return [(placement, self._estimate_kernel_cycles(runtime_context, kernel_stats.kernel_desc, placement, domain))]
        ai_stats = self.get_ai_stats(runtime_context, workload, kernel_stats, tuple(domain.ccg_tile_mesh.shape), persistent_state_placements, domain)
        per_core_flops = runtime_context.total_flops / len(runtime_context.ccg_tile_ids)
        domain_bandwidth = runtime_context.total_mem_bandwidth * len(domain.dma_ids) / len(runtime_context.dma_tile_ids)
        plans = []
        for mesh_shape, placement in ai_stats.items():
            core_count = math.prod(mesh_shape)
            intensity = placement.arithmetic_intensity
            if kernel_stats.kernel_desc.kernel_type == MeshKernelType.MEMCOPY and intensity == 0 and core_count <= len(domain.dma_ids):
                plans.append((placement, self._estimate_kernel_cycles(runtime_context, kernel_stats.kernel_desc, placement, domain)))
            elif intensity > 0 and (math.isinf(intensity) or core_count <= max(1, math.ceil(domain_bandwidth * intensity / per_core_flops))):
                plans.append((placement, self._estimate_kernel_cycles(runtime_context, kernel_stats.kernel_desc, placement, domain)))
        if not plans and (1, 1) in ai_stats:
            placement = ai_stats[(1, 1)]
            plans.append((placement, self._estimate_kernel_cycles(runtime_context, kernel_stats.kernel_desc, placement, domain)))
        return sorted(plans, key=lambda item: (item[1], -item[0].ccg_tile_mesh.size, tuple(item[0].ccg_tile_mesh.flatten())))

    def _get_singleton_plans(self, runtime_context, workload, kernel_stats, persistent_state_placements, domain):
        max_cores = self._get_candidate_core_limit(domain)
        plans = (item for item in self._get_shape_plans(runtime_context, workload, kernel_stats, persistent_state_placements, domain) if item[0].ccg_tile_mesh.size <= max_cores)
        return sorted(plans, key=lambda item: (item[0].ccg_tile_mesh.size, tuple(item[0].ccg_tile_mesh.shape)), reverse=True)

    @staticmethod
    def _diverse_shape_plans(plans, domain_core_count):
        selected = list(plans[:3])
        selected.extend(item for item in plans if item[0].ccg_tile_mesh.size <= max(1, domain_core_count // 2))
        selected = selected[:6]
        unique = {}
        for item in selected:
            unique.setdefault((item[0].ccg_tile_mesh.shape, tuple(item[0].ccg_tile_mesh.flatten())), item)
        return tuple(unique.values())

    def _estimate_kernel_cycles(self, runtime_context, kernel_desc, placement, domain) -> int:
        total_ops = self._get_total_ops(kernel_desc)
        core_count = placement.ccg_tile_mesh.size
        per_core_flops = runtime_context.total_flops / len(runtime_context.ccg_tile_ids)
        compute_cycles = total_ops / max(1.0, per_core_flops * core_count)
        memory_traffic = self._estimate_memory_traffic(kernel_desc, placement, total_ops)
        bandwidth = runtime_context.total_mem_bandwidth * len(domain.dma_ids) / len(runtime_context.dma_tile_ids)
        noc_cycles = self._estimate_noc_cycles(runtime_context, kernel_desc, placement) if self._use_joint_planning() else 0
        return max(1, math.ceil(max(compute_cycles, memory_traffic / max(bandwidth, 1e-9), noc_cycles)))

    def _estimate_memcopy_noc_cycles(self, runtime_context, kernel_desc, placement) -> int:
        tensor_placements = {id(stats.tensor_desc): tiles for stats, tiles in placement.tensor_placements}
        metadata_count = kernel_desc.get_required_kwargs("metadata_input_count")
        has_source = len(kernel_desc.input_tensors) > metadata_count
        source = kernel_desc.input_tensors[0] if has_source else None
        destination = kernel_desc.output_tensors[0] if kernel_desc.output_tensors else None
        source_coords = list(np.ndindex(*source.tile_grid_shape)) if source is not None else []
        destination_coords = list(np.ndindex(*destination.tile_grid_shape)) if destination is not None else []
        gather = kernel_desc.get_required_kwargs("gather")
        count = len(destination_coords) if gather else max(len(source_coords), len(destination_coords))
        source_bytes = kernel_desc.get_required_kwargs("src_traffic_bytes")
        destination_bytes = kernel_desc.get_required_kwargs("dst_traffic_bytes")
        source_sizes = [min(source.get_tile_size(), max(0, source_bytes - index * source.get_tile_size())) for index in range(count)] if source is not None else []
        destination_sizes = [min(destination.get_tile_size(), max(0, destination_bytes - index * destination.get_tile_size())) for index in range(count)] if destination is not None else []
        if source_sizes and sum(source_sizes) < source_bytes:
            source_sizes[-1] += source_bytes - sum(source_sizes)
        if destination_sizes and sum(destination_sizes) < destination_bytes:
            destination_sizes[-1] += destination_bytes - sum(destination_sizes)
        core_ids = tuple(int(core_id) for core_id in placement.ccg_tile_mesh.flat)
        mem_context = runtime_context.device.mem_context
        transfers = {}

        def add_bank_transfer(bank, core_id, size, is_source):
            if bank is None or size <= 0:
                return
            other_id = mem_context.get_dma_id_with_address(bank.addr) if bank.mem_type == MeshMemoryType.DEVICE_MEMORY else bank.owner_id
            src, dst = (other_id, core_id) if is_source else (core_id, other_id)
            if src != dst:
                transfers[src, dst] = transfers.get((src, dst), 0) + size

        for index in range(count):
            source_coord = source_coords[(index * 1315423911) % len(source_coords)] if gather else source_coords[min(len(source_coords) - 1, index * len(source_coords) // count)] if source_coords else None
            destination_coord = destination_coords[min(len(destination_coords) - 1, index * len(destination_coords) // count)] if destination_coords else None
            source_bank = tensor_placements[id(source)][source_coord] if source_coord is not None else None
            destination_bank = tensor_placements[id(destination)][destination_coord] if destination_coord is not None else None
            local_bank = next((bank for bank in (destination_bank, source_bank) if bank is not None and bank.mem_type == MeshMemoryType.LOCAL_CACHE and bank.owner_id in core_ids), None)
            core_id = int(local_bank.owner_id) if local_bank is not None else core_ids[index % len(core_ids)]
            add_bank_transfer(source_bank, core_id, source_sizes[index] if source_sizes else 0, True)
            add_bank_transfer(destination_bank, core_id, destination_sizes[index] if destination_sizes else 0, False)
        for metadata_index, tensor in enumerate(kernel_desc.input_tensors[int(has_source):]):
            core_id = core_ids[metadata_index % len(core_ids)]
            remaining = kernel_desc.get_required_kwargs("metadata_traffic_bytes")[metadata_index]
            for coord in np.ndindex(*tensor.tile_grid_shape):
                size = min(tensor.get_tile_size(), remaining)
                remaining -= size
                if size <= 0:
                    break
                add_bank_transfer(tensor_placements[id(tensor)][coord], core_id, size, True)
        return self._estimate_noc_transfer_cycles(runtime_context, transfers)

    def _estimate_noc_cycles(self, runtime_context, kernel_desc, placement) -> int:
        kind = kernel_desc.kernel_type
        if kind == MeshKernelType.MEMCOPY:
            return self._estimate_memcopy_noc_cycles(runtime_context, kernel_desc, placement)
        helpers = {MeshKernelType.LINEAR: create_linear_mapping_requistes, MeshKernelType.CONV2D: create_conv2d_mapping_requistes, MeshKernelType.ELEMENTWISE: create_elementwise_mapping_requistes, MeshKernelType.REDUCTION: create_reduction_mapping_requistes, MeshKernelType.SDPA: create_sdpa_mapping_requistes}
        if kind not in helpers:
            return 0
        cache_key = id(kernel_desc)
        if cache_key not in self._noc_ops_cache:
            _, _, ops = helpers[kind](kernel_desc)
            output_coords = tuple(ops)
            self._noc_ops_cache[cache_key] = ops, output_coords
        ops, output_coords = self._noc_ops_cache[cache_key]
        mesh = placement.ccg_tile_mesh
        mesh_h, mesh_w = mesh.shape
        n_cores = mesh.size
        tensor_placements = {id(stats.tensor_desc): tiles for stats, tiles in placement.tensor_placements}
        inputs = kernel_desc.input_tensors
        output = kernel_desc.output_tensors[0]
        # Resolve view coordinates once per estimate. Ordinary tensors already use
        # their own tile coordinates as storage coordinates.
        storage_keys = {
            id(tensor): (tensor.storage_id, {} if tensor.is_view else None)
            for tensor in inputs
        }
        if kind == MeshKernelType.LINEAR:
            ifm_bank = next(iter(tensor_placements[id(inputs[0])].values()))
            wgt_bank = next(iter(tensor_placements[id(inputs[1])].values()))
            shared_axis = -2 if ifm_bank.mem_type == MeshMemoryType.DEVICE_MEMORY and wgt_bank.mem_type != MeshMemoryType.DEVICE_MEMORY else -1
            output_coords = tuple(sorted(output_coords, key=lambda coord: (coord[:-2], coord[shared_axis], coord[-1 if shared_axis == -2 else -2])))
        base, remainder = divmod(len(output_coords), n_cores)
        counts = [0] * n_cores
        linear_owner = 0
        width = output.tile_grid_shape[-1]
        transfers = {}
        seen_inputs = [set() for _ in range(n_cores)]
        mem_context = runtime_context.device.mem_context

        def add_transfer(src, dst, size):
            if src != dst:
                transfers[src, dst] = transfers.get((src, dst), 0) + size

        for index, output_coord in enumerate(output_coords):
            if kind == MeshKernelType.LINEAR:
                while counts[linear_owner] >= base + int(linear_owner < remainder):
                    linear_owner += 1
                owner_index = linear_owner
            else:
                preferred = (index // width % mesh_h) * mesh_w + index % width % mesh_w
                owner_index = next((candidate for offset in range(n_cores) if counts[candidate := (preferred + offset) % n_cores] < base + int(candidate < remainder)), None)
                if owner_index is None:
                    raise RuntimeError("No output tile owner found for NoC estimation.")
            counts[owner_index] += 1
            core_id = int(mesh.flat[owner_index])
            for tensor_id, coords in ops[output_coord].items():
                tensor = inputs[tensor_id]
                tiles = tensor_placements[id(tensor)]
                storage_id, view_coords = storage_keys[id(tensor)]
                for coord in coords:
                    coord = tuple(coord)
                    if view_coords is None:
                        storage_coord = coord
                    else:
                        storage_coord = view_coords.get(coord)
                        if storage_coord is None:
                            storage_coord = tensor.map_tile_coord_to_storage(coord)
                            view_coords[coord] = storage_coord
                    storage_key = (storage_id, storage_coord)
                    if storage_key in seen_inputs[owner_index]:
                        continue
                    seen_inputs[owner_index].add(storage_key)
                    bank = tiles[coord]
                    src = mem_context.get_dma_id_with_address(bank.addr) if bank.mem_type == MeshMemoryType.DEVICE_MEMORY else bank.owner_id
                    add_transfer(core_id if bank.mem_type == MeshMemoryType.DEVICE_MEMORY else src, src if bank.mem_type == MeshMemoryType.DEVICE_MEMORY else core_id, tensor.get_tile_size())
            bank = tensor_placements[id(output)][output_coord]
            dst = mem_context.get_dma_id_with_address(bank.addr) if bank.mem_type == MeshMemoryType.DEVICE_MEMORY else bank.owner_id
            add_transfer(core_id, dst, output.get_tile_size())
        return self._estimate_noc_transfer_cycles(runtime_context, transfers)

    @staticmethod
    def _estimate_noc_transfer_cycles(runtime_context, transfers) -> int:
        if not transfers:
            return 0
        icnt = runtime_context.device.icnt_context
        config = icnt.config
        link_flits = {}
        injection_flits = {}
        egress_flits = {}
        longest_path = 0
        for (src, dst), size in transfers.items():
            src_coord, dst_coord = icnt.core_id_to_coord(src), icnt.core_id_to_coord(dst)
            src_node = src_coord[0] * config.shape[1] + src_coord[1]
            dst_node = dst_coord[0] * config.shape[1] + dst_coord[1]
            flits = math.ceil(size / config.flit_size)
            payload_count = math.ceil(flits / config.max_payload_size)
            hops = abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
            longest_path = max(longest_path, config.lightweight_packet_startup_cycles + (payload_count - 1) * config.lightweight_payload_issue_gap_cycles + hops * (config.lightweight_router_allocation_cycles + config.lightweight_router_latency_cycles + config.lightweight_link_latency_cycles) + 2 * math.ceil(min(flits, config.max_payload_size) / config.lightweight_flits_per_cycle_per_channel))
            for payload in range(payload_count):
                subnet = (src_node + dst_node + payload) % config.subnets
                payload_flits = min(config.max_payload_size, flits - payload * config.max_payload_size)
                injection_flits[src_coord, subnet] = injection_flits.get((src_coord, subnet), 0) + payload_flits
                egress_flits[dst_coord, subnet] = egress_flits.get((dst_coord, subnet), 0) + payload_flits
                y, x = src_coord
                while x != dst_coord[1]:
                    next_x = x + (1 if dst_coord[1] > x else -1)
                    endpoints = ((y, x), (y, next_x))
                    key = (*sorted(endpoints), subnet) if config.lightweight_channel_mode == "bidirectional_shared" else (*endpoints, subnet)
                    link_flits[key] = link_flits.get(key, 0) + payload_flits
                    x = next_x
                while y != dst_coord[0]:
                    next_y = y + (1 if dst_coord[0] > y else -1)
                    endpoints = ((y, x), (next_y, x))
                    key = (*sorted(endpoints), subnet) if config.lightweight_channel_mode == "bidirectional_shared" else (*endpoints, subnet)
                    link_flits[key] = link_flits.get(key, 0) + payload_flits
                    y = next_y
        link_cycles = max((math.ceil(flits / config.lightweight_flits_per_cycle_per_channel) for flits in link_flits.values()), default=0)
        injection_cycles = max((math.ceil(flits / config.lightweight_injection_flits_per_cycle) for flits in injection_flits.values()), default=0)
        egress_cycles = max((math.ceil(flits / config.lightweight_egress_flits_per_cycle) for flits in egress_flits.values()), default=0)
        return max(longest_path, link_cycles, injection_cycles, egress_cycles)

    def _estimate_memory_utilization(self, runtime_context, kernel_desc, placement, domain, cycles) -> float:
        traffic = self._estimate_memory_traffic(kernel_desc, placement, self._get_total_ops(kernel_desc))
        bandwidth = runtime_context.total_mem_bandwidth * len(domain.dma_ids) / len(runtime_context.dma_tile_ids)
        return min(1.0, traffic / max(1.0, cycles * bandwidth))

    @staticmethod
    def _estimate_memory_traffic(kernel_desc, placement, total_ops) -> float:
        if math.isinf(placement.arithmetic_intensity):
            return 0.0
        if placement.arithmetic_intensity > 0 and total_ops > 0:
            return total_ops / placement.arithmetic_intensity
        placement_map = {id(stats.tensor_desc): tile_placement for stats, tile_placement in placement.tensor_placements}
        return sum(tensor.get_size() for tensor in kernel_desc.input_tensors + kernel_desc.output_tensors if any(bank.mem_type == MeshMemoryType.DEVICE_MEMORY for bank in placement_map.get(id(tensor), {}).values()))

    @staticmethod
    def _get_total_ops(kernel_desc) -> float:
        if kernel_desc.kernel_type == MeshKernelType.LINEAR:
            ifm = kernel_desc.input_tensors[0]
            ofm = kernel_desc.output_tensors[0]
            reduction = ifm.shape[-2] if kernel_desc.get_kwargs("transpose_ifm", default=False) else ifm.shape[-1]
            return ofm.get_numel() * (2 * reduction + int(kernel_desc.base_input_count == 3) + kernel_desc.get_kwargs("extra_ops_per_output_element", default=0.0))
        if kernel_desc.kernel_type == MeshKernelType.CONV2D:
            operation, kernel_size = kernel_desc.get_required_kwargs("operation", "kernel_size")
            output_elements = kernel_desc.output_tensors[0].get_numel()
            extra_ops = kernel_desc.get_kwargs("extra_ops_per_output_element", default=0.0)
            if operation in ("max_pool2d", "min_pool2d", "avg_pool2d"):
                window = math.prod(kernel_size)
                return output_elements * ((window if operation == "avg_pool2d" else max(1, window - 1)) + extra_ops)
            weight = kernel_desc.input_tensors[1]
            return output_elements * (2 * weight.shape[0] * weight.shape[1] * weight.shape[3] + int(kernel_desc.base_input_count == 3) + extra_ops)
        if kernel_desc.kernel_type == MeshKernelType.ELEMENTWISE:
            return kernel_desc.output_tensors[0].get_numel() * kernel_desc.get_required_kwargs("ops_per_element")
        if kernel_desc.kernel_type == MeshKernelType.REDUCTION:
            return sum(tensor.get_numel() for tensor in kernel_desc.input_tensors) * kernel_desc.get_required_kwargs("ops_per_input_element") + sum(tensor.get_numel() for tensor in kernel_desc.output_tensors) * kernel_desc.get_kwargs("extra_ops_per_output_element", default=0.0)
        if kernel_desc.kernel_type == MeshKernelType.SDPA:
            q, k = kernel_desc.input_tensors[:2]
            batch_size, n_q_heads, q_length, head_dim = q.shape
            kv_length = k.shape[-2]
            if kernel_desc.get_required_kwargs("is_causal"):
                past_length = kv_length - q_length
                score_elements = sum(min(kv_length, past_length + index + 1) for index in range(q_length))
            else:
                score_elements = q_length * kv_length
            return batch_size * n_q_heads * score_elements * (4 * head_dim + kernel_desc.get_required_kwargs("softmax_ops_per_score"))
        return 0.0

    @staticmethod
    def _snapshot_placement_state(runtime_context, action_map, persistent_state_placements):
        tensor_stats = set(persistent_state_placements.values())
        kernel_stats = set()
        for action, workload_state, _, _ in action_map.values():
            workload = workload_state.compiled_workload
            stats = workload.get_kernel_stats(action.kernel_id)
            kernel_stats.add(stats)
            for tensor_desc in stats.kernel_desc.input_tensors + stats.kernel_desc.output_tensors:
                tensor_stats.add(workload.get_tensor_stats(tensor_desc))
                storage_stats = next((candidate for candidate in workload.tensor_stats_map.values() if candidate.tensor_desc is tensor_desc.storage_desc), None)
                if storage_stats is not None:
                    tensor_stats.add(storage_stats)
        return {"device": {owner_id: list(ranges) for owner_id, ranges in runtime_context.device_memory_vacancy.items()}, "local": {owner_id: list(ranges) for owner_id, ranges in runtime_context.local_cache_vacancy.items()}, "ccg": dict(runtime_context.ccg_kernel_vacancy), "tensors": {stats: dict(stats.tile_placement) if stats.is_placed else None for stats in tensor_stats}, "kernels": {stats: stats.ccg_tile_mesh.copy() if stats.is_placed else None for stats in kernel_stats}, "persistent": dict(persistent_state_placements)}

    @staticmethod
    def _restore_placement_state(runtime_context, action_map, persistent_state_placements, snapshot):
        current_persistent_stats = set(persistent_state_placements.values())
        runtime_context.device_memory_vacancy.clear()
        runtime_context.device_memory_vacancy.update({owner_id: list(ranges) for owner_id, ranges in snapshot["device"].items()})
        runtime_context.local_cache_vacancy.clear()
        runtime_context.local_cache_vacancy.update({owner_id: list(ranges) for owner_id, ranges in snapshot["local"].items()})
        runtime_context.ccg_kernel_vacancy.clear()
        runtime_context.ccg_kernel_vacancy.update(snapshot["ccg"])
        for stats in set(snapshot["tensors"]) | current_persistent_stats:
            stats.unplace()
        for stats, placement in snapshot["tensors"].items():
            if placement is not None:
                stats.place(dict(placement))
        for stats, placement in snapshot["kernels"].items():
            stats.unplace()
            if placement is not None:
                stats.place(placement.copy())
        persistent_state_placements.clear()
        persistent_state_placements.update(snapshot["persistent"])

    def get_ai_stats(self, runtime_context, workload, kernel_stats, ccg_mesh_shape: tuple[int, int], persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats] | None = None, domain: MeshSchedulingDomain | None = None) -> dict[tuple[int, int], MeshKernelAIStats]:
        persistent_state_placements = {} if persistent_state_placements is None else persistent_state_placements
        domain = self._full_device_domain(runtime_context) if domain is None else domain
        ccg_mesh_h, ccg_mesh_w = ccg_mesh_shape
        close_to = self._get_close_to(runtime_context, workload, kernel_stats)
        stats = {}

        for h in range(1, ccg_mesh_h + 1):
            for w in range(1, ccg_mesh_w + 1):
                mesh_shape = (h, w)
                if not self._supports_mesh_shape(kernel_stats.kernel_desc, mesh_shape):
                    continue
                ccg_tile_mesh = self._find_ccg_mesh(runtime_context, domain, mesh_shape, close_to)
                if ccg_tile_mesh is None:
                    continue
                placement_plan = self._plan_tensor_placements(runtime_context, workload, kernel_stats.kernel_desc, ccg_tile_mesh, persistent_state_placements, domain)
                if placement_plan is None:
                    continue
                tensor_placements, allocated_banks, persistent_placements = placement_plan
                tensor_placements = list(tensor_placements)
                planned_stats = {id(tensor_stats) for tensor_stats, _ in tensor_placements}
                for tensor_desc in kernel_stats.kernel_desc.input_tensors + kernel_stats.kernel_desc.output_tensors:
                    tensor_stats = workload.get_tensor_stats(tensor_desc)
                    if tensor_stats.is_placed and id(tensor_stats) not in planned_stats:
                        tensor_placements.append((tensor_stats, tensor_stats.tile_placement))
                        planned_stats.add(id(tensor_stats))
                tensor_placements = tuple(tensor_placements)
                placement_map = {id(tensor_stats.tensor_desc): placement for tensor_stats, placement in tensor_placements}
                arithmetic_intensity = self.get_ai_with_shape(kernel_stats.kernel_desc, mesh_shape, placement_map)
                stats[mesh_shape] = MeshKernelAIStats(arithmetic_intensity, ccg_tile_mesh.copy(), tensor_placements, allocated_banks, persistent_placements)

        return stats

    def get_ai_with_shape(self, kernel_desc: MeshKernelDescriptor, ccg_mesh_shape: tuple[int, int], tensor_placements: dict[int, dict] | None = None) -> float:
        ccg_mesh_h, ccg_mesh_w = ccg_mesh_shape
        tensor_placements = {} if tensor_placements is None else tensor_placements

        def uses_device_memory(tensor_desc):
            placement = tensor_placements.get(id(tensor_desc))
            return placement is None or any(bank.mem_type == MeshMemoryType.DEVICE_MEMORY for bank in placement.values())

        def traffic(tensor_desc):
            return tensor_desc.get_size() if uses_device_memory(tensor_desc) else 0

        def intensity(total_flops, total_traffic):
            return math.inf if total_traffic == 0 else total_flops / total_traffic

        if kernel_desc.kernel_type == MeshKernelType.LINEAR:
            ifm = kernel_desc.input_tensors[0]
            ofm = kernel_desc.output_tensors[0]
            K = ifm.shape[-2] if kernel_desc.get_kwargs("transpose_ifm", default=False) else ifm.shape[-1]
            extra_ops = kernel_desc.get_kwargs("extra_ops_per_output_element", default=0.0)
            total_flops = ofm.get_numel() * (2 * K + int(kernel_desc.base_input_count == 3) + extra_ops)
            total_traffic = sum(traffic(tensor) for tensor in kernel_desc.input_tensors) + traffic(ofm)
            return intensity(total_flops, total_traffic)

        if kernel_desc.kernel_type == MeshKernelType.CONV2D:
            operation, kernel_size, extra_ops_per_output_element = kernel_desc.get_required_kwargs("operation", "kernel_size", "extra_ops_per_output_element")
            ofm = kernel_desc.output_tensors[0]
            core_count = ccg_mesh_h * ccg_mesh_w
            spatial_tiles = math.prod(ofm.tile_grid_shape[:-1])
            spatial_partitions = min(core_count, spatial_tiles)
            channel_partitions = math.ceil(core_count / spatial_partitions)
            output_elements = ofm.get_numel()
            extra_traffic = sum(traffic(tensor) for tensor in kernel_desc.input_tensors[kernel_desc.base_input_count:])

            if operation in ("max_pool2d", "min_pool2d", "avg_pool2d"):
                window_elements = math.prod(kernel_size)
                ops_per_output_element = window_elements if operation == "avg_pool2d" else max(1, window_elements - 1)
                total_flops = output_elements * (ops_per_output_element + extra_ops_per_output_element)
                total_traffic = traffic(kernel_desc.input_tensors[0]) * channel_partitions + extra_traffic + traffic(ofm)
                return intensity(total_flops, total_traffic)

            wgt = kernel_desc.input_tensors[1]
            reduction_elements = wgt.shape[0] * wgt.shape[1] * wgt.shape[3]
            has_bias = kernel_desc.base_input_count == 3
            total_flops = output_elements * (2 * reduction_elements + int(has_bias) + extra_ops_per_output_element)
            total_traffic = traffic(kernel_desc.input_tensors[0]) * channel_partitions + traffic(wgt) * spatial_partitions + traffic(ofm) + extra_traffic
            if has_bias:
                total_traffic += traffic(kernel_desc.input_tensors[2]) * spatial_partitions
            return intensity(total_flops, total_traffic)

        if kernel_desc.kernel_type == MeshKernelType.ELEMENTWISE:
            total_flops = kernel_desc.output_tensors[0].get_numel() * kernel_desc.get_required_kwargs("ops_per_element")
            total_traffic = sum(traffic(tensor) for tensor in kernel_desc.input_tensors + kernel_desc.output_tensors)
            return intensity(total_flops, total_traffic)

        if kernel_desc.kernel_type == MeshKernelType.MEMCOPY:
            return 0.0

        if kernel_desc.kernel_type == MeshKernelType.REDUCTION:
            ops_per_input_element = kernel_desc.get_required_kwargs("ops_per_input_element")
            extra_ops_per_output_element = kernel_desc.get_kwargs("extra_ops_per_output_element", default=0.0)
            total_flops = sum(tensor.get_numel() for tensor in kernel_desc.input_tensors) * ops_per_input_element + sum(tensor.get_numel() for tensor in kernel_desc.output_tensors) * extra_ops_per_output_element
            total_traffic = sum(traffic(tensor) for tensor in kernel_desc.input_tensors + kernel_desc.output_tensors)
            return intensity(total_flops, total_traffic)

        if kernel_desc.kernel_type == MeshKernelType.SDPA:
            q, k, v = kernel_desc.input_tensors[:3]
            ofm = kernel_desc.output_tensors[0]
            mask = kernel_desc.input_tensors[3] if len(kernel_desc.input_tensors) == 4 else None
            is_causal, q_chunk_size, kv_chunk_size, max_cores_per_head, head_group_size, softmax_ops_per_score = kernel_desc.get_required_kwargs("is_causal", "q_chunk_size", "kv_chunk_size", "max_cores_per_head", "head_group_size", "softmax_ops_per_score")
            batch_size, n_q_heads, q_length, head_dim = q.shape
            _, n_kv_heads, kv_length, _ = k.shape
            q_chunks = tuple((start, min(start + q_chunk_size, q_length)) for start in range(0, q_length, q_chunk_size))
            kv_chunks = tuple((start, min(start + kv_chunk_size, kv_length)) for start in range(0, kv_length, kv_chunk_size))
            task_order = tuple((batch, q_head, q_chunk_index) for batch in range(batch_size) for q_chunk_index in range(len(q_chunks)) for kv_head in range(n_kv_heads) for q_head in range(kv_head * head_group_size, (kv_head + 1) * head_group_size))
            core_count = ccg_mesh_h * ccg_mesh_w
            cores_per_head = min(max_cores_per_head, len(kv_chunks), max(1, core_count // len(task_order)))
            tasks_per_wave = max(1, core_count // cores_per_head)
            waves = tuple(task_order[start:start + tasks_per_wave] for start in range(0, len(task_order), tasks_per_wave))
            past_length = kv_length - q_length

            def get_score_elements(task, kv_chunk):
                q_start, q_end = q_chunks[task[2]]
                kv_start, kv_end = kv_chunk
                if not is_causal:
                    return (q_end - q_start) * (kv_end - kv_start)
                partial_start = max(q_start, kv_start - past_length)
                full_start = max(partial_start, kv_end - past_length - 1)
                partial_end = min(q_end, full_start)
                partial_count = max(0, partial_end - partial_start)
                partial_pairs = partial_count * (past_length + 1 - kv_start) + (partial_start + partial_end - 1) * partial_count // 2
                full_count = max(0, q_end - max(q_start, full_start))
                return partial_pairs + full_count * (kv_end - kv_start)

            total_flops = 0
            total_kv_elements = 0
            total_mask_traffic = 0
            for wave in waves:
                for kv_chunk in kv_chunks:
                    active_tasks = tuple(task for task in wave if get_score_elements(task, kv_chunk) > 0)
                    total_flops += sum(get_score_elements(task, kv_chunk) for task in active_tasks) * (4 * head_dim + softmax_ops_per_score)
                    active_kv_groups = {(task[0], task[1] // head_group_size) for task in active_tasks}
                    total_kv_elements += len(active_kv_groups) * (kv_chunk[1] - kv_chunk[0]) * head_dim
                    if mask is not None and uses_device_memory(mask):
                        total_mask_traffic += sum(math.prod(1 if mask_dim == 1 else target_dim for mask_dim, target_dim in zip(mask.shape, (1, 1, q_chunks[task[2]][1] - q_chunks[task[2]][0], kv_chunk[1] - kv_chunk[0])[-len(mask.shape):])) for task in active_tasks)

            total_traffic = traffic(q) + total_kv_elements * (k.dtype.itemsize * uses_device_memory(k) + v.dtype.itemsize * uses_device_memory(v)) + total_mask_traffic * (mask.dtype.itemsize if mask is not None else 0) + traffic(ofm)
            return intensity(total_flops, total_traffic)

        raise ValueError(f"Unsupported kernel type: {kernel_desc.kernel_type}")

    def _place_kernel(self, runtime_context, workload, kernel_stats, persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats], domain: MeshSchedulingDomain | None = None) -> bool:
        domain = self._full_device_domain(runtime_context) if domain is None else domain
        for placement, _ in self._get_singleton_plans(runtime_context, workload, kernel_stats, persistent_state_placements, domain):
            if self._commit_kernel_plan(runtime_context, kernel_stats, placement, persistent_state_placements):
                return True
        return False

    def _get_candidate_core_limit(self, domain: MeshSchedulingDomain) -> int:
        return int(domain.ccg_tile_mesh.size)

    @staticmethod
    def _full_device_domain(runtime_context) -> MeshSchedulingDomain:
        return MeshSchedulingDomain("device", runtime_context.device.get_ccg_tile_mesh(), tuple(runtime_context.dma_tile_ids))

    @staticmethod
    def _find_ccg_mesh(runtime_context, domain: MeshSchedulingDomain, mesh_shape: tuple[int, int], close_to: list[int]) -> np.ndarray | None:
        mesh_h, mesh_w = mesh_shape
        domain_mesh = domain.ccg_tile_mesh
        if mesh_h > domain_mesh.shape[0] or mesh_w > domain_mesh.shape[1]:
            return None
        close_coords = tuple(runtime_context.device.icnt_context.core_id_to_coord(core_id) for core_id in close_to)
        candidates = []
        for y in range(domain_mesh.shape[0] - mesh_h + 1):
            for x in range(domain_mesh.shape[1] - mesh_w + 1):
                submesh = domain_mesh[y:y + mesh_h, x:x + mesh_w]
                core_ids = submesh.flatten().tolist()
                if any(core_id not in runtime_context.ccg_kernel_vacancy or not runtime_context.ccg_kernel_vacancy[core_id] for core_id in core_ids):
                    continue
                distance = sum(abs(cy - ty) + abs(cx - tx) for core_id in core_ids for cy, cx in (runtime_context.device.icnt_context.core_id_to_coord(core_id),) for ty, tx in close_coords)
                candidates.append((distance, y, x, submesh.copy()))
        return min(candidates, key=lambda item: item[:3])[3] if candidates else None

    @staticmethod
    def _supports_mesh_shape(kernel_desc: MeshKernelDescriptor, mesh_shape: tuple[int, int], allow_idle_cores: bool = False) -> bool:
        core_count = math.prod(mesh_shape)
        if allow_idle_cores:
            return kernel_desc.kernel_type != MeshKernelType.MEMCOPY or mesh_shape == (1, 1)
        if kernel_desc.kernel_type == MeshKernelType.MEMCOPY:
            return mesh_shape == (1, 1)
        if kernel_desc.kernel_type in (MeshKernelType.LINEAR, MeshKernelType.CONV2D, MeshKernelType.ELEMENTWISE, MeshKernelType.REDUCTION):
            output_tiles = max((tensor.get_n_tiles() for tensor in kernel_desc.output_tensors), default=1)
            return core_count <= output_tiles
        return True

    def _plan_tensor_placements(self, runtime_context, workload, kernel_desc, ccg_tile_mesh, persistent_state_placements, domain: MeshSchedulingDomain | None = None):
        domain = self._full_device_domain(runtime_context) if domain is None else domain
        local_vacancy = {owner_id: list(ranges) for owner_id, ranges in runtime_context.local_cache_vacancy.items()}
        device_vacancy = {owner_id: list(ranges) for owner_id, ranges in runtime_context.device_memory_vacancy.items()}
        ccg_ids = ccg_tile_mesh.flatten().tolist()
        dma_ids = list(domain.dma_ids)
        owner_placements = {}
        tensor_placements = []
        allocated_banks = []
        persistent_updates = {}
        planned_stats = set()

        def allocate(vacancy, owner_ids, size, count, local):
            trial = {owner_id: list(vacancy[owner_id]) for owner_id in owner_ids}
            banks = []
            for index in range(count):
                owner_id = owner_ids[index % len(owner_ids)]
                addr = runtime_context._allocate_memory_from_vacancy_info(trial[owner_id], size)
                if addr is None:
                    return None
                runtime_context._update_memory_allocation_info(trial[owner_id], addr, size)
                banks.append(MeshMemoryBankDescriptor.LOCAL_CACHE(owner_id, addr, size) if local else MeshMemoryBankDescriptor.DEVICE_MEMORY(addr, size))
            vacancy.update(trial)
            return banks

        def allocate_owner(owner_stats):
            key = id(owner_stats)
            if owner_stats.is_placed:
                local_owners = {bank.owner_id for bank in owner_stats.tile_placement.values() if bank.mem_type == MeshMemoryType.LOCAL_CACHE}
                return owner_stats.tile_placement if local_owners.issubset(set(domain.ccg_tile_mesh.flatten().tolist())) else None
            if key in owner_placements:
                return owner_placements[key]
            descriptor = owner_stats.tensor_desc
            size = descriptor.get_tile_size()
            count = descriptor.get_reserved_n_tiles()
            coords = list(np.ndindex(*descriptor.reserved_tile_grid_shape))
            banks = allocate(local_vacancy, ccg_ids, size, count, True) if descriptor.preferred_mem == MeshMemoryType.LOCAL_CACHE else None
            if banks is None:
                banks = allocate(device_vacancy, dma_ids, size, count, False)
            if banks is None:
                return None
            placement = dict(zip(coords, banks))
            owner_placements[key] = placement
            allocated_banks.extend(banks)
            tensor_placements.append((owner_stats, placement))
            planned_stats.add(key)
            return placement

        for tensor_desc in kernel_desc.input_tensors + kernel_desc.output_tensors:
            tensor_stats = workload.get_tensor_stats(tensor_desc)
            if tensor_stats.is_placed or id(tensor_stats) in planned_stats:
                continue
            storage_desc = tensor_desc.storage_desc
            if storage_desc.is_persistent:
                state_id = storage_desc.persistent_state_id
                owner_stats = persistent_state_placements.get(state_id) or persistent_updates.get(state_id)
                if owner_stats is None:
                    owner_stats = MeshDeviceCompiledTensorStats(storage_desc)
                    persistent_updates[state_id] = owner_stats
            elif tensor_desc.is_view:
                owner_stats = next((stats for stats in workload.tensor_stats_map.values() if stats.tensor_desc is storage_desc), None)
                if owner_stats is None:
                    raise RuntimeError("Tensor view storage is not registered in the compiled workload.")
            else:
                owner_stats = tensor_stats
            owner_placement = allocate_owner(owner_stats)
            if owner_placement is None:
                return None
            if tensor_stats is not owner_stats:
                placement = {coord: owner_placement[tensor_desc.map_tile_coord_to_storage(coord)] for coord in tensor_desc.get_tile_coords()}
                tensor_placements.append((tensor_stats, placement))
                planned_stats.add(id(tensor_stats))

        return tuple(tensor_placements), tuple(allocated_banks), tuple(persistent_updates.items())

    @staticmethod
    def _commit_kernel_plan(runtime_context, kernel_stats, plan: MeshKernelAIStats, persistent_state_placements) -> bool:
        ccg_ids = plan.ccg_tile_mesh.flatten().tolist()
        if any(not runtime_context.ccg_kernel_vacancy[ccg_id] for ccg_id in ccg_ids):
            return False
        device_snapshot = {owner_id: list(ranges) for owner_id, ranges in runtime_context.device_memory_vacancy.items()}
        local_snapshot = {owner_id: list(ranges) for owner_id, ranges in runtime_context.local_cache_vacancy.items()}
        placed_stats = []
        try:
            for bank in plan.allocated_banks:
                if bank.mem_type == MeshMemoryType.LOCAL_CACHE:
                    runtime_context._update_memory_allocation_info(runtime_context.local_cache_vacancy[bank.owner_id], bank.addr, bank.size)
                else:
                    dma_id = runtime_context.device.mem_context.get_dma_id_with_address(bank.addr)
                    runtime_context._update_memory_allocation_info(runtime_context.device_memory_vacancy[dma_id], bank.addr, bank.size)
            for ccg_id in ccg_ids:
                runtime_context.ccg_kernel_vacancy[ccg_id] = False
            for tensor_stats, placement in plan.tensor_placements:
                if not tensor_stats.is_placed:
                    tensor_stats.place(placement)
                    placed_stats.append(tensor_stats)
            for state_id, tensor_stats in plan.persistent_placements:
                persistent_state_placements[state_id] = tensor_stats
            if not kernel_stats.is_placed:
                kernel_stats.place(ccg_tile_mesh=plan.ccg_tile_mesh.copy())
            return True
        except (ValueError, KeyError):
            runtime_context.device_memory_vacancy.clear()
            runtime_context.device_memory_vacancy.update(device_snapshot)
            runtime_context.local_cache_vacancy.clear()
            runtime_context.local_cache_vacancy.update(local_snapshot)
            for ccg_id in ccg_ids:
                runtime_context.ccg_kernel_vacancy[ccg_id] = True
            for tensor_stats in placed_stats:
                tensor_stats.unplace()
            for state_id, tensor_stats in plan.persistent_placements:
                if persistent_state_placements.get(state_id) is tensor_stats:
                    persistent_state_placements.pop(state_id)
            return False

    @staticmethod
    def _get_close_to(runtime_context, workload, kernel_stats) -> list[int]:
        close_to = []
        for tensor_desc in kernel_stats.kernel_desc.input_tensors:
            tensor_stats = workload.get_tensor_stats(tensor_desc)
            if not tensor_stats.is_placed:
                continue
            for bank in tensor_stats.tile_placement.values():
                owner_id = bank.owner_id if bank.mem_type == MeshMemoryType.LOCAL_CACHE else runtime_context.device.mem_context.get_dma_id_with_address(bank.addr)
                if owner_id not in close_to:
                    close_to.append(owner_id)
        return close_to or list(runtime_context.dma_tile_ids)

    @staticmethod
    def _execution_id(key: tuple[str, int, str]) -> str:
        return f"{key[0]}/{key[1]}/{key[2]}"

    def _continue_after_placement_failure(self) -> bool:
        return True

    def _on_action_scheduled(self, workload_id: str):
        return None

    def order_ready(self, context: MeshSchedulingContext) -> tuple[str, ...]:
        ordered = sorted(context.ready_kernels, key=self._fcfs_key)
        starvation_anchor = self._starvation_anchor(context)
        if starvation_anchor is not None:
            ordered = [starvation_anchor] + [
                info for info in ordered if info.execution_id != starvation_anchor.execution_id
            ]
        return tuple(info.execution_id for info in ordered)

    def plan_search_anchors(self, context: MeshSchedulingContext) -> tuple[str, ...] | None:
        return None

    def select_plan(
        self, context: MeshSchedulingContext, candidates: tuple[MeshPlanCandidate, ...]
    ) -> int | None:
        candidate = candidates[0] if candidates else None
        self._record_decision(candidate, None, False, "first_dispatchable")
        return candidate.plan_id if candidate is not None else None

    def on_dispatch(self, context: MeshSchedulingContext, candidate: MeshPlanCandidate):
        return None

    def on_complete(self, workload_id: str, kernel_id: str, completion_cycle: int):
        return None

    def on_workload_complete(self, workload_id: str, completion_cycle: int):
        return None

    @staticmethod
    def _fcfs_key(info: MeshReadyKernelInfo) -> tuple:
        return (
            -info.priority,
            info.ready_cycle,
            info.arrival_cycle,
            info.submission_index,
            info.kernel_index,
            info.execution_id,
        )

    @staticmethod
    def _candidate_key(candidate: MeshPlanCandidate) -> tuple:
        return (
            candidate.benefit,
            candidate.utilization,
            -candidate.makespan_cycles,
            candidate.kernel_ids,
        )

    def _starvation_anchor(self, context: MeshSchedulingContext) -> MeshReadyKernelInfo | None:
        starved = []
        for info in context.ready_kernels:
            limit = (
                info.max_wait_cycles if info.max_wait_cycles is not None else self.starvation_cycles
            )
            if limit is not None and info.waiting_cycles >= limit:
                starved.append(info)
        return (
            min(starved, key=lambda info: (-info.waiting_cycles, self._fcfs_key(info)))
            if starved
            else None
        )

    @staticmethod
    def _anchor_candidates(
        candidates: tuple[MeshPlanCandidate, ...], execution_id: str
    ) -> tuple[MeshPlanCandidate, ...]:
        return tuple(candidate for candidate in candidates if execution_id in candidate.kernel_ids)

    def _record_decision(
        self,
        candidate: MeshPlanCandidate | None,
        anchor: MeshReadyKernelInfo | None,
        starvation_override: bool,
        reason: str,
        rr_cursor: str | None = None,
    ):
        self._decision_metadata = {
            "anchor_workload_id": anchor.workload_id if anchor is not None else None,
            "anchor_kernel_id": anchor.execution_id if anchor is not None else None,
            "ready_wait_cycles": anchor.waiting_cycles if anchor is not None else 0,
            "starvation_override": starvation_override,
            "selection_reason": reason,
            "rr_cursor": rr_cursor,
            "selected_plan_id": candidate.plan_id if candidate is not None else None,
        }

    def _select_for_anchor(
        self,
        candidates: tuple[MeshPlanCandidate, ...],
        anchor: MeshReadyKernelInfo,
        starvation_override: bool,
        reason: str,
        rr_cursor: str | None = None,
    ) -> int | None:
        eligible = self._anchor_candidates(candidates, anchor.execution_id)
        candidate = max(eligible, key=self._candidate_key) if eligible else None
        self._record_decision(candidate, anchor, starvation_override, reason, rr_cursor)
        return candidate.plan_id if candidate is not None else None


class MeshUtilizationScheduler(MeshDeviceScheduler):
    def order_ready(self, context: MeshSchedulingContext) -> tuple[str, ...]:
        ordered = sorted(
            context.ready_kernels, key=lambda info: (-info.priority, info.ready_cycle, info.execution_id)
        )
        starvation_anchor = self._starvation_anchor(context)
        if starvation_anchor is not None:
            ordered = [starvation_anchor] + [
                info for info in ordered if info.execution_id != starvation_anchor.execution_id
            ]
        return tuple(info.execution_id for info in ordered)

    def select_plan(
        self, context: MeshSchedulingContext, candidates: tuple[MeshPlanCandidate, ...]
    ) -> int | None:
        starvation_anchor = self._starvation_anchor(context)
        if starvation_anchor is not None:
            return self._select_for_anchor(candidates, starvation_anchor, True, "starvation_anchor")
        multi = tuple(candidate for candidate in candidates if len(candidate.kernel_ids) >= 3)
        pairs = tuple(candidate for candidate in candidates if len(candidate.kernel_ids) == 2)
        if multi:
            candidate = max(multi, key=self._candidate_key)
            reason = "multi_kernel_utilization"
        elif pairs:
            candidate = max(pairs, key=self._candidate_key)
            reason = "pair_utilization"
        else:
            by_kernel = {
                candidate.kernel_ids[0]: candidate
                for candidate in candidates
                if len(candidate.kernel_ids) == 1
            }
            candidate = next(
                (
                    by_kernel[execution_id]
                    for execution_id in self.order_ready(context)
                    if execution_id in by_kernel
                ),
                None,
            )
            reason = "singleton_fallback"
        self._record_decision(candidate, None, False, reason)
        return candidate.plan_id if candidate is not None else None

    def plan_search_anchors(self, context: MeshSchedulingContext) -> tuple[str, ...] | None:
        starvation_anchor = self._starvation_anchor(context)
        return (starvation_anchor.execution_id,) if starvation_anchor is not None else None


class MeshFCFSScheduler(MeshDeviceScheduler):
    def _continue_after_placement_failure(self) -> bool:
        return False

    def plan_search_anchors(self, context: MeshSchedulingContext) -> tuple[str, ...] | None:
        if not context.ready_kernels:
            return ()
        starvation_anchor = self._starvation_anchor(context)
        anchor = (
            starvation_anchor
            if starvation_anchor is not None
            else min(context.ready_kernels, key=self._fcfs_key)
        )
        return (anchor.execution_id,)

    def select_plan(
        self, context: MeshSchedulingContext, candidates: tuple[MeshPlanCandidate, ...]
    ) -> int | None:
        if not context.ready_kernels:
            self._record_decision(None, None, False, "no_ready_kernel")
            return None
        starvation_anchor = self._starvation_anchor(context)
        anchor = (
            starvation_anchor
            if starvation_anchor is not None
            else min(context.ready_kernels, key=self._fcfs_key)
        )
        return self._select_for_anchor(
            candidates,
            anchor,
            starvation_anchor is not None,
            "starvation_anchor" if starvation_anchor is not None else "fcfs_head",
        )


class MeshFRFCFSScheduler(MeshDeviceScheduler):
    def plan_search_anchors(self, context: MeshSchedulingContext) -> tuple[str, ...] | None:
        starvation_anchor = self._starvation_anchor(context)
        return (
            (starvation_anchor.execution_id,)
            if starvation_anchor is not None
            else self.order_ready(context)
        )

    def select_plan(
        self, context: MeshSchedulingContext, candidates: tuple[MeshPlanCandidate, ...]
    ) -> int | None:
        starvation_anchor = self._starvation_anchor(context)
        if starvation_anchor is not None:
            return self._select_for_anchor(candidates, starvation_anchor, True, "starvation_anchor")
        ready_by_id = {info.execution_id: info for info in context.ready_kernels}
        for execution_id in self.order_ready(context):
            if self._anchor_candidates(candidates, execution_id):
                return self._select_for_anchor(
                    candidates, ready_by_id[execution_id], False, "first_dispatchable_fcfs"
                )
        self._record_decision(None, None, False, "no_dispatchable_kernel")
        return None


class MeshRoundRobinScheduler(MeshDeviceScheduler):
    def __init__(self, starvation_cycles: int | None = 200_000, candidate_window: int = 16):
        super().__init__(starvation_cycles, candidate_window)
        self._workload_queue: list[str] = []

    def reset(self):
        super().reset()
        self._workload_queue = []

    def order_ready(self, context: MeshSchedulingContext) -> tuple[str, ...]:
        self._register_workloads(context)
        ready_by_workload: dict[str, list[MeshReadyKernelInfo]] = {}
        for info in context.ready_kernels:
            ready_by_workload.setdefault(info.workload_id, []).append(info)
        ordered = []
        queue_index = {workload_id: index for index, workload_id in enumerate(self._workload_queue)}
        workload_order = sorted(ready_by_workload, key=lambda workload_id: (-max(info.priority for info in ready_by_workload[workload_id]), queue_index[workload_id]))
        for workload_id in workload_order:
            ordered.extend(sorted(ready_by_workload.get(workload_id, ()), key=self._fcfs_key))
        starvation_anchor = self._starvation_anchor(context)
        if starvation_anchor is not None:
            ordered = [starvation_anchor] + [
                info for info in ordered if info.execution_id != starvation_anchor.execution_id
            ]
        return tuple(info.execution_id for info in ordered)

    def plan_search_anchors(self, context: MeshSchedulingContext) -> tuple[str, ...] | None:
        starvation_anchor = self._starvation_anchor(context)
        return (
            (starvation_anchor.execution_id,)
            if starvation_anchor is not None
            else self.order_ready(context)
        )

    def select_plan(
        self, context: MeshSchedulingContext, candidates: tuple[MeshPlanCandidate, ...]
    ) -> int | None:
        self._register_workloads(context)
        starvation_anchor = self._starvation_anchor(context)
        rr_cursor = self._workload_queue[0] if self._workload_queue else None
        if starvation_anchor is not None:
            return self._select_for_anchor(
                candidates, starvation_anchor, True, "starvation_anchor", rr_cursor
            )
        ready_by_id = {info.execution_id: info for info in context.ready_kernels}
        for execution_id in self.order_ready(context):
            if self._anchor_candidates(candidates, execution_id):
                return self._select_for_anchor(
                    candidates, ready_by_id[execution_id], False, "round_robin", rr_cursor
                )
        self._record_decision(None, None, False, "no_dispatchable_kernel", rr_cursor)
        return None

    def on_dispatch(self, context: MeshSchedulingContext, candidate: MeshPlanCandidate):
        self._register_workloads(context)
        selected = set(candidate.workload_ids)
        served = [workload_id for workload_id in self._workload_queue if workload_id in selected]
        waiting = [
            workload_id for workload_id in self._workload_queue if workload_id not in selected
        ]
        self._workload_queue = waiting + served

    def on_workload_complete(self, workload_id: str, completion_cycle: int):
        self._workload_queue = [
            queued_id for queued_id in self._workload_queue if queued_id != workload_id
        ]

    def _on_action_scheduled(self, workload_id: str):
        if workload_id in self._workload_queue:
            self._workload_queue.remove(workload_id)
            self._workload_queue.append(workload_id)

    def _register_workloads(self, context: MeshSchedulingContext):
        ordered_infos = sorted(
            context.ready_kernels,
            key=lambda info: (info.arrival_cycle, info.submission_index, info.workload_id),
        )
        for info in ordered_infos:
            if info.workload_id not in self._workload_queue:
                self._workload_queue.append(info.workload_id)
