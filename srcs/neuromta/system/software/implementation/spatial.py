import math

import numpy as np

from neuromta.system.software.utils.compiler import MeshDeviceCompiler
from neuromta.system.software.utils.scheduler import MeshDeviceScheduler, MeshPlanCandidate, MeshSchedulingContext
from neuromta.system.software.utils.descriptor import MeshKernelType, MeshMemoryType
from neuromta.system.software.utils.runtime_utils import create_linear_mapping_requistes, create_conv2d_mapping_requistes, create_elementwise_mapping_requistes, create_reduction_mapping_requistes, create_sdpa_mapping_requistes
from neuromta.system.software.utils.runtime import MeshDeviceRuntime

from ._common import MeshFCFSScheduler, MeshFRFCFSScheduler, MeshRoundRobinScheduler


__all__ = [
    "SpatialCompiler", 
    "SpatialScheduler", 
    "SpatialRuntime",
    
    # common schedulers avilable
    "MeshFCFSScheduler",
    "MeshFRFCFSScheduler",
    "MeshRoundRobinScheduler"
]


class SpatialCompiler(MeshDeviceCompiler):
    pass


class SpatialScheduler(MeshDeviceScheduler):
    def __init__(self, policy=None, starvation_cycles=200_000, candidate_window=16, decision_hook=None):
        super().__init__(starvation_cycles=starvation_cycles, candidate_window=candidate_window, decision_hook=decision_hook)
        self._noc_ops_cache = {}
        self.policy = policy

    def reset(self):
        super().reset()
        self._noc_ops_cache = {}
        if self.policy is not None:
            self.policy.reset()

    def order_ready(self, context):
        if self.policy is not None:
            return self.policy.order_ready(context)
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

    def _continue_after_placement_failure(self) -> bool:
        return True if self.policy is None else self.policy._continue_after_placement_failure()

    def _on_action_scheduled(self, workload_id: str):
        if self.policy is not None:
            self.policy._on_action_scheduled(workload_id)

    def on_dispatch(self, context, candidate):
        if self.policy is not None:
            self.policy.on_dispatch(context, candidate)

    def on_complete(self, workload_id: str, kernel_id: str, completion_cycle: int):
        if self.policy is not None:
            self.policy.on_complete(workload_id, kernel_id, completion_cycle)

    def on_workload_complete(self, workload_id: str, completion_cycle: int):
        super().on_workload_complete(workload_id, completion_cycle)
        if self.policy is not None:
            self.policy.on_workload_complete(workload_id, completion_cycle)

    def _use_joint_planning(self) -> bool:
        return True

    def _schedule_joint_actions(self, runtime_context, context, ordered_ids, action_map, persistent_state_placements):
        # Called from `schedule_actions()`, when `_use_joint_planning()`

        ordered_ids = tuple(ordered_ids[:self.candidate_window])
        base_snapshot = self._snapshot_placement_state(runtime_context, action_map, persistent_state_placements)
        plans = []
        baseline_cycles = {}
        plan_id = 0
        trace = [] if self.decision_hook is not None else None

        def record(phase, reason, kernel_ids, placements=(), predicted_cycles=(), sequential_cycles=None, makespan_cycles=None, benefit=None, peak_dma_utilization=None, candidate_id=None):
            """Record a scheduling decision for debugging and analysis."""
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
            """Return a list of feasible shape plans for the given execution ID, or an empty list if none are feasible."""
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

        # STEP 1: Evaluate each ready kernel individually to find feasible shape plans and record their baseline cycles and memory utilization.
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

        # STEP 2: Evaluate pairs of ready kernels to find feasible joint placements and record their makespan, sequential cycles, benefit, and peak DMA utilization.
        for first_index, first_id in enumerate(ordered_ids):
            if first_id not in baseline_cycles or not any(second_id in baseline_cycles for second_id in ordered_ids[first_index + 1:]):
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

        # Extend only the eight highest-ranked pairs with a third kernel.
        # The triple DMA check conservatively sums all three utilizations.
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

        # Search is complete: return to the incoming resource state before
        # asking the policy to choose and committing its selected placements.
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
        # Commit the chosen group atomically. If any placement fails, restore
        # the entire group and leave every action suspended for a retry.
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

    @staticmethod
    def _diverse_shape_plans(plans, domain_core_count):
        selected = list(plans[:3])
        selected.extend(item for item in plans if item[0].ccg_tile_mesh.size <= max(1, domain_core_count // 2))
        selected = selected[:6]
        unique = {}
        for item in selected:
            unique.setdefault((item[0].ccg_tile_mesh.shape, tuple(item[0].ccg_tile_mesh.flatten())), item)
        return tuple(unique.values())

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


class SpatialRuntime(MeshDeviceRuntime):
    def __init__(self, device_desc, kernel_materializer=None, scheduler=None, enable_debug_log=False):
        effective_scheduler = None if kernel_materializer is not None else scheduler if isinstance(scheduler, SpatialScheduler) else SpatialScheduler(policy=scheduler, starvation_cycles=scheduler.starvation_cycles if scheduler is not None else 200_000, candidate_window=scheduler.candidate_window if scheduler is not None else 16)
        super().__init__(device_desc, kernel_materializer=kernel_materializer, scheduler=effective_scheduler, enable_debug_log=enable_debug_log)
