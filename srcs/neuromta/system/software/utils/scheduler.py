from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from neuromta.system.software.utils.compiler import MeshDeviceActionType, MeshDeviceCompiledAction, MeshDeviceCompiledTensorStats
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshKernelType, MeshMemoryBankDescriptor, MeshMemoryType


__all__ = [
    "MeshKernelAIStats",
    "MeshPlanCandidate",
    "MeshSchedulingDomain",
    "MeshReadyKernelInfo",
    "MeshDeviceScheduler",
    "MeshSchedulingContext",
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
        if starvation_cycles is not None and (not isinstance(starvation_cycles, int) or isinstance(starvation_cycles, bool) or starvation_cycles <= 0):
            raise ValueError("starvation_cycles must be a positive integer or None.")
        if not isinstance(candidate_window, int) or isinstance(candidate_window, bool) or candidate_window <= 0:
            raise ValueError("candidate_window must be positive.")
        if decision_hook is not None and not callable(decision_hook):
            raise TypeError("decision_hook must be callable or None.")

        self.starvation_cycles = starvation_cycles
        self.candidate_window = candidate_window
        self.decision_hook = decision_hook

        self._decision_metadata: dict = {}
        self._ready_cycles: dict[tuple[str, int, str], int] = {}            # key: (workload_id, kernel_index, kernel_id) -> timestamp when the kernel became ready
        self._submission_indices: dict[tuple[str, int, str], int] = {}      # key: (workload_id, kernel_index, kernel_id) -> submission index
        self._next_submission_index = 0                                     # next submission index to assign for a new ready kernel

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

    def schedule_actions(self, runtime_context, actions: list[tuple], persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats] | None = None, running_kernel_ids: tuple[str, ...] = ()) -> tuple[list[tuple[MeshDeviceCompiledAction, Any]], list[tuple[MeshDeviceCompiledAction, Any]]]:
        """Choose ready RUN_KERNEL actions and reserve resources for this dispatch.

        ``actions`` contains (action, workload_state[, domain]) tuples. Missing
        domains mean the whole device. A successful action is returned as
        (action, workload_state); a deferred one retains its domain in
        (action, workload_state, domain) so the runtime can retry it later.
        Persistent tensor placements and device vacancy are shared mutable
        state: successful placement commits reservations before returning.

        The default path orders candidates, lets the policy choose the first
        one to try, then greedily places more kernels within the search window
        and dispatch limit. Joint planning instead evaluates compatible
        placements together and commits only the selected plan.
        """
        if runtime_context is None:
            raise ValueError("A runtime context must be registered before scheduling kernel actions.")

        persistent_state_placements = {} if persistent_state_placements is None else persistent_state_placements
        cycle = int(runtime_context.device.timestamp)

        # STEP 1: Collect ready kernels and set scheduling metadata
        action_map      = {}    # execution_id -> (action, workload_state, domain, key)
        ready_kernels   = []    # list of MeshReadyKernelInfo for policy ordering
        domains         = {}    # domain_id -> MeshSchedulingDomain for policy context

        for item in actions:
            # 1-1: Unpack action, workload state and domain (optional)
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

            # 1-2: Record ready cycle and assign submission index to the kernel
            #   - Note that the kernel may be re-submitted, and may exist in `ready_cycles` record
            #   - If then, we do not update the ready cycle and submission index
            key = (workload_state.workload_id, workload_state.cursor, action.kernel_id)
            if key not in self._ready_cycles:
                self._ready_cycles[key] = cycle
                self._submission_indices[key] = self._next_submission_index
                self._next_submission_index += 1

            # 1-3: Register action to the action map and create a MeshReadyKernelInfo for policy ordering
            hint = workload_state.scheduling_hint or MeshWorkloadSchedulingHint()
            execution_id = self._execution_id(key)
            ready_cycle = self._ready_cycles[key]

            action_map[execution_id] = (action, workload_state, domain, key)
            ready_kernels.append(MeshReadyKernelInfo(
                execution_id=execution_id,
                workload_id=workload_state.workload_id,
                ready_cycle=ready_cycle,
                waiting_cycles=max(0, cycle - ready_cycle),
                kernel_index=workload_state.cursor,
                arrival_cycle=workload_state.arrival_cycle,
                submission_index=self._submission_indices[key],
                kernel_id=action.kernel_id,
                domain_id=domain.domain_id,
                priority=hint.priority,
                weight=hint.weight,
                max_wait_cycles=hint.max_wait_cycles
            ))
            domains[domain.domain_id] = domain

        # STEP 2: Create a scheduling context and order kernels by policy
        context = MeshSchedulingContext(cycle=cycle, ready_kernels=tuple(ready_kernels), running_kernel_ids=tuple(running_kernel_ids), domains=tuple(domains.values()))

        # 2-1: Order by policy, then promote any search anchors into the finite candidate window (for example, a kernel past its wait threshold).
        ordered_ids = self.order_ready(context)                 # 2-1-1: Order ready kernels by readiness (with respect to the policy)
        search_anchors = self.plan_search_anchors(context)      # 2-1-2: Identify any search anchors (e.g., starved kernels) that should be prioritized
        if search_anchors is not None:
            anchor_ids = tuple(execution_id for execution_id in search_anchors if execution_id in action_map)
            ordered_ids = anchor_ids + tuple(execution_id for execution_id in ordered_ids if execution_id not in set(anchor_ids))

        # 2-2: Check dispatch limit and check whether to use joint planning
        #   - `dispatch_limit` stands for the maximun number of kernels that can be dispatched in a single scheduling call
        #   - if `dispatch_limit` is None, it implies that there is no limit on the number of kernels that can be dispatched in a single scheduling call
        #   - joint planning is a spatial co-location planning that evaluates multiple kernels together to find the best placement and scheduling plan
        dispatch_limit = self._get_dispatch_limit(context)
        if dispatch_limit is not None and dispatch_limit <= 0:
            # 2-2-1: negative dispatch limit -> no kernels can be dispatched, return all ready kernels as suspended
            return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        if self._use_joint_planning():
            # 2-2-2: joint planning -> evaluate multiple kernels together to find the best placement and scheduling plan
            return self._schedule_joint_actions(runtime_context, context, ordered_ids, action_map, persistent_state_placements)

        # 2-3: Select a plan from the ordered candidates and attempt to place kernels in order of the selected plan
        selection_candidates = tuple(
            MeshPlanCandidate(
                # plan metadata
                plan_id=index,
                kernel_ids=(execution_id,),
                workload_ids=(action_map[execution_id][1].workload_id,),
                # performance metrics (used for ??)
                makespan_cycles=0,
                sequential_cycles=0,
                utilization=0.0,
                benefit=0.0,
                peak_dma_utilization=0.0
            )
            for index, execution_id in enumerate(ordered_ids[:self.candidate_window])
        )

        selected_plan_id = self.select_plan(context, selection_candidates)  # select a plan based on the policy and the candidates (first one is selected by default)
        if selected_plan_id is None:
            return [], [(action, workload_state, domain) for action, workload_state, domain, _ in action_map.values()]
        selected_candidate = next((candidate for candidate in selection_candidates if candidate.plan_id == selected_plan_id), None)
        if selected_candidate is None:
            raise RuntimeError(f"Scheduler {self.name} selected unknown plan ID {selected_plan_id}.")
        selected_execution_id = selected_candidate.kernel_ids[0]
        ordered_ids = (selected_execution_id,) + tuple(execution_id for execution_id in ordered_ids if execution_id != selected_execution_id)   # reorder the candidates to prioritize the selected one

        # STEP 3: Attempt to place kernels in order of the selected plan, respecting the candidate window and dispatch limit
        scheduled = []
        suspended = []

        for index, execution_id in enumerate(ordered_ids):
            # 3-1: Unpack the action, workload state, domain, and key for the current execution ID
            action, workload_state, domain, key = action_map[execution_id]
            if index >= self.candidate_window or dispatch_limit is not None and len(scheduled) >= dispatch_limit:
                suspended.append((action, workload_state, domain))  # exceeded candidate window or dispatch limit, suspend the remaining kernels
                continue

            # 3-2: Attempt to place the kernel on the device.
            #   - If successful, add it to the scheduled list
            #   - Otherwise, add it to the suspended list
            kernel_stats = workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
            if self._place_kernel(runtime_context, workload_state.compiled_workload, kernel_stats, persistent_state_placements, domain):
                # 3-2-1: Successfully placed the kernel, add it to the scheduled list and update the ready cycles and submission indices
                scheduled.append((action, workload_state))
                self._ready_cycles.pop(key, None)
                self._submission_indices.pop(key, None)
                self._on_action_scheduled(workload_state.workload_id)
            else:
                # 3-2-2: Failed to place the kernel, add it to the suspended list. If the policy indicates not to continue after placement failure, suspend all remaining kernels.
                suspended.append((action, workload_state, domain))
                if not self._continue_after_placement_failure():
                    suspended.extend((action_map[remaining_id][0], action_map[remaining_id][1], action_map[remaining_id][2]) for remaining_id in ordered_ids[index + 1:])
                    break

        # STEP 4: If any kernels were successfully scheduled, notify the policy and update the decision metadata with the selected kernel IDs and their utilization
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
        raise NotImplementedError("Joint planning must be implemented by the scheduler subclass.")

    def _get_shape_plans(self, runtime_context, workload, kernel_stats, persistent_state_placements, domain):
        # STEP 1: If the kernel is pre-placed, validate its placement and return a single plan if valid
        if kernel_stats.is_placed:
            # 1-1: Validate that the pre-placed kernel mesh is within the scheduling domain and supported by the kernel type
            ccg_tile_mesh = kernel_stats.ccg_tile_mesh
            ccg_ids = ccg_tile_mesh.flatten().tolist()
            domain_ids = set(domain.ccg_tile_mesh.flatten().tolist())
        
            if any(ccg_id not in runtime_context.ccg_kernel_vacancy or ccg_id not in domain_ids for ccg_id in ccg_ids):
                raise ValueError("The pre-placed kernel mesh contains CCG IDs outside its scheduling domain.")
            if not self._supports_mesh_shape(kernel_stats.kernel_desc, ccg_tile_mesh.shape, allow_idle_cores=True):
                raise ValueError(f"Kernel type {kernel_stats.kernel_desc.kernel_type.name} does not support pre-placed mesh shape {ccg_tile_mesh.shape}.")
            if any(not runtime_context.ccg_kernel_vacancy[ccg_id] for ccg_id in ccg_ids):
                return []

            # 1-2: Plan tensor placements that are not already pre-placed (e.g., intermediate tensors)
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
        
        # STEP 2: If the kernel is not pre-placed, evaluate all supported mesh shapes within the scheduling domain and return a list of valid plans
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

    def _estimate_kernel_cycles(self, runtime_context, kernel_desc, placement, domain) -> int:
        total_ops = self._get_total_ops(kernel_desc)
        core_count = placement.ccg_tile_mesh.size
        per_core_flops = runtime_context.total_flops / len(runtime_context.ccg_tile_ids)
        compute_cycles = total_ops / max(1.0, per_core_flops * core_count)
        memory_traffic = self._estimate_memory_traffic(kernel_desc, placement, total_ops)
        bandwidth = runtime_context.total_mem_bandwidth * len(domain.dma_ids) / len(runtime_context.dma_tile_ids)
        noc_cycles = self._estimate_noc_cycles(runtime_context, kernel_desc, placement) if self._use_joint_planning() else 0
        return max(1, math.ceil(max(compute_cycles, memory_traffic / max(bandwidth, 1e-9), noc_cycles)))

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
            """returns a list of MeshMemoryBankDescriptor for the allocated banks, or None if allocation fails"""
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
            """returns the tile placement for the owner_stats, or None if allocation fails (reuse existing placement if already placed)"""
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
            # STEP 1: Skip tensor placements that are already placed or have been planned in this iteration
            tensor_stats = workload.get_tensor_stats(tensor_desc)
            if tensor_stats.is_placed or id(tensor_stats) in planned_stats:
                continue
            
            # STEP 2: Determine the owner of the tensor stats (persistent state, view, or the tensor itself) and allocate its placement
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
            
            # STEP 3: Allocate the placement for the owner stats and map the tensor's tile coordinates to the allocated banks
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
            ordered = [starvation_anchor] + [info for info in ordered if info.execution_id != starvation_anchor.execution_id]
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
            limit = (info.max_wait_cycles if info.max_wait_cycles is not None else self.starvation_cycles)
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
