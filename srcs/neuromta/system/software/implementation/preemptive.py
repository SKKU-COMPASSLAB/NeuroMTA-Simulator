from neuromta.system.software.utils.compiler import MeshDeviceCompiler
from neuromta.system.software.utils.scheduler import MeshSchedulingContext
from neuromta.system.software.utils.runtime import MeshDeviceRuntime

from ._common import MeshFCFSScheduler, MeshFRFCFSScheduler, MeshRoundRobinScheduler


__all__ = [
    "PreemptiveCompiler", 
    "PreemptiveScheduler", 
    "PreemptiveRuntime",
    
    # common schedulers avilable
    "MeshFCFSScheduler",
    "MeshFRFCFSScheduler",
    "MeshRoundRobinScheduler"
]


class PreemptiveCompiler(MeshDeviceCompiler):
    pass


class PreemptiveScheduler(MeshFRFCFSScheduler):
    def __init__(self, policy=None, starvation_cycles=200_000, candidate_window=16):
        super().__init__(starvation_cycles=starvation_cycles, candidate_window=candidate_window)
        
        self.policy = policy    # additional scheduler, use `MeshFRFCFSScheduler` if not provided

    def reset(self):
        super().reset()
        if self.policy is not None:
            self.policy.reset()

    def order_ready(self, context):
        return super().order_ready(context) if self.policy is None else self.policy.order_ready(context)

    def plan_search_anchors(self, context):
        return super().plan_search_anchors(context) if self.policy is None else self.policy.plan_search_anchors(context)

    def select_plan(self, context, candidates):
        if self.policy is None:
            return super().select_plan(context, candidates)
        plan_id = self.policy.select_plan(context, candidates)
        self._decision_metadata = self.policy.decision_metadata
        return plan_id

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

    def _get_dispatch_limit(self, context: MeshSchedulingContext) -> int:
        return max(0, 1 - len(context.running_kernel_ids))


class PreemptiveRuntime(MeshDeviceRuntime):
    def __init__(self, device_desc, kernel_materializer=None, scheduler=None, enable_debug_log=False):
        effective_scheduler = None if kernel_materializer is not None else scheduler if isinstance(scheduler, PreemptiveScheduler) else PreemptiveScheduler(policy=scheduler, starvation_cycles=scheduler.starvation_cycles if scheduler is not None else 200_000, candidate_window=scheduler.candidate_window if scheduler is not None else 16)
        super().__init__(device_desc, kernel_materializer=kernel_materializer, scheduler=effective_scheduler, enable_debug_log=enable_debug_log)
