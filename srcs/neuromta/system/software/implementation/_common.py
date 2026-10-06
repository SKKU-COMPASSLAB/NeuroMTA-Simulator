from neuromta.system.software.utils.scheduler import MeshDeviceScheduler, MeshSchedulingContext, MeshPlanCandidate, MeshReadyKernelInfo


__all__ = [
    "MeshFCFSScheduler", 
    "MeshFRFCFSScheduler", 
    "MeshRoundRobinScheduler"
]


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
