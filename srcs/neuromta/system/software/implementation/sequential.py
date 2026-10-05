from neuromta.system.software.utils.compiler import MeshDeviceCompiler
from neuromta.system.software.utils.scheduler import MeshDeviceScheduler
from neuromta.system.software.utils.runtime import MeshDeviceRuntime, MeshDeviceRuntimeWorkloadState


__all__ = ["SequentialCompiler", "SequentialScheduler", "SequentialRuntime"]


class SequentialCompiler(MeshDeviceCompiler):
    pass


class SequentialScheduler(MeshDeviceScheduler):
    pass


class SequentialRuntime(MeshDeviceRuntime):
    def __init__(self, device_desc, kernel_materializer=None, scheduler=None, enable_debug_log=False):
        super().__init__(device_desc, kernel_materializer=kernel_materializer, scheduler=SequentialScheduler() if scheduler is None and kernel_materializer is None else scheduler, enable_debug_log=enable_debug_log)

    def _reset_policy_state(self) -> None:
        self._active_workload_id = None

    def _select_dispatchable_workloads(self, ready_states: tuple[MeshDeviceRuntimeWorkloadState, ...]) -> tuple[MeshDeviceRuntimeWorkloadState, ...]:
        if self._active_workload_id is not None:
            return tuple(state for state in ready_states if state.workload_id == self._active_workload_id)
        if not ready_states:
            return ()
        selected = min(ready_states, key=lambda state: (state.arrival_cycle, tuple(self._workload_states).index(state.workload_id), state.workload_id))
        self._active_workload_id = selected.workload_id
        return (selected,)

    def _on_workload_completed(self, workload_state: MeshDeviceRuntimeWorkloadState) -> None:
        if self._active_workload_id == workload_state.workload_id:
            self._active_workload_id = None
