from typing import  Sequence

from neuromta.framework.logger import logger, LogLevel, _LOG_LEVEL_COLORS, _COLOR_RESET, set_global_monitoring_window, unset_global_monitoring_window
from neuromta.framework.core import Core, Kernel, Command
from neuromta.framework.device import Device

try:
    from neuromta_monitor.client import MonitorClient, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_FREQ, DEFAULT_TIMEOUT, DEFAULT_SIM_NAME
    from neuromta_monitor.core_tracker import CoreClusterTrackerBase, CoreTrackerBase
    from neuromta_monitor.profiler import GroupedProfilerBase, ProfilerBase
except ImportError as e:
    logger.error(f"Failed to import neuromta_monitor. MonitoringWindow requires the neuromta_monitor package to be properly installed and accessible.")
    raise e
    

__all__ = [
    "MonitoringWindow",
]


class MonitoringWindow(MonitorClient):
    def __init__(
        self, 
        device: Device,
        core_groups: list[list[int]] | list[int],  # list of core groups, each core group is a list of core IDs
        profilers: list[ProfilerBase | GroupedProfilerBase]=None,
        sim_name: str=DEFAULT_SIM_NAME, 
        host: str=DEFAULT_HOST, 
        port: int=DEFAULT_PORT, 
        freq: float=DEFAULT_FREQ, 
        timeout: float=DEFAULT_TIMEOUT,
        disable: bool=False,
    ):  
        self._device = device
        self._binded_core_ids: set[int] = set()
        self._core_group_mappings: dict[int, list[int]] = {}  # bind_core -> Core Group 0 | bind_core_group -> Core Group 1 ...  
        self._command_debug_hook_handles: dict[int, list[int]] = {}
        self._kernel_debug_hook_handles: dict[int, list[int]] = {}
        self._core_trackers: dict[int, CoreTrackerBase] = {}
        self._profilers = profilers if profilers is not None else []
        self._disable = disable
        
        if core_groups is None:
            raise ValueError("core_groups cannot be None.")
        
        if all(isinstance(cg, Sequence) for cg in core_groups):
            for core_group in core_groups:
                self._bind_core_group(core_group)
        elif all(not isinstance(cg, Sequence) for cg in core_groups):
            # if core_groups is a list of individual core IDs, bind them to a single core group
            self._bind_core_group(core_groups) 
        else:
            raise ValueError("Invalid core_groups format. It should be either a list of core groups (each core group is a list of core IDs) or a list of individual core IDs.")
        
        n_core_groups = len(self._core_group_mappings)
        n_cores_per_group = [len(core_ids) for core_ids in self._core_group_mappings.values()]
        
        core_cluster_tracker = CoreClusterTrackerBase(n_core_groups, n_cores_per_group)
        
        for group_id, core_ids in self._core_group_mappings.items():
            for core_idx, core_id in enumerate(core_ids):
                self._core_trackers[core_id] = core_cluster_tracker.trackers[group_id].trackers[core_idx]
                
        super().__init__(core_cluster_tracker, profilers, sim_name, host, port, freq, timeout)
        
    def _kernel_progress_debug_hook(self, core: Core, kernel: Kernel):
        if core.core_id not in self._core_trackers:
            raise Exception(f"Core ID {core.core_id} is not found in the core progress status. This should not happen if the monitoring window is properly initialized.")
        
        tracker = self._core_trackers[core.core_id]
                
        _amt = 0
        for slot_id, _kernel in core._dispatched_main_kernels.items():
            if _kernel.is_finished(core) or _kernel is kernel:
                continue
            _cursor = _kernel._execution_cursor
            _totals = len(_kernel._execution_steps)
            _n_kernels = tracker.totals
            _amt += 0 if (_totals == 0 or _n_kernels == 0) else (_cursor / _totals / _n_kernels)

        tracker.update(1, acc=True)
        tracker.update(_amt, temporary=True)
        self.update(cycle=self._device.timestamp)
    
    def _command_progress_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        if core.core_id not in self._core_trackers:
            raise Exception(f"Core ID {core.core_id} is not found in the core progress status. This should not happen if the monitoring window is properly initialized.")
        
        tracker = self._core_trackers[core.core_id]
        
        _amt = 0
        for slot_id, _kernel in core._dispatched_main_kernels.items():
            if _kernel.is_finished(core):
                continue
            _cursor = _kernel._execution_cursor
            _totals = len(_kernel._execution_steps)
            _n_kernels = tracker.totals
            _amt += 0 if (_totals == 0 or _n_kernels == 0) else (_cursor / _totals / _n_kernels)
        
        tracker.update(_amt, temporary=True)
        self.update(cycle=self._device.timestamp)
        
    def add_log(self, message: str, level: LogLevel=LogLevel.INFO):
        self.message(
            msg_type=level.value,
            msg=message
        )
    
    def _bind_core(self, core_id: int, _core_group_id: int=0):
        if core_id not in self._device.initialized_cores.keys():
            raise ValueError(f"Core ID {core_id} is not valid or the core has not been initialized in the device.")
        
        if core_id in self._binded_core_ids:
            logger.warning(f"Core ID {core_id} is already bound to the monitoring window. Ignoring duplicate binding.")
            return
        
        self._binded_core_ids.add(core_id)
        core = self._device.initialized_cores[core_id]
        
        self._core_group_mappings[_core_group_id].append(core_id)
        
        if core_id not in self._command_debug_hook_handles:
            self._command_debug_hook_handles[core_id] = []
        hook_handle = core.register_command_debug_hook(self._command_progress_debug_hook)
        self._command_debug_hook_handles[core_id].append(hook_handle)
        
        if core_id not in self._kernel_debug_hook_handles:
            self._kernel_debug_hook_handles[core_id] = []
        hook_handle = core.register_kernel_debug_hook(self._kernel_progress_debug_hook, slot_level=0, filter_rpc=True)  # only monitor the progress of main kernels (slot_level=0)
        self._kernel_debug_hook_handles[core_id].append(hook_handle)
        
        return self
        
    def _bind_core_group(self, core_ids: list[int]):
        core_group_id = len(self._core_group_mappings)
        self._core_group_mappings[core_group_id] = []
        
        for core_id in core_ids:
            self._bind_core(core_id, _core_group_id=core_group_id)
            
        return self
            
    def _unbind_cores(self):
        for core_id, hook_handles in self._command_debug_hook_handles.items():
            core = self._device.initialized_cores[core_id]
            for handle in hook_handles:
                core.unregister_command_debug_hook(handle)
        self._command_debug_hook_handles.clear()
        
        for core_id, hook_handles in self._kernel_debug_hook_handles.items():
            core = self._device.initialized_cores[core_id]
            for handle in hook_handles:
                core.unregister_kernel_debug_hook(handle)
        self._kernel_debug_hook_handles.clear()
        
        self._core_group_mappings.clear()
        self._n_cores_binded = 0
        self._binded_core_ids.clear()
        
    def initialize(self):
        if self._disable:
            return self
        
        set_global_monitoring_window(self)
        
        super().initialize(per_core_queue_limit=2, per_core_queue_resume=1)
        
        for core_id in self._binded_core_ids:
            core = self._device.initialized_cores[core_id]
            self._core_trackers[core_id].initialize(total=core.n_dispatched_main_kernels)
        
        self.update(cycle=self._device.timestamp, enforce=True)
        
        return self
    
    def close(self):
        for tracker in self._core_trackers.values():
            tracker.update(tracker.totals)
        self.update(cycle=self._device.timestamp, enforce=True)
        
        logger.info("Simulation finished with NeuroMTA Simulator! Closing client ...")
        
        super().close()
        
        self._unbind_cores()
        unset_global_monitoring_window()
        
        return self
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()