import enum
from typing import Callable

from neuromta.framework import *
from neuromta.component.core.npu_core import NPUCore
from neuromta.component.core.dma_core import DMACore
from neuromta.component.implementation.hardware import MCA_DeviceBase, MCA_CoreGroup
from neuromta.component.utils.profiler_base import GroupedProfilerBase, ProfilerBase


__all__ = [
    "_ProfilerTemplate",
    "_GroupedProfilerTemplate",
    "DRAMBandwidthProfiler",
    "InterconnectBandwidthProfiler",
    "ThreadUtilizationProfiler",
]


class _ProfilerTemplate(ProfilerBase):
    def __init__(self, device: MCA_DeviceBase, target_core_ids: list[int], metric_name: str, metric_unit: str, n_max_entries: int, profiler_type: ProfilerBase.Type):
        if isinstance(target_core_ids, (int, str)):
            target_core_ids = [target_core_ids]
            
        self.device = device
        self.target_core_ids = target_core_ids
        
        super().__init__(
            metric_name=metric_name,
            metric_unit=metric_unit,
            n_max_entries=n_max_entries,
            profiler_type=profiler_type,
        )
        
        for core_id in target_core_ids:
            core = device.initialized_cores[core_id]
            core.register_command_debug_hook(self.command_debug_hook)
        
    def command_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        raise NotImplementedError("command_debug_hook must be implemented by subclasses of _ProfilerTemplate")


class _GroupedProfilerTemplate(GroupedProfilerBase):
    def __init__(self, device: MCA_DeviceBase, target_core_ids: list[int], metric_name: str, metric_unit: str, n_max_entries: int, profiler_type: ProfilerBase.Type):
        if isinstance(target_core_ids, (int, str)):
            target_core_ids = [target_core_ids]
            
        self.device = device
        self.target_core_ids = target_core_ids
        
        super().__init__(
            n_agents=len(self.target_core_ids),
            metric_name=metric_name,
            metric_unit=metric_unit,
            n_max_entries=n_max_entries,
            profiler_type=profiler_type,
        )
        
        for core_id in target_core_ids:
            core = device.initialized_cores[core_id]
            core.register_command_debug_hook(self.command_debug_hook)
        
    def command_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        raise NotImplementedError("command_debug_hook must be implemented by subclasses of _GroupedProfilerTemplate")
    
    
class ProfilerFileSaver:
    def __init__(self):
        pass


class DRAMBandwidthProfiler(_ProfilerTemplate):
    class RecordType(enum.Enum):
        READ  = 0
        WRITE = 1
        BOTH  = 2
    
    def __init__(self, device: MCA_DeviceBase, record_type: RecordType, n_max_entries: int=512):
        super().__init__(
            device=device,
            target_core_ids=device.companion_core.core_id,
            metric_name="DRAM Bandwidth",
            metric_unit="B/cycle",
            n_max_entries=n_max_entries,
            profiler_type=ProfilerBase.Type.BANDWIDTH,
        )
        
        if isinstance(record_type, str):
            record_type = self.RecordType[record_type.upper()]
        elif isinstance(record_type, int):
            record_type = self.RecordType(record_type)
        
        self.record_type = record_type
    
    def command_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        if not isinstance(core, CompanionCore):
            raise Exception(f"DRAMBandwidthProfiler can only be registered to Companion cores. Core ID {core.core_id} is not a Companion core.")
        
        if command.cmd_id != "dispatch_command_with_module":
            return
        
        module_id = None
        companion_cmd = None
        
        if len(command.args) > 0:
            module_id = command.args[0]
        elif "module_id" in command.kwargs:
            module_id = command.kwargs["module_id"]
            
        if len(command.args) > 1:
            companion_cmd = command.args[1]
        elif "cmd" in command.kwargs:
            companion_cmd = command.kwargs["cmd"]
            
        if module_id is None or companion_cmd is None:
            logger.warning(f"Received a dispatch_command_with_module command without module_id or cmd information. Skipping profiling for this command. Command args: {command.args}, kwargs: {command.kwargs}")
            return
        
        if module_id != "DRAMSIM":
            return
        
        size: int = companion_cmd.kwargs.get("size", None)
        is_write: bool = companion_cmd.kwargs.get("is_write", None)
        
        if size is None or is_write is None:
            logger.warning(f"Received a DRAMSIM command without size or is_write information. Skipping profiling for this command. Command kwargs: {companion_cmd.kwargs}")
            return
        
        flag = (self.record_type == self.RecordType.WRITE and is_write) \
            or (self.record_type == self.RecordType.READ and not is_write) \
            or (self.record_type == self.RecordType.BOTH)
            
        if flag:
            self.add_entry(issue_time, commit_time, size)
    
    
class InterconnectBandwidthProfiler(_ProfilerTemplate):
    def __init__(self, device: MCA_DeviceBase, n_max_entries: int=512):
        super().__init__(
            device=device,
            target_core_ids=device.companion_core.core_id,
            metric_name="Interconnect Bandwidth",
            metric_unit="flits/cycle",
            n_max_entries=n_max_entries,
            profiler_type=ProfilerBase.Type.BANDWIDTH,
        )
    
    def command_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        if not isinstance(core, CompanionCore):
            raise Exception(f"InterconnectBandwidthProfiler can only be registered to Companion cores. Core ID {core.core_id} is not a Companion core.")
        
        if command.cmd_id != "dispatch_command_with_module":
            return
        
        module_id = None
        companion_cmd = None
        
        if len(command.args) > 0:
            module_id = command.args[0]
        elif "module_id" in command.kwargs:
            module_id = command.kwargs["module_id"]
            
        if len(command.args) > 1:
            companion_cmd = command.args[1]
        elif "cmd" in command.kwargs:
            companion_cmd = command.kwargs["cmd"]
            
        if module_id is None or companion_cmd is None:
            logger.warning(f"Received a dispatch_command_with_module command without module_id or cmd information. Skipping profiling for this command. Command args: {command.args}, kwargs: {command.kwargs}")
            return
        
        if module_id != "BOOKSIM":
            return
        
        n_flits: int = companion_cmd.kwargs.get("n_flits", None)
        
        if n_flits is None:
            logger.warning(f"Received a BOOKSIM command without n_flits information. Skipping profiling for this command. Command kwargs: {companion_cmd.kwargs}")
            return
        
        self.add_entry(issue_time, commit_time, n_flits)
        

class ThreadUtilizationProfiler(_GroupedProfilerTemplate):
    ALL_COMMANDS = lambda x: True
    NO_VAR_WAIT = lambda cmd: cmd.cmd_id not in ["var_atomic_barrier", "var_conditional_wait", "var_atomic_wait", "var_atomic_compare_and_swap"]

    def __init__(self, device: MCA_DeviceBase, core_group: MCA_CoreGroup, slot_id: str, n_max_entries: int=1024, command_filter: Callable[[Command,], bool]=NO_VAR_WAIT):
        super().__init__(
            device=device,
            target_core_ids=[core_id for core_id in core_group.core_ids],
            metric_name=f"{slot_id} Thread Utilization",
            metric_unit="%",
            n_max_entries=n_max_entries,
            profiler_type=ProfilerBase.Type.UTILIZATION,
        )
        
        self.slot_id = slot_id
        self.core_id_to_agent_id = {core_id: agent_id for agent_id, core_id in enumerate(core_group.core_ids)}
        self.command_filter = command_filter
    
    def command_debug_hook(self, core: Core, kernel: Kernel, command: Command, issue_time: int, commit_time: int):
        if kernel.slot_id != self.slot_id:
            return
        if core.core_id not in self.core_id_to_agent_id:
            logger.warning(f"Received a command from core ID {core.core_id} which is not in the target core group for this profiler. Skipping profiling for this command.")
            return
        if not self.command_filter(command):
            return
        
        agent_id = self.core_id_to_agent_id[core.core_id]
        self.add_entry(agent_id, issue_time, commit_time)
