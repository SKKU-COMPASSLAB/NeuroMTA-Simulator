import os
import sys
import shutil
from typing import Sequence, Callable  #, Any

from neuromta.framework.core import Core, Kernel, Command, RPCMessage
from neuromta.framework.companion import CompanionCore
from neuromta.framework.logger import logger
# from neuromta.framework.tracer import Tracer, TraceEntry


__all__ = [
    "Device",
]


class Device:
    def __init__(self):
        self._cores: dict[str, Core] = None
        self._verbose_hook_ids: dict[str, str] = {}

        self._rpc_req_send_inbox: dict[str, list[RPCMessage]] = {}
        self._rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = {}
        
        self._companion_core = CompanionCore()
        
    def get_core_from_id(self, core_id: int) -> Core:
        return self._cores[core_id]
        
    @property
    def companion_core(self) -> CompanionCore:
        return self._companion_core 
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        for core in self._cores.values():
            core.change_sim_model_options(use_cycle_model, use_functional_model)
        
    def _register_core(self, name: str, core: Core | Sequence[Core]):
        if isinstance(core, Sequence):
            for idx, item in enumerate(core):
                if isinstance(item, (Core, Sequence)):
                    self._register_core(f"{name}[{idx}]", item)
        elif isinstance(core, Core):
            if core.core_id in self._cores.keys():
                raise Exception(f"[ERROR] Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core

    def initialize(self):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if name == "_cores":
                continue    # skip internal members
            
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._rpc_req_send_inbox = {core.core_id: [] for core in self._cores.values()}
        self._rpc_rsp_send_inbox = {core.core_id: [] for core in self._cores.values()}

        for core in self._cores.values():
            core.initialize_kernel_dispatch_queue()
            core.initialize_mp_queue_inbox(rpc_req_send_inbox=self._rpc_req_send_inbox, rpc_rsp_send_inbox=self._rpc_rsp_send_inbox)
            
        for core in self._cores.values():
            hook_id = core.register_command_debug_hook(self._verbose_command_debug_hook)
            self._verbose_hook_ids[core.core_id] = hook_id
        
        return self
    
    def _verbose_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        callstack = kernel.callstack if kernel else "N/A"
        if len(callstack) > 100:
            callstack = callstack[:47] + " ... " + callstack[-47:]
        logger.debug(f"{issue_time:<6d} - {commit_time:<6d} | {core.core_id.__str__():<10s} | {callstack:<100s} | command: {cmd.cmd_id}")

    def run_single_step(self, cycle_resolution: int = 1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        remaining_cycles = None
            
        for core_id, core in self.cores.items():
            c = core.get_remaining_cycles()
            
            if remaining_cycles is None:
                remaining_cycles = c
            elif c is not None:
                remaining_cycles = min(remaining_cycles, c)
                
        for core_id, core in self.cores.items():
            core.rpc_update_routine()

        if remaining_cycles == 0 or remaining_cycles is None:
            remaining_cycles = self.companion_core.update_cycle_time_until_cmd_executed()
                
            if remaining_cycles == 0 or remaining_cycles is None:
                remaining_cycles = cycle_resolution
        else:
            self.companion_core.update_cycle_time_companion_modules(cycle_time=remaining_cycles)

        for core_id, core in self.cores.items():
            core.update_cycle_time(cycle_time=remaining_cycles)

    def run_kernels(
        self, 
        cycle_resolution:   int  = 1,   # the number of cycles to update when all the cores are waiting and returning (0 | None) as the minimum remaining cycles
        max_steps:          int  = -1,  # the maximum number of steps to run
        max_timestamp:      int  = -1,  # the maximum timestamp to run
    ):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        step_cnt = 0

        while not all(core.is_idle for core in self._cores.values()):  
            step_cnt += 1
            if step_cnt >= max_steps > 0:
                logger.info(f"Reached maximum steps: {max_steps}. Stopping simulation.")
                break
            
            if self.timestamp >= max_timestamp > 0:
                logger.info(f"Reached maximum timestamp: {max_timestamp}. Stopping simulation.")
                break
            
            self.run_single_step(cycle_resolution=cycle_resolution)

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)

    @property
    def timestamp(self) -> int:
        t = [core.timestamp for core in self._cores.values()]
        return max(t) if t else 0

    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())
    
    @property
    def cores(self) -> dict[str, Core]:
        return self._cores