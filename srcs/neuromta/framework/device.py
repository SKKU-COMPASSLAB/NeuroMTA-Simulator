import os
import sys
import shutil
from typing import Sequence, Callable  #, Any

from neuromta.framework.core import Core, Kernel, Command, RPCMessage, ThreadGroup
from neuromta.framework.companion import CompanionCore
from neuromta.framework.logger import logger


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
        
        self._timestamp = 0
        
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
                raise Exception(f"Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
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
        
        return self
    
    def reset_simulation(self):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.reset_simulation()
    
    def set_command_debug_verbosity(self, verbose: bool=True):
        if verbose:
            for core in self._cores.values():
                if core.core_id in self._verbose_hook_ids:
                    continue    # already registered
                
                hook_id = core.register_command_debug_hook(self._verbose_command_debug_hook)
                self._verbose_hook_ids[core.core_id] = hook_id
        else:
            for core in self._cores.values():
                if core.core_id not in self._verbose_hook_ids:
                    continue    # not registered
                
                hook_id = self._verbose_hook_ids[core.core_id]
                core.unregister_command_debug_hook(hook_id)
                del self._verbose_hook_ids[core.core_id]
    
    def _verbose_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        callstack = kernel.callstack if kernel else "N/A"
        if len(callstack) > 100:
            callstack = callstack[:47] + " ... " + callstack[-47:]
        logger.debug(f"{issue_time:<6d} - {commit_time:<6d} | {core.core_id.__str__():<10s} | {callstack:<100s} | command: {cmd.cmd_id}")

    def run_single_step(self, cycle_resolution: int = 1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        remaining_cycles = None
            
        for core_id, core in self.initialized_cores.items():
            core.rpc_update_routine()
        
        for core_id, core in self.initialized_cores.items():
            if core.is_idle:
                continue
            c = core.get_remaining_cycles()
            
            if remaining_cycles is None:
                remaining_cycles = c
            elif c is not None:
                remaining_cycles = min(remaining_cycles, c)

        if remaining_cycles == 0 or remaining_cycles is None:
            remaining_cycles = self.companion_core.update_cycle_time_until_cmd_executed()
        
            if remaining_cycles == 0 or remaining_cycles is None:
                remaining_cycles = cycle_resolution
        else:
            self.companion_core.update_cycle_time_companion_modules(cycle_time=remaining_cycles)

        for core_id, core in self.initialized_cores.items():
            if core.is_idle:
                core._timestamp += remaining_cycles
            else:
                core.update_cycle_time(cycle_time=remaining_cycles)
                
        self._timestamp += remaining_cycles

    def run_kernels(
        self, 
        cycle_resolution:   int  = 1,   # the number of cycles to update when all the cores are waiting and returning (0 | None) as the minimum remaining cycles
        max_steps:          int  = -1,  # the maximum number of steps to run
        max_timestamp:      int  = -1,  # the maximum timestamp to run
        
        sync_target_core_groups: Sequence[Sequence[int]] = None,  # the target core groups to synchronize after each step; if None, all cores are synchronized
    ):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        if sync_target_core_groups is None:
            core_ids = list(self._cores.keys())
            sync_target_core_groups = [core_ids]
        else:
            core_ids = []
            for group in sync_target_core_groups:
                core_ids.extend(group)
                
        step_cnt = 0
        deadlock_cnt = 0

        # while not all(self.initialized_cores[core_id].is_idle for core_id in core_ids):
        while True:
            self.run_single_step(cycle_resolution=cycle_resolution)
            
            # break condition: step count  
            step_cnt += 1
            if step_cnt >= max_steps > 0:
                logger.debug(f"Reached maximum steps: {max_steps}. Stopping simulation.")
                self.debug_current_execution_state(*core_ids)
                break
            
            # break condition: timestamp
            if self.timestamp >= max_timestamp > 0:
                logger.debug(f"Reached maximum timestamp: {max_timestamp}. Stopping simulation.")
                self.debug_current_execution_state(*core_ids)
                break
            
            # break condition: idle synchronization targets
            is_idle = False
            for sync_group in sync_target_core_groups:
                is_idle = all(self.initialized_cores[core_id].is_idle for core_id in sync_group)
                if not is_idle:
                    break
            if is_idle:
                logger.debug(f"Synchronization target cores are idle. Stopping simulation.")
                break
            
            # break condition: deadlock
            if self.check_deadlock(*core_ids):
                deadlock_cnt += 1
                if deadlock_cnt >= 10:  # threshold for deadlock detection
                    logger.error(f"Deadlock detected among cores {core_ids}. Stopping simulation.")
                    self.debug_current_execution_state(*core_ids)
                    break
            else:
                deadlock_cnt = 0

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
            
    def register_kernel_debug_hook(self, hook: Callable, slot_id: str, core_ids: list[int]=None):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        if isinstance(core_ids, (int, str)):
            core_ids = [core_ids]
        
        if core_ids is None:
            core_ids = list(self._cores.keys())
        
        for core_id in core_ids:
            core = self._cores[core_id]
            core.register_kernel_debug_hook(hook, slot_id)
            
    def check_deadlock(self, *core_ids: int | str) -> bool:
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        is_not_deadlock = False
        
        for core_id in core_ids:
            core = self._cores[core_id]
            if not core.check_all_blocked():
                is_not_deadlock = True
                break
            if core.core_id == self.companion_core.core_id and not self.companion_core.check_idle():
                is_not_deadlock = True
                break
            
        return not is_not_deadlock
    
    def debug_current_execution_state(self, *core_ids: int | str):
        for core_id in core_ids:
            core = self.initialized_cores[core_id]
            if len(core._dispatched_main_kernels) == 0:
                continue
            logger.debug(f"Core '{core_id}'")
            for slot_id, kernel in core._dispatched_main_kernels.items():
                if kernel.is_finished(core):
                    continue
                
                logger.debug(f"  KERNEL {kernel.callstack}")
                step = kernel.current_step(core)
                
                def print_step_info(step):
                    if isinstance(step, Kernel):
                        if step.is_finished(core):
                            return
                        logger.debug(f"    calls KERNEL {step.callstack}")
                        print_step_info(step.current_step(core))
                    elif isinstance(step, ThreadGroup):
                        for sub_step in step:
                            if not sub_step.is_finished(core):
                                logger.debug(f"    calls THREAD {sub_step.callstack}" + (" (blocked)" if sub_step.is_blocked else ""))
                                print_step_info(sub_step.current_step(core))
                    else:
                        logger.debug(f"     -> COMMAND {step}")
                
                print_step_info(step)
        
    @property
    def timestamp(self) -> int:
        # t = [core.timestamp for core in self._cores.values()]
        # return max(t) if t else 0
        return self._timestamp

    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())
    
    @property
    def initialized_cores(self) -> dict[str, Core]:
        return self._cores