import abc
import functools
from typing import Any, Callable

from neuromta.framework.logger import logger
from neuromta.framework.core import *


__all__ = [
    "CompanionModule",
    "CompanionCore",
    "COMPANION_CORE_ID",
]


COMPANION_CORE_ID = "COMPANION"


class CompanionModule(metaclass=abc.ABCMeta):
    def __init__(self):
        self.module_id = None
    
    @abc.abstractmethod
    def update_cycle_time(self, cycle_time: int):
        pass
    
    @abc.abstractmethod
    def create_command(self, *args, callback: Callable, **kwargs) -> Any:
        pass
    
    @abc.abstractmethod
    def dispatch_command(self, cmd: Any, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        pass

    @abc.abstractmethod
    def check_command_executed(self, cmd: Any) -> bool:
        pass


class CompanionCore(Core):
    def __init__(self):
        super().__init__(core_id=COMPANION_CORE_ID)
        
        self._companion_modules: dict[str, CompanionModule] = {}
        # self._suspended_dispatch_context: dict[int, Kernel] = {}
        # self._ongoing_command_context: dict[int, Kernel] = {}
        self._command_execution_context: dict[int, list[Kernel, int]] = {}
        self._is_any_cmd_retired: bool = False
        
    def register_companion_module(self, module_id: str, module: CompanionModule):
        if not isinstance(module, CompanionModule):
            raise Exception(f"[ERROR] The module must be an instance of CompanionModule, but got {type(module)}")
        
        self._companion_modules[module_id] = module
        module.module_id = module_id
        
    def get_companion_module(self, module_id: str) -> CompanionModule:
        return self._companion_modules.get(module_id, None)

    def update_cycle_time_companion_modules(self, cycle_time: int):
        for cmod in self._companion_modules.values():
            cmod.update_cycle_time(cycle_time=cycle_time)
            
    def update_cycle_time_until_cmd_executed(self) -> int:
        if len(self._companion_modules) == 0:
            return

        cycle_time = 0
        
        if len(self._command_execution_context) == 0:
            return cycle_time
        
        self._is_any_cmd_retired = False
        
        while True:
            self.update_cycle_time_companion_modules(1)
            cycle_time += 1

            if self._is_any_cmd_retired:
                break
            
        return cycle_time
    
    @core_command_method
    def dispatch_command_with_module(self, module_id: str, cmd):
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")
        
        cmd_id = id(cmd)
        
        if cmd_id in self._command_execution_context:
            raise Exception(f"[ERROR] Command id {cmd_id} is already being tracked in core '{self.core_id}'")
        
        kernel = get_global_kernel_context()
        self._command_execution_context[cmd_id] = [kernel, 1]  # 2 means dispatch and execute
        kernel.set_blocked(True)
        
        dispatch_callback = functools.partial(self._callback_common, module_id)
        execute_callback = functools.partial(self._callback_common, module_id)
        cmod.dispatch_command(cmd, dispatch_callback, execute_callback)
        
    @core_command_method
    def wait_command_with_module(self, module_id: str, cmd):
        cmd_id = id(cmd)
        
        kernel, cnt = self._command_execution_context.get(cmd_id, (None, 0))
        if kernel is None:
            return  # The command has already been executed and there is no need to wait.
        
        if cnt <= 0:
            kernel.set_blocked(False)
            self._is_any_cmd_retired = True
            del self._command_execution_context[cmd_id]
        else:
            kernel.set_blocked(True)
            self._command_execution_context[cmd_id][1] = cnt + 1  # increase the count of waiters
        
    def _callback_common(self, module_id: str, cmd):
        cmd_id = id(cmd)
        kernel, cnt = self._command_execution_context.get(cmd_id, (None, 0))
        
        if kernel is None:
            return
        
        cnt -= 1
        
        if cnt <= 0:
            kernel.set_blocked(False)
            self._is_any_cmd_retired = True
            del self._command_execution_context[cmd_id]
    
    def send_companion_command(self, module_id: str, *args, **kwargs) -> Any:
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")

        cmd = cmod.create_command(*args, **kwargs)
        self.dispatch_command_with_module(module_id, cmd)
        self.wait_command_with_module(module_id, cmd)