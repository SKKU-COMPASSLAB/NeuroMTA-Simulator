import abc
from typing import Any

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
        self._ongoing_cmds = []
    
    @abc.abstractmethod
    def update_cycle_time(self, cycle_time: int):
        pass
    
    @abc.abstractmethod
    def create_command(self, *args, **kwargs) -> Any:
        pass
    
    @abc.abstractmethod
    def dispatch_command(self, cmd: Any) -> bool:
        pass

    @abc.abstractmethod
    def check_command_executed(self, cmd: Any) -> bool:
        pass
    
    def register_command(self, cmd: Any):
        self._ongoing_cmds.append(cmd)
        
    def retire_executed_commands(self):
        self._ongoing_cmds = [cmd for cmd in self._ongoing_cmds if not self.check_command_executed(cmd)]

    @property
    def is_busy(self):
        return len(self._ongoing_cmds) > 0

    @property
    def has_executed_ongoing_cmd(self) -> bool:
        for cmd in self._ongoing_cmds:
            if self.check_command_executed(cmd):
                return True
        return False


class CompanionCore(Core):
    def __init__(self):
        super().__init__(core_id=COMPANION_CORE_ID)
        
        self._companion_modules: dict[str, CompanionModule] = {}
        
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
        
        while True:
            if all(not cmod.is_busy for cmod in self._companion_modules.values()):
                break

            if any(cmod.has_executed_ongoing_cmd for cmod in self._companion_modules.values()):
                break

            self.update_cycle_time_companion_modules(1)
            cycle_time += 1

        for cmod in self._companion_modules.values():
            cmod.retire_executed_commands()
            
        return cycle_time
            
    @core_conditional_command_method
    def dispatch_command_with_module(self, module_id: str, cmd):
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")
        
        return cmod.dispatch_command(cmd)
    
    @core_command_method
    def register_command_with_module(self, module_id: str, cmd: Any):
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")

        cmod.register_command(cmd)

    @core_conditional_command_method
    def wait_command_with_module(self, module_id: str, cmd) -> bool:
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")

        return cmod.check_command_executed(cmd)
    
    @core_kernel_method
    def send_companion_command(self, module_id: str, *args, **kwargs) -> Any:
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"[ERROR] Companion module '{module_id}' not found in core '{self.core_id}'")
        
        cmd = cmod.create_command(*args, **kwargs)

        self.dispatch_command_with_module(module_id, cmd)
        self.register_command_with_module(module_id, cmd)
        self.wait_command_with_module(module_id, cmd)