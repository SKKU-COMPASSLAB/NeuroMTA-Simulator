import abc
import functools
from typing import Any, Callable
from neuromta.framework.core import *


__all__ = [
    "CompanionCommandSignature",
    "CompanionModule",
    "CompanionCore",
    "COMPANION_CORE_ID",
]


COMPANION_CORE_ID = "COMPANION"


class CompanionCommandSignature:
    def __init__(self, module_id: str, capsule: Any, kwargs: dict[str, Any]=None):
        self.module_id = module_id
        self.capsule = capsule
        self.kwargs = kwargs if kwargs is not None else {}

        self.issue_time = 0
        self.commit_time = 0
        
    @property
    def capsule_id(self) -> int:
        return id(self.capsule)  # TODO: better way to identify the capsule?
        

class CompanionModule(metaclass=abc.ABCMeta):
    def __init__(self):
        self.module_id = None
        self.companion_core: CompanionCore = None
    
    @abc.abstractmethod
    def update_cycle_time(self, cycle_time: int):
        pass
    
    @abc.abstractmethod
    def create_command(self, *args, **kwargs) -> CompanionCommandSignature:
        pass
    
    @abc.abstractmethod
    def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        pass


class CompanionCore(Core):
    def __init__(self):
        super().__init__(core_id=COMPANION_CORE_ID)
        
        self._companion_modules: dict[str, CompanionModule] = {}
        self._command_execution_context: dict[int, list[Kernel, int]] = {}
        self._capsule_to_signature_mapping: dict[int, CompanionCommandSignature] = {}
        self._is_any_cmd_retired: bool = False
        
        self._registered_companion_debug_hooks: dict[str, dict[str, Callable]] = {}
        
    def dump_core_states(self):
        return {}
    
    def load_core_states(self, states: dict):
        pass
        
    def register_companion_module_hook(self, module_id: str, hook: Callable) -> str:
        def create_hook_id(i: int) -> str:
            return f"hook_{i}"
        
        MAX_HOOK_NUM = 32
        
        for i in range(MAX_HOOK_NUM):
            hook_id = create_hook_id(i)
            if hook_id not in self._registered_companion_debug_hooks[module_id]:
                self._registered_companion_debug_hooks[module_id][hook_id] = hook
                return hook_id
        
        raise Exception(f"Cannot register command debug hook since the maximum number of hooks ({MAX_HOOK_NUM}) is reached. Please remove some hooks before adding new ones.")
    
    def unregister_companion_module_hook(self, module_id: str, hook_id: str):
        if module_id not in self._registered_companion_debug_hooks.keys():
            raise Exception(f"Companion module '{module_id}' not found in core '{self.core_id}'")
        
        if hook_id not in self._registered_companion_debug_hooks[module_id].keys():
            raise Exception(f"Hook id '{hook_id}' not found in companion module '{module_id}' in core '{self.core_id}'")
        
        del self._registered_companion_debug_hooks[module_id][hook_id]
        
    def register_companion_module(self, module_id: str, module: CompanionModule):
        if not isinstance(module, CompanionModule):
            raise Exception(f"The module must be an instance of CompanionModule, but got {type(module)}")
        
        self._companion_modules[module_id] = module
        self._registered_companion_debug_hooks[module_id] = {}
        module.module_id = module_id
        module.companion_core = self
        
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
    def dispatch_command_with_module(self, module_id: str, cmd: CompanionCommandSignature):
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"Companion module '{module_id}' not found in core '{self.core_id}'")

        cmd_id = cmd.capsule_id
        
        if cmd_id in self._command_execution_context:
            raise Exception(f"Command id {cmd_id} is already being tracked in core '{self.core_id}'")
        
        kernel = get_global_kernel_context()
        self._command_execution_context[cmd_id] = [kernel, 1]  # 2 means dispatch and execute
        self._capsule_to_signature_mapping[cmd_id] = cmd
        kernel.set_blocked(self, True)
        
        callback = functools.partial(self._callback_common, module_id)
        cmod.dispatch_command(cmd, None, callback)
        
        cmd.issue_time = self.timestamp
        
    def _callback_common(self, module_id: str, capsule: Any):
        cmd_id = id(capsule)
        
        cmd = self._capsule_to_signature_mapping.get(cmd_id, None)
        if cmd is None:
            raise Exception(f"Command with id {cmd_id} not found in core '{self.core_id}'")
        
        kernel, cnt = self._command_execution_context.get(cmd_id, (None, 0))
        if kernel is None:
            return
        
        cnt -= 1
        
        if cnt <= 0:
            kernel.set_blocked(self, False)
            self._is_any_cmd_retired = True
            cmd.commit_time = self.timestamp
            
            for hook in self._registered_companion_debug_hooks.get(module_id, {}).values():
                hook(cmd)
            
            del self._command_execution_context[cmd_id]
            del self._capsule_to_signature_mapping[cmd_id]
    
    def send_companion_command(self, module_id: str, *args, **kwargs) -> Any:
        cmod = self.get_companion_module(module_id)
        if cmod is None:
            raise ValueError(f"Companion module '{module_id}' not found in core '{self.core_id}'")

        cmd = cmod.create_command(*args, **kwargs)
        self.dispatch_command_with_module(module_id, cmd)
        