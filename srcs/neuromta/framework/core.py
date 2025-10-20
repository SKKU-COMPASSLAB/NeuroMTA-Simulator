import enum
import functools
import itertools
from typing import Callable, Sequence, Any

from neuromta.framework.logger import logger
from neuromta.framework.debug_utils import *
from neuromta.framework.memory_handle import *
from neuromta.framework.parser_utils import parse_arguments


__all__ = [
    "set_global_context",
    "get_global_core_context",
    "get_global_kernel_context",
    "get_global_pid",
    "get_global_context_mode",
    "GlobalContextMode",
    "new_global_context",

    "RPCMessage",
    "Command",
    "ConditionalCommand",
    
    "KernelPrototype",
    "Kernel",

    "CoreCycleModel",
    "Core",
    
    "jit_prototype",
    "core_command_method",
    "core_conditional_command_method",
    "new_parallel_thread",
]


MAX_COMMAND_NUM_PER_KERNEL = 2 ** 20


#################################################
# Global Context Management
#################################################

class GlobalContextMode(enum.Enum):
    IDLE    = enum.auto()
    COMPILE = enum.auto()
    EXECUTE = enum.auto()
    
_context_mode: GlobalContextMode = GlobalContextMode.IDLE
_core_context: 'Core' = None
_kernel_context: 'Kernel' = None
_parent_kernel_callstack: list[tuple['GlobalContextMode', 'Core', 'Kernel']] = []
    
class new_global_context:
    def __init__(self, context_mode: GlobalContextMode, core_context: 'Core' = None, kernel_context: 'Kernel' = None):
        self.context_mode = context_mode
        self.core_context = core_context
        self.kernel_context = kernel_context
        
        self._history_context_mode   = None
        self._history_core_context   = None
        self._history_kernel_context = None

    def __enter__(self):
        self._history_context_mode   = get_global_context_mode()
        self._history_core_context   = get_global_core_context()
        self._history_kernel_context = get_global_kernel_context()
        
        set_global_context(self.context_mode, self.core_context, self.kernel_context)

    def __exit__(self, exc_type, exc_value, traceback):
        set_global_context(self._history_context_mode, self._history_core_context, self._history_kernel_context)

def set_global_context(context_mode: GlobalContextMode, core: 'Core', kernel: 'Kernel'):
    global _core_context, _kernel_context, _context_mode
    
    if isinstance(context_mode, str):
        context_mode = GlobalContextMode.__members__.get(context_mode.upper())
    if context_mode == GlobalContextMode.COMPILE and not isinstance(kernel, Kernel):
        raise Exception(f"Cannot set global context to COMPILE mode with a non-Kernel object: {kernel}")
    
    _context_mode = context_mode
    _core_context = core
    _kernel_context = kernel
    
def get_global_context_mode() -> GlobalContextMode:
    global _context_mode
    return _context_mode

def get_global_core_context() -> 'Core':
    global _core_context
    return _core_context

def get_global_kernel_context() -> 'Kernel':
    global _kernel_context
    return _kernel_context

def get_global_pid() -> str:
    cid = get_global_core_context().core_id
    kid = get_global_kernel_context().root_kernel.kernel_id
    return f"{cid}.{kid}"   # this is the global PID format: <core_id>.<root kernel_id>

def store_global_parent_kernel_callstack():
    global _parent_kernel_callstack
    
    context_mode = get_global_context_mode()
    core_context = get_global_core_context()
    kernel_context = get_global_kernel_context()

    history = (context_mode, core_context, kernel_context)
    _parent_kernel_callstack.append(history)

def restore_global_parent_kernel_callstack() -> 'Kernel':
    global _parent_kernel_callstack
    
    if len(_parent_kernel_callstack) == 0:
        raise Exception("[ERROR] Cannot pop from parent kernel callstack since it is empty")

    history = _parent_kernel_callstack.pop()
    set_global_context(*history)
    
def get_global_current_parent_kernel_callstack() -> tuple[GlobalContextMode, 'Core', 'Kernel']:
    global _parent_kernel_callstack
    
    if len(_parent_kernel_callstack) == 0:
        raise Exception("[ERROR] Cannot get current parent kernel callstack since it is empty")
    
    return _parent_kernel_callstack[-1]


#################################################
# Decorators for Command and Kernel Methods
#################################################

def jit_prototype(_func: Callable):
    @functools.wraps(_func)
    def __jit_prototype_wrapper(*_args, **_kwargs) -> KernelPrototype:
        prototype = KernelPrototype(func=_func, args=_args, kwargs=_kwargs)

        if get_global_context_mode() == GlobalContextMode.COMPILE:  # the jit prototype is called inside another kernel function
            kernel_context = get_global_kernel_context()
            kernel_context.add_execution_step(prototype)
        
        return prototype
    return __jit_prototype_wrapper

def core_command_method(_func: Callable):
    @functools.wraps(_func)
    def __core_command_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Command:
        if get_global_context_mode() == GlobalContextMode.IDLE:
            raise Exception(f"Command method '{_func.__name__}' cannot be called in IDLE context since it is neither in COMPILE nor EXECUTE context.")
        if get_global_context_mode() == GlobalContextMode.EXECUTE:
            # with print_log_execution_time(f"RUNNING COMMAND '{_func.__name__}'"):
                return _func(_core, *_args, **_kwargs)

        if not isinstance(_core, Core):
            raise Exception(f"Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        elif not isinstance(kernel_context, Kernel):
            raise Exception(f"Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function. The current kernel context is not an instance of Kernel, but {type(kernel_context).__name__}")
        
        cmd = Command(
            _func.__name__,     # the command ID is the name of the function
            *_args,             # the arguments of the command
            **_kwargs           # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            logger.warning(f"Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    
    __core_command_method_wrapper._is_command_method = True  # mark this function as a command method
    __core_command_method_wrapper._is_conditional = False  # mark this function as a non-conditional command method
    return __core_command_method_wrapper

def core_conditional_command_method(_func: Callable):
    def __core_command_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Command:
        if get_global_context_mode() == GlobalContextMode.IDLE:
            raise Exception(f"Command method '{_func.__name__}' cannot be called in IDLE context since it is neither in COMPILE nor EXECUTE context.")
        if get_global_context_mode() == GlobalContextMode.EXECUTE:
            # with print_log_execution_time(f"RUNNING COMMAND '{_func.__name__}'"):
                return _func(_core, *_args, **_kwargs)

        if not isinstance(_core, Core):
            raise Exception(f"Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        elif not isinstance(kernel_context, Kernel):
            raise Exception(f"Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function. The current kernel context is not an instance of Kernel, but {type(kernel_context).__name__}")

        cmd = ConditionalCommand(
            _func.__name__,     # the command ID is the name of the function
            *_args,             # the arguments of the command
            **_kwargs           # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            logger.warning(f"Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    
    __core_command_method_wrapper._is_command_method = True  # mark this function as a command method
    __core_command_method_wrapper._is_conditional = True  # mark this function as a conditional command method
    return __core_command_method_wrapper

class new_parallel_thread:
    def __init__(self, p_kernel_id: str=None):
        self.p_kernel_id = p_kernel_id
    
    def __enter__(self):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception("[ERROR] Cannot create a parallel kernel since the global context mode is not COMPILE")
        
        kernel_context = get_global_kernel_context()
        parallel_kernel = kernel_context.add_parallel_kernel_step(p_kernel_id=self.p_kernel_id)
        
        store_global_parent_kernel_callstack()
        set_global_context(GlobalContextMode.COMPILE, get_global_core_context(), parallel_kernel)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception("[ERROR] Cannot end parallel kernel since the global context mode is not COMPILE")

        restore_global_parent_kernel_callstack()


#################################################
# Implementation
#################################################
        
class RPCMessage:
    def __init__(self, src_core_id: str, dst_core_id: str, cmd_id: str):
        self.msg_type = 0        # 0 for request, 1 for response
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id
        
        kernel_context = get_global_kernel_context()
        if kernel_context is None:
            self.kernel_id = "UNKNOWN"
            self.root_kernel_id = "UNKNOWN"
        else:
            self.kernel_id = kernel_context.kernel_id
            self.root_kernel_id = kernel_context.root_callstack
        
        self.cmd_id = cmd_id
        self.args = []
        self.kwargs = {}
        
        self.msg_id:     str = None  # message ID is None by default

    def with_args(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        return self
        
    def response(self, rpc_kernel: 'Kernel') -> 'RPCMessage':
        msg = RPCMessage(
            src_core_id=self.src_core_id,
            dst_core_id=self.dst_core_id,
            cmd_id=self.cmd_id,
        ).with_args(
            *self.args,
            **self.kwargs
        )
        
        msg.msg_id = self.msg_id
        
        return msg
        
    def copy_args_from_rsp(self, rsp_msg: 'RPCMessage'):
        self.args = rsp_msg.args
        self.kwargs = rsp_msg.kwargs
    
    def __str__(self):
        return f"RPCMessage(msg_id={self.msg_id}, src_core_id={self.src_core_id}, dst_core_id={self.dst_core_id}, kernel_id={self.kernel_id}, cmd_id={self.cmd_id})"


class Command:
    def __init__(
        self, 
        cmd_id: str,
        *args,
        **kwargs
    ):
        self.cmd_id = cmd_id
        self.args   = args
        self.kwargs = kwargs

        self._cached_cycle: int = None
        self._cached_cycle_slack: int = 0
        self._cached_issue_time: int = None
        self._is_behavioral_model_called: bool = False
        
    def get_remaining_cycles(self, core: 'Core', kernel: 'Kernel') -> int:
        if self._cached_cycle is None:
            self._cached_cycle = self.run_cycle_model(core, kernel)
            
            if self._cached_cycle is None:
                self._cached_cycle = 1
            
            self._cached_cycle = max(0, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, core: 'Core', kernel: 'Kernel', cycle_time: int):
        if self._cached_issue_time is None:
            self._cached_issue_time = core.timestamp
            
        if cycle_time < 0:
            raise ValueError(f"Cycle time cannot be negative: {cycle_time}")

        self._cached_cycle_slack += cycle_time
        
        if self.get_remaining_cycles(core, kernel) <= 0 and not self._is_behavioral_model_called:
            flag = self.run_behavioral_model(core, kernel)
            
            if flag is not None:
                raise Exception(f"Behavioral model for command '{self.cmd_id}' returned '{flag}' even though the command is not conditional. Use 'core_conditional_command_method' instead if you want to implement any retry operation.")

        if self.is_finished(core, kernel):
            core.run_command_debug_hook(kernel=kernel, cmd=self, issue_time=self._cached_issue_time, commit_time=core.timestamp+cycle_time)

    def run_behavioral_model(self, core: 'Core', kernel: 'Kernel'):
        with new_global_context(GlobalContextMode.EXECUTE, core, kernel):
            model = core.get_behavioral_model(self.cmd_id)
            try:
                self._is_behavioral_model_called = True
                return model(*self.args, **self.kwargs)
            except Exception as e:
                logger.error(f"Exception occurred while executing behavioral model for command '{self.cmd_id}': {e}")
                logger.error(f"  - Core: {type(core).__name__}(id={core.core_id}) | kernel: {kernel.callstack} | args: {self.args} | kwargs: {self.kwargs}")
                raise e
        
    def run_cycle_model(self, core: 'Core', kernel: 'Kernel') -> int:
        with new_global_context(GlobalContextMode.EXECUTE, core, kernel):
            model = core.get_cycle_model(self.cmd_id)

            if model is None:
                return None
            elif isinstance(model, int):
                return model
            elif callable(model):
                try:
                    return model(*self.args, **self.kwargs)
                except Exception as e:
                    logger.error(f"Exception occurred while executing cycle model for command '{self.cmd_id}': {e}")
                    logger.error(f"  - Core: {type(core).__name__}(id={core.core_id}) | kernel: {kernel.callstack} | args: {self.args} | kwargs: {self.kwargs}")
                    raise e
            
            return None
        
    def is_finished(self, core: 'Core', kernel: 'Kernel') -> bool:
        return self.get_remaining_cycles(core, kernel) <= 0 and self._is_behavioral_model_called

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"
    
    
class ConditionalCommand(Command):
    def __init__(self, cmd_id: str, *args, **kwargs):
        super().__init__(cmd_id, *args, **kwargs)
        
        self._is_async_finished = False
    
    def get_remaining_cycles(self, core: 'Core', kernel: 'Kernel') -> int:
        return None  # TODO: currently, the conditional command is not used for remaining cycle estimation. However, it would be better if we can predict the execution time of the conditional command...

    def update_cycle_time(self, core: 'Core', kernel: 'Kernel', cycle_time: int):
        if self._cached_issue_time is None:
            self._cached_issue_time = core.timestamp

        self._is_async_finished = self.run_behavioral_model(core, kernel)

        if self.is_finished(core, kernel):
            core.run_command_debug_hook(kernel=kernel, cmd=self, issue_time=self._cached_issue_time, commit_time=core.timestamp+cycle_time)

    def is_finished(self, core: 'Core', kernel: 'Kernel') -> bool:
        return self._is_async_finished and self._is_behavioral_model_called


class ThreadGroup(list['Kernel']):
    def __init__(self):
        super().__init__()
    
    def append(self, kernel: 'Kernel'):
        if not isinstance(kernel, Kernel):
            raise TypeError(f"Cannot add kernel '{kernel}' to the parallel kernel group since it is not an instance of Kernel")
        return super().append(kernel)

    def get_remaining_cycles(self, core: 'Core') -> int:
        remaining_cycles = None
        for kernel in self:
            tmp = kernel.get_remaining_cycles(core)
            if remaining_cycles is None:
                remaining_cycles = tmp
            elif tmp is not None:
                remaining_cycles = min(remaining_cycles, tmp)
        return remaining_cycles

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        for kernel in self:
            kernel.update_cycle_time(core, cycle_time)
    
    def is_finished(self, core: 'Core') -> bool:
        return all(kernel.is_finished(core) for kernel in self)
    
    
class KernelPrototype:
    def __init__(self, func: Callable, args: Sequence[Any], kwargs: dict[str, Any]):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.compiled_kernel_id = func.__name__
        
    def compile(self) -> 'Kernel':
        with Kernel(kernel_id=self.compiled_kernel_id) as kernel:
            try:
                self.func(*self.args, **self.kwargs)
            except Exception as e:
                logger.error(f"Exception occurred while compiling kernel '{kernel.kernel_id}': {e}")
                logger.error(f"  - args: {self.args} | kwargs: {self.kwargs}")
                raise e

        return kernel


class Kernel:
    def __init__(self, kernel_id: str):
        self.kernel_id = kernel_id
        self.root_kernel: Kernel = None
        
        self._execution_steps: list[Command | ThreadGroup | KernelPrototype] = []
        self._execution_cursor: int = 0
        
        self._is_blocked: bool = False
        
    def set_blocked(self, flag: bool=True):
        self._is_blocked = flag
        
    @property
    def is_blocked(self) -> bool:
        return self._is_blocked
        
    def __enter__(self):
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            raise Exception(f"Cannot enter kernel '{self.kernel_id}' since the global context mode is already COMPILE")
        
        store_global_parent_kernel_callstack()
        set_global_context(GlobalContextMode.COMPILE, get_global_core_context(), self)
        
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"Cannot exit kernel '{self.kernel_id}' since the global context mode is not COMPILE")
        
        restore_global_parent_kernel_callstack()
        
    def add_execution_step(self, step: Command | ThreadGroup | KernelPrototype):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"Cannot add execution step '{step}' to the kernel '{self.kernel_id}' since it is not in compile mode")
        if isinstance(step, Kernel):
            raise Exception(f"Cannot add a kernel '{step.kernel_id}' as an execution step to the kernel '{self.kernel_id}'. Use 'add_parallel_kernel_step()' instead to add a parallel kernel step.")
        
        self._execution_steps.append(step)
            
    def add_parallel_kernel_step(self, p_kernel_id: str=None) -> 'Kernel':
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"Cannot add parallel kernel step to the kernel '{self.kernel_id}' since it is not in compile mode")
        
        if len(self._execution_steps) == 0:
            self._execution_steps.append(ThreadGroup())
        elif not isinstance(self._execution_steps[-1], ThreadGroup):
            self._execution_steps.append(ThreadGroup())

        parallel_kernel_idx = len(self._execution_steps[-1])
        if p_kernel_id is None:
            p_kernel_id = parallel_kernel_idx

        parallel_kernel = Kernel(kernel_id=f"{p_kernel_id}")
        parallel_kernel.root_kernel = self

        self._execution_steps[-1].append(parallel_kernel)

        return parallel_kernel
        
    def get_remaining_cycles(self, core: 'Core') -> int:
        if self.is_blocked:
            return None
        
        cycle = None
        
        while not self.is_finished(core):        
            step = self.current_step(core)

            if isinstance(step, ConditionalCommand):
                cycle = step.get_remaining_cycles(core, kernel=self)
            elif isinstance(step, Command):
                cycle = step.get_remaining_cycles(core, kernel=self)
            else:
                cycle = step.get_remaining_cycles(core=core)
                
            if cycle is None:
                break
            elif cycle == 0:
                self.update_cycle_time(core, cycle_time=0)
            else:
                break
            
        return cycle

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        if self.is_finished(core) or self.is_blocked:
            return
        
        step = self.current_step(core)

        if isinstance(step, Command):
            step.update_cycle_time(core, self, cycle_time)
            if step.is_finished(core, self):
                self._execution_cursor += 1
        else:
            step.update_cycle_time(core, cycle_time)
            if step.is_finished(core):
                self._execution_cursor += 1
            
    def current_step(self, core: 'Core') -> 'Command | ThreadGroup':
        if self.is_finished(core):
            return None
        
        if isinstance(self._execution_steps[self._execution_cursor], KernelPrototype):
            kernel_step = self._execution_steps[self._execution_cursor].compile()
            kernel_step.root_kernel = self
            self._execution_steps[self._execution_cursor] = kernel_step
        
        return self._execution_steps[self._execution_cursor]
    
    def recursive_current_commands(self, core: 'Core') -> list[Command]:
        if self.is_finished(core):
            return []
        
        commands = []
        step = self._execution_steps[self._execution_cursor]
        
        if isinstance(step, Command):
            commands.append(step)
        elif isinstance(step, Kernel):
            commands = step.recursive_current_commands(core)
        elif isinstance(step, ThreadGroup):
            for k in step:
                commands.extend(k.recursive_current_commands(core))
                
        return commands
    
    def is_finished(self, core: 'Core') -> bool:
        return (self._execution_cursor >= len(self._execution_steps)) and (not self.is_blocked)
    
    @property
    def root_callstack(self) -> str | None:
        if self.root_kernel is None:
            return None
        return self.root_kernel.callstack
    
    @property
    def root_kernel_id(self) -> str | None:
        if self.root_kernel is None:
            return self.kernel_id
        if self.root_kernel.root_kernel is None:
            return self.root_kernel.kernel_id
        return self.root_kernel.root_kernel_id
    
    @root_kernel_id.setter
    def root_kernel_id(self, value):
        if isinstance(value, str):
            self.root_kernel = Kernel(kernel_id=value)
        else:
            self.root_kernel = None
            
    @property
    def callstack(self) -> str:
        if self.root_callstack is None:
            return self.kernel_id
        return f"{self.root_callstack}::{self.kernel_id}"
    

class CoreCycleModel:
    def __init__(self):
        pass

class Core:
    def __init__(self, core_id: int, cycle_model: CoreCycleModel=None):
        self.core_id = core_id

        self._cycle_model: CoreCycleModel = cycle_model

        self._dispatched_main_kernels:      dict[str, Kernel] = {}
        self._dispatched_rpc_kernels:       dict[str, Kernel] = {}
        self._dispatched_rpc_msg_mappings:  dict[str, RPCMessage] = {}  # RPC kernel -> RPC request message (given from the source core)

        self._suspended_main_kernels: dict[str, list[Kernel | KernelPrototype]] = {}
        self._suspended_rpc_req_msg: dict[str, RPCMessage] = {}
        # self._suspended_rpc_rsp_msg: dict[str, RPCMessage] = {}
        self._suspended_rpc_to_main_kernels_mapping: dict[str, str] = {}  # RPC request message ID -> main kernel slot ID (to resume the main kernel when the RPC response is received)
        self._suspended_rpc_kernel_blocking_condition: dict[str, list[Kernel]] = {}  # RPC kernel ID -> blocking condition (RPC request message)

        self._rpc_req_recv_queue: list[RPCMessage] = None               # queue to receive RPC request messages
        self._rpc_rsp_recv_queue: list[RPCMessage] = None               # queue to receive RPC response messages
        self._rpc_req_send_inbox: dict[str, list[RPCMessage]] = None    # inbox to send RPC request messages (will be initialized by initialize() method)
        self._rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = None    # inbox to send RPC response messages (will be initialized by initialize() method)

        self._registered_command_debug_hooks: dict[str, Callable] = {}
        self._registered_kernel_debug_hooks: dict[str, Callable] = {}
        
        self._use_cycle_model = True
        self._use_functional_model = True

        self._timestamp = 0

    ###########################################################################
    # Initialization
    ###########################################################################
    
    def initialize_kernel_dispatch_queue(self):
        self._dispatched_main_kernels.clear()
        self._dispatched_rpc_kernels.clear()
        self._dispatched_rpc_msg_mappings.clear()
        self._suspended_main_kernels.clear()
        self._suspended_rpc_req_msg.clear()
        # self._suspended_rpc_rsp_msg.clear()
        
        return self

    def initialize_mp_queue_inbox(self, rpc_req_send_inbox: dict[str, list[RPCMessage]] = None, rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = None):
        self._rpc_req_recv_queue = rpc_req_send_inbox[self.core_id]
        self._rpc_rsp_recv_queue = rpc_rsp_send_inbox[self.core_id]
        self._rpc_req_send_inbox = rpc_req_send_inbox
        self._rpc_rsp_send_inbox = rpc_rsp_send_inbox

        return self
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        self._use_cycle_model = use_cycle_model if use_cycle_model is not None else self._use_cycle_model
        self._use_functional_model = use_functional_model if use_functional_model is not None else self._use_functional_model

    ###########################################################################
    # Kernel Dispatch / Execute / Update Timestamp
    ###########################################################################
    
    def dispatch_main_kernel(self, slot_id: Any, kernel: Kernel | KernelPrototype):
        if not isinstance(kernel, (Kernel, KernelPrototype)):
            raise Exception(f"Cannot dispatch kernel '{kernel}' to the core since it is not an instance of Kernel")
        
        if slot_id in self._dispatched_main_kernels:
            if slot_id not in self._suspended_main_kernels:
                self._suspended_main_kernels[slot_id] = []
            self._suspended_main_kernels[slot_id].append(kernel)
        else:
            if isinstance(kernel, KernelPrototype):
                kernel = kernel.compile()
            kernel.root_kernel_id = f"MAIN<{slot_id}>"
            self._dispatched_main_kernels[slot_id] = kernel
        
        self.run_kernel_debug_hook(kernel=kernel, issue_time=self._timestamp)

    def dispatch_rpc_kernel(self, kernel: Kernel, msg: RPCMessage):
        if not isinstance(kernel, Kernel):
            raise Exception(f"Cannot dispatch kernel '{kernel}' to the core since it is not an instance of Kernel")
        
        kernel_name = f"{kernel.kernel_id}.0"
        i = 0
        
        while kernel_name in self._dispatched_rpc_kernels.keys():
            kernel_name = f"{kernel.kernel_id}.{i}"
            i += 1
        
        kernel.kernel_id = kernel_name  # rename the kernel ID to avoid name collision
        self._dispatched_rpc_kernels[kernel_name] = kernel
        self._dispatched_rpc_msg_mappings[kernel_name] = msg
    
    @core_command_method
    def dispatch_process_kernel(self, slot_id: str, kernel: Kernel):
        self.dispatch_main_kernel(slot_id, kernel)
        
    def get_remaining_cycles(self) -> int:        
        remaining_cycles = None
        
        for kernel in itertools.chain(self._dispatched_main_kernels.values(), self._dispatched_rpc_kernels.values()):
            kernel_remaining_cycles = kernel.get_remaining_cycles(self)
            
            if remaining_cycles is None:
                remaining_cycles = kernel_remaining_cycles
            elif kernel_remaining_cycles is not None:
                remaining_cycles = min(remaining_cycles, kernel_remaining_cycles)
                
        return remaining_cycles
    
    def rpc_update_routine(self):
        self._rpc_req_kernel_dispatch_routine()  # dispatch RPC kernel if the RPC request queue is not empty
        self._rpc_rsp_msg_receive_routine()      # receive RPC response message and register them as suspended
        
    def update_cycle_time(self, cycle_time: int):
        main_kernel_slot_ids = list(self._dispatched_main_kernels.keys())
        rpc_kernel_slot_ids  = list(self._dispatched_rpc_kernels.keys())

        for slot_id in main_kernel_slot_ids:
            kernel = self._dispatched_main_kernels[slot_id]
            kernel.update_cycle_time(self, cycle_time)

            if kernel.is_finished(self):
                self._dispatched_main_kernels.pop(slot_id) # if the kernel is main kernel, simply remove the kernel from the "dispatched_kernels" dictionary
                self.run_kernel_debug_hook(kernel=kernel, commit_time=self._timestamp + cycle_time)
                
                if slot_id in self._suspended_main_kernels:
                    if len(self._suspended_main_kernels[slot_id]) > 0:
                        suspended_kernel = self._suspended_main_kernels[slot_id].pop(0)
                        if isinstance(suspended_kernel, KernelPrototype):
                            suspended_kernel = suspended_kernel.compile()
                        suspended_kernel.root_kernel_id = f"MAIN<{slot_id}>"
                        self._dispatched_main_kernels[slot_id] = suspended_kernel  # TODO: directly dispatch the suspended kernel without going through the dispatch_main_kernel() method

        for slot_id in rpc_kernel_slot_ids:
            kernel = self._dispatched_rpc_kernels[slot_id]
            kernel.update_cycle_time(self, cycle_time)

            if kernel.is_finished(self):
                self._rpc_req_kernel_remove_and_rsp_send_routine(slot_id)  # generate RPC response if the current ongoing RPC message is properly handled
            
        self._timestamp += cycle_time
    
    ###########################################################################
    # Debugging Methods
    ###########################################################################
    
    def register_command_debug_hook(self, hook: Callable[[Command], None]) -> str:
        def create_hook_id(i: int) -> str:
            return f"hook_{i}"
        
        MAX_HOOK_NUM = 1000
        
        for i in range(MAX_HOOK_NUM):
            hook_id = create_hook_id(i)
            if hook_id not in self._registered_command_debug_hooks:
                self._registered_command_debug_hooks[hook_id] = hook
                return hook_id
        
        raise Exception(f"Cannot register command debug hook since the maximum number of hooks ({MAX_HOOK_NUM}) is reached. Please remove some hooks before adding new ones.")
            
    def unregister_command_debug_hook(self, hook_id: str):
        if hook_id in self._registered_command_debug_hooks:
            del self._registered_command_debug_hooks[hook_id]
        else:
            raise Exception(f"Hook ID '{hook_id}' is not registered")
        
    def run_command_debug_hook(self, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        for hook_id, hook in self._registered_command_debug_hooks.items():
            try:
                hook(self, kernel, cmd, issue_time, commit_time)
            except Exception as e:
                logger.error(f"Command debug hook '{hook_id}' failed with error: {e}")
                raise e
            
    def register_kernel_debug_hook(self, hook: Callable[[Kernel], None]) -> str:
        def create_hook_id(i: int) -> str:
            return f"hook_{i}"
        
        MAX_HOOK_NUM = 1000
        
        for i in range(MAX_HOOK_NUM):
            hook_id = create_hook_id(i)
            if hook_id not in self._registered_kernel_debug_hooks:
                self._registered_kernel_debug_hooks[hook_id] = hook
                return hook_id
        
        raise Exception(f"Cannot register kernel debug hook since the maximum number of hooks ({MAX_HOOK_NUM}) is reached. Please remove some hooks before adding new ones.")
    
    def unregister_kernel_debug_hook(self, hook_id: str):
        if hook_id in self._registered_kernel_debug_hooks:
            del self._registered_kernel_debug_hooks[hook_id]
        else:
            raise Exception(f"Hook ID '{hook_id}' is not registered")
        
    def run_kernel_debug_hook(self, kernel: Kernel, issue_time: int=None, commit_time: int=None):
        for hook_id, hook in self._registered_kernel_debug_hooks.items():
            try:
                hook(self, kernel, issue_time, commit_time)
            except Exception as e:
                logger.error(f"Kernel debug hook '{hook_id}' failed with error: {e}")
                raise e
                
    @core_command_method
    def debug_core_with_ambiguous_func(self, func: Callable, *args, **kwargs):
        if isinstance(func, str):
            logger.debug(f"{func} args: {args} kwargs: {kwargs}")
        else:
            func(*args, **kwargs)
    
    ###########################################################################
    # Cycle / Behavioral Model
    ###########################################################################
      
    def get_cycle_model(self, cmd_id: str) -> Callable:
        return getattr(self._cycle_model, cmd_id) if (self._use_cycle_model and hasattr(self._cycle_model, cmd_id)) else None

    def get_behavioral_model(self, cmd_id: str) -> Callable:
        if not hasattr(self, cmd_id):
            raise Exception(f"Command '{cmd_id}' is not registered in the core '{self.core_id}'")
        return getattr(self, cmd_id)
    
    ###########################################################################
    # Parallelization
    ###########################################################################
    
    @core_conditional_command_method
    def parallel_merge(self):
        # NOTE: This command is a dummy command for merging parallel threads. Since the core executes the command in order, this command will be executed
        # after all the parallel threads are successfully executed. This command does not actually merges all the preceding parallel threads. However, this
        # command will automatically be dispatched as a new step for the current kernel context, preventing other subsequent steps from being executed until 
        # this command is finished.
        return True  # dummy: always conditional true!

    ###########################################################################
    # Asynchronous RPC Methods (Inter-Core Communication)
    ###########################################################################
    
    def check_rpc_inbox(self, target_core_id: str):
        return target_core_id in self._rpc_req_send_inbox

    @core_command_method
    def async_rpc_send_req_msg(self, req_msg: RPCMessage):
        msg_id_fmt = f"{self.core_id}.{req_msg.dst_core_id}.{req_msg.kernel_id}.{req_msg.cmd_id}.{self.timestamp}"
        msg_id = msg_id_fmt

        tmp = 0
        while msg_id in self._suspended_rpc_req_msg:
            msg_id = msg_id_fmt + f"_{tmp}"
            tmp += 1

        req_msg.msg_id = msg_id
        
        self._rpc_req_send_inbox[req_msg.dst_core_id].append(req_msg)
        self._suspended_rpc_req_msg[msg_id] = req_msg
        self._suspended_rpc_to_main_kernels_mapping[msg_id] = get_global_kernel_context().root_kernel_id
        
    @core_command_method
    def async_rpc_wait_rsp_msg(self, req_msg: RPCMessage):
        msg_id = req_msg.msg_id
        
        if msg_id not in self._suspended_rpc_kernel_blocking_condition:
            self._suspended_rpc_kernel_blocking_condition[msg_id] = []
            
        context = get_global_kernel_context()
        
        if context is None:
            raise Exception(f"Cannot suspend the current kernel since there is no kernel context")
        elif not isinstance(context, Kernel):
            raise Exception(f"Cannot suspend the current kernel since the current context is not an instance of Kernel, but {type(context).__name__}")
        
        context.set_blocked(True)
        
        self._suspended_rpc_kernel_blocking_condition[msg_id].append(context)

    def _rpc_req_kernel_dispatch_routine(self):
        while len(self.rpc_req_recv_queue):
            msg: RPCMessage = self.rpc_req_recv_queue.pop(0)

            if not isinstance(msg, RPCMessage):
                raise Exception(f"Received message is not an instance of RPCMessage: {type(msg).__name__}")
            if msg.msg_type != 0:
                raise Exception(f"Received message is not a request message: {msg.msg_type}. This exception may caused by the faulty implementation of RPC.")
            
            func = getattr(self, msg.cmd_id, None)
            
            if func is None:
                raise Exception(f"Command '{msg.cmd_id}' is not registered in the core '{type(self).__name__}(core_id={self.core_id})' for RPC processing")
            elif hasattr(func, "_is_command_method") and func._is_command_method:
                kernel = Kernel(f"AUTO_REMOTE")
                with new_global_context(GlobalContextMode.COMPILE, self, kernel):
                    if func._is_conditional:
                        cmd = ConditionalCommand(cmd_id=msg.cmd_id, *msg.args, **msg.kwargs)
                    else:
                        cmd = Command(cmd_id=msg.cmd_id, *msg.args, **msg.kwargs)
                    kernel.add_execution_step(cmd)  # Add the command as an execution step
                kernel.root_kernel_id = f"RPC<{msg.src_core_id}>"
            else:
                kernel = Kernel(func.__name__)
                with new_global_context(GlobalContextMode.COMPILE, self, kernel):
                    func(*msg.args, **msg.kwargs)
                kernel.root_kernel_id = f"RPC<{msg.src_core_id}>::{msg.kernel_id}"
            
            self.dispatch_rpc_kernel(kernel=kernel, msg=msg)
        
    def _rpc_rsp_msg_receive_routine(self):
        while len(self.rpc_rsp_recv_queue):
            rsp_msg: RPCMessage = self.rpc_rsp_recv_queue.pop(0)
            msg_id = rsp_msg.msg_id

            req_msg = self._suspended_rpc_req_msg[msg_id]
            req_msg.copy_args_from_rsp(rsp_msg)

            self._suspended_rpc_req_msg.pop(msg_id)  # remove the request message from the suspended RPC request message list
            self._suspended_rpc_to_main_kernels_mapping.pop(msg_id)  # remove the mapping from the suspended RPC to main kernel mapping
            
            if msg_id in self._suspended_rpc_kernel_blocking_condition:
                for kernel in self._suspended_rpc_kernel_blocking_condition[msg_id]:
                    kernel.set_blocked(False)  # unblock the RPC kernel
                self._suspended_rpc_kernel_blocking_condition.pop(msg_id)
        
    def _rpc_req_kernel_remove_and_rsp_send_routine(self, slot_id: str):
        kernel = self._dispatched_rpc_kernels[slot_id]
        req_msg = self._dispatched_rpc_msg_mappings[slot_id]
        rsp_msg = req_msg.response(rpc_kernel=kernel)

        self._rpc_rsp_send_inbox[req_msg.src_core_id].append(rsp_msg)

        self._dispatched_rpc_kernels.pop(slot_id)       # remove the kernel from the dispatched RPC kernels
        self._dispatched_rpc_msg_mappings.pop(slot_id)  # remove the message
    
    ###########################################################################
    # Properties
    ###########################################################################
    
    @property
    def is_idle(self) -> bool:
        return self.is_idle_main and self.is_idle_rpc
    
    @property
    def is_idle_main(self) -> bool:
        for kernel_queue in self._suspended_main_kernels.values():
            if len(kernel_queue) > 0:
                return False
        
        for kernel in self._dispatched_main_kernels.values():
            if not kernel.is_finished(self):
                return False
        
        return True

    @property
    def is_idle_rpc(self) -> bool:
        for kernel in self._dispatched_rpc_kernels.values():
            if not kernel.is_finished(self):
                return False
        return True

    @property
    def use_cycle_model(self) -> bool:
        return self._use_cycle_model
    
    @property
    def use_functional_model(self) -> bool:
        return self._use_functional_model
    
    @property
    def rpc_req_recv_queue(self) -> list[RPCMessage]:
        return self._rpc_req_recv_queue

    @property
    def rpc_rsp_recv_queue(self) -> list[RPCMessage]:
        return self._rpc_rsp_recv_queue

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def n_dispatched_main_kernels(self) -> int:
        return len(self._dispatched_main_kernels) + sum(len(v) for v in self._suspended_main_kernels.values())