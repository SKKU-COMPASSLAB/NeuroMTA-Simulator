import enum
import itertools
from typing import Callable, Sequence, Any

from neuromta.framework.memory_handle import *


__all__ = [
    "set_global_context",
    "get_global_core_context",
    "get_global_kernel_context",
    "get_global_pid",

    "DataContainer",
    "RPCMessage",
    "Command",
    "ConditionalCommand",
    
    "Kernel",

    "CoreCycleModel",
    "Core",
    
    "core_kernel_method",
    "core_command_method",
    "core_conditional_command_method",
    "new_parallel_thread",
    "start_parallel_thread",
    "end_parallel_thread",
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
        raise Exception(f"[ERROR] Cannot set global context to COMPILE mode with a non-Kernel object: {kernel}")
    
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
    
    if context_mode != GlobalContextMode.COMPILE:
        raise Exception(f"[ERROR] Cannot add to parent kernel callstack since the global context mode is not COMPILE, but {context_mode.name}")

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

def core_kernel_method(_func: Callable):
    def __core_kernel_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Kernel:
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel = Kernel(
            _func.__name__,     # the kernel ID is the name of the function
            _func,              # the behavioral model is the function itself
            *_args,             # the arguments of the kernel
            **_kwargs           # the keyword arguments of the kernel
        )

        if get_global_context_mode() == GlobalContextMode.IDLE:
            pass  # do not automatically dispatch kernel
        elif get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context = get_global_kernel_context()
            
            if kernel_context is None:
                raise Exception(f"[ERROR] Cannot register kernel '{kernel.kernel_id}' to the compiled kernel since it is called outside of a low-level kernel function")
            
            kernel_context.add_execution_step(kernel)
        else:
            print(f"[WARNING] Kernel method '{_func.__name__}' is called outside of the compile or idle context. It implies that the kernel is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Kernel method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return kernel
        
    return __core_kernel_method_wrapper

def core_command_method(_func: Callable):
    def __core_command_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Command:
        if get_global_context_mode() in (GlobalContextMode.IDLE, GlobalContextMode.EXECUTE):
            return _func(_core, *_args, **_kwargs)

        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        elif not isinstance(kernel_context, Kernel):
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function. The current kernel context is not an instance of Kernel, but {type(kernel_context).__name__}")
        
        cmd = Command(
            _func.__name__,     # the command ID is the name of the function
            *_args,             # the arguments of the command
            **_kwargs           # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            print(f"[WARNING] Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    return __core_command_method_wrapper

def core_conditional_command_method(_func: Callable):
    def __core_command_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Command:
        if get_global_context_mode() in (GlobalContextMode.IDLE, GlobalContextMode.EXECUTE):
            return _func(_core, *_args, **_kwargs)

        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        elif not isinstance(kernel_context, Kernel):
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function. The current kernel context is not an instance of Kernel, but {type(kernel_context).__name__}")

        cmd = ConditionalCommand(
            _func.__name__,     # the command ID is the name of the function
            *_args,             # the arguments of the command
            **_kwargs           # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            print(f"[WARNING] Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    return __core_command_method_wrapper

class new_parallel_thread:
    def __enter__(self):
        start_parallel_thread()
    
    def __exit__(self, exc_type, exc_value, traceback):
        end_parallel_thread()

def start_parallel_thread():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot create a parallel kernel since the global context mode is not COMPILE")
    
    kernel_context = get_global_kernel_context()
    parallel_kernel = kernel_context.add_parallel_kernel_step()
    
    store_global_parent_kernel_callstack()
    set_global_context(GlobalContextMode.COMPILE, get_global_core_context(), parallel_kernel)
    
def end_parallel_thread():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot end parallel kernel since the global context mode is not COMPILE")

    restore_global_parent_kernel_callstack()


#################################################
# Implementation
#################################################

class DataContainer:
    def __init__(self, data: Any=None):
        self.data = data
    
    def copy_from(self, other: 'DataContainer'):
        if not isinstance(other, DataContainer):
            raise Exception(f"[ERROR] Cannot copy from non-DataContainer object of type {type(other).__name__}")
        self.data = other.data

    def __getstate__(self) -> dict[str, Any]:
        return {
            "data": self.data
        }
        
    def __setstate__(self, state: dict):
        self.data = state["data"]
        

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
            self.root_kernel_id = kernel_context.root_kernel_id
        
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
            *rpc_kernel.args,
            **rpc_kernel.kwargs
        )
        
        msg.msg_id = self.msg_id
        
        return msg
        
    def copy_args_from_rsp(self, rsp_msg: 'RPCMessage'):
        for arg_idx in range(len(self.args)):
            req_arg = self.args[arg_idx]
            rsp_arg = rsp_msg.args[arg_idx]

            if isinstance(req_arg, DataContainer):
                req_arg.copy_from(rsp_arg)  # copy the response arguments to the request arguments
            else:
                self.args[arg_idx] = rsp_arg

        for arg_name in self.kwargs.keys():
            req_arg = self.kwargs[arg_name]
            rsp_arg = rsp_msg.kwargs[arg_name]

            if isinstance(req_arg, DataContainer):
                req_arg.copy_from(rsp_arg)  # copy the response arguments to the request arguments
            else:
                self.kwargs[arg_name] = rsp_arg
    
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
        
    def get_remaining_cycles(self, core: 'Core', kernel: 'Kernel') -> int:
        if self._cached_cycle is None:
            self._cached_cycle = self.run_cycle_model(core, kernel)
            
            if self._cached_cycle is None:
                self._cached_cycle = 1
            
            self._cached_cycle = max(1, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, core: 'Core', kernel: 'Kernel', cycle_time: int):
        if self._cached_issue_time is None:
            self._cached_issue_time = core.timestamp
            
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        self._cached_cycle_slack += cycle_time
        
        if self.is_finished(core, kernel):
            flag = self.run_behavioral_model(core, kernel)
            
            if flag is not None:
                raise Exception(f"[ERROR] Behavioral model for command '{self.cmd_id}' returned '{flag}' even though the command is not conditional. Use 'core_conditional_command_method' instead if you want to implement any retry operation.")

        if self.is_finished(core, kernel):
            core.run_command_debug_hook(kernel=kernel, cmd=self, issue_time=self._cached_issue_time, commit_time=core.timestamp+cycle_time)

    def run_behavioral_model(self, core: 'Core', kernel: 'Kernel'):
        with new_global_context(GlobalContextMode.EXECUTE, core, kernel):
            model = core.get_behavioral_model(self.cmd_id)
            return model(*self.args, **self.kwargs)
        
    def run_cycle_model(self, core: 'Core', kernel: 'Kernel') -> int:
        with new_global_context(GlobalContextMode.EXECUTE, core, kernel):
            model = core.get_cycle_model(self.cmd_id)

            if model is None:
                return None
            elif isinstance(model, int):
                return model
            elif callable(model):
                return model(*self.args, **self.kwargs)
            
            return None
        
    def is_finished(self, core: 'Core', kernel: 'Kernel') -> bool:
        remaining_cycles = self.get_remaining_cycles(core, kernel)
        if remaining_cycles is None:
            return False
        return remaining_cycles <= 0

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"
    
    
class ConditionalCommand(Command):
    def __init__(self, cmd_id: str, *args, **kwargs):
        super().__init__(cmd_id, *args, **kwargs)
        
        self._is_async_finished = False
    
    def get_remaining_cycles(self, core: 'Core', kernel: 'Kernel') -> int:
        return None

    def update_cycle_time(self, core: 'Core', kernel: 'Kernel', cycle_time: int):
        if self._cached_issue_time is None:
            self._cached_issue_time = core.timestamp

        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        self._is_async_finished = self.run_behavioral_model(core, kernel)

        if self.is_finished(core, kernel):
            core.run_command_debug_hook(kernel=kernel, cmd=self, issue_time=self._cached_issue_time, commit_time=core.timestamp+cycle_time)

    def run_behavioral_model(self, core: 'Core', kernel: 'Kernel'):
        with new_global_context(GlobalContextMode.EXECUTE, core, kernel):
            model = core.get_behavioral_model(self.cmd_id)
            return model(*self.args, **self.kwargs)
        
    def is_finished(self, core: 'Core', kernel: 'Kernel') -> bool:
        return self._is_async_finished


class ThreadGroup(list['Kernel']):
    def __init__(self):
        super().__init__()
    
    def append(self, kernel: 'Kernel'):
        if not isinstance(kernel, Kernel):
            raise TypeError(f"[ERROR] Cannot add kernel '{kernel}' to the parallel kernel group since it is not an instance of Kernel")
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

class Kernel:
    def __init__(self, kernel_id: str, func: Callable, *args, **kwargs):
        self.kernel_id = kernel_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
        self.root_kernel: Kernel = None
        
        self._is_compiled = False
        self._is_parallel = False
        
        self._execution_steps: list[Command | Kernel | ThreadGroup] = []
        self._execution_cursor: int = 0
        
    def set_parallel(self):
        self._is_compiled = True
        self._is_parallel = True
        return self
        
    def add_execution_step(self, step: 'Command | Kernel'):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"[ERROR] Cannot add execution step '{step}' to the kernel '{self.kernel_id}' since it is not in compile mode")
        if not isinstance(step, (Command, Kernel)):
            raise TypeError(f"[ERROR] Execution step must be an instance of Command or Kernel, but got {type(step).__name__}")
        
        self._execution_steps.append(step)
        
        if isinstance(step, Kernel):
            step.root_kernel = self
            
    def add_parallel_kernel_step(self) -> 'Kernel':
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"[ERROR] Cannot add parallel kernel step to the kernel '{self.kernel_id}' since it is not in compile mode")
        
        if len(self._execution_steps) == 0:
            self._execution_steps.append(ThreadGroup())
        elif not isinstance(self._execution_steps[-1], ThreadGroup):
            self._execution_steps.append(ThreadGroup())

        parallel_kernel_idx = len(self._execution_steps[-1])

        parallel_kernel = Kernel(kernel_id=f"{self.kernel_id}.{parallel_kernel_idx}", func=None, core=None).set_parallel()
        parallel_kernel.root_kernel = self.root_kernel

        self._execution_steps[-1].append(parallel_kernel)

        return parallel_kernel

    def compile(self, core: 'Core'):
        with new_global_context(GlobalContextMode.COMPILE, core, self):
            self.func(core, *self.args, **self.kwargs)
        self._is_compiled = True
        
    def get_remaining_cycles(self, core: 'Core') -> int:
        if not self.is_compiled:
            self.compile(core)
        if self.is_finished(core):
            return None
        
        step = self.current_step(core)

        if isinstance(step, ConditionalCommand):
            return step.get_remaining_cycles(core, kernel=self)
        elif isinstance(step, Command):
            return step.get_remaining_cycles(core, kernel=self)
        else:
            return step.get_remaining_cycles(core=core)

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        if not self.is_compiled:
            self.compile(core)
            
        if self.is_finished(core):
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
            
    def current_step(self, core: 'Core') -> 'Command | Kernel | ThreadGroup':
        if not self.is_compiled:
            self.compile(core)
        if self.is_finished(core):
            return None
        
        return self._execution_steps[self._execution_cursor]
        
    @property
    def is_compiled(self) -> bool:
        return self._is_compiled
    
    def is_finished(self, core: 'Core') -> bool:
        if not self.is_compiled:
            self.compile(core)
        return self._execution_cursor >= len(self._execution_steps)
    
    @property
    def root_kernel_id(self) -> str | None:
        if self.root_kernel is None:
            return None
        return self.root_kernel.kernel_id_full
    
    @root_kernel_id.setter
    def root_kernel_id(self, value):
        if isinstance(value, str):
            self.root_kernel = Kernel(kernel_id=value, func=None)
        else:
            self.root_kernel = None
            
    @property
    def kernel_id_full(self) -> str:
        if self.root_kernel_id is None:
            return self.kernel_id
        return f"{self.root_kernel_id}::{self.kernel_id}"
    

class CoreCycleModel:
    def __init__(self):
        pass

class Core:
    def __init__(self, core_id: str, cycle_model: CoreCycleModel=None):
        self.core_id = core_id

        self._cycle_model: CoreCycleModel = cycle_model

        self._dispatched_main_kernels:      dict[str, Kernel] = {}
        self._dispatched_rpc_kernels:       dict[str, Kernel] = {}
        self._dispatched_rpc_msg_mappings:  dict[str, RPCMessage] = {}  # RPC kernel -> RPC request message (given from the source core)

        self._suspended_main_kernels: dict[str, list[Kernel]] = {}
        self._suspended_rpc_req_msg: dict[str, RPCMessage] = {}
        self._suspended_rpc_rsp_msg: dict[str, RPCMessage] = {}

        self._rpc_req_recv_queue: list[RPCMessage] = None               # queue to receive RPC request messages
        self._rpc_rsp_recv_queue: list[RPCMessage] = None               # queue to receive RPC response messages
        self._rpc_req_send_inbox: dict[str, list[RPCMessage]] = None    # inbox to send RPC request messages (will be initialized by initialize() method)
        self._rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = None    # inbox to send RPC response messages (will be initialized by initialize() method)

        self._registered_command_debug_hooks: dict[str, Callable[[Command], None]] = {}
        
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
        self._suspended_rpc_rsp_msg.clear()
        
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
    
    def dispatch_main_kernel(self, slot_id: Any, kernel: Kernel):
        if not isinstance(kernel, Kernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")

        if slot_id in self._dispatched_main_kernels:
            if slot_id not in self._suspended_main_kernels:
                self._suspended_main_kernels[slot_id] = []
            self._suspended_main_kernels[slot_id].append(kernel)
        else:
            # print(f"[DEBUG] [{self.core_id}] Dispatching kernel '{kernel.kernel_id_full}' to slot {slot_id}")
            self._dispatched_main_kernels[slot_id] = kernel

    def dispatch_rpc_kernel(self, kernel: Kernel, msg: RPCMessage):
        if not isinstance(kernel, Kernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")
        
        kernel_name = "rpc_kernel"
        i = 0
        
        while kernel_name in self._dispatched_rpc_kernels.keys():
            kernel_name = f"rpc_kernel_{i}"
            i += 1
            
        self._dispatched_rpc_kernels[kernel_name] = kernel
        self._dispatched_rpc_msg_mappings[kernel_name] = msg
        
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
                # print(f"[DEBUG] [{self.core_id}] Terminating kernel '{kernel.kernel_id_full}' to slot {slot_id}")
                
                if slot_id in self._suspended_main_kernels:
                    if len(self._suspended_main_kernels[slot_id]) > 0:
                        suspended_kernel = self._suspended_main_kernels[slot_id].pop(0)
                        # self.dispatch_main_kernel(slot_id, suspended_kernel)
                        # print(f"[DEBUG] [{self.core_id}] Dispatching suspended kernel '{suspended_kernel.kernel_id_full}' to slot {slot_id}")
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
        
        raise Exception(f"[ERROR] Cannot register command debug hook since the maximum number of hooks ({MAX_HOOK_NUM}) is reached. Please remove some hooks before adding new ones.")
            
    def unregister_command_debug_hook(self, hook_id: str):
        if hook_id in self._registered_command_debug_hooks:
            del self._registered_command_debug_hooks[hook_id]
        else:
            raise Exception(f"[ERROR] Hook ID '{hook_id}' is not registered")
        
    def run_command_debug_hook(self, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        for hook_id, hook in self._registered_command_debug_hooks.items():
            try:
                hook(self, kernel, cmd, issue_time, commit_time)
            except Exception as e:
                print(f"[ERROR] Command debug hook '{hook_id}' failed with error: {e}")
                
    @core_command_method
    def debug_core_with_ambiguous_func(self, func: Callable, *args, **kwargs):
        return func(*args, **kwargs)
    
    ###########################################################################
    # Cycle / Behavioral Model
    ###########################################################################
      
    def get_cycle_model(self, cmd_id: str) -> Callable:
        return getattr(self._cycle_model, cmd_id) if (self._use_cycle_model and hasattr(self._cycle_model, cmd_id)) else None

    def get_behavioral_model(self, cmd_id: str) -> Callable:
        if not hasattr(self, cmd_id):
            raise Exception(f"[ERROR] Command '{cmd_id}' is not registered in the core '{self.core_id}'")
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
        
    @core_conditional_command_method
    def async_rpc_wait_rsp_msg(self, req_msg: RPCMessage):
        msg_id = req_msg.msg_id

        if msg_id not in self._suspended_rpc_rsp_msg:
            return False
        
        rsp_msg = self._suspended_rpc_rsp_msg[msg_id]
        req_msg.copy_args_from_rsp(rsp_msg)

        self._suspended_rpc_rsp_msg.pop(msg_id)  # remove the response message from the suspended RPC response message list
        self._suspended_rpc_req_msg.pop(msg_id)  # remove the request message from the suspended RPC request message list

        return True

    def _rpc_req_kernel_dispatch_routine(self):
        while len(self.rpc_req_recv_queue):
            msg: RPCMessage = self.rpc_req_recv_queue.pop(0)

            if not isinstance(msg, RPCMessage):
                raise Exception(f"[ERROR] Received message is not an instance of RPCMessage: {type(msg).__name__}")
            if msg.msg_type != 0:
                raise Exception(f"[ERROR] Received message is not a request message: {msg.msg_type}. This exception may caused by the faulty implementation of RPC.")
            
            func = getattr(self, msg.cmd_id, None)
            
            if func is None:
                raise Exception(f"[ERROR] Command '{msg.cmd_id}' is not registered in the core '{self.core_id}' for RPC processing")
            elif func.__name__ == "__core_command_method_wrapper":
                kernel = Kernel(kernel_id="__auto_remote", func=func, *msg.args, **msg.kwargs)
                with new_global_context(GlobalContextMode.COMPILE, self, kernel):
                    cmd = Command(cmd_id=msg.cmd_id, *msg.args, **msg.kwargs)
                    kernel.add_execution_step(cmd)  # Add the command as an execution step
                kernel._is_compiled = True      # Mark the kernel as compiled
                kernel.root_kernel_id = f"{msg.root_kernel_id}::{msg.cmd_id}"
            elif func.__name__ == "__core_kernel_method_wrapper":
                kernel: Kernel = func(*msg.args, **msg.kwargs)
                kernel.root_kernel_id = f"{msg.root_kernel_id}::{msg.kernel_id}"
            else:
                raise Exception(f"[ERROR] Command '{msg.cmd_id}' is not a valid command for RPC processing. It must be a core command or a kernel method.")
            
            self.dispatch_rpc_kernel(kernel=kernel, msg=msg)
        
    def _rpc_rsp_msg_receive_routine(self):
        while len(self.rpc_rsp_recv_queue):
            rsp_msg: RPCMessage = self.rpc_rsp_recv_queue.pop(0)
            self._suspended_rpc_rsp_msg[rsp_msg.msg_id] = rsp_msg
        
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
