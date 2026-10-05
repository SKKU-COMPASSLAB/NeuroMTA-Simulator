import enum
import functools
from typing import Any, Callable

import torch

from neuromta.system.hardware.base_accelerator import HostJob
from neuromta.system.hardware.mesh_accelerator import MeshAccelerator
from neuromta.system.software.utils.compiler import MeshDeviceCompiledWorkload, MeshDeviceCompiler
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshKernelDescriptor, MeshTensorDescriptor, MeshMemoryType, MeshTensorType
from neuromta.system.software.utils.runtime import MeshDeviceRuntime, MeshKernelState, MeshDeviceRuntimeWorkloadState
from neuromta.system.software.utils.scheduler import MeshDeviceScheduler, MeshFCFSScheduler, MeshFRFCFSScheduler, MeshRoundRobinScheduler, MeshWorkloadSchedulingHint


__all__ = [
    "MeshDeviceCompiledWorkload",
    "MeshDeviceRuntimeContext",
    # "MeshDeviceSchedulerType",
    "MeshKernelState",
    "MeshMemoryType",
    "MeshTensorDescriptor",
    "MeshTensorType",
    "MeshWorkloadSchedulingHint",
    "MeshDeviceRuntimeWorkloadState",

    "get_global_mesh_device_context",

    "_require_context",
    "_require_tensor",
    "_mesh_device_op_method",
]


_global_mesh_device_context: "MeshDeviceRuntimeContext | None" = None

def get_global_mesh_device_context() -> "MeshDeviceRuntimeContext | None":
    return _global_mesh_device_context


# class MeshDeviceSchedulerType(enum.Enum):
#     RR = "RR"
#     FCFS = "FCFS"
#     FRFCFS = "FRFCFS"


class _MeshDeviceCompilerContext:
    def __init__(self, context: "MeshDeviceRuntimeContext", compiler: MeshDeviceCompiler, arrival_cycle: int=0, workload_id: str=None, dependent_workload_ids: tuple[str, ...]=(), scheduling_hint: MeshWorkloadSchedulingHint=None, warmup: bool=False, domain_id: str=None):
        self.context = context
        self.compiler = compiler
        self.arrival_cycle = arrival_cycle
        self.workload_id = workload_id
        self.dependent_workload_ids = dependent_workload_ids
        self.scheduling_hint = scheduling_hint
        self.warmup = warmup
        self.domain_id = domain_id
        self.compiled_workload: MeshDeviceCompiledWorkload | None = None
        self.submitted_workload_id: str | None = None

    def __enter__(self) -> MeshDeviceCompiler:
        return self.context.open_compiler(compiler=self.compiler, arrival_cycle=self.arrival_cycle, workload_id=self.workload_id, dependent_workload_ids=self.dependent_workload_ids, scheduling_hint=self.scheduling_hint, warmup=self.warmup, domain_id=self.domain_id)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.context.abort_compiler()
            return False
        self.compiled_workload, self.submitted_workload_id = self.context.close_compiler()
        return False


class MeshDeviceRuntimeContext:
    def __init__(
        self, 
        device: MeshAccelerator, 
        default_tile_shape: tuple[int, int], 
        default_dtype: torch.dtype, 
        # scheduler_type: MeshDeviceSchedulerType=MeshDeviceSchedulerType.FRFCFS, 
        runtime: MeshDeviceRuntime,
        # enable_debug_log: bool=False,
    ):
        if not isinstance(device, MeshAccelerator):
            raise TypeError(f"Expected MeshAccelerator, got {type(device).__name__}.")
        # if not isinstance(scheduler_type, MeshDeviceSchedulerType):
        #     raise TypeError(f"Expected MeshDeviceSchedulerType, got {type(scheduler_type).__name__}.")
        self.device = device
        self.default_tile_shape = tuple(int(dim) for dim in default_tile_shape)
        self.default_dtype = default_dtype
        # self.scheduler_type = scheduler_type
        # self.enable_debug_log = enable_debug_log
        self._device_desc = MeshDeviceDescriptor(device)
        self._compiler: MeshDeviceCompiler | None = None
        self._compiler_arrival_cycle = 0
        self._compiler_workload_id: str | None = None
        self._compiler_dependent_workload_ids: tuple[str, ...] = ()
        self._compiler_scheduling_hint: MeshWorkloadSchedulingHint | None = None
        self._compiler_warmup = False
        self._compiler_domain_id: str | None = None
        self._state_counter = 0
        self._state_handles: list[Any] = []
        self._compiler_state_snapshots: dict[Any, tuple[int, ...]] = {}
        # scheduler_classes = {MeshDeviceSchedulerType.RR: MeshRoundRobinScheduler, MeshDeviceSchedulerType.FCFS: MeshFCFSScheduler, MeshDeviceSchedulerType.FRFCFS: MeshFRFCFSScheduler}
        # self._runtime = MeshDeviceRuntime(
        #     device_desc=self._device_desc, scheduler=scheduler_classes[scheduler_type](), enable_debug_log=enable_debug_log)
        self._runtime = runtime
        
    def open(self):
        global _global_mesh_device_context
        if _global_mesh_device_context is not None and _global_mesh_device_context is not self:
            raise RuntimeError("Another MeshDeviceRuntimeContext is already open.")
        _global_mesh_device_context = self

    def close(self):
        global _global_mesh_device_context
        if self._compiler is not None:
            self.abort_compiler()
        if _global_mesh_device_context is self:
            _global_mesh_device_context = None

    def new_compiler_context(self, compiler: MeshDeviceCompiler, arrival_cycle: int=0, workload_id: str=None, dependent_workload_ids: tuple[str, ...]=(), scheduling_hint: MeshWorkloadSchedulingHint=None, warmup: bool=False, domain_id: str=None) -> _MeshDeviceCompilerContext:
        return _MeshDeviceCompilerContext(self, compiler=compiler, arrival_cycle=arrival_cycle, workload_id=workload_id, dependent_workload_ids=dependent_workload_ids, scheduling_hint=scheduling_hint, warmup=warmup, domain_id=domain_id)

    def open_compiler(self, compiler: MeshDeviceCompiler, arrival_cycle: int=0, workload_id: str=None, dependent_workload_ids: tuple[str, ...]=(), scheduling_hint: MeshWorkloadSchedulingHint=None, warmup: bool=False, domain_id: str=None) -> MeshDeviceCompiler:
        if self._compiler is not None:
            raise RuntimeError("A compiler context is already active.")
        if not isinstance(arrival_cycle, int) or isinstance(arrival_cycle, bool) or arrival_cycle < 0:
            raise ValueError(f"Invalid arrival cycle: {arrival_cycle}")
        if scheduling_hint is not None and not isinstance(scheduling_hint, MeshWorkloadSchedulingHint):
            raise TypeError(f"Expected MeshWorkloadSchedulingHint, got {type(scheduling_hint).__name__}.")
        self._compiler = compiler
        self._compiler_arrival_cycle = arrival_cycle
        self._compiler_workload_id = workload_id
        self._compiler_dependent_workload_ids = dependent_workload_ids
        self._compiler_scheduling_hint = scheduling_hint
        self._compiler_warmup = bool(warmup)
        self._compiler_domain_id = domain_id
        self._compiler_state_snapshots = {state: state.layer_lengths for state in self._state_handles}
        return self._compiler

    def close_compiler(self) -> tuple[MeshDeviceCompiledWorkload, str]:
        if self._compiler is None:
            raise RuntimeError("No active compiler to close.")
        compiler = self._compiler
        arrival_cycle = self._compiler_arrival_cycle
        workload_id = self._compiler_workload_id
        dependent_workload_ids = self._compiler_dependent_workload_ids
        scheduling_hint = self._compiler_scheduling_hint
        warmup = self._compiler_warmup
        domain_id = self._compiler_domain_id
        state_snapshots = self._compiler_state_snapshots
        self._compiler = None
        self._compiler_arrival_cycle = 0
        self._compiler_workload_id = None
        self._compiler_dependent_workload_ids = ()
        self._compiler_scheduling_hint = None
        self._compiler_warmup = False
        self._compiler_domain_id = None
        self._compiler_state_snapshots = {}
        try:
            compiled_workload = compiler.compile()
            submitted_workload_id = self.submit(compiled_workload, arrival_cycle=arrival_cycle, workload_id=workload_id, dependent_workload_ids=dependent_workload_ids, scheduling_hint=scheduling_hint, warmup=warmup, domain_id=domain_id)
        except Exception:
            for state, lengths in state_snapshots.items():
                state._lengths = list(lengths)
            raise
        return compiled_workload, submitted_workload_id

    def abort_compiler(self):
        for state, lengths in self._compiler_state_snapshots.items():
            state._lengths = list(lengths)
        self._compiler = None
        self._compiler_arrival_cycle = 0
        self._compiler_workload_id = None
        self._compiler_dependent_workload_ids = ()
        self._compiler_scheduling_hint = None
        self._compiler_warmup = False
        self._compiler_domain_id = None
        self._compiler_state_snapshots = {}

    def compile(self, compiler: MeshDeviceCompiler=None) -> MeshDeviceCompiledWorkload:
        target = self._compiler if compiler is None else compiler
        if target is None:
            raise RuntimeError("No compiler was provided and no compiler context is active.")
        if not isinstance(target, MeshDeviceCompiler):
            raise TypeError(f"Expected MeshDeviceCompiler, got {type(target).__name__}.")
        return target.compile()

    def submit(self, compiled_workload: MeshDeviceCompiledWorkload, arrival_cycle: int=0, workload_id: str=None, dependent_workload_ids: tuple[str, ...]=(), scheduling_hint: MeshWorkloadSchedulingHint=None, warmup: bool=False, domain_id: str=None) -> str:
        return self._runtime.submit(compiled_workload, arrival_cycle=arrival_cycle, workload_id=workload_id, dependent_workload_ids=dependent_workload_ids, scheduling_hint=scheduling_hint, warmup=warmup, domain_id=domain_id)

    def warmup(self, compiled_workload: MeshDeviceCompiledWorkload):
        workload_ids = tuple(state.workload_id for state in self._runtime.workloads if state.compiled_workload is compiled_workload)
        if not workload_ids:
            raise ValueError("The compiled workload has not been submitted to this runtime.")
        return self._runtime.warmup(*workload_ids)

    def run(self) -> list[HostJob]:
        return self._runtime.run()

    def deallocate_weights(self, compiled_workload: MeshDeviceCompiledWorkload=None) -> int:
        return self._runtime.deallocate_weights(compiled_workload)

    def reserve_state(self, states):
        descriptors = states.storage_descriptors if hasattr(states, "storage_descriptors") else states
        return self._runtime.reserve_state(descriptors)

    def deallocate_state(self, states=None) -> int:
        descriptors = states.storage_descriptors if hasattr(states, "storage_descriptors") else states
        return self._runtime.deallocate_state(descriptors)

    @property
    def compiler(self) -> MeshDeviceCompiler | None:
        return self._compiler

    @property
    def runtime(self) -> MeshDeviceRuntime:
        return self._runtime

    @property
    def device_descriptor(self) -> MeshDeviceDescriptor:
        return self._device_desc

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _require_context(function_name: str, require_compiler: bool=False) -> MeshDeviceRuntimeContext:
    if _global_mesh_device_context is None:
        raise RuntimeError(f"{function_name}() must be called within a MeshDeviceRuntimeContext.")
    if require_compiler and _global_mesh_device_context.compiler is None:
        raise RuntimeError(f"{function_name}() requires an active compiler context.")
    return _global_mesh_device_context


def _require_tensor(tensor: MeshTensorDescriptor, name: str) -> MeshTensorDescriptor:
    if not isinstance(tensor, MeshTensorDescriptor):
        raise TypeError(f"{name} must be a MeshTensorDescriptor.")
    return tensor

def _mesh_device_op_method(function: Callable):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        context = _require_context(function.__name__, require_compiler=True)
        result = function(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(f"Invalid leaf operator result from {function.__name__}.")
        kernels = (result[0],) if isinstance(result[0], MeshKernelDescriptor) else tuple(result[0])
        if not kernels or any(not isinstance(kernel, MeshKernelDescriptor) for kernel in kernels):
            raise TypeError(f"Leaf operator {function.__name__} returned invalid kernel descriptors.")
        for kernel in kernels:
            context.compiler.add_kernel(kernel)
        return result[1] if len(result) == 2 else result[1:]
    return wrapper
