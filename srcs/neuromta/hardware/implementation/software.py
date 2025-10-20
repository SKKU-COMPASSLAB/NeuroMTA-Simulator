import enum
import torch
import math
import copy
import abc
import functools
from typing import Sequence, Callable

from neuromta.framework import *
from neuromta.hardware.core.npu_core import NPUCore
from neuromta.hardware.implementation.hardware import MCA_DeviceBase, MTA_DeviceBase


__all__ = [
    "MCA_TensorMemoryType",
    "MCA_TensorMemoryLayout",
    "MCA_TensorBuffer",
    
    "MCA_RT_JIT_COMPILE_REGION",
    "MCA_RT_KERNEL",
    "MCA_RT_OPERATOR",
]
    
    
class MCA_TensorMemoryType(enum.Enum):
    L1 = enum.auto()
    MAIN = enum.auto()


class MCA_TensorMemoryLayout:
    def __init__(
        self, 
        mem_type: MCA_TensorMemoryType,
        page_shape: int | Sequence[int],
    ):        
        self.mem_type = mem_type
        
        if isinstance(page_shape, int):
            page_shape = (1, page_shape,)
        elif isinstance(page_shape, Sequence):
            if len(page_shape) == 1:
                page_shape = (1, page_shape[0],)
            elif len(page_shape) > 2:
                raise Exception("[ERROR] Invalid page_shape: page_shape must be a tuple of (y, x).")
        
        self.y_page_size, self.x_page_size = page_shape
                
    def copy(self):
        return copy.deepcopy(self)

    def overrides(
        self, 
        mem_type: MCA_TensorMemoryType  = None,
        page_shape: int | Sequence[int] = None,
    ) -> 'MCA_TensorMemoryLayout':
        
        return MCA_TensorMemoryLayout(
            mem_type   = self.mem_type if mem_type is None else mem_type,
            page_shape = (self.y_page_size, self.x_page_size) if page_shape is None else page_shape,
        )
        
    def __str__(self):
        return f"MCA_TensorMemoryLayout(mem_type={self.mem_type}, page_shape=({self.y_page_size}, {self.x_page_size}))"


class MCA_TensorBuffer:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, layout: MCA_TensorMemoryLayout, device: MCA_DeviceBase, core_ids: list[int] | int=None):
        # STEP 1: Setup
        self.tensor_shape   = tuple(shape)  # (iter, y, x)
        self.tensor_dtype   = dtype
        self.layout         = layout.copy()
        self.device         = device
        self.core_ids       = core_ids
        
        self.i_dim = 1 if (len(self.tensor_shape) < 3) else math.prod(self.tensor_shape[:-2])
        self.y_dim = 1 if (len(self.tensor_shape) < 2) else self.tensor_shape[-2]
        self.x_dim = self.tensor_shape[-1]
        
        self.y_page = self.layout.y_page_size
        self.x_page = self.layout.x_page_size
        
        self.y_pad = (self.y_page - (self.y_dim % self.y_page)) % self.y_page
        self.x_pad = (self.x_page - (self.x_dim % self.x_page)) % self.x_page
        
        self.y_dim += self.y_pad
        self.x_dim += self.x_pad
        
        self.y_n_pages = self.y_dim // self.y_page
        self.x_n_pages = self.x_dim // self.x_page
        
        self._reference: BufferPointer = None  # type: BufferPointer (will be set in allocate())
        
        self._n_pages = self.i_dim * self.y_n_pages * self.x_n_pages
        self._page_size = self.y_page * self.x_page * self.tensor_dtype.itemsize
    
    @property
    def buffer_size(self) -> int:
        return self._n_pages * self._page_size
    
    @property    
    def buffer_segment_size(self) -> int:
        if self.layout.mem_type == MCA_TensorMemoryType.L1:
            return math.ceil(self.n_pages / len(self.core_ids)) * self._page_size
        elif self.layout.mem_type == MCA_TensorMemoryType.MAIN:
            return self.buffer_size
        else:
            raise Exception(f"[ERROR] Unsupported memory type: {self.layout.mem_type}.")

    def allocate(self, initial: torch.Tensor=None):
        if self.layout.mem_type == MCA_TensorMemoryType.L1:
            if self.core_ids is None:
                raise Exception("[ERROR] core_ids must be specified when the memory type is L1.")
            elif isinstance(self.core_ids, int):
                self.core_ids = [self.core_ids,]
                
            if len(self.core_ids) == 0:
                raise Exception("[ERROR] core_ids must contain at least one core ID when the memory type is L1.")
            
            if len(self.core_ids) > 1:
                if not isinstance(self.device, MTA_DeviceBase):
                    raise Exception("[ERROR] The device must be a MTA_DeviceBase when allocating sharded L1 buffer.")

                self._reference: BufferPointer = self.device.create_sharded_l1_buffer(page_size=self._page_size, n_pages=self._n_pages, core_ids=self.core_ids, contiguous_n_pages=1)
            else:
                self._reference: BufferPointer = self.device.create_local_l1_buffer(page_size=self._page_size, n_pages=self._n_pages, core_ids=self.core_ids)
        else:
            self._reference: BufferPointer = self.device.create_sharded_main_buffer(page_size=self._page_size, n_pages=self._n_pages)  # TODO: support selective channel interleaving for each page
            
        if self._reference is None:
            raise Exception("[ERROR] Failed to allocate tensor buffer. This exception is may derived by the out-of-memory situation.")
        
        if initial is not None:
            self.update(initial)
        
        return self
    
    def deallocate(self):
        if not self.is_allocated:
            raise Exception("Cannot deallocate the tensor buffer since it is not allocated yet.")
        
        self.device.remove_buffer(self.reference)
        self._reference = None
        
        return self
    
    def update(self, tensor: torch.Tensor):
        tensor = tensor.to(dtype=self.tensor_dtype).reshape((self.i_dim, self.y_dim-self.y_pad, self.x_dim-self.x_pad))
        tensor = torch.nn.functional.pad(tensor, (0, self.x_pad, 0, self.y_pad, 0, 0), mode='constant', value=0)
        tensor = tensor.reshape(self.i_dim, self.y_n_pages, self.y_page, self.x_n_pages, self.x_page)
        tensor = tensor.permute(0, 1, 3, 2, 4)
        tensor = tensor.reshape(self.i_dim * self.y_n_pages * self.x_n_pages, self.y_page * self.x_page)
        
        buffer_handle = self.reference.resolve(is_read=False)
        
        for page_idx in range(self.n_pages):
            page_ptr = buffer_handle.page_ptrs[page_idx]
            self.device.set_ptr_content(page_ptr, tensor[page_idx, :])
            
        return self
            
    def restore(self) -> torch.Tensor:
        tensor = self.device.get_ptr_content(self.reference.resolve(is_read=True), shape=(-1,), dtype=self.tensor_dtype)
        tensor = tensor.reshape(self.i_dim, self.y_n_pages, self.x_n_pages, self.y_page, self.x_page)
        tensor = tensor.permute(0, 1, 3, 2, 4)
        tensor = tensor.reshape(self.i_dim, self.y_n_pages * self.y_page, self.x_n_pages * self.x_page)
        tensor = tensor[:, :self.y_dim - self.y_pad, :self.x_dim - self.x_pad]
        tensor = tensor.reshape(self.tensor_shape)
        
        return tensor
            
    def get_page_idx_by_owner(self, core_id: int) -> list[int]:
        if self.layout.mem_type == MCA_TensorMemoryType.MAIN:
            raise Exception("[ERROR] get_page_idx_by_owner is only available for L1 memory type.")
        
        if core_id not in self.core_ids:
            raise Exception(f"core_id {core_id} is not in the core_ids of this buffer.")
        
        if len(self.core_ids) == 1:
            return list(range(self.n_pages))
        
        return [i for i in range(self.n_pages) if (i % len(self.core_ids)) == self.core_ids.index(core_id)]
    
    def get_reference_by_page_idx(self, *page_idx: int) -> BufferPointer:
        buffer_handle = self.reference.resolve(is_read=True)
        page_ptrs = [buffer_handle.page_ptrs[i] for i in page_idx]
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=len(page_ptrs), page_ptrs=page_ptrs)
        
        return BufferPointer(new_buffer_handle)
    
    def get_row_contiguous_reference(self, i_page_idx: int, y_page_idx: int) -> BufferPointer:
        buffer_handle = self.reference.resolve(is_read=True)
        
        offset = i_page_idx * self.y_n_pages * self.x_n_pages
        st = offset + y_page_idx * self.x_n_pages
        ed = offset + (y_page_idx + 1) * self.x_n_pages
        
        page_ptrs = [buffer_handle.page_ptrs[i] for i in range(st, ed, 1)]
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=len(page_ptrs), page_ptrs=page_ptrs)
        
        return BufferPointer(new_buffer_handle)
    
    def get_page_reference(self, i_page_idx: int, y_page_idx: int, x_page_idx: int) -> BufferPointer:
        buffer_handle = self.reference.resolve(is_read=True)
        
        offset = i_page_idx * self.y_n_pages * self.x_n_pages + y_page_idx * self.x_n_pages + x_page_idx
        page_ptrs = [buffer_handle.page_ptrs[offset],]
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=1, page_ptrs=page_ptrs)
        
        return BufferPointer(new_buffer_handle)

    @property
    def reference(self) -> BufferPointer:
        if not self.is_allocated:
            raise Exception("Cannot obtain the reference of the tensor buffer since it is not allocated yet.")
        return self._reference[:self.n_pages]  # TODO: prevent out-of-bound access (channel-interleaved buffer may have more pages than the number of pages required by the tensor shape)
    
    @property
    def n_pages(self) -> int:
        return self.i_dim * self.y_n_pages * self.x_n_pages
    
    @property
    def buffer_shape(self) -> tuple[int, ...]:
        shape = list(self.tensor_shape)
        if len(shape) >= 2: shape[-2] += self.y_pad
        if len(shape) >= 1: shape[-1] += self.x_pad
        return tuple(shape)
    
    @property
    def is_allocated(self) -> bool:
        return self._reference is not None
    
    def __str__(self):
        return f"MCA_TensorBuffer(mem_type={self.layout.mem_type}, shape={self.tensor_shape}, dtype={self.tensor_dtype}, page_shape=({self.y_page}, {self.x_page}), page_grid=({self.i_dim}, {self.y_n_pages}, {self.x_n_pages}), device={type(self.device).__name__}, core_ids={self.core_ids})"


_global_mca_rt_op_id: str = None

def activate_global_mca_rt_op(rt_op_id: str):
    global _global_mca_rt_op_id
    if _global_mca_rt_op_id is not None:
        raise Exception("[ERROR] The global MCA runtime operator has already been activated. This exception is mainly caused by the recursive call of MCA_RT_OPERATOR. Note that MCA_RT_OPERATOR cannot be called inside another MCA_RT_OPERATOR.")
    _global_mca_rt_op_id = rt_op_id
    
def deactivate_global_mca_rt_op():
    global _global_mca_rt_op_id
    _global_mca_rt_op_id = None
    
def check_global_mca_rt_op_active() -> bool:
    global _global_mca_rt_op_id
    return _global_mca_rt_op_id is not None

def get_global_mca_rt_op_id() -> str:
    global _global_mca_rt_op_id
    return _global_mca_rt_op_id

class MCA_RT_JIT_COMPILE_REGION:
    def __init__(self, core: NPUCore, kernel_id: str=None):
        self._core = core
        self._kernel = Kernel(kernel_id=kernel_id)
        
        self._history_context_mode   = None
        self._history_core_context   = None
        self._history_kernel_context = None
        
        if not isinstance(core, NPUCore):
            raise Exception(f"The argument of MCA_RT_JIT_REGION_AUTO_DISPATCH must be a NPUCore instance, but got {type(core)}.")

    def __enter__(self):
        self._history_context_mode   = get_global_context_mode()
        self._history_core_context   = get_global_core_context()
        self._history_kernel_context = get_global_kernel_context()
        
        if self._history_context_mode == GlobalContextMode.COMPILE:
            logger.warning(f"Calling MCA_RT_JIT_REGION_AUTO_DISPATCH with COMPILE context may cause unexpected behavior.")
            if self._history_core_context.core_id != self._core.core_id:
                raise Exception(f"Nested MCA_RT_JIT_REGION_AUTO_DISPATCH with different core context is not allowed. (current core: {self._core.core_id}, history core: {self._history_core_context.core_id})")
        
        set_global_context(GlobalContextMode.COMPILE, self._core, kernel=self._kernel)
        
    def __exit__(self, exc_type, exc_value, traceback):
        set_global_context(self._history_context_mode, self._history_core_context, kernel=self._history_kernel_context)

        if self._history_context_mode == GlobalContextMode.IDLE and check_global_mca_rt_op_active():
            self._kernel.kernel_id = f"{get_global_mca_rt_op_id()}::{self._kernel.kernel_id}"
            self._core.dispatch_main_kernel(slot_id="RT", kernel=self._kernel)
        elif self._history_context_mode == GlobalContextMode.COMPILE:
            for step in self._kernel._execution_steps:
                self._history_kernel_context.add_execution_step(step)
                
        return False  # Do not suppress exceptions

def MCA_RT_KERNEL(func: Callable):
    @functools.wraps(func)
    def __wrapper(*args, **kwargs):
        pargs = parse_arguments(args, kwargs, ["core"])
        core: NPUCore = pargs["core"]
        
        if not isinstance(core, NPUCore):
            raise Exception(f"The first argument of the MCA_RT_KERNEL-decorated function or the keyword argument 'core' must be a NPUCore instance, but got {type(core)}.")
        
        rt_kernel = KernelPrototype(func=func, args=args, kwargs=kwargs)
        rt_kernel.compiled_kernel_id = func.__name__

        if check_global_mca_rt_op_active():
            rt_kernel.compiled_kernel_id = f"{get_global_mca_rt_op_id()}::{rt_kernel.compiled_kernel_id}"
        
        core.dispatch_main_kernel(slot_id="RT", kernel=rt_kernel)  # TODO: currently, all runtime kernels are dispatched to the "RT" slot (inter-dependency between consecutive runtime kernels are determined by the order of the kernel calls)
        
        return rt_kernel
    return __wrapper

def MCA_RT_OPERATOR(func: Callable):
    @functools.wraps(func)
    def __wrapper(*args, **kwargs) -> None:
        try:
            activate_global_mca_rt_op(rt_op_id=func.__name__)
            
            pargs = parse_arguments(args, kwargs, ["device"])
            device: MCA_DeviceBase = pargs["device"]
            
            if not isinstance(device, MCA_DeviceBase):
                raise Exception(f"The first argument of the MCA_RT_OPERATOR-decorated function or the keyword argument 'device' must be a MCA_DeviceBase instance, but got {type(device)}.")
            
            ret = func(*args, **kwargs)
        finally:
            deactivate_global_mca_rt_op()
        
        if ret is not None:
            raise Exception(f"The MCA_RT_OPERATOR-decorated function must return None, but got {type(ret)}.")
        return ret
    return __wrapper