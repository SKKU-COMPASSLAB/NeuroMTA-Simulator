import enum
import torch
import math
import copy
import functools
from typing import Sequence, Callable

from neuromta.framework import *
from neuromta.hardware.implementation.hardware import MultiCoreAccelerator, MultiTileAccelerator


__all__ = [
    "TensorShardType",
    "TensorPagingOrder",
    "TensorMemoryType",
    "TensorMemoryLayout",
    "TensorBuffer",
    
    "RuntimeOperator",
    "runtime_kernel_method",
]


class TensorShardType(enum.Enum):
    BLOCK  = enum.auto()
    WIDTH  = enum.auto()
    HEIGHT = enum.auto()
    
    
class TensorPagingOrder(enum.Enum):
    ROW_MAJOR = enum.auto()
    COLUMN_MAJOR = enum.auto()
    
    
class TensorMemoryType(enum.Enum):
    L1 = enum.auto()
    MAIN = enum.auto()


class TensorMemoryLayout:
    def __init__(
        self, 
        mem_type: TensorMemoryType, 
        shard_type: TensorShardType, 
        paging_order: TensorPagingOrder,
        grid_shape: int | Sequence[int], 
        shard_shape: int | Sequence[int],
        page_shape: int | Sequence[int],
    ):        
        self.mem_type = mem_type
        self.shard_type = shard_type
        self.paging_order = paging_order
        
        if isinstance(grid_shape, int):
            grid_shape = (1, grid_shape,)
        elif isinstance(grid_shape, Sequence):
            if len(grid_shape) == 1:
                grid_shape = (1, grid_shape[0],)
            elif len(grid_shape) > 2:
                raise Exception("[ERROR] Invalid grid_shape: grid_shape must be a tuple of (y, x).")

        if isinstance(page_shape, int):
            page_shape = (1, page_shape,)
        elif isinstance(page_shape, Sequence):
            if len(page_shape) == 1:
                page_shape = (1, page_shape[0],)
            elif len(page_shape) > 2:
                raise Exception("[ERROR] Invalid page_shape: page_shape must be a tuple of (y, x).")
            
        self.y_grid, self.x_grid = grid_shape
        self.y_page, self.x_page = page_shape
        self.y_shard,  self.x_shard  = -1, -1    # indicating block shard shape is not determined by the memory layout
        
        if self.shard_type == TensorShardType.BLOCK:
            if shard_shape is None:
                raise Exception("[ERROR] shard_shape must be specified when shard_type is BLOCK.")
            if isinstance(shard_shape, int):
                shard_shape = (1, shard_shape,)
            elif isinstance(shard_shape, Sequence):
                if len(shard_shape) == 1:
                    shard_shape = (1, shard_shape[0],)
                elif len(shard_shape) > 2:
                    raise Exception("[ERROR] Invalid shard_shape: shard_shape must be a tuple of (y, x).")
                    
            self.y_shard, self.x_shard = shard_shape
        elif self.shard_type == TensorShardType.WIDTH:
            if shard_shape is None:
                raise Exception("[ERROR] shard_shape must be specified when shard_type is WIDTH.")
            if isinstance(shard_shape, int):
                self.y_shard = -1
                self.x_shard = shard_shape
            elif isinstance(shard_shape, Sequence):
                if len(shard_shape) == 1:
                    self.y_shard = -1
                    self.x_shard = shard_shape[0]
                elif len(shard_shape) >= 2:
                    raise Exception("[ERROR] Invalid shard_shape: shard_shape must be an integer for the shard_type WIDTH.")
        elif self.shard_type == TensorShardType.HEIGHT:
            if shard_shape is None:
                raise Exception("[ERROR] shard_shape must be specified when shard_type is HEIGHT.")
            if isinstance(shard_shape, int):
                self.y_shard = shard_shape
                self.x_shard = -1
            elif isinstance(shard_shape, Sequence):
                if len(shard_shape) == 1:
                    self.y_shard = shard_shape[0]
                    self.x_shard = -1
                elif len(shard_shape) >= 2:
                    raise Exception("[ERROR] Invalid shard_shape: shard_shape must be an integer for the shard_type HEIGHT.")
                
    def copy(self):
        return copy.deepcopy(self)


class TensorBuffer:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, layout: TensorMemoryLayout, device: MultiCoreAccelerator, core_ids: list[int] | int=None):
        # STEP 1: Setup
        self.tensor_shape   = tuple(shape)
        self.tensor_dtype   = dtype
        self.layout         = layout.copy()
        self.device         = device
        self.core_ids       = core_ids
        
        if self.layout.mem_type == TensorMemoryType.L1:
            if isinstance(self.core_ids, int):
                self.core_ids = [self.core_ids]
            if self.layout.y_grid * self.layout.x_grid != len(self.core_ids):
                raise Exception(f"[ERROR] The number of core_ids ({len(self.core_ids)}) must match the grid shape ({self.layout.y_grid} * {self.layout.x_grid} = {self.layout.y_grid * self.layout.x_grid}).")
        
        # STEP 2: Reshape the original tensor into memory layout format (matrix of (y, x))
        self.y_dim = sum(self.tensor_shape[:-1]) if len(self.tensor_shape) > 1 else 1
        self.x_dim = self.tensor_shape[-1]
        
        self.y_pad = (self.layout.y_page - (self.y_dim % self.layout.y_page)) if (self.y_dim % self.layout.y_page) != 0 else 0
        self.x_pad = (self.layout.x_page - (self.x_dim % self.layout.x_page)) if (self.x_dim % self.layout.x_page) != 0 else 0
        
        if self.y_pad > 0 or self.x_pad > 0:
            self.y_dim += self.y_pad
            self.x_dim += self.x_pad
        
        # STEP 4: Determine the shard shape if not specified in the memory layout
        self.layout.y_shard = self.y_dim if (self.layout.y_shard == -1) else self.layout.y_shard
        self.layout.x_shard = self.x_dim if (self.layout.x_shard == -1) else self.layout.x_shard
        
        # STEP 5: Validate the memory layout
        if (self.layout.y_shard % self.layout.y_page) != 0 or (self.layout.x_shard % self.layout.x_page) != 0:
            raise Exception(f"[ERROR] Invalid memory layout: shard shape must be multiples of page shape. (shard_shape=({self.layout.y_shard}, {self.layout.x_shard}), page_shape=({self.layout.y_page}, {self.layout.x_page}))")
        if (self.y_dim % self.layout.y_shard) != 0 or (self.x_dim % self.layout.x_shard) != 0:
            raise Exception(f"[ERROR] Invalid memory layout: tensor shape must be multiples of shard shape. (tensor_shape=({self.y_dim}, {self.x_dim}), shard_shape=({self.layout.y_shard}, {self.layout.x_shard}))")
        
        # STEP 6: Reshape the tensor into memory layout format
        self.y_shard_num = self.y_dim // self.layout.y_shard
        self.x_shard_num = self.x_dim // self.layout.x_shard
        self.y_page_num  = self.layout.y_shard // self.layout.y_page
        self.x_page_num  = self.layout.x_shard // self.layout.x_page
        
        # STEP 7: Allocate the tensor buffer in the given address space and create reference
        if self.layout.mem_type == TensorMemoryType.L1:
            if self.core_ids is None or len(self.core_ids) == 0:
                raise Exception("[ERROR] core_ids must be specified when mem_type is L1.")

            n_channel = len(self.core_ids)
            n_contiguous_page = self.y_page_num * self.x_page_num
            n_shard_per_channel = math.ceil((self.y_shard_num * self.x_shard_num) / n_channel)
            n_pages = n_contiguous_page * n_shard_per_channel * n_channel

            if n_channel > 1:
                if not isinstance(self.device, MultiTileAccelerator):
                    raise Exception("[ERROR] The device must be a MultiTileAccelerator when allocating sharded L1 buffer.")
                
                self._reference: Reference = self.device.create_sharded_l1_buffer(
                    page_size=self.layout.y_page * self.layout.x_page * self.tensor_dtype.itemsize,
                    n_pages=n_pages,
                    core_ids=self.core_ids,
                    contiguous_n_pages=n_contiguous_page
                )
            else:
                self._reference: Reference = self.device.create_local_l1_buffer(
                    page_size=self.layout.y_page * self.layout.x_page * self.tensor_dtype.itemsize,
                    n_pages=n_pages,
                    core_ids=self.core_ids
                )
                
                if isinstance(self._reference, list):
                    self._reference = self._reference[0]  # get the buffer of the first core
                
        elif self.layout.mem_type == TensorMemoryType.MAIN:
            n_channel = self.device.mem_context.main_config.ch_num  # TODO: the buffer should always be distributed across all memory channels
            n_contiguous_page = 1   # TODO: only support page-level channel interleaving
            n_shard_per_channel = math.ceil((self.y_page_num * self.x_page_num * self.y_shard_num * self.x_shard_num) / n_channel)
            n_pages = n_contiguous_page * n_shard_per_channel * n_channel
            
            self._reference: Reference = self.device.create_sharded_main_buffer(
                page_size=self.layout.y_page * self.layout.x_page * self.tensor_dtype.itemsize,
                n_pages=n_pages,
                channel_id=list(range(n_channel)),
            )
        
        else:
            raise Exception(f"[ERROR] Invalid memory type {self.layout.mem_type}.")
    
    def update(self, tensor: torch.Tensor):
        # STEP 1: Reshape the original tensor into memory layout format
        #   - flatten the tensor to 2D (y, x)
        #   - pad the tensor to be multiples of page shape
        #   - reshape the tensor to (y_shard_num, y_page_num, y_page, x_shard_num, x_page_num, x_page)
        #   - permute the tensor to (y_shard_idx, x_shard_idx, y_page_idx, x_page_idx, y_page_shape, x_page_shape)
        #     * if paging_order is ROW_MAJOR: (y_si, x_si, y_pi, x_pi, y_ps, x_ps) = (0, 1, 2, 3, 4, 5)
        #     * if paging_order is COLUMN_MAJOR: (y_si, x_si, x_pi, y_pi, y_ps, x_ps) = (0, 1, 3, 2, 4, 5)
        #   - reshape the tensor to (y_shard_num, x_shard_num, y_page_num * x_page_num, y_page_shape, x_page_shape)
        if tensor.dtype != self.tensor_dtype:
            raise Exception(f"[ERROR] Invalid tensor dtype: expected {self.tensor_dtype}, got {tensor.dtype}.")
        if tensor.numel() != torch.prod(torch.tensor(self.tensor_shape)):
            raise Exception(f"[ERROR] Invalid tensor shape: expected {self.tensor_shape}, got {tensor.shape}.")
            
        tensor = tensor.reshape(self.y_dim - self.y_pad, self.x_dim - self.x_pad)  # flatten the tensor to 2D (y, x)
        
        if self.y_pad > 0 or self.x_pad > 0:
            tensor = torch.nn.functional.pad(tensor, (0, self.x_pad, 0, self.y_pad), mode='constant', value=0)

        tensor = tensor.reshape(self.y_shard_num, self.y_page_num, self.layout.y_page, self.x_shard_num, self.x_page_num, self.layout.x_page)        
        if self.layout.paging_order == TensorPagingOrder.ROW_MAJOR:
            tensor = tensor.permute(0, 3, 1, 4, 2, 5)  # (y_si, x_si, y_pi, x_pi, y_ps, x_ps)
        elif self.layout.paging_order == TensorPagingOrder.COLUMN_MAJOR:
            tensor = tensor.permute(0, 3, 4, 1, 2, 5)  # (y_si, x_si, x_pi, y_pi, y_ps, x_ps)
        else:
            raise Exception("[ERROR] Invalid paging order.")
        tensor = tensor.reshape(self.y_shard_num, self.x_shard_num, self.y_page_num * self.x_page_num, self.layout.y_page, self.layout.x_page)  # (y_si, x_si, yx_pi, y_ps, x_ps)
        
        # STEP 2: Copy each page to the allocated buffer
        buffer_handle = self._reference.resolve(is_read=False)
        
        for page_idx in range(buffer_handle.n_pages):
            page_ptr = buffer_handle.page_ptrs[page_idx]
            
            y_si  = page_idx // (self.x_shard_num * self.y_page_num * self.x_page_num)
            x_si  = (page_idx - (y_si * self.x_shard_num * self.y_page_num * self.x_page_num)) // (self.y_page_num * self.x_page_num)
            yx_pi = page_idx % (self.y_page_num * self.x_page_num)
            
            if y_si >= self.y_shard_num or x_si >= self.x_shard_num:
                break   # if the X/Y shard index exceeds the number of shards, stop copying
            
            page_data = tensor[y_si, x_si, yx_pi, :, :].reshape(self.layout.y_page, self.layout.x_page).contiguous()
            
            self.device.set_ptr_content(page_ptr, page_data)
            
    def restore(self) -> torch.Tensor:
        # STEP 1: Create an empty tensor in memory layout format
        tensor = torch.zeros((self.y_shard_num, self.x_shard_num, self.y_page_num * self.x_page_num, self.layout.y_page, self.layout.x_page), dtype=self.tensor_dtype)
        
        # STEP 2: Copy each page from the allocated buffer to the tensor
        buffer_handle = self._reference.resolve(is_read=True)
        
        for page_idx in range(buffer_handle.n_pages):
            page_ptr = buffer_handle.page_ptrs[page_idx]
            
            y_si  = page_idx // (self.x_shard_num * self.y_page_num * self.x_page_num)
            x_si  = (page_idx - (y_si * self.x_shard_num * self.y_page_num * self.x_page_num)) // (self.y_page_num * self.x_page_num)
            yx_pi = page_idx % (self.y_page_num * self.x_page_num)
            
            if y_si >= self.y_shard_num or x_si >= self.x_shard_num:
                break   # if the X/Y shard index exceeds the number of shards, stop copying
            
            page_data = self.device.get_ptr_content(page_ptr, shape=(self.layout.y_page, self.layout.x_page), dtype=self.tensor_dtype)
            tensor[y_si, x_si, yx_pi, :, :] = page_data
        
        # STEP 3: Reshape the tensor back to original format
        if self.layout.paging_order == TensorPagingOrder.ROW_MAJOR:
            tensor = tensor.reshape(self.y_shard_num, self.x_shard_num, self.y_page_num, self.x_page_num, self.layout.y_page, self.layout.x_page)  # (y_si, x_si, y_pi, x_pi, y_ps, x_ps)
        elif self.layout.paging_order == TensorPagingOrder.COLUMN_MAJOR:
            tensor = tensor.reshape(self.y_shard_num, self.x_shard_num, self.x_page_num, self.y_page_num, self.layout.y_page, self.layout.x_page)  # (y_si, x_si, x_pi, y_pi, y_ps, x_ps)
            tensor = tensor.permute(0, 1, 3, 2, 4, 5)  # (y_si, x_si, y_pi, x_pi, y_ps, x_ps)
        else:
            raise Exception("[ERROR] Invalid paging order.")
        
        tensor = tensor.permute(0, 2, 4, 1, 3, 5).reshape(self.y_dim, self.x_dim)  # (y, x)
        
        return tensor.view(dtype=self.tensor_dtype).reshape(shape=self.tensor_shape).clone().contiguous()
            
    def get_shard_reference(self, shard_idx: tuple[int, int] | int) -> Reference:
        if isinstance(shard_idx, int):
            y_si = shard_idx // self.x_shard_num
            x_si = shard_idx % self.x_shard_num
        elif isinstance(shard_idx, tuple) and len(shard_idx) == 2:
            y_si, x_si = shard_idx
        else:
            raise Exception(f"[ERROR] Invalid shard_idx: must be an integer or a tuple of (y_shard_idx, x_shard_idx), but got {type(shard_idx)}.")
        
        if not (0 <= y_si < self.y_shard_num) or not (0 <= x_si < self.x_shard_num):
            raise Exception(f"[ERROR] Invalid shard_idx: out of range. (y_shard_idx={y_si}, x_shard_idx={x_si}), (y_shard_num={self.y_shard_num}, x_shard_num={self.x_shard_num})")
        
        buffer_handle = self._reference.resolve(is_read=True)
        
        page_st = (y_si * self.x_shard_num + x_si) * (self.y_page_num * self.x_page_num)
        page_ed = page_st + (self.y_page_num * self.x_page_num)
        
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=page_ed - page_st, page_ptrs=buffer_handle.page_ptrs[page_st:page_ed])
        
        return Reference(new_buffer_handle)

    @property
    def reference(self) -> Reference:
        return self._reference
    
    
def runtime_kernel_method(_func: Callable):
    def __wrapper(_rt: 'RuntimeOperator', _core: 'Core', *_args, **_kwargs) -> Kernel:
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        if len(_args):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' does not accept positional arguments")
        if len(_kwargs):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' does not accept keyword arguments")
        
        kernel = Kernel(
            _func.__name__,                 # the kernel ID is the name of the function
            functools.partial(_func, _rt),  # the behavioral model is the function itself
        )

        if get_global_context_mode() == GlobalContextMode.IDLE:
            pass  # do not automatically dispatch kernel
        elif get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context = get_global_kernel_context()
            
            if kernel_context is None:
                raise Exception(f"[ERROR] Cannot register kernel '{kernel.kernel_id}' to the compiled kernel since it is called outside of a low-level kernel function")
            
            kernel_context.add_execution_step(kernel)
        else:
            logger.warning(f"Kernel method '{_func.__name__}' is called outside of the compile or idle context. It implies that the kernel is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Kernel method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return kernel

    __wrapper._is_kernel_method = True  # mark this function as a kernel method (for debugging and profiling purposes)
    __wrapper._is_rt_kernel_method = True

    return __wrapper
    
    
class RuntimeOperator:
    def dispatch_runtime_to_core(self, core: Core):
        rt_kernel_methods: dict[str, Callable] = {attr: getattr(self, attr) for attr in dir(self) if callable(getattr(self, attr)) and hasattr(getattr(self, attr), '_is_rt_kernel_method')}
        rt_kernels: dict[str, Kernel] = {method_name: method(core) for method_name, method in rt_kernel_methods.items()}
        
        for slot_id, kernel in rt_kernels.items():
            core.dispatch_main_kernel(slot_id, kernel)
