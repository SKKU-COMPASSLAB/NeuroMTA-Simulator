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
    
    "MCA_RT_KERNEL_THREAD",
    "MCA_RT_OPERATOR",
    "MCA_RuntimeKernel",
]
    
    
class MCA_TensorMemoryType(enum.Enum):
    L1 = enum.auto()
    MAIN = enum.auto()


class MCA_TensorMemoryLayout:
    def __init__(
        self, 
        mem_type: MCA_TensorMemoryType, 
        grid_shape: int | Sequence[int], 
        page_shape: int | Sequence[int],
    ):        
        self.mem_type = mem_type
        
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
        self.y_page_size, self.x_page_size = page_shape
                
    def copy(self):
        return copy.deepcopy(self)

    def overrides(
        self, 
        mem_type: MCA_TensorMemoryType  = None,
        grid_shape: int | Sequence[int] = None,
        page_shape: int | Sequence[int] = None,
    ) -> 'MCA_TensorMemoryLayout':
        
        return MCA_TensorMemoryLayout(
            mem_type   = self.mem_type if mem_type is None else mem_type,
            grid_shape = (self.y_grid, self.x_grid) if grid_shape is None else grid_shape,
            page_shape = (self.y_page_size, self.x_page_size) if page_shape is None else page_shape,
        )


class MCA_TensorBuffer:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, layout: MCA_TensorMemoryLayout, device: MCA_DeviceBase, core_ids: list[int] | int=None):
        # STEP 1: Setup
        self.tensor_shape   = tuple(shape)
        self.tensor_dtype   = dtype
        self.layout         = layout.copy()
        self.device         = device
        self.core_ids       = core_ids
        
        if len(self.tensor_shape) == 1:
            self.tensor_shape = (1, self.tensor_shape[0])
        
        if self.layout.mem_type == MCA_TensorMemoryType.L1:
            if isinstance(self.core_ids, int):
                self.core_ids = [self.core_ids]
            if not isinstance(self.core_ids, list):
                raise Exception(f"[ERROR] In case of L1 memory, core_ids must be a list of integers or a single integer, but got {type(self.core_ids)}.")
            if self.layout.y_grid * self.layout.x_grid != len(self.core_ids):
                raise Exception(f"[ERROR] The number of core_ids ({len(self.core_ids)}) must match the grid shape ({self.layout.y_grid} * {self.layout.x_grid} = {self.layout.y_grid * self.layout.x_grid}).")
        
        # STEP 2: Reshape the original tensor into memory layout format (matrix of (y, x))
        self.y_dim = sum(self.tensor_shape[:-1]) if len(self.tensor_shape) > 1 else 1
        self.x_dim = self.tensor_shape[-1]
        
        self.y_pad = (self.layout.y_page_size - (self.y_dim % self.layout.y_page_size)) if (self.y_dim % self.layout.y_page_size) != 0 else 0
        self.x_pad = (self.layout.x_page_size - (self.x_dim % self.layout.x_page_size)) if (self.x_dim % self.layout.x_page_size) != 0 else 0
        
        if self.y_pad > 0 or self.x_pad > 0:
            self.y_dim += self.y_pad
            self.x_dim += self.x_pad
        
        if self.layout.mem_type == MCA_TensorMemoryType.L1:
            if self.core_ids is None or len(self.core_ids) == 0:
                raise Exception("[ERROR] core_ids must be specified when mem_type is L1.")
            
            if self.y_dim % self.layout.y_page_size != 0 or self.x_dim % self.layout.x_page_size != 0:
                raise Exception(f"[ERROR] The tensor shape (y={self.y_dim}, x={self.x_dim}) must be multiples of page_shape (y_page_size={self.layout.y_page_size}, x_page_size={self.layout.x_page_size}).")
            
            if self.layout.y_grid > (self.y_dim // self.layout.y_page_size):
                self.layout.y_grid = self.y_dim // self.layout.y_page_size
            if self.layout.x_grid > (self.x_dim // self.layout.x_page_size):
                self.layout.x_grid = self.x_dim // self.layout.x_page_size
                
            if self.y_dim % (self.layout.y_grid * self.layout.y_page_size) != 0 or self.x_dim % (self.layout.x_grid * self.layout.x_page_size) != 0:
                raise Exception(f"[ERROR] The tensor shape (y={self.y_dim}, x={self.x_dim}) must be multiples of (grid_shape * page_shape) = ({self.layout.y_grid * self.layout.y_page_size}, {self.layout.x_grid * self.layout.x_page_size}).")
            
            if len(self.core_ids) > (self.layout.y_grid * self.layout.x_grid):
                self.core_ids = self.core_ids[:(self.layout.y_grid * self.layout.x_grid)]  # TODO: This code does not considers the shape of the core grid (MCA does not have any core grid concept, only MTA has ...)
            
            self.y_shard_grid = self.layout.y_grid
            self.x_shard_grid = self.layout.x_grid
            self.y_page_num_per_shard = self.y_dim // (self.layout.y_grid * self.layout.y_page_size)
            self.x_page_num_per_shard = self.x_dim // (self.layout.x_grid * self.layout.x_page_size)

            n_channel = len(self.core_ids)
            n_contiguous_page = self.y_page_num_per_shard * self.x_page_num_per_shard
            n_shard_per_channel = math.ceil((self.y_shard_grid * self.x_shard_grid) / n_channel)
            n_pages = n_contiguous_page * n_shard_per_channel * n_channel

            if n_channel > 1:
                if not isinstance(self.device, MTA_DeviceBase):
                    raise Exception("[ERROR] The device must be a MTA_DeviceBase when allocating sharded L1 buffer.")
                
                self._reference: BufferPointer = self.device.create_sharded_l1_buffer(
                    page_size=self.layout.y_page_size * self.layout.x_page_size * self.tensor_dtype.itemsize,
                    n_pages=n_pages,
                    core_ids=self.core_ids,
                    contiguous_n_pages=n_contiguous_page
                )
            else:
                self._reference: BufferPointer = self.device.create_local_l1_buffer(
                    page_size=self.layout.y_page_size * self.layout.x_page_size * self.tensor_dtype.itemsize,
                    n_pages=n_pages,
                    core_ids=self.core_ids
                )
                
                if isinstance(self._reference, list):
                    self._reference = self._reference[0]  # get the buffer of the first core
                
        elif self.layout.mem_type == MCA_TensorMemoryType.MAIN:
            self.y_shard_grid = self.y_dim // self.layout.y_page_size
            self.x_shard_grid = self.x_dim // self.layout.x_page_size
            self.y_page_num_per_shard = 1
            self.x_page_num_per_shard = 1
            
            n_channel = self.device.mem_context.main_config.ch_num  # TODO: the buffer should always be distributed across all memory channels
            n_contiguous_page = 1   # TODO: only support page-level channel interleaving
            n_shard_per_channel = math.ceil((self.y_page_num_per_shard * self.x_page_num_per_shard * self.y_shard_grid * self.x_shard_grid) / n_channel)
            n_pages = n_contiguous_page * n_shard_per_channel * n_channel
            
            self._reference: BufferPointer = self.device.create_sharded_main_buffer(
                page_size=self.layout.y_page_size * self.layout.x_page_size * self.tensor_dtype.itemsize,
                n_pages=n_pages,
                channel_id=list(range(n_channel)),
            )
        
        else:
            raise Exception(f"[ERROR] Invalid memory type {self.layout.mem_type}.")
        
        if self._reference is None:
            raise Exception("[ERROR] Failed to allocate tensor buffer. This exception is may derived by the out-of-memory situation.")
    
    def update(self, tensor: torch.Tensor):
        # STEP 1: Reshape the original tensor into memory layout format
        #   - flatten the tensor to 2D (y, x)
        #   - pad the tensor to be multiples of page shape
        #   - reshape the tensor to (y_shard_num, y_page_num, y_page, x_shard_num, x_page_num, x_page)
        #   - permute the tensor to (y_shard_idx, x_shard_idx, y_page_idx, x_page_idx, y_page_shape, x_page_shape)
        #     * ROW_MAJOR paging order: (y_si, x_si, y_pi, x_pi, y_ps, x_ps) = (0, 1, 2, 3, 4, 5)
        #   - reshape the tensor to (y_shard_num, x_shard_num, y_page_num * x_page_num, y_page_shape, x_page_shape)
        if tensor.dtype != self.tensor_dtype:
            raise Exception(f"[ERROR] Invalid tensor dtype: expected {self.tensor_dtype}, got {tensor.dtype}.")
        if tensor.numel() != torch.prod(torch.tensor(self.tensor_shape)):
            raise Exception(f"[ERROR] Invalid tensor shape: expected {self.tensor_shape}, got {tensor.shape}.")
            
        tensor = tensor.reshape(self.y_dim - self.y_pad, self.x_dim - self.x_pad)  # flatten the tensor to 2D (y, x)
        
        if self.y_pad > 0 or self.x_pad > 0:
            tensor = torch.nn.functional.pad(tensor, (0, self.x_pad, 0, self.y_pad), mode='constant', value=0)

        tensor = tensor.reshape(self.y_shard_grid, self.y_page_num_per_shard, self.layout.y_page_size, self.x_shard_grid, self.x_page_num_per_shard, self.layout.x_page_size)        
        tensor = tensor.permute(0, 3, 1, 4, 2, 5)  # (y_si, x_si, y_pi, x_pi, y_ps, x_ps)
        tensor = tensor.reshape(self.y_shard_grid, self.x_shard_grid, self.y_page_num_per_shard * self.x_page_num_per_shard, self.layout.y_page_size, self.layout.x_page_size)  # (y_si, x_si, yx_pi, y_ps, x_ps)
        
        # STEP 2: Copy each page to the allocated buffer
        buffer_handle = self._reference.resolve(is_read=False)
        
        for page_idx in range(buffer_handle.n_pages):
            page_ptr = buffer_handle.page_ptrs[page_idx]
            
            y_si  = page_idx // (self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)
            x_si  = (page_idx - (y_si * self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)) // (self.y_page_num_per_shard * self.x_page_num_per_shard)
            yx_pi = page_idx % (self.y_page_num_per_shard * self.x_page_num_per_shard)
            
            if y_si >= self.y_shard_grid or x_si >= self.x_shard_grid:
                break   # if the X/Y shard index exceeds the number of shards, stop copying
            
            page_data = tensor[y_si, x_si, yx_pi, :, :].reshape(self.layout.y_page_size, self.layout.x_page_size).contiguous()
            
            self.device.set_ptr_content(page_ptr, page_data)
            
    def restore(self) -> torch.Tensor:
        # STEP 1: Create an empty tensor in memory layout format
        tensor = torch.zeros((self.y_shard_grid, self.x_shard_grid, self.y_page_num_per_shard * self.x_page_num_per_shard, self.layout.y_page_size, self.layout.x_page_size), dtype=self.tensor_dtype)
        
        # STEP 2: Copy each page from the allocated buffer to the tensor
        buffer_handle = self._reference.resolve(is_read=True)
        
        for page_idx in range(buffer_handle.n_pages):
            page_ptr = buffer_handle.page_ptrs[page_idx]
            
            y_si  = page_idx // (self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)
            x_si  = (page_idx - (y_si * self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)) // (self.y_page_num_per_shard * self.x_page_num_per_shard)
            yx_pi = page_idx % (self.y_page_num_per_shard * self.x_page_num_per_shard)
            
            if y_si >= self.y_shard_grid or x_si >= self.x_shard_grid:
                break   # if the X/Y shard index exceeds the number of shards, stop copying
            
            page_data = self.device.get_ptr_content(page_ptr, shape=(self.layout.y_page_size, self.layout.x_page_size), dtype=self.tensor_dtype)
            tensor[y_si, x_si, yx_pi, :, :] = page_data
        
        # STEP 3: Reshape the tensor back to original format
        tensor = tensor.reshape(self.y_shard_grid, self.x_shard_grid, self.y_page_num_per_shard, self.x_page_num_per_shard, self.layout.y_page_size, self.layout.x_page_size)  # (y_si, x_si, y_pi, x_pi, y_ps, x_ps)
        tensor = tensor.permute(0, 2, 4, 1, 3, 5).reshape(self.y_dim, self.x_dim)  # (y, x)
        
        return tensor.view(dtype=self.tensor_dtype).reshape(shape=self.tensor_shape).clone().contiguous()
            
    def get_shard_reference(self, shard_idx: tuple[int, int] | int) -> BufferPointer:
        if isinstance(shard_idx, int):
            y_si = shard_idx // self.x_shard_grid
            x_si = shard_idx % self.x_shard_grid
        elif isinstance(shard_idx, tuple) and len(shard_idx) == 2:
            y_si, x_si = shard_idx
        else:
            raise Exception(f"[ERROR] Invalid shard_idx: must be an integer or a tuple of (y_shard_idx, x_shard_idx), but got {type(shard_idx)}.")
        
        if not (0 <= y_si < self.y_shard_grid) or not (0 <= x_si < self.x_shard_grid):
            raise Exception(f"[ERROR] Invalid shard_idx: out of range. (y_shard_idx={y_si}, x_shard_idx={x_si}), (y_shard_num={self.y_shard_grid}, x_shard_num={self.x_shard_grid})")
        
        buffer_handle = self._reference.resolve(is_read=True)
        
        n_contiguous_page = self.y_page_num_per_shard * self.x_page_num_per_shard
        shard_start_page_idx = (y_si * self.x_shard_grid + x_si) * n_contiguous_page
        page_st = shard_start_page_idx
        page_ed = shard_start_page_idx + n_contiguous_page
        
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=page_ed - page_st, page_ptrs=buffer_handle.page_ptrs[page_st:page_ed])
        
        return BufferPointer(new_buffer_handle)
    
    def get_row_contiguous_reference(self, row_page_idx: int) -> BufferPointer:
        if not (0 <= row_page_idx < self.y_shard_grid * self.y_page_num_per_shard):
            raise Exception(f"[ERROR] Invalid row_page_idx: out of range. (row_page_idx={row_page_idx}), (max_row_page_idx={self.y_shard_grid * self.y_page_num_per_shard - 1})")
        
        buffer_handle = self._reference.resolve(is_read=True)
        page_ptrs = []
        
        for c_i in range(self.x_shard_grid * self.x_page_num_per_shard):
            y_si = row_page_idx // self.y_page_num_per_shard
            x_si = c_i // self.x_page_num_per_shard
            y_pi = row_page_idx % self.y_page_num_per_shard
            x_pi = c_i % self.x_page_num_per_shard
            
            page_idx = y_si * (self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard) + x_si * (self.y_page_num_per_shard * self.x_page_num_per_shard) + y_pi * self.x_page_num_per_shard + x_pi
            page_ptrs.append(buffer_handle.page_ptrs[page_idx])
            
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=len(page_ptrs), page_ptrs=page_ptrs)
        
        return BufferPointer(new_buffer_handle)
    
    def get_page_reference(self, shard_idx: tuple[int, int] | int, page_idx: tuple[int, int] | int) -> BufferPointer:
        if isinstance(shard_idx, int):
            y_si = shard_idx // self.x_shard_grid
            x_si = shard_idx % self.x_shard_grid
        elif isinstance(shard_idx, tuple) and len(shard_idx) == 2:
            y_si, x_si = shard_idx
        else:
            raise Exception(f"[ERROR] Invalid shard_idx: must be an integer or a tuple of (y_shard_idx, x_shard_idx), but got {type(shard_idx)}.")
        
        if not (0 <= y_si < self.y_shard_grid) or not (0 <= x_si < self.x_shard_grid):
            raise Exception(f"[ERROR] Invalid shard_idx: out of range. (y_shard_idx={y_si}, x_shard_idx={x_si}), (y_shard_num={self.y_shard_grid}, x_shard_num={self.x_shard_grid})")
        
        if isinstance(page_idx, int):
            y_pi = page_idx // self.x_page_num_per_shard
            x_pi = page_idx % self.x_page_num_per_shard
        elif isinstance(page_idx, tuple) and len(page_idx) == 2:
            y_pi, x_pi = page_idx
        else:
            raise Exception(f"[ERROR] Invalid page_idx: must be an integer or a tuple of (y_page_idx, x_page_idx), but got {type(page_idx)}.")
        
        if not (0 <= y_pi < self.y_page_num_per_shard) or not (0 <= x_pi < self.x_page_num_per_shard):
            raise Exception(f"[ERROR] Invalid page_idx: out of range. (y_page_idx={y_pi}, x_page_idx={x_pi}), (y_page_num={self.y_page_num_per_shard}, x_page_num={self.x_page_num_per_shard})")
        
        page_idx = (y_si * self.x_shard_grid + x_si) * (self.y_page_num_per_shard * self.x_page_num_per_shard) + y_pi * self.x_page_num_per_shard + x_pi
        
        buffer_handle = self._reference.resolve(is_read=True)
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=1, page_ptrs=[buffer_handle.page_ptrs[page_idx]])
        
        return BufferPointer(new_buffer_handle)

    def get_core_id_to_page_coord_mapping(self) -> dict[int, list[tuple[int, int]]]:
        if self.layout.mem_type != MCA_TensorMemoryType.L1:
            raise Exception("[ERROR] get_page_idx_core_id_mapping is only available for L1 memory type.")
        
        buffer_handle = self._reference.resolve(is_read=True)
        page_idx_core_id_map: dict[int, list[tuple[int, int]]] = {i: [] for i in self.core_ids}
        
        for page_idx, page_ptr in enumerate(buffer_handle.page_ptrs):
            c_id = self.device.get_npu_core(addr=page_ptr.addr).core_id
            
            y_si  = page_idx // (self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)
            x_si  = (page_idx - (y_si * self.x_shard_grid * self.y_page_num_per_shard * self.x_page_num_per_shard)) // (self.y_page_num_per_shard * self.x_page_num_per_shard)
            y_pi = (page_idx % (self.y_page_num_per_shard * self.x_page_num_per_shard)) // self.x_page_num_per_shard
            x_pi = (page_idx % (self.y_page_num_per_shard * self.x_page_num_per_shard)) % self.x_page_num_per_shard
            
            y_coord = y_si * self.y_page_num_per_shard + y_pi
            x_coord = x_si * self.x_page_num_per_shard + x_pi

            page_idx_core_id_map[c_id].append((y_coord, x_coord))

        return page_idx_core_id_map
    
    def get_multiple_pages_reference(self, *page_coords: tuple[int, int]) -> BufferPointer:
        buffer_handle = self._reference.resolve(is_read=True)
        page_ptrs = []

        for y_coord, x_coord in page_coords:
            y_si = y_coord // self.y_page_num_per_shard
            x_si = x_coord // self.x_page_num_per_shard
            y_pi = y_coord % self.y_page_num_per_shard
            x_pi = x_coord % self.x_page_num_per_shard
            
            page_idx = (y_si * self.x_shard_grid + x_si) * (self.y_page_num_per_shard * self.x_page_num_per_shard) + y_pi * self.x_page_num_per_shard + x_pi
                        
            page_ptrs.append(buffer_handle.page_ptrs[page_idx])
        
        new_buffer_handle = BufferHandle(page_size=buffer_handle.page_size, n_pages=len(page_ptrs), page_ptrs=page_ptrs)
        
        return BufferPointer(new_buffer_handle)

    @property
    def reference(self) -> BufferPointer:
        return self._reference
    
    @property
    def n_pages(self) -> int:
        return self._reference.resolve(is_read=True).n_pages


_global_mca_rt_op_id: str = None
# _global_mca_rt_kernel_queue: list['MCA_RuntimeKernel'] = []

def activate_global_mca_rt_op(rt_op_id: str):
    global _global_mca_rt_op_id
    if _global_mca_rt_op_id is not None:
        raise Exception("[ERROR] The global MCA runtime operator has already been activated. This exception is mainly caused by the recursive call of MCA_RuntimeOperator. Note that MCA_RuntimeOperator cannot be used inside another MCA_RuntimeOperator.")
    _global_mca_rt_op_id = rt_op_id
    reset_global_mca_rt_kernel_queue()
    
def deactivate_global_mca_rt_op():
    global _global_mca_rt_op_id
    for rt_kernel in get_global_mca_rt_kernel_queue():
        rt_kernel.dispatch(_global_mca_rt_op_id)
    _global_mca_rt_op_id = None
    reset_global_mca_rt_kernel_queue()
    
def check_global_mca_rt_op_active() -> bool:
    global _global_mca_rt_op_id
    return _global_mca_rt_op_id is not None

def get_global_mca_rt_kernel_queue() -> list['MCA_RuntimeKernel']:
    global _global_mca_rt_kernel_queue
    return _global_mca_rt_kernel_queue

def reset_global_mca_rt_kernel_queue():
    global _global_mca_rt_kernel_queue
    _global_mca_rt_kernel_queue = []
    
def register_global_mca_rt_kernel(kernel: 'MCA_RuntimeKernel'):
    global _global_mca_rt_kernel_queue
    _global_mca_rt_kernel_queue.append(kernel)


def MCA_RT_KERNEL_THREAD(func: Callable):
    @functools.wraps(func)
    def __wrapper(self):
        return func(self)
    __wrapper._is_mca_rt_kernel_thread = True
    return __wrapper

def MCA_RT_OPERATOR(func: Callable):
    @functools.wraps(func)
    def __wrapper(*args, **kwargs):
        activate_global_mca_rt_op(rt_op_id=func.__name__)
        try:
            pargs = parse_arguments(args, kwargs, ["device"])
            device: MCA_DeviceBase = pargs["device"]
            
            if not isinstance(device, MCA_DeviceBase):
                raise Exception(f"[ERROR] The first argument of the MCA_RT_OPERATOR-decorated function or the keyword argument 'device' must be a MCA_DeviceBase instance, but got {type(device)}.")
            
            ret = func(*args, **kwargs)
        finally:
            deactivate_global_mca_rt_op()
            
        device.run_kernels()
        return ret
    return __wrapper

    
class MCA_RuntimeKernel:
    def __init__(self, core: NPUCore):
        self.core = core
        
        if check_global_mca_rt_op_active():
            register_global_mca_rt_kernel(self)
    
    @jit_prototype
    def __call__(self):
        threads = self._get_instance_kernel_threads()
            
        if len(threads) == 0:
            raise Exception("[ERROR] No threads defined in the MCA_RuntimeKernel instance. Please define at least one method decorated with @MCA_RT_KERNEL_THREAD.")
        elif len(threads) == 1:
            for name, thread in threads.items():
                thread()
        else:
            for name, thread in threads.items():
                with new_parallel_thread(name):
                    thread()
            self.core.parallel_merge()
    
    def _get_instance_kernel_threads(self):
        kernel_threads = {}
        
        for attr_name in dir(self):
            if attr_name in ("_get_instance_kernel_threads", "dispatch"):
                continue
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            
            attr = getattr(self, attr_name)
            
            if callable(attr) and hasattr(attr, '_is_mca_rt_kernel_thread'):
                kernel_threads[attr_name] = attr

        return kernel_threads
    
    def dispatch(self, slot_id: str="MAIN"):
        try:
            rt_main_kernel = self()
        except Exception as e:
            logger.error(f"Failed to compile the MCA runtime kernel '{type(self).__name__}': {e}")
            raise e
            
        self.core.dispatch_main_kernel(slot_id, rt_main_kernel)
