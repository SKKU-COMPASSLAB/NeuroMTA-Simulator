import math
import torch
import functools
from typing import Sequence

from neuromta.framework import *
from neuromta.component.context.global_context import *
from neuromta.component.implementation.hardware import *


__all__ = [
    "MCA_TensorBuffer",
]


class MCA_TensorBuffer:
    def __init__(
        self, 
        
        mem_space: MCA_MemorySpace,
        shape: Sequence[int],    # shape of the tensor
        dtype: torch.dtype,      # data type of the tensor
        
        shard_shape: Sequence[int]=None,  # (shard_height, shard_width)
        
        blocked_mapping: bool=False,    # whether to use blocked mapping for height and width shards (only if the buffer is L1 memory and mem_ids are provided as MTA_CoreGrid)
    ):
        # Tensor Information
        self._mem_space = mem_space
        self._shape = shape if len(shape) >= 2 else (1, *shape)
        self._dtype = dtype
        
        # Sharding Factors
        if shard_shape is None:
            shard_shape = (self._shape[-2], self._shape[-1])  # no sharding by default
        elif isinstance(shard_shape, int):
            shard_shape = (shard_shape, shard_shape)
        elif len(shard_shape) != 2:
            raise ValueError("shard_shape must be a sequence of two integers: (shard_height, shard_width).")
        else:
            shard_shape = tuple(shard_shape)
        
        self._shard_y = shard_shape[0]
        self._shard_x = shard_shape[1]
        
        if self._shape[-2] % self._shard_y != 0:
            raise ValueError(f"Height {self._shape[-2]} is not divisible by shard height {self._shard_y}.")
        if self._shape[-1] % self._shard_x != 0:
            raise ValueError(f"Width {self._shape[-1]} is not divisible by shard width {self._shard_x}.")
        
        self._layout_y = functools.reduce(lambda x, y: x * y, self._shape[:-2], 1) * self._shape[-2]
        self._layout_x = self._shape[-1]
        
        self._n_outer_shards = functools.reduce(lambda x, y: x * y, self._shape[:-2], 1)
        self._n_y_shards = self._layout_y // self._shard_y
        self._n_x_shards = self._layout_x // self._shard_x
        
        # Tiling Factors
        self._tile_y = self._shard_y
        self._tile_x = self._shard_x
        
        self._n_y_tiles_per_shard = self._shard_y // self._tile_y
        self._n_x_tiles_per_shard = self._shard_x // self._tile_x
        
        # Memory Owners and Layout
        self._blocked_mapping = blocked_mapping
        if self._blocked_mapping and not isinstance(self.owner_ids, MTA_CoreGrid):
            self._blocked_mapping = False  # fallback to non-blocked mapping if mem_ids is not MTA_CoreGrid
        
        self._shard_size = self._shard_y * self._shard_x * self._dtype.itemsize
        self._shard_ptrs: list[list[Pointer]] = [[
            Pointer() 
            for _ in range(self._n_x_shards)] 
            for _ in range(self._n_y_shards)]
        
        self._is_allocated = False
        
    def reshape(self, *new_shape: int) -> "MCA_TensorBuffer":
        if new_shape[-1] != self._shape[-1]:
            raise ValueError("Reshaping is only supported for leading dimensions. The last dimension (width) must remain unchanged.")
        
        new_total_y_size = functools.reduce(lambda x, y: x * y, new_shape[:-1], 1)
        cur_total_y_size = functools.reduce(lambda x, y: x * y, self._shape[:-1], 1)
        
        if new_total_y_size != cur_total_y_size:
            raise ValueError("Reshaping must preserve the total number of elements in the tensor.")
        
        if new_shape[-2] % self._shard_y != 0:
            raise ValueError(f"New height {new_shape[-2]} is not divisible by shard height {self._shard_y}. It implies that the reshaping would change the memory layout of the tensor, which requires additional memory copy operations.")
        
        new_n_y_shards = new_shape[-2] // self._shard_y  # shard size should be preserved
        
        new_buffer = MCA_TensorBuffer(
            mem_space=self._mem_space,
            shape=new_shape,
            dtype=self._dtype,
            shard_grid=(new_n_y_shards, self._n_x_shards),
            blocked_mapping=self._blocked_mapping
        )
        
        for y in range(new_buffer.shard_grid[0]):
            for x in range(new_buffer.shard_grid[1]):
                new_buffer._shard_ptrs[y][x].addr = self._shard_ptrs[y][x].addr
                
        new_buffer.tiling(tile_shape=(self._tile_y, self._tile_x))
        
        return new_buffer
        
    def copy(self) -> "MCA_TensorBuffer":
        new_buffer = MCA_TensorBuffer(
            mem_space=self._mem_space,
            shape=self._shape,
            dtype=self._dtype,
            shard_shape=(self._shard_y, self._shard_x),
            blocked_mapping=self._blocked_mapping
        )
        
        for y in range(self._n_y_shards):
            for x in range(self._n_x_shards):
                new_buffer._shard_ptrs[y][x].addr = self._shard_ptrs[y][x].addr
        new_buffer._is_allocated = self._is_allocated
        
        new_buffer.tiling(tile_shape=(self._tile_y, self._tile_x))
        
        return new_buffer
        
    def tiling(self, tile_shape: Sequence[int]=None):
        if tile_shape is None:
            tile_shape = (self._shard_y, self._shard_x)
        if not isinstance(tile_shape, Sequence):
            tile_shape = (1, tile_shape)
        if len(tile_shape) != 2:
            raise ValueError("tile_shape must be a sequence of two integers: (tile_height, tile_width).")
        
        self._tile_y = tile_shape[0]
        self._tile_x = tile_shape[1]
        
        self._n_y_tiles_per_shard = math.ceil(self._shard_y / self._tile_y)  # enable non-divisible tiling (padding is applied when "args" are generated)
        self._n_x_tiles_per_shard = math.ceil(self._shard_x / self._tile_x)  # enable non-divisible tiling (padding is applied when "args" are generated)
        
        return self
        
    def required_mem_space_per_id(self) -> int:
        n_shards_per_mem_id = math.ceil(self._n_y_shards * self._n_x_shards / len(self.owner_ids))
        required_size_per_mem_id = n_shards_per_mem_id * self._shard_size
        return required_size_per_mem_id
    
    def check_mem_vacancy(self) -> bool:
        required_mem_space_per_id = self.required_mem_space_per_id()    

        for mem_id in self.owner_ids:
            available_size = self._mem_space.empty_space(mem_id)
            if required_mem_space_per_id > available_size:
                return False
        return True
    
    def allocate(self):
        if not self.check_mem_vacancy():
            raise RuntimeError("Not enough memory available for allocation. Create larger global memory context or reduce tensor buffer size.")
        
        if self._blocked_mapping:
            if self.mem_type != GlobalContextMemType.L1:
                self._blocked_mapping = False
                logger.warning(f"Unable to use blocked mapping: blocked mapping is only supported for L1 memory type.")
            if not isinstance(self.owner_ids, MTA_CoreGrid):
                self._blocked_mapping = False
                logger.warning(f"Unable to use blocked mapping: blocked mapping requires mem_ids to be of type MTA_CoreGrid.")
            if self._n_y_shards % self.owner_ids.shape[0] != 0:
                self._blocked_mapping = False
                logger.warning(f"Unable to use blocked mapping: number of height shards {self._n_y_shards} is not divisible by core grid height {self.owner_ids.shape[0]}.")
            if self._n_x_shards % self.owner_ids.shape[1] != 0:
                self._blocked_mapping = False
                logger.warning(f"Unable to use blocked mapping: number of width shards {self._n_x_shards} is not divisible by core grid width {self.owner_ids.shape[1]}.")
        
        if self._blocked_mapping:
            h_block_size = self._n_y_shards // self.owner_ids.shape[0]
            w_block_size = self._n_x_shards // self.owner_ids.shape[1]
            
            for h in range(self._n_y_shards):
                for w in range(self._n_x_shards):
                    core_grid_y = h // h_block_size
                    core_grid_x = w // w_block_size
                    core_id = self.owner_ids.core_ids[core_grid_y * self.owner_ids.shape[1] + core_grid_x]
                    
                    ptr = self._mem_space.allocate(
                        core_id,
                        size=self._shard_size
                    )
                    self._shard_ptrs[h][w] = ptr
        else:
            for h in range(self._n_y_shards):
                for w in range(self._n_x_shards):
                    mem_id_index = (h * self._n_x_shards + w) % len(self.owner_ids)
                    
                    ptr = self._mem_space.allocate(
                        self.owner_ids[mem_id_index],
                        size=self._shard_size
                    )
                    
                    self._shard_ptrs[h][w] = ptr
                    
        self._is_allocated = True
        
        return self
    
    def update(self, tensor: torch.Tensor):
        if not self._is_allocated:
            raise RuntimeError("Tensor buffer is not allocated. Call allocate() before update().")
        
        # tensor = torch.nn.functional.pad(tensor, (0, self._pad_x, 0, self._pad_y))  # pad width and height dimensions
        tensor = tensor.flatten().to(dtype=self._dtype)
        tensor = tensor.reshape(self._layout_y, self._layout_x)
        tensor = tensor.reshape(self._n_y_shards, self._shard_y, self._n_x_shards, self._shard_x)
        tensor = tensor.permute(0, 2, 1, 3)  # (n_height_shards, n_width_shards, shard_y, shard_x)
        
        for h in range(self._n_y_shards):
            for w in range(self._n_x_shards):
                shard_data = tensor[h, w].contiguous().flatten()
                ptr = self._shard_ptrs[h][w]
                self.device.mem_set_data(ptr, size=self._shard_size, data=shard_data)
                
        return self
                
    def restore(self) -> torch.Tensor:
        tensor = torch.empty((self._layout_y, self._layout_x), dtype=self._dtype)
        tensor = tensor.reshape(self._n_y_shards, self._shard_y, self._n_x_shards, self._shard_x)
        tensor = tensor.permute(0, 2, 1, 3)  # (n_height_shards, n_width_shards, shard_y, shard_x)
        
        for h in range(self._n_y_shards):
            for w in range(self._n_x_shards):
                ptr = self._shard_ptrs[h][w]
                shard_data = self.device.mem_get_data(ptr, size=self._shard_size, dtype=self._dtype)
                shard_data = shard_data.reshape(self._shard_y, self._shard_x)
                tensor[h, w] = shard_data
        
        tensor = tensor.permute(0, 2, 1, 3).reshape(self._layout_y, self._layout_x)
        tensor = tensor.reshape(self._shape)
        # tensor = tensor.flatten().reshape((self._shape[:-2] + (self._shape[-2] + self._pad_y, self._shape[-1] + self._pad_x)))
        # tensor = tensor[..., :self._shape[-2], :self._shape[-1]]  # remove padding
        return tensor
    
    def get_shard_ptr(self, y_shard_idx: int, x_shard_idx: int) -> Pointer:
        return self._shard_ptrs[y_shard_idx][x_shard_idx]
    
    def get_tile_ptr_read_args(self, y_shard_idx: int, x_shard_idx: int, y_tile_in_shard_idx: int, x_tile_in_shard_idx: int) -> tuple[Pointer, int, int, int, int, int]:
        y_shard_offset = y_tile_in_shard_idx * self._tile_y
        x_shard_offset = x_tile_in_shard_idx * self._tile_x
        
        total_offset = y_shard_offset * self._shard_x + x_shard_offset
        
        actual_tile_x = min(self._tile_x, self._shard_x - x_shard_offset)  # if the tile exceeds shard boundary (used for automatic padding)
        actual_tile_y = min(self._tile_y, self._shard_y - y_shard_offset)  # if the tile exceeds shard boundary (used for automatic padding)
    
        src_ptr = self._shard_ptrs[y_shard_idx][x_shard_idx] + (total_offset * self._dtype.itemsize)
        row_size = actual_tile_x * self._dtype.itemsize
        row_num = actual_tile_y
        src_row_stride = self._shard_x * self._dtype.itemsize
        dst_row_stride = self._tile_x  * self._dtype.itemsize
        dst_row_zero_pad = (self._tile_x - actual_tile_x) * self._dtype.itemsize  # if the tile exceeds shard boundary (used for automatic padding)
        
        return src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad
    
    def get_tile_ptr_write_args(self, y_shard_idx: int, x_shard_idx: int, y_tile_in_shard_idx: int, x_tile_in_shard_idx: int) -> tuple[Pointer, int, int, int, int]:
        y_shard_offset = y_tile_in_shard_idx * self._tile_y
        x_shard_offset = x_tile_in_shard_idx * self._tile_x
        
        total_offset = y_shard_offset * self._shard_x + x_shard_offset
        
        actual_tile_x = min(self._tile_x, self._shard_x - x_shard_offset)  # if the tile exceeds shard boundary (used for automatic padding)
        actual_tile_y = min(self._tile_y, self._shard_y - y_shard_offset)  # if the tile exceeds shard boundary (used for automatic padding)
        
        dst_ptr = self._shard_ptrs[y_shard_idx][x_shard_idx] + (total_offset * self._dtype.itemsize)
        row_size = actual_tile_x * self._dtype.itemsize
        row_num = actual_tile_y
        src_row_stride = self._tile_x  * self._dtype.itemsize
        dst_row_stride = self._shard_x * self._dtype.itemsize
        
        return dst_ptr, row_size, row_num, src_row_stride, dst_row_stride
    
    def get_shard_grid_from_tile_grid_idx(self, y_tile_idx: int, x_tile_idx: int) -> tuple[int, int, int, int]:
        y_shard_idx = y_tile_idx // self._n_y_tiles_per_shard
        x_shard_idx = x_tile_idx // self._n_x_tiles_per_shard
        
        y_tile_in_shard_idx = y_tile_idx % self._n_y_tiles_per_shard
        x_tile_in_shard_idx = x_tile_idx % self._n_x_tiles_per_shard
        
        return y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx
    
    def get_raw_data(self, y_shard_idx: int, x_shard_idx: int, y_tile_in_shard_idx: int, x_tile_in_shard_idx: int) -> torch.Tensor:
        src_ptr, row_size, row_num, src_row_stride, _, _ = self.get_tile_ptr_read_args(y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx)
        data = self.device.mem_get_data(src_ptr, size=src_row_stride * row_num, dtype=torch.uint8)
        data = data.reshape(row_num, src_row_stride)[:, :row_size].flatten().view(self._dtype)
        
        tile_raw_data = torch.zeros((self._tile_y, self._tile_x), dtype=self._dtype)
        tile_raw_data[:row_num, :row_size // self._dtype.itemsize] = data.reshape(row_num, row_size // self._dtype.itemsize)
        
        return tile_raw_data
    
    @property
    def shape(self) -> Sequence[int]:
        return self._shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def numel(self) -> int:
        return functools.reduce(lambda x, y: x * y, self._shape, 1)
    
    @property
    def layout_shape(self) -> tuple[int, int]:
        return (self._layout_y, self._layout_x)
    
    @property
    def shard_grid(self) -> tuple[int, int]:
        return (self._n_y_shards, self._n_x_shards)
    
    @property
    def n_outer_shards(self) -> int:
        return self._n_outer_shards  # number of shards in the leading dimensions -> n_y_shards // n_outer_shards = (# of actual height shards)
    
    @property
    def tile_grid(self) -> tuple[int, int]:
        return (self._n_y_tiles_per_shard * self._n_y_shards, self._n_x_tiles_per_shard * self._n_x_shards)
    
    @property
    def tile_grid_per_shard(self) -> tuple[int, int]:
        return (self._n_y_tiles_per_shard, self._n_x_tiles_per_shard)
    
    @property
    def shard_shape(self) -> tuple[int, int]:
        return (self._shard_y, self._shard_x)
    
    @property
    def tile_shape(self) -> tuple[int, int]:
        return (self._tile_y, self._tile_x)
    
    @property
    def total_size(self) -> int:
        return self._layout_y * self._layout_x * self._dtype.itemsize
    
    @property
    def shard_size(self) -> int:
        return self._shard_size
    
    @property
    def tile_size(self) -> int:
        return self._tile_y * self._tile_x * self._dtype.itemsize
    
    @property
    def n_tiles(self) -> int:
        return (self._n_y_tiles_per_shard * self._n_y_shards) * (self._n_x_tiles_per_shard * self._n_x_shards)
    
    @property
    def mem_space(self) -> MCA_MemorySpace:
        return self._mem_space
    
    @property
    def device(self) -> MCA_DeviceBase:
        return self._mem_space.device
    
    @property
    def mem_type(self) -> GlobalContextMemType:
        return self._mem_space.mem_type
    
    @property
    def owner_ids(self) -> MCA_CoreGroup:
        return self._mem_space.owner_ids
    
    @property
    def is_allocated(self) -> bool:
        return self._is_allocated
