import math
import numpy as np
import enum
import torch
import functools
from typing import Any

from neuromta.framework import *
from neuromta.component.context.mem_context import MemoryContext
from neuromta.component.context.global_context import GlobalContext
from neuromta.component.context.compute_tile_context import ComputeTileContext
from neuromta.system.hardware.mesh_accelerator import MeshAccelerator


__all__ = [
    "MeshTensorDescriptor",
    "MeshTensorType",
    "MeshKernelType",
    "MeshKernelDescriptor",
    "MeshDeviceDescriptor",
    "MeshMemoryType",
    "MeshMemoryBankDescriptor",
    "MeshMemoryDescriptor",
]


class MeshTensorType(enum.Enum):
    WEIGHT = "WEIGHT"
    INTERMEDIATE = "INTERMEDIATE"

class MeshTensorTileLayout:
    def __init__(self, layout: str="ROW_MAJOR"):
        if not MeshTensorTileLayout.is_valid(layout):
            raise ValueError(f"Invalid tile layout: {layout}")

        self.layout = layout

    @classmethod
    def BLOCKED(cls, x: int, y: int) -> 'MeshTensorTileLayout':
        return MeshTensorTileLayout(f"BLOCKED_{x}_{y}")

    @classmethod
    def ROW_MAJOR(cls) -> 'MeshTensorTileLayout':
        return MeshTensorTileLayout("ROW_MAJOR")

    @classmethod
    def COL_MAJOR(cls) -> 'MeshTensorTileLayout':
        return MeshTensorTileLayout("COL_MAJOR")

    @staticmethod
    def is_valid(layout: str) -> bool:
        if layout in ("ROW_MAJOR", "COL_MAJOR"):
            return True
        if layout.startswith("BLOCKED_"):
            parts = layout.split("_")
            if len(parts) == 3 and all(part.isdigit() and int(part) > 0 for part in parts[1:]):
                return True
        return False

    def is_row_major(self) -> bool:
        return self.layout == "ROW_MAJOR"

    def is_col_major(self) -> bool:
        return self.layout == "COL_MAJOR"

    def is_blocked(self) -> bool:
        return self.layout.startswith("BLOCKED_")

    def parse_blocked(self) -> tuple[int, int]:
        if not self.is_blocked():
            raise ValueError(f"Tile layout {self.layout} is not a blocked layout.")
        parts = self.layout.split("_")
        return int(parts[1]), int(parts[2])

class MeshKernelType(enum.Enum):
    LINEAR = "linear"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    MEMCOPY = "memcopy"
    CONV2D = "conv2d"
    SDPA = "sdpa"

class MeshMemoryType(enum.Enum):
    DEVICE_MEMORY = "DEVICE_MEMORY"
    LOCAL_CACHE = "LOCAL_CACHE"


class MeshTensorDescriptor:
    def __init__(
        self,
        shape: tuple[int, ...],
        tile_shape: tuple[int, ...],
        dtype: torch.dtype,
        tensor_type: MeshTensorType | str=MeshTensorType.INTERMEDIATE,
        reserved_shape: tuple[int, ...] | None=None
    ):
        self.shape = (1, shape) if isinstance(shape, int) else tuple(int(dim) for dim in shape)
        self.tile_shape = tile_shape
        self.dtype = dtype

        if isinstance(self.tile_shape, int):
            self.tile_shape = (self.tile_shape,)
        elif len(self.tile_shape) > len(self.shape):
            raise ValueError(f"Tile shape rank {len(self.tile_shape)} exceeds tensor rank {len(self.shape)}.")
        elif len(self.tile_shape) > 2 and sum(int(dim) != 1 for dim in self.tile_shape) > 2:
            raise ValueError(f"Invalid tile_shape: {self.tile_shape}. At most two tiled dimensions are supported.")
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        if len(self.tile_shape) < len(self.shape):
            self.tile_shape = (1,) * (len(self.shape) - len(self.tile_shape)) + self.tile_shape
        self.tile_shape = tuple(int(dim) for dim in self.tile_shape)
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid tensor shape: {self.shape}")
        if any(dim <= 0 for dim in self.tile_shape):
            raise ValueError(f"Invalid tile shape: {self.tile_shape}")
        if isinstance(tensor_type, str):
            try:
                tensor_type = MeshTensorType[tensor_type.upper()]
            except KeyError as exc:
                raise ValueError(f"Invalid tensor type: {tensor_type}") from exc
        if not isinstance(tensor_type, MeshTensorType):
            raise TypeError(f"Expected MeshTensorType, got {type(tensor_type).__name__}")

        self.tensor_type = tensor_type
        self.reserved_shape = self.shape if reserved_shape is None else tuple(int(dim) for dim in reserved_shape)

        if len(self.reserved_shape) != len(self.shape) or any(dim <= 0 for dim in self.reserved_shape):
            raise ValueError(f"Invalid reserved shape: {self.reserved_shape}")
        if any(reserved_dim < dim for dim, reserved_dim in zip(self.shape, self.reserved_shape)):
            raise ValueError(f"Reserved shape {self.reserved_shape} cannot be smaller than tensor shape {self.shape}.")

        self.preferred_mem: MeshMemoryType = MeshMemoryType.DEVICE_MEMORY if (self.tensor_type == MeshTensorType.WEIGHT) else MeshMemoryType.LOCAL_CACHE
        self.persistent_state_id: str | None = None
        self.fusion_barrier = False
        self._storage_desc = self
        self._view_parent: MeshTensorDescriptor | None = None
        self._view_kind: str | None = None
        self._view_params: tuple | None = None
        self.storage_offset = 0
        self.element_offset = (0,) * len(self.shape)
        self.strides = self._row_major_strides(self.reserved_tile_grid_shape)

    def get_tile_coords(self, tile_layout: MeshTensorTileLayout | str=None) -> list[tuple[int, ...]]:
        if tile_layout is None:
            tile_layout = MeshTensorTileLayout.ROW_MAJOR()

        tile_layout = MeshTensorTileLayout(tile_layout) if isinstance(tile_layout, str) else tile_layout
        if not isinstance(tile_layout, MeshTensorTileLayout):
            raise TypeError(f"Expected MeshTensorTileLayout, got {type(tile_layout).__name__}")

        grid_shape = self.tile_grid_shape
        columns = list(np.ndindex(*grid_shape[:-1]))

        if tile_layout.is_row_major():
            return list(np.ndindex(*grid_shape))
        elif tile_layout.is_col_major():
            return [column + (row,) for row in range(grid_shape[-1]) for column in columns]
        elif tile_layout.is_blocked():
            block_width, block_height = tile_layout.parse_blocked()
            return [
                columns[column] + (row,)
                for column_start in range(0, len(columns), block_height)
                for row_start in range(0, grid_shape[-1], block_width)
                for column in range(column_start, min(column_start + block_height, len(columns)))
                for row in range(row_start, min(row_start + block_width, grid_shape[-1]))
            ]
        else:
            raise Exception(f"Invalid tile layout received: {tile_layout}")

    @staticmethod
    def _row_major_strides(grid_shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = []
        stride = 1
        for dim in reversed(grid_shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))

    @property
    def tile_grid_shape(self) -> tuple[int, ...]:
        if getattr(self, "_view_kind", None) == "slice" and len(self._view_params) > 1:
            element_offsets = self._view_params[1]
            return tuple(math.ceil((offset % tile_dim + dim) / tile_dim) for offset, dim, tile_dim in zip(element_offsets, self.shape, self.tile_shape))
        return tuple(math.ceil(dim / tile_dim) for dim, tile_dim in zip(self.shape, self.tile_shape))

    @property
    def reserved_tile_grid_shape(self) -> tuple[int, ...]:
        return tuple(math.ceil(dim / tile_dim) for dim, tile_dim in zip(self.reserved_shape, self.tile_shape))

    @property
    def storage_desc(self) -> 'MeshTensorDescriptor':
        return self._storage_desc

    @property
    def storage_id(self) -> int:
        return id(self._storage_desc)

    @property
    def is_view(self) -> bool:
        return self._view_parent is not None

    @property
    def is_persistent(self) -> bool:
        return self.storage_desc.persistent_state_id is not None

    def bind_view(self, parent: 'MeshTensorDescriptor', view_kind: str, view_params: tuple=()) -> 'MeshTensorDescriptor':
        if not isinstance(parent, MeshTensorDescriptor):
            raise TypeError(f"Expected MeshTensorDescriptor, got {type(parent).__name__}")
        if parent is self:
            raise ValueError("A tensor descriptor cannot be its own view parent.")
        if self.dtype != parent.dtype or self.get_tile_size() != parent.get_tile_size():
            raise ValueError("A tensor view must preserve dtype and tile size.")
        if view_kind not in ("reshape", "permute", "slice", "expand"):
            raise ValueError(f"Invalid view kind: {view_kind}")
        self._storage_desc = parent.storage_desc
        self._view_parent = parent
        self._view_kind = view_kind
        self._view_params = tuple(view_params)
        self.tensor_type = parent.tensor_type
        self.preferred_mem = parent.preferred_mem
        self.persistent_state_id = parent.storage_desc.persistent_state_id
        if view_kind == "reshape":
            self.storage_offset = parent.storage_offset
            self.element_offset = (0,) * len(self.shape)
            self.strides = self._row_major_strides(self.tile_grid_shape)
        elif view_kind == "permute":
            permutation = self._view_params[0]
            self.storage_offset = parent.storage_offset
            self.element_offset = tuple(parent.element_offset[index] for index in permutation)
            self.strides = tuple(parent.strides[index] for index in permutation)
        elif view_kind == "slice":
            offsets = self._view_params[0]
            element_offsets = self._view_params[1] if len(self._view_params) > 1 else tuple(offset * tile_dim for offset, tile_dim in zip(offsets, parent.tile_shape))
            self.storage_offset = parent.storage_offset + sum(offset * stride for offset, stride in zip(offsets, parent.strides))
            self.element_offset = element_offsets
            self.strides = parent.strides
        else:
            leading = len(self.shape) - len(parent.shape)
            aligned_strides = (0,) * leading + parent.strides
            aligned_parent_grid = (1,) * leading + parent.tile_grid_shape
            self.storage_offset = parent.storage_offset
            self.element_offset = (0,) * leading + parent.element_offset
            self.strides = tuple(0 if parent_dim == 1 and view_dim > 1 else stride for parent_dim, view_dim, stride in zip(aligned_parent_grid, self.tile_grid_shape, aligned_strides))
        return self

    def map_tile_coord_to_storage(self, tile_coord: tuple[int, ...]) -> tuple[int, ...]:
        coord = tuple(int(value) for value in tile_coord)
        if len(coord) != len(self.tile_grid_shape) or any(value < 0 or value >= dim for value, dim in zip(coord, self.tile_grid_shape)):
            raise IndexError(f"Tile coordinate {coord} is outside grid {self.tile_grid_shape}.")
        if self._view_parent is None:
            return coord
        parent_grid = self._view_parent.tile_grid_shape
        if self._view_kind == "reshape":
            linear_index = sum(value * stride for value, stride in zip(coord, self._row_major_strides(self.tile_grid_shape)))
            parent_coord = []
            for stride, dim in zip(self._row_major_strides(parent_grid), parent_grid):
                parent_coord.append((linear_index // stride) % dim)
            return self._view_parent.map_tile_coord_to_storage(tuple(parent_coord))
        if self._view_kind == "permute":
            permutation = self._view_params[0]
            parent_coord = [0] * len(coord)
            for output_dim, parent_dim in enumerate(permutation):
                parent_coord[parent_dim] = coord[output_dim]
            return self._view_parent.map_tile_coord_to_storage(tuple(parent_coord))
        if self._view_kind == "slice":
            element_offsets = self._view_params[1] if len(self._view_params) > 1 else tuple(offset * tile_dim for offset, tile_dim in zip(self._view_params[0], self._view_parent.tile_shape))
            parent_coord = tuple((offset + value * tile_dim) // parent_tile_dim for offset, value, tile_dim, parent_tile_dim in zip(element_offsets, coord, self.tile_shape, self._view_parent.tile_shape))
            return self._view_parent.map_tile_coord_to_storage(parent_coord)
        leading = len(coord) - len(parent_grid)
        aligned_coord = coord[leading:]
        return self._view_parent.map_tile_coord_to_storage(tuple(0 if dim == 1 else value for value, dim in zip(aligned_coord, parent_grid)))

    def get_size(self) -> int:
        return self.dtype.itemsize * self.get_numel()

    def get_tile_size(self) -> int:
        return self.dtype.itemsize * self.get_tile_numel()

    def get_numel(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.shape, 1)

    def get_tile_numel(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.tile_shape, 1)

    def get_n_tiles(self) -> int:
        return math.prod(self.tile_grid_shape)

    def get_reserved_size(self) -> int:
        return self.dtype.itemsize * self.get_reserved_numel()

    def get_reserved_numel(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.reserved_shape, 1)

    def get_reserved_n_tiles(self) -> int:
        return functools.reduce(lambda x, y: x * y, (math.ceil(s / t) for s, t in zip(self.reserved_shape, self.tile_shape)), 1)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, tile_shape: tuple[int, ...], tensor_type: MeshTensorType | str=MeshTensorType.INTERMEDIATE, reserved_shape: tuple[int, ...] | None=None) -> 'MeshTensorDescriptor':
        return cls(shape=tensor.shape, tile_shape=tile_shape, dtype=tensor.dtype, tensor_type=tensor_type, reserved_shape=reserved_shape)

    def to_local_cache(self):
        self.preferred_mem = MeshMemoryType.LOCAL_CACHE
        self.storage_desc.preferred_mem = MeshMemoryType.LOCAL_CACHE
        return self

    def to_device_memory(self):
        self.preferred_mem = MeshMemoryType.DEVICE_MEMORY
        self.storage_desc.preferred_mem = MeshMemoryType.DEVICE_MEMORY
        return self

    def as_weight(self):
        self.tensor_type = MeshTensorType.WEIGHT
        self.preferred_mem = MeshMemoryType.DEVICE_MEMORY
        self.storage_desc.tensor_type = MeshTensorType.WEIGHT
        self.storage_desc.preferred_mem = MeshMemoryType.DEVICE_MEMORY
        return self

    def as_intermediate(self):
        self.tensor_type = MeshTensorType.INTERMEDIATE
        self.preferred_mem = MeshMemoryType.LOCAL_CACHE
        self.storage_desc.tensor_type = MeshTensorType.INTERMEDIATE
        self.storage_desc.preferred_mem = MeshMemoryType.LOCAL_CACHE
        return self

    def as_persistent(self, state_id: str):
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string.")
        self.tensor_type = MeshTensorType.INTERMEDIATE
        self.storage_desc.tensor_type = MeshTensorType.INTERMEDIATE
        self.persistent_state_id = state_id
        self.storage_desc.persistent_state_id = state_id
        return self

    def __repr__(self) -> str:
        return f"MeshTensorDescriptor(shape={self.shape}, tile_shape={self.tile_shape}, dtype={self.dtype}, tensor_type={self.tensor_type.value}, reserved_shape={self.reserved_shape}, is_view={self.is_view}, persistent_state_id={self.persistent_state_id})"


class MeshKernelDescriptor:
    def __init__(
        self,
        kernel_type: MeshKernelType,
        input_tensors: list[MeshTensorDescriptor],
        output_tensors: list[MeshTensorDescriptor],
        kernel_kwargs: dict | None = None,
        enable_fusion: bool = None,
    ):
        self.kernel_type = kernel_type
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.base_input_count = len(input_tensors)
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        self.fused_operations: tuple[MeshKernelDescriptor, ...] = ()

        self.enable_fusion = (kernel_type == MeshKernelType.ELEMENTWISE) if enable_fusion is None else enable_fusion

    def set_kwargs(self, **kwargs):
        self.kernel_kwargs.update(kwargs)
        return self

    def get_kwargs(self, *keys: str, default: Any=None):
        if len(keys) == 1:
            return self.kernel_kwargs.get(keys[0], default)
        else:
            return [self.kernel_kwargs.get(key, default) for key in keys]

    def get_required_kwargs(self, *keys: str):
        missing_keys = [key for key in keys if key not in self.kernel_kwargs]
        if missing_keys:
            raise KeyError(f"Missing required kernel kwargs: {', '.join(missing_keys)}")
        return self.get_kwargs(*keys)

    def __repr__(self):
        return f"MeshKernelDescriptor(kernel_type={self.kernel_type.name})"

class MeshMemoryBankDescriptor:
    def __init__(
        self,
        mem_type: MeshMemoryType,
        owner_id: int,
        addr: int,
        size: int,
    ):
        self.mem_type = mem_type
        self.owner_id = owner_id
        self.addr = addr
        self.size = size

    @classmethod
    def DEVICE_MEMORY(cls, addr: int, size: int) -> 'MeshMemoryBankDescriptor':
        return cls(MeshMemoryType.DEVICE_MEMORY, 0, addr, size)

    @classmethod
    def LOCAL_CACHE(cls, tile_id: int, addr: int, size: int) -> 'MeshMemoryBankDescriptor':
        return cls(MeshMemoryType.LOCAL_CACHE, tile_id, addr, size)

class MeshMemoryDescriptor:
    def __init__(
        self,
        mem_type: MeshMemoryType,
        banks: list[MeshMemoryBankDescriptor],
    ):
        self.mem_type = mem_type
        self.banks = banks

    @classmethod
    def from_main_mem_context(cls, mem_context: MemoryContext) -> 'MeshMemoryDescriptor':
        return cls(mem_type=MeshMemoryType.DEVICE_MEMORY, banks=[
            MeshMemoryBankDescriptor(
                mem_type=MeshMemoryType.DEVICE_MEMORY,
                owner_id=mem_context.get_dma_id_with_instance_id(inst_id),
                addr=mem_context.get_instance_addr(inst_id, 0),
                size=mem_context.channel_size_per_instance,
            )
            for inst_id in range(mem_context.n_instance)
        ])

    @classmethod
    def from_local_cache_context(cls, global_context: GlobalContext, compute_tile_context: ComputeTileContext) -> 'MeshMemoryDescriptor':
        return cls(mem_type=MeshMemoryType.LOCAL_CACHE, banks=[
            MeshMemoryBankDescriptor(
                mem_type=MeshMemoryType.LOCAL_CACHE,
                owner_id=tile_id,
                addr=0,  # Local cache address is relative to the tile
                size=compute_tile_context.config.local_cache,
            )
            for tile_id in global_context.config.ccg_tile_ids
        ])


class MeshDeviceDescriptor:
    def __init__(self, device: MeshAccelerator):
        self.device = device
        self.ccg_tile_mesh = device.get_ccg_tile_mesh()
        self.dma_tile_mesh = device.get_dma_tile_mesh()

        compute_config = device.ccg_context.config
        memory_config = device.mem_context.config
        self.compute_throughput_per_core = float(compute_config.tops) * 1.0e12
        self.compute_tile_shape = (
            int(compute_config.tile_y_dim),
            int(compute_config.tile_x_dim),
        )

        if memory_config.dramsim3_enable:
            total_memory_bandwidth = memory_config.dramsim3_config.peak_bandwidth()
            self.memory_bandwidth_per_bank = total_memory_bandwidth / memory_config.n_instance
        else:
            self.memory_bandwidth_per_bank = (
                memory_config.lightweight_channel_bandwidth_bytes_per_cycle
                * memory_config.n_channel_per_instance
                * compute_config.processor_clock_freq
            )
