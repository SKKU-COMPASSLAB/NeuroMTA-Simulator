import math
import torch

from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshMemoryType, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_MEMCOPY

from neuromta.system.software.api.common import *


__all__ = [
    "_normalize_shape",
    "_base_tile_shape",
    "_normalize_dim",
    "_normalize_view_shape",
    "_view_tile_shape",
    "_create_view",
    "_broadcast_shape",
    "_validate_broadcast_to",
]


def _normalize_shape(shape: tuple) -> tuple[int, ...]:
    values = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else tuple(shape)
    normalized = tuple(int(dim) for dim in values)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError(f"Invalid tensor shape: {normalized}")
    return normalized


def _base_tile_shape(tensor: MeshTensorDescriptor, rank: int) -> tuple[int, ...]:
    return tensor.tile_shape if rank == len(tensor.shape) else tensor.tile_shape[-min(2, rank):]


def _normalize_dim(dim: int, rank: int, allow_end: bool=False) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer.")
    limit = rank + 1 if allow_end else rank
    normalized = dim + limit if dim < 0 else dim
    if normalized < 0 or normalized >= limit:
        raise IndexError(f"Dimension {dim} is out of range for rank {rank}.")
    return normalized


def _normalize_view_shape(shape: tuple, numel: int) -> tuple[int, ...]:
    values = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else tuple(shape)
    normalized = tuple(int(dim) for dim in values)
    if not normalized or sum(dim == -1 for dim in normalized) > 1 or any(dim == 0 or dim < -1 for dim in normalized):
        raise ValueError(f"Invalid view shape: {normalized}")
    known_numel = math.prod(dim for dim in normalized if dim != -1)
    if -1 in normalized:
        if known_numel == 0 or numel % known_numel:
            raise ValueError(f"Cannot infer view shape {normalized} for {numel} elements.")
        normalized = tuple(numel // known_numel if dim == -1 else dim for dim in normalized)
    if math.prod(normalized) != numel:
        raise ValueError(f"View shape {normalized} does not preserve {numel} elements.")
    return normalized


def _view_tile_shape(full_tile_shape: tuple[int, ...]) -> tuple[int, ...]:
    if sum(dim != 1 for dim in full_tile_shape) > 2:
        raise ValueError(f"View tile shape {full_tile_shape} cannot be represented by the current tiled tensor descriptor.")
    return full_tile_shape


def _create_view(parent: MeshTensorDescriptor, shape: tuple[int, ...], full_tile_shape: tuple[int, ...], view_kind: str, view_params: tuple=()) -> MeshTensorDescriptor:
    output = MeshTensorDescriptor(shape=shape, tile_shape=_view_tile_shape(full_tile_shape), dtype=parent.dtype, tensor_type=parent.tensor_type)
    return output.bind_view(parent, view_kind, view_params)


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        raise ValueError("At least one tensor shape is required for broadcasting.")
    rank = max(len(shape) for shape in shapes)
    result = []
    for offset in range(1, rank + 1):
        dims = {shape[-offset] for shape in shapes if len(shape) >= offset and shape[-offset] != 1}
        if len(dims) > 1:
            raise ValueError(f"Tensor shapes are not broadcastable: {shapes}")
        result.append(next(iter(dims)) if dims else 1)
    return tuple(reversed(result))


def _validate_broadcast_to(source_shape: tuple[int, ...], target_shape: tuple[int, ...], name: str):
    if _broadcast_shape(source_shape, target_shape) != target_shape:
        raise ValueError(f"{name} shape {source_shape} cannot broadcast to {target_shape}.")
