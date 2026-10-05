import math
import torch

from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshMemoryType, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_MEMCOPY

from neuromta.system.software.api.common import *
from neuromta.system.software.api.utils import *


__all__ = [
    "mesh_tensor",
    "mesh_parameter",
    "mesh_empty_like",
    "mesh_to_local_cache",
    "mesh_to_device_memory",
    
    "mesh_reshape",
    "mesh_view",
    "mesh_flatten",
    "mesh_squeeze",
    "mesh_unsqueeze",
    "mesh_permute",
    "mesh_transpose",
    "mesh_narrow",
    "mesh_slice",
    "mesh_select",
    "mesh_split",
    "mesh_chunk",
    "mesh_expand",
    "mesh_copy",
    "mesh_prefetch",
    "mesh_store",
    "mesh_contiguous",
    "mesh_cat",
    "mesh_stack",
    "mesh_head_split",
    "mesh_head_merge",
    "mesh_embedding",
    
    # "_normalize_shape",
    # "_base_tile_shape",
    # "_normalize_dim",
    # "_normalize_view_shape",
    # "_view_tile_shape",
    # "_create_view",
    # "_broadcast_shape",
    # "_validate_broadcast_to",
]




def mesh_tensor(*shape: int, tile_shape: tuple[int, ...]=None, dtype: torch.dtype=None, tensor_type: MeshTensorType | str=MeshTensorType.INTERMEDIATE, reserved_shape: tuple[int, ...]=None, preferred_mem: MeshMemoryType=None) -> MeshTensorDescriptor:
    context = _require_context("mesh_tensor", require_compiler=True)
    normalized_shape = _normalize_shape(shape)
    selected_tile_shape = context.default_tile_shape if tile_shape is None else ((tile_shape,) if isinstance(tile_shape, int) else tuple(tile_shape))
    selected_tile_shape = selected_tile_shape[-min(len(selected_tile_shape), len(normalized_shape)):]
    tensor = MeshTensorDescriptor(shape=normalized_shape, tile_shape=selected_tile_shape, dtype=context.default_dtype if dtype is None else dtype, tensor_type=tensor_type, reserved_shape=reserved_shape)
    if preferred_mem is not None:
        if not isinstance(preferred_mem, MeshMemoryType):
            raise TypeError(f"Expected MeshMemoryType, got {type(preferred_mem).__name__}.")
        tensor.preferred_mem = preferred_mem
    return tensor


def mesh_parameter(*shape: int, tile_shape: tuple[int, ...]=None, dtype: torch.dtype=None, preferred_mem: MeshMemoryType=MeshMemoryType.DEVICE_MEMORY) -> MeshTensorDescriptor:
    return mesh_tensor(*shape, tile_shape=tile_shape, dtype=dtype, tensor_type=MeshTensorType.WEIGHT, preferred_mem=preferred_mem)


def mesh_empty_like(x: MeshTensorDescriptor, shape: tuple[int, ...]=None, tile_shape: tuple[int, ...]=None, dtype: torch.dtype=None, tensor_type: MeshTensorType | str=MeshTensorType.INTERMEDIATE, preferred_mem: MeshMemoryType=None) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    output_shape = x.shape if shape is None else _normalize_shape((shape,))
    selected_tile_shape = _base_tile_shape(x, len(output_shape)) if tile_shape is None else tile_shape
    return mesh_tensor(*output_shape, tile_shape=selected_tile_shape, dtype=x.dtype if dtype is None else dtype, tensor_type=tensor_type, preferred_mem=preferred_mem)


def mesh_to_local_cache(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _require_tensor(x, "x").to_local_cache()


def mesh_to_device_memory(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _require_tensor(x, "x").to_device_memory()


def mesh_reshape(x: MeshTensorDescriptor, *shape: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    output_shape = _normalize_view_shape(shape, x.get_numel())
    if len(output_shape) == 1:
        full_tile_shape = (x.get_tile_numel(),)
    else:
        tile_tail = x.tile_shape[-2:] if len(x.tile_shape) >= 2 else (1, x.tile_shape[-1])
        full_tile_shape = (1,) * (len(output_shape) - 2) + tile_tail
    output = _create_view(x, output_shape, full_tile_shape, "reshape")
    if output.get_n_tiles() != x.get_n_tiles():
        raise ValueError(f"Reshape from {x.shape} to {output_shape} changes the tile count; materialize a compatible layout first.")
    return output


def mesh_view(x: MeshTensorDescriptor, *shape: int) -> MeshTensorDescriptor:
    return mesh_reshape(x, *shape)


def mesh_flatten(x: MeshTensorDescriptor, start_dim: int=0, end_dim: int=-1) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    start = _normalize_dim(start_dim, len(x.shape))
    end = _normalize_dim(end_dim, len(x.shape))
    if start > end:
        raise ValueError("start_dim must not be greater than end_dim.")
    output_shape = x.shape[:start] + (math.prod(x.shape[start:end + 1]),) + x.shape[end + 1:]
    output_tile_shape = x.tile_shape[:start] + (math.prod(x.tile_shape[start:end + 1]),) + x.tile_shape[end + 1:]
    output = _create_view(x, output_shape, output_tile_shape, "reshape")
    if output.get_n_tiles() != x.get_n_tiles():
        raise ValueError("mesh_flatten cannot preserve the current tile layout.")
    return output


def mesh_squeeze(x: MeshTensorDescriptor, dim: int=None) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if dim is None:
        output_shape = tuple(size for size in x.shape if size != 1)
        if not output_shape:
            raise ValueError("Scalar tensor descriptors are not supported.")
        if output_shape == x.shape:
            return x
        removed_dims = tuple(index for index, size in enumerate(x.shape) if size == 1)
        if any(x.tile_shape[index] != 1 for index in removed_dims):
            raise ValueError("mesh_squeeze cannot remove a tiled dimension.")
        output_tile_shape = tuple(size for index, size in enumerate(x.tile_shape) if index not in removed_dims)
        return _create_view(x, output_shape, output_tile_shape, "reshape")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    if x.shape[normalized_dim] != 1:
        return x
    if x.tile_shape[normalized_dim] != 1:
        raise ValueError("mesh_squeeze cannot remove a tiled dimension.")
    output_shape = x.shape[:normalized_dim] + x.shape[normalized_dim + 1:]
    if not output_shape:
        raise ValueError("Scalar tensor descriptors are not supported.")
    output_tile_shape = x.tile_shape[:normalized_dim] + x.tile_shape[normalized_dim + 1:]
    return _create_view(x, output_shape, output_tile_shape, "reshape")


def mesh_unsqueeze(x: MeshTensorDescriptor, dim: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    normalized_dim = _normalize_dim(dim, len(x.shape), allow_end=True)
    output_shape = x.shape[:normalized_dim] + (1,) + x.shape[normalized_dim:]
    output_tile_shape = x.tile_shape[:normalized_dim] + (1,) + x.tile_shape[normalized_dim:]
    return _create_view(x, output_shape, output_tile_shape, "reshape")


def mesh_permute(x: MeshTensorDescriptor, dims: tuple[int, ...]) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    permutation = tuple(int(dim) for dim in dims)
    if len(permutation) != len(x.shape):
        raise ValueError(f"Expected {len(x.shape)} dimensions, got {permutation}.")
    permutation = tuple(_normalize_dim(dim, len(x.shape)) for dim in permutation)
    if len(set(permutation)) != len(permutation):
        raise ValueError(f"Invalid permutation: {permutation}")
    output_shape = tuple(x.shape[dim] for dim in permutation)
    output_tile_shape = tuple(x.tile_shape[dim] for dim in permutation)
    return _create_view(x, output_shape, output_tile_shape, "permute", (permutation,))


def mesh_transpose(x: MeshTensorDescriptor, dim0: int, dim1: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    first = _normalize_dim(dim0, len(x.shape))
    second = _normalize_dim(dim1, len(x.shape))
    permutation = list(range(len(x.shape)))
    permutation[first], permutation[second] = permutation[second], permutation[first]
    return mesh_permute(x, tuple(permutation))


def mesh_narrow(x: MeshTensorDescriptor, dim: int, start: int, length: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("start and length must be integers.")
    if start < 0:
        start += x.shape[normalized_dim]
    if start < 0 or length <= 0 or start + length > x.shape[normalized_dim]:
        raise ValueError(f"Invalid narrow range start={start}, length={length} for dimension size {x.shape[normalized_dim]}.")
    output_shape = x.shape[:normalized_dim] + (length,) + x.shape[normalized_dim + 1:]
    offsets = tuple(start // x.tile_shape[normalized_dim] if index == normalized_dim else 0 for index in range(len(x.shape)))
    element_offsets = tuple(start if index == normalized_dim else 0 for index in range(len(x.shape)))
    return _create_view(x, output_shape, x.tile_shape, "slice", (offsets, element_offsets))


def mesh_slice(x: MeshTensorDescriptor, dim: int, start: int=0, end: int=None, step: int=1) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    if step != 1:
        raise NotImplementedError("mesh_slice currently supports only step=1.")
    size = x.shape[normalized_dim]
    normalized_start = start + size if start < 0 else start
    normalized_end = size if end is None else end + size if end < 0 else end
    if normalized_start < 0 or normalized_end > size or normalized_start >= normalized_end:
        raise ValueError(f"Invalid slice range [{start}:{end}] for dimension size {size}.")
    return mesh_narrow(x, normalized_dim, normalized_start, normalized_end - normalized_start)


def mesh_select(x: MeshTensorDescriptor, dim: int, index: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    normalized_index = index + x.shape[normalized_dim] if index < 0 else index
    return mesh_squeeze(mesh_narrow(x, normalized_dim, normalized_index, 1), normalized_dim)


def mesh_split(x: MeshTensorDescriptor, split_size_or_sections, dim: int=0) -> tuple[MeshTensorDescriptor, ...]:
    x = _require_tensor(x, "x")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    dimension_size = x.shape[normalized_dim]
    if isinstance(split_size_or_sections, int) and not isinstance(split_size_or_sections, bool):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be positive.")
        sections = [min(split_size_or_sections, dimension_size - start) for start in range(0, dimension_size, split_size_or_sections)]
    else:
        sections = [int(section) for section in split_size_or_sections]
        if not sections or any(section <= 0 for section in sections) or sum(sections) != dimension_size:
            raise ValueError(f"Split sections {sections} do not partition dimension size {dimension_size}.")
    outputs = []
    start = 0
    for section in sections:
        outputs.append(mesh_narrow(x, normalized_dim, start, section))
        start += section
    return tuple(outputs)


def mesh_chunk(x: MeshTensorDescriptor, chunks: int, dim: int=0) -> tuple[MeshTensorDescriptor, ...]:
    x = _require_tensor(x, "x")
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks <= 0:
        raise ValueError("chunks must be a positive integer.")
    normalized_dim = _normalize_dim(dim, len(x.shape))
    return mesh_split(x, math.ceil(x.shape[normalized_dim] / chunks), normalized_dim)


def mesh_expand(x: MeshTensorDescriptor, *shape: int) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    values = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else tuple(shape)
    if len(values) < len(x.shape):
        raise ValueError("Expanded rank cannot be smaller than the input rank.")
    leading = len(values) - len(x.shape)
    aligned_shape = (1,) * leading + x.shape
    output_shape = tuple(source_dim if int(target_dim) == -1 else int(target_dim) for source_dim, target_dim in zip(aligned_shape, values))
    if any(target_dim <= 0 or source_dim not in (1, target_dim) for source_dim, target_dim in zip(aligned_shape, output_shape)):
        raise ValueError(f"Cannot expand shape {x.shape} to {output_shape}.")
    return _create_view(x, output_shape, (1,) * leading + x.tile_shape, "expand")

@_mesh_device_op_method
def mesh_copy(x: MeshTensorDescriptor, preferred_mem: MeshMemoryType=None) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    output = MeshTensorDescriptor(shape=x.shape, tile_shape=x.tile_shape, dtype=x.dtype)
    if preferred_mem is not None:
        if not isinstance(preferred_mem, MeshMemoryType):
            raise TypeError(f"Expected MeshMemoryType, got {type(preferred_mem).__name__}.")
        output.preferred_mem = preferred_mem
    return MESH_KERNEL_MEMCOPY(src=x, dst=output), output


@_mesh_device_op_method
def mesh_prefetch(x: MeshTensorDescriptor) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    return MESH_KERNEL_MEMCOPY(src=x), x


@_mesh_device_op_method
def mesh_store(x: MeshTensorDescriptor, dst: MeshTensorDescriptor) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    dst = _require_tensor(dst, "dst")
    return MESH_KERNEL_MEMCOPY(src=x, dst=dst, range_aware=True), dst


def mesh_contiguous(x: MeshTensorDescriptor, preferred_mem: MeshMemoryType=None) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if preferred_mem is not None and not isinstance(preferred_mem, MeshMemoryType):
        raise TypeError(f"Expected MeshMemoryType, got {type(preferred_mem).__name__}.")
    return mesh_copy(x, preferred_mem=preferred_mem) if x.is_view or preferred_mem is not None and x.preferred_mem != preferred_mem else x


@_mesh_device_op_method
def mesh_cat(tensors: list[MeshTensorDescriptor] | tuple[MeshTensorDescriptor, ...], dim: int=0) -> tuple[list[MeshKernelDescriptor], MeshTensorDescriptor]:
    tensors = tuple(_require_tensor(tensor, f"tensors[{index}]") for index, tensor in enumerate(tensors))
    if not tensors:
        raise ValueError("mesh_cat requires at least one tensor.")
    rank = len(tensors[0].shape)
    normalized_dim = _normalize_dim(dim, rank)
    if any(len(tensor.shape) != rank or tensor.dtype != tensors[0].dtype or tensor.tile_shape != tensors[0].tile_shape for tensor in tensors):
        raise ValueError("mesh_cat requires matching ranks, dtypes, and tile shapes.")
    if any(any(left != right for index, (left, right) in enumerate(zip(tensor.shape, tensors[0].shape)) if index != normalized_dim) for tensor in tensors[1:]):
        raise ValueError("mesh_cat tensor shapes differ outside the concatenation dimension.")
    output_shape = tensors[0].shape[:normalized_dim] + (sum(tensor.shape[normalized_dim] for tensor in tensors),) + tensors[0].shape[normalized_dim + 1:]
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=tensors[0].tile_shape, dtype=tensors[0].dtype)
    kernels = []
    offset = 0
    for tensor in tensors:
        destination = mesh_narrow(output, normalized_dim, offset, tensor.shape[normalized_dim])
        kernels.append(MESH_KERNEL_MEMCOPY(src=tensor, dst=destination, range_aware=True))
        offset += tensor.shape[normalized_dim]
    return kernels, output


def mesh_stack(tensors: list[MeshTensorDescriptor] | tuple[MeshTensorDescriptor, ...], dim: int=0) -> MeshTensorDescriptor:
    tensors = tuple(_require_tensor(tensor, f"tensors[{index}]") for index, tensor in enumerate(tensors))
    if not tensors:
        raise ValueError("mesh_stack requires at least one tensor.")
    normalized_dim = _normalize_dim(dim, len(tensors[0].shape), allow_end=True)
    return mesh_cat(tuple(mesh_unsqueeze(tensor, normalized_dim) for tensor in tensors), dim=normalized_dim)


def mesh_head_split(x: MeshTensorDescriptor, num_heads: int, head_dim: int=None) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer.")
    inferred_head_dim = x.shape[-1] // num_heads if head_dim is None else int(head_dim)
    if inferred_head_dim <= 0 or num_heads * inferred_head_dim != x.shape[-1]:
        raise ValueError(f"Cannot split hidden dimension {x.shape[-1]} into {num_heads} heads of size {inferred_head_dim}.")
    output_shape = x.shape[:-1] + (num_heads, inferred_head_dim)
    output_tile_shape = x.tile_shape[:-1] + (1, x.tile_shape[-1])
    output = _create_view(x, output_shape, output_tile_shape, "reshape")
    if output.get_n_tiles() != x.get_n_tiles():
        raise ValueError("mesh_head_split cannot preserve the current tile layout.")
    return output


def mesh_head_merge(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if len(x.shape) < 2:
        raise ValueError("mesh_head_merge requires at least two dimensions.")
    output_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
    output_tile_shape = x.tile_shape[:-2] + (x.tile_shape[-2] * x.tile_shape[-1],)
    output = _create_view(x, output_shape, output_tile_shape, "reshape")
    if output.get_n_tiles() != x.get_n_tiles():
        raise ValueError("mesh_head_merge cannot preserve the current tile layout.")
    return output


@_mesh_device_op_method
def mesh_embedding(indices: MeshTensorDescriptor, weight: MeshTensorDescriptor) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    indices = _require_tensor(indices, "indices")
    weight = _require_tensor(weight, "weight")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("mesh_embedding indices must use torch.int32 or torch.int64.")
    if len(weight.shape) != 2:
        raise ValueError(f"mesh_embedding requires a rank-two weight, got {weight.shape}.")
    output_shape = indices.shape + (weight.shape[-1],)
    output_tile_shape = (1,) * max(0, len(output_shape) - 2) + (indices.tile_shape[-1], weight.tile_shape[-1])
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=output_tile_shape, dtype=weight.dtype)
    traffic_bytes = output.get_n_tiles() * output.get_tile_size()
    return MESH_KERNEL_MEMCOPY(src=weight, dst=output, metadata_inputs=[indices], range_aware=True, gather=True, src_traffic_bytes=traffic_bytes, dst_traffic_bytes=traffic_bytes), output