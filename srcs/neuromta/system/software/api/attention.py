import enum

from neuromta.system.software.api.common import _mesh_device_op_method, _require_tensor
from neuromta.system.software.api.tensor import mesh_narrow
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import MESH_KERNEL_ELEMENTWISE, MESH_KERNEL_LINEAR, MESH_KERNEL_REDUCTION, MESH_KERNEL_SDPA


__all__ = [
    "mesh_sdpa",
    "set_attention_type",
    "get_attention_type",
]


class _AttentionType(enum.Enum):
    SEQUENTIAL = "sequential"
    FUSED = "fused"

_global_neuromta_api_attention = _AttentionType.FUSED

def set_attention_type(attention_type: str | _AttentionType):
    global _global_neuromta_api_attention
    
    if isinstance(attention_type, str):
        try:
            attention_type = _AttentionType(attention_type.strip().lower())
        except ValueError:
            raise ValueError(f"Invalid attention type: {attention_type}")
        
    _global_neuromta_api_attention = attention_type
    
def get_attention_type() -> _AttentionType:
    global _global_neuromta_api_attention
    return _global_neuromta_api_attention


@_mesh_device_op_method
def mesh_sdpa(q: MeshTensorDescriptor, k: MeshTensorDescriptor, v: MeshTensorDescriptor, mask: MeshTensorDescriptor=None, is_causal: bool=False, scale: float=None, q_chunk_size: int=None, kv_chunk_size: int=None, max_cores_per_head: int=16) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    if get_attention_type() == _AttentionType.SEQUENTIAL:
        q = _require_tensor(q, "q")
        k = _require_tensor(k, "k")
        v = _require_tensor(v, "v")
        mask = None if mask is None else _require_tensor(mask, "mask")
        output = MeshTensorDescriptor(shape=q.shape, tile_shape=q.tile_shape, dtype=q.dtype)
        head_dim = q.shape[-1]
        scale = (head_dim**-0.5) if scale is None else float(scale)
        group_size = q.shape[1] // k.shape[1]
        kernels = []

        for head in range(k.shape[1] if group_size > 1 else 1):
            query = mesh_narrow(q, 1, head * group_size, group_size) if group_size > 1 else q
            key = mesh_narrow(k, 1, head, 1) if group_size > 1 else k
            value = mesh_narrow(v, 1, head, 1) if group_size > 1 else v
            result = mesh_narrow(output, 1, head * group_size, group_size) if group_size > 1 else output
            score_shape = query.shape[:-1] + (key.shape[-2],)
            score_tile = query.tile_shape[:-1] + (key.tile_shape[-2],)
            scores = MeshTensorDescriptor(shape=score_shape, tile_shape=score_tile, dtype=q.dtype)
            kernels.append(MESH_KERNEL_LINEAR(ifm=query, wgt=key, ofm=scores, transpose_wgt=False))

            group_mask = mask
            if mask is not None and group_size > 1 and len(mask.shape) >= 3 and mask.shape[-3] == q.shape[1]:
                group_mask = mesh_narrow(mask, len(mask.shape) - 3, head * group_size, group_size)
            if scale != 1 or group_mask is not None or is_causal:
                prepared = MeshTensorDescriptor(shape=score_shape, tile_shape=score_tile, dtype=q.dtype)
                inputs = [scores] + ([] if group_mask is None else [group_mask])
                kernels.append(MESH_KERNEL_ELEMENTWISE(inputs, prepared, ops_per_element=1 + int(group_mask is not None) + int(is_causal)).set_kwargs(operation="attention_scores", scale=scale, is_causal=is_causal))
                scores = prepared

            reduction_shape = score_shape[:-1] + (1,)
            reduction_tile = score_tile[:-1] + (1,)
            maximum = MeshTensorDescriptor(shape=reduction_shape, tile_shape=reduction_tile, dtype=q.dtype)
            kernels.append(MESH_KERNEL_REDUCTION(scores, maximum).set_kwargs(operation="max"))
            exponentials = MeshTensorDescriptor(shape=score_shape, tile_shape=score_tile, dtype=q.dtype)
            kernels.append(MESH_KERNEL_ELEMENTWISE([scores, maximum], exponentials, ops_per_element=9).set_kwargs(operation="exp_sub"))
            denominator = MeshTensorDescriptor(shape=reduction_shape, tile_shape=reduction_tile, dtype=q.dtype)
            kernels.append(MESH_KERNEL_REDUCTION(exponentials, denominator).set_kwargs(operation="sum"))
            probabilities = MeshTensorDescriptor(shape=score_shape, tile_shape=score_tile, dtype=q.dtype)
            kernels.append(MESH_KERNEL_ELEMENTWISE([exponentials, denominator], probabilities, ops_per_element=1).set_kwargs(operation="divide"))
            kernels.append(MESH_KERNEL_LINEAR(ifm=probabilities, wgt=value, ofm=result, transpose_wgt=True))

        return kernels, output
    elif get_attention_type() == _AttentionType.FUSED:
        q = _require_tensor(q, "q")
        k = _require_tensor(k, "k")
        v = _require_tensor(v, "v")
        mask = None if mask is None else _require_tensor(mask, "mask")
        output = MeshTensorDescriptor(shape=q.shape, tile_shape=q.tile_shape, dtype=q.dtype)
        return MESH_KERNEL_SDPA(q=q, k=k, v=v, ofm=output, mask=mask, is_causal=is_causal, scale=scale, q_chunk_size=q_chunk_size, kv_chunk_size=kv_chunk_size, max_cores_per_head=max_cores_per_head), output
    else:
        raise ValueError(f"Invalid attention type: {get_attention_type()}")