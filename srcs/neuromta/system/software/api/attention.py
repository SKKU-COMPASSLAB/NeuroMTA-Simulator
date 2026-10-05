from neuromta.system.software.api.common import _mesh_device_op_method, _require_tensor
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import MESH_KERNEL_SDPA


__all__ = [
    "mesh_sdpa",
]


@_mesh_device_op_method
def mesh_sdpa(q: MeshTensorDescriptor, k: MeshTensorDescriptor, v: MeshTensorDescriptor, mask: MeshTensorDescriptor=None, is_causal: bool=False, scale: float=None, q_chunk_size: int=None, kv_chunk_size: int=None, max_cores_per_head: int=16) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    q = _require_tensor(q, "q")
    k = _require_tensor(k, "k")
    v = _require_tensor(v, "v")
    mask = None if mask is None else _require_tensor(mask, "mask")
    output = MeshTensorDescriptor(shape=q.shape, tile_shape=q.tile_shape, dtype=q.dtype)
    return MESH_KERNEL_SDPA(q=q, k=k, v=v, ofm=output, mask=mask, is_causal=is_causal, scale=scale, q_chunk_size=q_chunk_size, kv_chunk_size=kv_chunk_size, max_cores_per_head=max_cores_per_head), output
