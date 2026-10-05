from numbers import Real
import torch

from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import *

from neuromta.system.software.api.common import *
from neuromta.system.software.api.utils import *
from neuromta.system.software.api.tensor import *


__all__ = [
    "mesh_linear",
    "mesh_matmul",
    "mesh_bmm",
    "mesh_add",
    "mesh_sub",
    "mesh_mul",
    "mesh_div",
    "mesh_neg",
    "mesh_abs",
    "mesh_square",
    "mesh_pow",
    "mesh_exp",
    "mesh_sqrt",
    "mesh_rsqrt",
    "mesh_scale",
    "mesh_where",
    "mesh_masked_fill",
    "mesh_clamp",
    "mesh_relu",
    "mesh_silu",
    "mesh_gelu",
    "mesh_quick_gelu",
    "mesh_sigmoid",
    "mesh_swiglu",
    "mesh_sum",
    "mesh_max",
    "mesh_mean",
    "mesh_argmax",
    "mesh_softmax",
    "mesh_rms_norm",
    "mesh_layer_norm",
    "mesh_rope",
]


@_mesh_device_op_method
def mesh_linear(x: MeshTensorDescriptor, weight: MeshTensorDescriptor, bias: MeshTensorDescriptor=None) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    weight = _require_tensor(weight, "weight")
    bias = None if bias is None else _require_tensor(bias, "bias")
    if len(x.shape) < 2:
        raise ValueError("mesh_linear requires an input rank of at least two.")
    if len(weight.shape) != 2:
        raise ValueError(f"mesh_linear requires a rank-two weight, got {weight.shape}.")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"Linear reduction dimensions do not match: x={x.shape}, weight={weight.shape}.")
    output_shape = x.shape[:-1] + (weight.shape[-2],)
    if bias is not None:
        _validate_broadcast_to(bias.shape, output_shape, "bias")
    if weight.dtype != x.dtype or bias is not None and bias.dtype != x.dtype:
        raise ValueError("mesh_linear requires matching input, weight, and bias dtypes.")
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=_base_tile_shape(x, len(output_shape)), dtype=x.dtype)
    return MESH_KERNEL_LINEAR(ifm=x, wgt=weight, ofm=output, bias=bias), output


@_mesh_device_op_method
def _mesh_matmul_leaf(a: MeshTensorDescriptor, b: MeshTensorDescriptor, transpose_a: bool, transpose_b: bool) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("mesh_matmul requires tensor ranks of at least two.")
    a_output_dim = a.shape[-1] if transpose_a else a.shape[-2]
    a_reduction_dim = a.shape[-2] if transpose_a else a.shape[-1]
    reduction_dim = b.shape[-1] if transpose_b else b.shape[-2]
    output_dim = b.shape[-2] if transpose_b else b.shape[-1]
    if a_reduction_dim != reduction_dim:
        raise ValueError(f"Matmul reduction dimensions do not match: a={a.shape}, b={b.shape}.")
    batch_shape = _broadcast_shape(a.shape[:-2], b.shape[:-2])
    output_shape = batch_shape + (a_output_dim, output_dim)
    if a.dtype != b.dtype:
        raise ValueError("mesh_matmul requires matching input dtypes.")
    output_tile_shape = (a.tile_shape[-1] if transpose_a else a.tile_shape[-2], b.tile_shape[-2] if transpose_b else b.tile_shape[-1])
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=output_tile_shape, dtype=a.dtype)
    return MESH_KERNEL_LINEAR(ifm=a, wgt=b, ofm=output, transpose_wgt=not transpose_b, transpose_ifm=transpose_a), output


def mesh_matmul(a: MeshTensorDescriptor, b: MeshTensorDescriptor, transpose_a: bool=False, transpose_b: bool=False) -> MeshTensorDescriptor:
    return _mesh_matmul_leaf(a, b, transpose_a, transpose_b)


def mesh_bmm(a: MeshTensorDescriptor, b: MeshTensorDescriptor, transpose_a: bool=False, transpose_b: bool=False) -> MeshTensorDescriptor:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError(f"mesh_bmm requires rank-three tensors, got {a.shape} and {b.shape}.")
    return mesh_matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


@_mesh_device_op_method
def _mesh_elementwise_leaf(inputs: list[MeshTensorDescriptor], output_shape: tuple[int, ...], ops_per_element: float, output_dtype: torch.dtype=None, enable_fusion: bool=True) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    tensors = [_require_tensor(tensor, f"inputs[{index}]") for index, tensor in enumerate(inputs)]
    if not tensors:
        raise ValueError("Elementwise operators require at least one tensor input.")
    for index, tensor in enumerate(tensors):
        _validate_broadcast_to(tensor.shape, output_shape, f"inputs[{index}]")
    reference = max(tensors, key=lambda tensor: len(tensor.shape))
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=_base_tile_shape(reference, len(output_shape)), dtype=tensors[0].dtype if output_dtype is None else output_dtype)
    return MESH_KERNEL_ELEMENTWISE(tensors, ofm=output, ops_per_element=ops_per_element, enable_fusion=enable_fusion), output


def _mesh_unary(x: MeshTensorDescriptor, ops_per_element: float) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    return _mesh_elementwise_leaf([x], x.shape, ops_per_element)


def _mesh_binary(x, y, ops_per_element: float) -> MeshTensorDescriptor:
    if any(not isinstance(operand, (MeshTensorDescriptor, Real)) for operand in (x, y)):
        raise TypeError("Binary operands must be MeshTensorDescriptor or real scalar values.")
    tensors = [operand for operand in (x, y) if isinstance(operand, MeshTensorDescriptor)]
    if not tensors:
        raise TypeError("At least one binary operand must be a MeshTensorDescriptor.")
    if any(tensor.dtype != tensors[0].dtype for tensor in tensors):
        raise ValueError("Binary tensor operands must have matching dtypes.")
    return _mesh_elementwise_leaf(tensors, _broadcast_shape(*(tensor.shape for tensor in tensors)), ops_per_element)


def mesh_add(x, y) -> MeshTensorDescriptor:
    return _mesh_binary(x, y, 1.0)


def mesh_sub(x, y) -> MeshTensorDescriptor:
    return _mesh_binary(x, y, 1.0)


def mesh_mul(x, y) -> MeshTensorDescriptor:
    return _mesh_binary(x, y, 1.0)


def mesh_div(x, y) -> MeshTensorDescriptor:
    return _mesh_binary(x, y, 1.0)


def mesh_neg(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 1.0)


def mesh_abs(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 1.0)


def mesh_square(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 1.0)


def mesh_pow(x: MeshTensorDescriptor, exponent: Real) -> MeshTensorDescriptor:
    if not isinstance(exponent, Real):
        raise TypeError("exponent must be a real scalar.")
    return _mesh_unary(x, 2.0)


def mesh_exp(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 8.0)


def mesh_sqrt(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 8.0)


def mesh_rsqrt(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 8.0)


def mesh_scale(x: MeshTensorDescriptor, scale: Real) -> MeshTensorDescriptor:
    if not isinstance(scale, Real):
        raise TypeError("scale must be a real scalar.")
    return _mesh_unary(x, 1.0)


def mesh_where(condition: MeshTensorDescriptor, x, y) -> MeshTensorDescriptor:
    condition = _require_tensor(condition, "condition")
    if any(not isinstance(operand, (MeshTensorDescriptor, Real)) for operand in (x, y)):
        raise TypeError("mesh_where values must be MeshTensorDescriptor or real scalar values.")
    values = [operand for operand in (x, y) if isinstance(operand, MeshTensorDescriptor)]
    if not values:
        raise TypeError("At least one mesh_where value must be a MeshTensorDescriptor.")
    if any(value.dtype != values[0].dtype for value in values):
        raise ValueError("mesh_where tensor values must have matching dtypes.")
    operands = [condition] + values
    return _mesh_elementwise_leaf(operands, _broadcast_shape(*(tensor.shape for tensor in operands)), 1.0, output_dtype=values[0].dtype)


def mesh_masked_fill(x: MeshTensorDescriptor, mask: MeshTensorDescriptor, value: Real) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    mask = _require_tensor(mask, "mask")
    if not isinstance(value, Real):
        raise TypeError("value must be a real scalar.")
    _validate_broadcast_to(mask.shape, x.shape, "mask")
    return _mesh_elementwise_leaf([x, mask], x.shape, 1.0, output_dtype=x.dtype)


def mesh_clamp(x: MeshTensorDescriptor, minimum: Real=None, maximum: Real=None) -> MeshTensorDescriptor:
    if minimum is None and maximum is None:
        raise ValueError("mesh_clamp requires minimum, maximum, or both.")
    if minimum is not None and not isinstance(minimum, Real) or maximum is not None and not isinstance(maximum, Real):
        raise TypeError("mesh_clamp bounds must be real scalar values.")
    return _mesh_unary(x, 2.0 if minimum is not None and maximum is not None else 1.0)


def mesh_relu(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 2.0)


def mesh_silu(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 6.0)


def mesh_gelu(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 10.0)


def mesh_quick_gelu(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 6.0)


def mesh_sigmoid(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return _mesh_unary(x, 4.0)


def mesh_swiglu(gate: MeshTensorDescriptor, up: MeshTensorDescriptor) -> MeshTensorDescriptor:
    return mesh_mul(mesh_silu(gate), up)


@_mesh_device_op_method
def _mesh_reduction_leaf(x: MeshTensorDescriptor, keepdim: bool, ops_per_input_element: float, output_dtype: torch.dtype=None) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    if not isinstance(keepdim, bool):
        raise TypeError("keepdim must be a boolean.")
    if not keepdim and len(x.shape) == 1:
        raise ValueError("Scalar reduction outputs are not supported; use keepdim=True.")
    output_shape = x.shape[:-1] + (1,) if keepdim else x.shape[:-1]
    source_tile_shape = x.tile_shape if keepdim else x.tile_shape[:-1]
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=source_tile_shape[-min(2, len(output_shape)):], dtype=x.dtype if output_dtype is None else output_dtype)
    return MESH_KERNEL_REDUCTION(ifm=x, ofm=output, ops_per_input_element=ops_per_input_element), output


def _validate_reduction_dim(x: MeshTensorDescriptor, dim: int):
    x = _require_tensor(x, "x")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer.")
    if dim not in (-1, len(x.shape) - 1):
        raise NotImplementedError("Phase 1 reduction operators support only the last dimension.")


def mesh_sum(x: MeshTensorDescriptor, dim: int=-1, keepdim: bool=True) -> MeshTensorDescriptor:
    _validate_reduction_dim(x, dim)
    return _mesh_reduction_leaf(x, keepdim, 1.0)


def mesh_max(x: MeshTensorDescriptor, dim: int=-1, keepdim: bool=True) -> MeshTensorDescriptor:
    _validate_reduction_dim(x, dim)
    return _mesh_reduction_leaf(x, keepdim, 1.0)


def mesh_mean(x: MeshTensorDescriptor, dim: int=-1, keepdim: bool=True) -> MeshTensorDescriptor:
    _validate_reduction_dim(x, dim)
    return mesh_scale(_mesh_reduction_leaf(x, keepdim, 1.0), 1.0 / x.shape[-1])


def mesh_argmax(x: MeshTensorDescriptor, dim: int=-1, keepdim: bool=False) -> MeshTensorDescriptor:
    _validate_reduction_dim(x, dim)
    return _mesh_reduction_leaf(x, keepdim, 1.0, output_dtype=torch.int64)


def mesh_softmax(x: MeshTensorDescriptor, dim: int=-1, mask: MeshTensorDescriptor=None, scale: Real=None) -> MeshTensorDescriptor:
    _validate_reduction_dim(x, dim)
    normalized_input = mesh_scale(x, scale) if scale is not None else x
    if mask is not None:
        normalized_input = mesh_masked_fill(normalized_input, mask, float("-inf"))
    maximum = mesh_max(normalized_input, dim=dim, keepdim=True)
    exponentials = mesh_exp(mesh_sub(normalized_input, maximum))
    denominator = mesh_sum(exponentials, dim=dim, keepdim=True)
    return mesh_div(exponentials, denominator)


def mesh_rms_norm(x: MeshTensorDescriptor, weight: MeshTensorDescriptor=None, eps: float=1e-6) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if not isinstance(eps, Real) or eps <= 0:
        raise ValueError("eps must be positive.")
    if weight is not None:
        weight = _require_tensor(weight, "weight")
        _validate_broadcast_to(weight.shape, x.shape, "weight")
    variance = mesh_mean(mesh_square(x), dim=-1, keepdim=True)
    normalized = mesh_mul(x, mesh_rsqrt(mesh_add(variance, eps)))
    return normalized if weight is None else mesh_mul(normalized, weight)


def mesh_layer_norm(x: MeshTensorDescriptor, weight: MeshTensorDescriptor=None, bias: MeshTensorDescriptor=None, eps: float=1e-5) -> MeshTensorDescriptor:
    x = _require_tensor(x, "x")
    if not isinstance(eps, Real) or eps <= 0:
        raise ValueError("eps must be positive.")
    if weight is not None:
        weight = _require_tensor(weight, "weight")
        _validate_broadcast_to(weight.shape, x.shape, "weight")
    if bias is not None:
        bias = _require_tensor(bias, "bias")
        _validate_broadcast_to(bias.shape, x.shape, "bias")
    centered = mesh_sub(x, mesh_mean(x, dim=-1, keepdim=True))
    variance = mesh_mean(mesh_square(centered), dim=-1, keepdim=True)
    normalized = mesh_mul(centered, mesh_rsqrt(mesh_add(variance, eps)))
    normalized = normalized if weight is None else mesh_mul(normalized, weight)
    return normalized if bias is None else mesh_add(normalized, bias)


def mesh_rope(q: MeshTensorDescriptor, k: MeshTensorDescriptor, cos: MeshTensorDescriptor, sin: MeshTensorDescriptor, position_ids: MeshTensorDescriptor=None, rotary_sections: tuple[int, ...]=None) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor]:
    q = _require_tensor(q, "q")
    k = _require_tensor(k, "k")
    cos = _require_tensor(cos, "cos")
    sin = _require_tensor(sin, "sin")
    if q.dtype != k.dtype or cos.dtype != q.dtype or sin.dtype != q.dtype:
        raise ValueError("mesh_rope requires matching q, k, cos, and sin dtypes.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("mesh_rope requires matching q and k head dimensions.")
    _validate_broadcast_to(cos.shape, q.shape, "cos")
    _validate_broadcast_to(sin.shape, q.shape, "sin")
    _validate_broadcast_to(cos.shape, k.shape, "cos")
    _validate_broadcast_to(sin.shape, k.shape, "sin")
    if position_ids is not None:
        _require_tensor(position_ids, "position_ids")
    if rotary_sections is not None:
        sections = tuple(int(section) for section in rotary_sections)
        if not sections or any(section <= 0 for section in sections) or sum(sections) > q.shape[-1]:
            raise ValueError(f"Invalid rotary sections {sections} for head dimension {q.shape[-1]}.")
    return _mesh_elementwise_leaf([q, cos, sin], q.shape, 6.0, enable_fusion=False), _mesh_elementwise_leaf([k, cos, sin], k.shape, 6.0, enable_fusion=False)
