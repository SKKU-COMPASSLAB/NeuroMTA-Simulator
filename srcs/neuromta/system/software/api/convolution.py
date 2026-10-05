import numpy as np

from neuromta.system.software.api.common import _mesh_device_op_method, _require_tensor
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import MESH_KERNEL_CONV2D, MESH_KERNEL_POOL2D


__all__ = [
    "mesh_conv2d",
    "mesh_max_pool2d",
    "mesh_min_pool2d",
    "mesh_avg_pool2d",
]


def _normalize_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    values = (int(value), int(value)) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) else tuple(value) if isinstance(value, (tuple, list)) else ()
    if len(values) != 2 or any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item <= 0 for item in values):
        raise ValueError(f"{name} must be a positive integer or pair.")
    return tuple(int(item) for item in values)


def _normalize_padding(value: int | tuple[int, ...]) -> tuple[int, int, int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        values = (int(value),) * 4
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        values = (value[0], value[0], value[1], value[1])
    elif isinstance(value, (tuple, list)) and len(value) == 4:
        values = tuple(value)
    else:
        values = ()
    if len(values) != 4 or any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item < 0 for item in values):
        raise ValueError("padding must be a non-negative integer, pair, or four-element tuple.")
    return tuple(int(item) for item in values)


def _output_shape(x: MeshTensorDescriptor, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int, int, int], dilation: tuple[int, int], output_channels: int) -> tuple[int, int, int, int]:
    output_height = (x.shape[1] + padding[0] + padding[1] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (x.shape[2] + padding[2] + padding[3] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("The convolution window does not produce a non-empty output.")
    return x.shape[0], output_height, output_width, output_channels


@_mesh_device_op_method
def mesh_conv2d(x: MeshTensorDescriptor, weight: MeshTensorDescriptor, bias: MeshTensorDescriptor=None, stride: int | tuple[int, int]=1, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, int]=1, groups: int=1) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    weight = _require_tensor(weight, "weight")
    bias = None if bias is None else _require_tensor(bias, "bias")
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("mesh_conv2d expects an NHWC input and an HWKC weight tensor.")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError("groups must be a positive integer.")
    kernel_height, kernel_width, output_channels, weight_channels = weight.shape
    if x.shape[-1] % groups or output_channels % groups or weight_channels != x.shape[-1] // groups:
        raise ValueError("Input, output, weight channels, and groups are incompatible.")
    if x.dtype != weight.dtype or bias is not None and bias.dtype != x.dtype:
        raise ValueError("mesh_conv2d requires matching input, weight, and bias dtypes.")
    if bias is not None and (bias.shape[-1] != output_channels or any(dim != 1 for dim in bias.shape[:-1])):
        raise ValueError("bias must be broadcastable over NHW and match output channels.")
    stride_pair = _normalize_pair(stride, "stride")
    padding_tuple = _normalize_padding(padding)
    dilation_pair = _normalize_pair(dilation, "dilation")
    output_shape = _output_shape(x, (kernel_height, kernel_width), stride_pair, padding_tuple, dilation_pair, output_channels)
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=x.tile_shape, dtype=x.dtype)
    return MESH_KERNEL_CONV2D(x, weight, output, bias=bias, stride=stride_pair, padding=padding_tuple, dilation=dilation_pair, groups=groups), output


@_mesh_device_op_method
def _mesh_pool2d(x: MeshTensorDescriptor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, padding: int | tuple[int, ...], dilation: int | tuple[int, int], mode: str, count_include_pad: bool) -> tuple[MeshKernelDescriptor, MeshTensorDescriptor]:
    x = _require_tensor(x, "x")
    if len(x.shape) != 4:
        raise ValueError("Pooling expects an NHWC input tensor.")
    if not isinstance(count_include_pad, bool):
        raise TypeError("count_include_pad must be a boolean.")
    kernel_pair = _normalize_pair(kernel_size, "kernel_size")
    stride_pair = kernel_pair if stride is None else _normalize_pair(stride, "stride")
    padding_tuple = _normalize_padding(padding)
    dilation_pair = _normalize_pair(dilation, "dilation")
    output_shape = _output_shape(x, kernel_pair, stride_pair, padding_tuple, dilation_pair, x.shape[-1])
    output = MeshTensorDescriptor(shape=output_shape, tile_shape=x.tile_shape, dtype=x.dtype)
    return MESH_KERNEL_POOL2D(x, output, kernel_size=kernel_pair, stride=stride_pair, padding=padding_tuple, dilation=dilation_pair, mode=mode, count_include_pad=count_include_pad), output


def mesh_max_pool2d(x: MeshTensorDescriptor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None=None, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, int]=1) -> MeshTensorDescriptor:
    return _mesh_pool2d(x, kernel_size, stride, padding, dilation, "max", False)


def mesh_min_pool2d(x: MeshTensorDescriptor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None=None, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, int]=1) -> MeshTensorDescriptor:
    return _mesh_pool2d(x, kernel_size, stride, padding, dilation, "min", False)


def mesh_avg_pool2d(x: MeshTensorDescriptor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None=None, padding: int | tuple[int, ...]=0, count_include_pad: bool=True) -> MeshTensorDescriptor:
    return _mesh_pool2d(x, kernel_size, stride, padding, 1, "avg", count_include_pad)
