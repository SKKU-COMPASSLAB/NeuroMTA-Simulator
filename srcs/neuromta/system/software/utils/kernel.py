import math
from numbers import Real

import numpy as np

from neuromta.framework import *

from neuromta.component.core.ccg_tile import *
from neuromta.system.software.utils.descriptor import (
    MeshKernelDescriptor,
    MeshKernelType,
    MeshTensorDescriptor,
)


__all__ = [
    "MESH_KERNEL_LINEAR",
    "MESH_KERNEL_CONV2D",
    "MESH_KERNEL_POOL2D",
    "MESH_KERNEL_ELEMENTWISE",
    "MESH_KERNEL_REDUCTION",
    "MESH_KERNEL_MEMCOPY",
    "MESH_KERNEL_SDPA",
]


def _validate_tensor(tensor: MeshTensorDescriptor, name: str) -> MeshTensorDescriptor:
    if not isinstance(tensor, MeshTensorDescriptor):
        raise TypeError(f"{name} must be a MeshTensorDescriptor.")
    return tensor


def _validate_operation_factor(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def MESH_KERNEL_SDPA(
    q: MeshTensorDescriptor,
    k: MeshTensorDescriptor,
    v: MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    mask: MeshTensorDescriptor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    q_chunk_size: int | None = None,
    kv_chunk_size: int | None = None,
    max_cores_per_head: int = 16,
) -> MeshKernelDescriptor:
    q = _validate_tensor(q, "q")
    k = _validate_tensor(k, "k")
    v = _validate_tensor(v, "v")
    ofm = _validate_tensor(ofm, "ofm")
    mask = None if mask is None else _validate_tensor(mask, "mask")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be a boolean.")
    if any(len(tensor.shape) != 4 for tensor in (q, k, v, ofm)):
        raise ValueError(
            "SDPA requires Q, K, V, and output tensors with shape [batch, heads, sequence, head_dim]."
        )
    if any(any(tile != 1 for tile in tensor.tile_shape[:-2]) for tensor in (q, k, v, ofm)):
        raise ValueError("SDPA batch and head dimensions must not be tiled.")
    batch_size, n_q_heads, q_length, head_dim = q.shape
    k_batch, n_kv_heads, kv_length, k_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError("SDPA K and V shapes must match.")
    if ofm.shape != q.shape:
        raise ValueError("SDPA output shape must match Q.")
    if k_batch != batch_size or k_head_dim != head_dim:
        raise ValueError("SDPA Q, K, and V batch and head dimensions are incompatible.")
    if n_q_heads % n_kv_heads:
        raise ValueError("SDPA query head count must be divisible by the K/V head count.")
    if any(tensor.dtype != q.dtype for tensor in (k, v, ofm)):
        raise ValueError("SDPA Q, K, V, and output dtypes must match.")
    if (
        k.tile_shape != v.tile_shape
        or q.tile_shape != ofm.tile_shape
        or q.tile_shape[-1] != k.tile_shape[-1]
    ):
        raise ValueError(
            "SDPA Q/output and K/V tile layouts must match, including the head-dimension tile width."
        )
    if is_causal and q_length > kv_length:
        raise ValueError("Causal SDPA requires query length to be no larger than K/V length.")
    score_shape = (batch_size, n_q_heads, q_length, kv_length)
    if mask is not None:
        if len(mask.shape) > len(score_shape):
            raise ValueError("SDPA mask rank cannot exceed the attention-score rank.")
        aligned_score_shape = score_shape[-len(mask.shape) :]
        if any(
            mask_dim not in (1, score_dim)
            for mask_dim, score_dim in zip(mask.shape, aligned_score_shape)
        ):
            raise ValueError(
                f"SDPA mask shape {mask.shape} cannot broadcast to score shape {score_shape}."
            )
        aligned_score_tiles = (1, 1, q.tile_shape[-2], k.tile_shape[-2])[-len(mask.tile_shape) :]
        if any(
            mask_dim != 1 and mask_tile != score_tile
            for mask_dim, mask_tile, score_tile in zip(
                mask.shape, mask.tile_shape, aligned_score_tiles
            )
        ):
            raise ValueError(
                "SDPA mask tile layout is incompatible with the attention-score layout."
            )
    if scale is not None and (
        not isinstance(scale, Real) or isinstance(scale, bool) or not math.isfinite(float(scale))
    ):
        raise ValueError("scale must be a finite real value or None.")
    q_chunk_size = q.tile_shape[-2] if q_chunk_size is None else q_chunk_size
    kv_chunk_size = k.tile_shape[-2] if kv_chunk_size is None else kv_chunk_size
    for value, tile, name in (
        (q_chunk_size, q.tile_shape[-2], "q_chunk_size"),
        (kv_chunk_size, k.tile_shape[-2], "kv_chunk_size"),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value <= 0
            or value % tile
        ):
            raise ValueError(
                f"{name} must be a positive multiple of its sequence tile dimension {tile}."
            )
    if (
        not isinstance(max_cores_per_head, (int, np.integer))
        or isinstance(max_cores_per_head, bool)
        or max_cores_per_head <= 0
    ):
        raise ValueError("max_cores_per_head must be a positive integer.")

    softmax_ops_per_score = 13 + int(mask is not None)

    descriptor = MeshKernelDescriptor(
        MeshKernelType.SDPA,
        [q, k, v] + ([] if mask is None else [mask]),
        [ofm],
    ).set_kwargs(
        is_causal=is_causal,
        scale=head_dim**-0.5 if scale is None else float(scale),
        q_chunk_size=int(q_chunk_size),
        kv_chunk_size=int(kv_chunk_size),
        max_cores_per_head=int(max_cores_per_head),
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_group_size=n_q_heads // n_kv_heads,
        softmax_ops_per_score=softmax_ops_per_score,
    )

    return descriptor


def MESH_KERNEL_LINEAR(
    ifm: MeshTensorDescriptor,
    wgt: MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    bias: MeshTensorDescriptor | None = None,
    extra_ops_per_output_element: float = 0.0,
    transpose_wgt: bool = False,
    transpose_ifm: bool = False,
) -> MeshKernelDescriptor:
    ifm = _validate_tensor(ifm, "ifm")
    wgt = _validate_tensor(wgt, "wgt")
    ofm = _validate_tensor(ofm, "ofm")
    bias = None if bias is None else _validate_tensor(bias, "bias")
    extra_ops_per_output_element = _validate_operation_factor(extra_ops_per_output_element, "extra_ops_per_output_element")

    if len(ifm.shape) < 2 or len(wgt.shape) < 2 or len(ofm.shape) < 2:
        raise ValueError("Linear tensors have invalid ranks.")
    if not isinstance(transpose_wgt, bool) or not isinstance(transpose_ifm, bool):
        raise TypeError("transpose_wgt and transpose_ifm must be booleans.")

    ifm_output_dim = ifm.shape[-1] if transpose_ifm else ifm.shape[-2]
    ifm_reduction_dim = ifm.shape[-2] if transpose_ifm else ifm.shape[-1]
    reduction_dim = wgt.shape[-2] if transpose_wgt else wgt.shape[-1]
    output_dim = wgt.shape[-1] if transpose_wgt else wgt.shape[-2]

    if ifm_reduction_dim != reduction_dim:
        raise ValueError("IFM and WGT reduction dimensions do not match.")
    if ofm.shape[-2:] != (ifm_output_dim, output_dim):
        raise ValueError("OFM and WGT output dimensions do not match.")
    if bias is not None and bias.shape[-1] != ofm.shape[-1]:
        raise ValueError("BIAS and OFM dimensions do not match.")

    inputs = [ifm, wgt] if bias is None else [ifm, wgt, bias]
    descriptor = MeshKernelDescriptor(
        MeshKernelType.LINEAR,
        inputs,
        [ofm],
    ).set_kwargs(
        transpose_wgt=transpose_wgt,
        transpose_ifm=transpose_ifm,
        extra_ops_per_output_element=extra_ops_per_output_element
    )

    return descriptor


def _normalize_spatial_pair(
    value: int | tuple[int, int], name: str, allow_zero: bool = False
) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        values = (int(value), int(value))
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in value
        )
    ):
        values = tuple(int(item) for item in value)
    else:
        values = ()
    minimum = 0 if allow_zero else 1
    if len(values) != 2 or any(item < minimum for item in values):
        raise ValueError(f"{name} must contain two integers greater than or equal to {minimum}.")
    return values


def _normalize_padding(value: int | tuple[int, ...]) -> tuple[int, int, int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        values = (int(value),) * 4
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in value
        )
    ):
        values = (int(value[0]), int(value[0]), int(value[1]), int(value[1]))
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 4
        and all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in value
        )
    ):
        values = tuple(int(item) for item in value)
    else:
        values = ()
    if len(values) != 4 or any(item < 0 for item in values):
        raise ValueError("padding must be a non-negative integer, pair, or four-element tuple.")
    return values


def MESH_KERNEL_CONV2D(
    ifm: MeshTensorDescriptor,
    wgt: MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    bias: MeshTensorDescriptor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    extra_ops_per_output_element: float = 0.0,
) -> MeshKernelDescriptor:
    ifm = _validate_tensor(ifm, "ifm")
    wgt = _validate_tensor(wgt, "wgt")
    ofm = _validate_tensor(ofm, "ofm")
    bias = None if bias is None else _validate_tensor(bias, "bias")
    stride = _normalize_spatial_pair(stride, "stride")
    padding = _normalize_padding(padding)
    dilation = _normalize_spatial_pair(dilation, "dilation")

    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError("groups must be a positive integer.")
    if len(ifm.shape) != 4 or len(wgt.shape) != 4 or len(ofm.shape) != 4:
        raise ValueError("Conv2d requires NHWC input/output tensors and an HWKC weight tensor.")

    batch_size, input_height, input_width, input_channels = ifm.shape
    kernel_height, kernel_width, output_channels, weight_channels = wgt.shape

    if wgt.tile_shape[0] != 1 or wgt.tile_shape[1] != 1:
        raise ValueError("Conv2d HWKC weights must be tiled over the K and C dimensions only.")
    if (input_channels % groups or output_channels % groups or weight_channels != input_channels // groups):
        raise ValueError("Conv2d input, output, weight channels, and groups are incompatible.")

    output_height = (input_height + padding[0] + padding[1] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width  = (input_width + padding[2] + padding[3] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1

    if (output_height <= 0 or output_width <= 0 or ofm.shape != (batch_size, output_height, output_width, output_channels)):
        raise ValueError(f"Invalid Conv2d output shape {ofm.shape}; expected {(batch_size, output_height, output_width, output_channels)}.")
    if (ifm.dtype != wgt.dtype or ofm.dtype != ifm.dtype or bias is not None and bias.dtype != ifm.dtype):
        raise ValueError("Conv2d input, weight, bias, and output dtypes must match.")
    if bias is not None and (bias.shape[-1] != output_channels or any(dim != 1 for dim in bias.shape[:-1])):
        raise ValueError("Conv2d bias must be broadcastable over NHW and match output channels.")

    descriptor = MeshKernelDescriptor(
        MeshKernelType.CONV2D,
        [ifm, wgt] if bias is None else [ifm, wgt, bias],
        [ofm]
    ).set_kwargs(
        operation="conv2d",
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        count_include_pad=False,
        extra_ops_per_output_element=extra_ops_per_output_element
    )

    return descriptor


def MESH_KERNEL_POOL2D(
    ifm: MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, int] = 1,
    mode: str = "max",
    count_include_pad: bool = True,
    extra_ops_per_output_element: float = 0.0,
) -> MeshKernelDescriptor:
    ifm = _validate_tensor(ifm, "ifm")
    ofm = _validate_tensor(ofm, "ofm")
    kernel_size = _normalize_spatial_pair(kernel_size, "kernel_size")
    stride = kernel_size if stride is None else _normalize_spatial_pair(stride, "stride")
    padding = _normalize_padding(padding)
    dilation = _normalize_spatial_pair(dilation, "dilation")

    if mode not in ("max", "min", "avg"):
        raise ValueError(f"Unsupported pooling mode: {mode}")
    if not isinstance(count_include_pad, bool):
        raise TypeError("count_include_pad must be a boolean.")
    if len(ifm.shape) != 4 or len(ofm.shape) != 4:
        raise ValueError("Pooling requires NHWC input and output tensors.")

    batch_size, input_height, input_width, channels = ifm.shape
    output_height = (input_height + padding[0] + padding[1] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (input_width + padding[2] + padding[3] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

    if (output_height <= 0 or output_width <= 0 or ofm.shape != (batch_size, output_height, output_width, channels)):
        raise ValueError(f"Invalid Pool2d output shape {ofm.shape}; expected {(batch_size, output_height, output_width, channels)}.")
    if ofm.dtype != ifm.dtype:
        raise ValueError("Pooling input and output dtypes must match.")

    descriptor = MeshKernelDescriptor(
        MeshKernelType.CONV2D, [ifm], [ofm]
    ).set_kwargs(
        operation=f"{mode}_pool2d",
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        count_include_pad=count_include_pad,
        extra_ops_per_output_element=extra_ops_per_output_element
    )

    return descriptor


def MESH_KERNEL_ELEMENTWISE(
    ifms: list[MeshTensorDescriptor] | MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    ops_per_element: float = 1.0,
    enable_fusion: bool = True,
) -> MeshKernelDescriptor:
    ifms = [ifms] if isinstance(ifms, MeshTensorDescriptor) else list(ifms)
    if not ifms:
        raise ValueError("Elementwise kernels require at least one input tensor.")

    ifms = [_validate_tensor(ifm, f"ifms[{index}]") for index, ifm in enumerate(ifms)]
    ofm = _validate_tensor(ofm, "ofm")
    ops_per_element = _validate_operation_factor(ops_per_element, "ops_per_element")

    if not isinstance(enable_fusion, bool):
        raise TypeError("enable_fusion must be a boolean.")

    descriptor = MeshKernelDescriptor(
        MeshKernelType.ELEMENTWISE, ifms, [ofm], enable_fusion=enable_fusion
    ).set_kwargs(
        ops_per_element=ops_per_element,
    )

    return descriptor


def MESH_KERNEL_REDUCTION(
    ifm: MeshTensorDescriptor,
    ofm: MeshTensorDescriptor,
    ops_per_input_element: float = 1.0,
    extra_ops_per_output_element: float = 0.0,
) -> MeshKernelDescriptor:
    ifm = _validate_tensor(ifm, "ifm")
    ofm = _validate_tensor(ofm, "ofm")
    ops_per_input_element = _validate_operation_factor(ops_per_input_element, "ops_per_input_element")

    descriptor = MeshKernelDescriptor(
        MeshKernelType.REDUCTION, [ifm], [ofm]
    ).set_kwargs(
        ops_per_input_element=ops_per_input_element,
        extra_ops_per_output_element=extra_ops_per_output_element,
    )

    return descriptor


def MESH_KERNEL_MEMCOPY(
    src: MeshTensorDescriptor | None = None,
    dst: MeshTensorDescriptor | None = None,
    metadata_inputs: list[MeshTensorDescriptor] = None,
    range_aware: bool = False,
    gather: bool = False,
    src_traffic_bytes: int = None,
    dst_traffic_bytes: int = None,
) -> MeshKernelDescriptor:
    src = None if src is None else _validate_tensor(src, "src")
    dst = None if dst is None else _validate_tensor(dst, "dst")
    metadata_inputs = (
        []
        if metadata_inputs is None
        else [
            _validate_tensor(tensor, f"metadata_inputs[{index}]")
            for index, tensor in enumerate(metadata_inputs)
        ]
    )
    if src is None and dst is None:
        raise ValueError("Memory-copy kernels require a source, a destination, or both.")
    if not isinstance(range_aware, bool) or not isinstance(gather, bool):
        raise TypeError("range_aware and gather must be booleans.")
    if gather and (src is None or dst is None):
        raise ValueError("Gather memory copies require both source and destination tensors.")

    _is_grid_mismatch = src is not None and dst is not None and (src.tile_grid_shape != dst.tile_grid_shape or src.get_tile_size() != dst.get_tile_size())
    if (src is not None and dst is not None and not range_aware and not gather and _is_grid_mismatch):
        raise ValueError("Memory-copy source and destination tile layouts must match unless range_aware=True.")

    inputs = ([] if src is None else [src]) + metadata_inputs
    outputs = [] if dst is None else [dst]

    if src_traffic_bytes is None and src is not None:
        if gather:
            src_traffic_bytes = dst.get_n_tiles() * src.get_tile_size()
        else:
            src_traffic_bytes = src.get_n_tiles() * src.get_tile_size()
    elif src_traffic_bytes is None:
        src_traffic_bytes = 0

    if dst_traffic_bytes is None and dst is not None:
        dst_traffic_bytes = dst.get_n_tiles() * dst.get_tile_size()
    elif dst_traffic_bytes is None:
        dst_traffic_bytes = 0

    if src_traffic_bytes < 0 or dst_traffic_bytes < 0:
        raise ValueError("Memory-copy traffic cannot be negative.")

    descriptor = MeshKernelDescriptor(
        MeshKernelType.MEMCOPY, inputs, outputs
    ).set_kwargs(
        range_aware=range_aware,
        gather=gather,
        metadata_input_count=len(metadata_inputs),
        metadata_traffic_bytes=tuple(tensor.get_size() for tensor in metadata_inputs),
        src_traffic_bytes=src_traffic_bytes,
        dst_traffic_bytes=dst_traffic_bytes,
    )

    return descriptor
