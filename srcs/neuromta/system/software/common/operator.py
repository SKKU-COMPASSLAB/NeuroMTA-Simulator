from typing import Sequence, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *

import neuromta.system.software.common.kernel as common_kernel_lib
import neuromta.system.software.common.mapping as common_mapping_lib


__all__ = [
    "MCA_OP_LINEAR",
    "MCA_OP_RELU",
    "MCA_OP_LINEAR_RELU",
    "MCA_OP_CONV2D",
    "MCA_OP_MAXPOOL2D",
    "MCA_OP_AVGPOOL2D",
    "MCA_OP_ADAPTIVE_MAXPOOL2D",
    "MCA_OP_ADAPTIVE_AVGPOOL2D",
]


@mca_operator_method     
def MCA_OP_LINEAR(
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="LINEAR",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_LINEAR()
    )
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    k_tile = ifm.mem_space.device.mxu_config.k_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, k_tile)),  is_input=True)
    op_sig.add_buffer("wgt",  wgt.tiling((n_tile, k_tile)),  is_param=True)
    op_sig.add_buffer("bias", bias.tiling((1, n_tile)),      is_param=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_LINEAR(op_sig)


@mca_operator_method    
def MCA_OP_RELU(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="RELU",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_RELU()
    )
    
    m_tile = ofm.mem_space.device.mxu_config.m_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm", ifm.tiling((m_tile, n_tile)), is_input=True)
    op_sig.add_buffer("ofm", ofm.tiling((m_tile, n_tile)), is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_UNARY(op_sig)


@mca_operator_method  
def MCA_OP_LINEAR_RELU(
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="LINEAR_RELU",
        kernel_template=common_kernel_lib.MCA_KERNEL_MERGED_LINEAR_RELU()
    )
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    k_tile = ifm.mem_space.device.mxu_config.k_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, k_tile)),  is_input=True)
    op_sig.add_buffer("wgt",  wgt.tiling((n_tile, k_tile)),  is_param=True)
    op_sig.add_buffer("bias", bias.tiling((1, n_tile)),      is_param=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_LINEAR(op_sig)


@mca_operator_method 
def MCA_OP_CONV2D(
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    stride: Sequence[int],
    padding: Sequence[int]=(0, 0),
    dilation: Sequence[int]=(1, 1),
    groups: int=1,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="CONV2D",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_CONV2D()
    )
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    k_tile = ifm.mem_space.device.mxu_config.k_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, k_tile)),  is_input=True)
    op_sig.add_buffer("wgt",  wgt.tiling((n_tile, k_tile)),  is_param=True)
    op_sig.add_buffer("bias", bias.tiling((1, n_tile)),      is_param=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    op_sig.global_kwargs["groups"] = groups
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig, 
        is_conv2d=True, 
    )


@mca_operator_method 
def MCA_OP_MAXPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int]=(0, 0),
    dilation: Sequence[int]=(1, 1),
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="MAXPOOL2D",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_MAXPOOL2D(),
    )
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, n_tile)),  is_input=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
    )


@mca_operator_method 
def MCA_OP_AVGPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="AVGPOOL2D",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_AVGPOOL2D(),
    )
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, n_tile)),  is_input=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
    )
    

@mca_operator_method
def MCA_OP_ADAPTIVE_MAXPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="ADAPTIVE_MAXPOOL2D",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_MAXPOOL2D(),
    )
    
    N, H, W, C = ifm.shape
    N, OH, OW, C = ofm.shape
    
    window = (H // OH, W // OW)
    stride = window
    padding = (0, 0)
    dilation = (1, 1)
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, n_tile)),  is_input=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
    )
    
    
@mca_operator_method
def MCA_OP_ADAPTIVE_AVGPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="ADAPTIVE_AVGPOOL2D",
        kernel_template=common_kernel_lib.MCA_KERNEL_TILED_AVGPOOL2D(),
    )
    
    N, H, W, C = ifm.shape
    N, OH, OW, C = ofm.shape
    
    window = (H // OH, W // OW)
    stride = window
    padding = (0, 0)
    dilation = (1, 1)
    
    m_tile = ifm.mem_space.device.mxu_config.m_tile
    n_tile = ofm.mem_space.device.mxu_config.n_tile
    
    op_sig.add_buffer("ifm",  ifm.tiling((m_tile, n_tile)),  is_input=True)
    op_sig.add_buffer("ofm",  ofm.tiling((m_tile, n_tile)),  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
    )
