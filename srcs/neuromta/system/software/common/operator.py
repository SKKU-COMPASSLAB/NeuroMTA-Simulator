from typing import Sequence, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *

import neuromta.component.implementation.kernel as mca_kernel_lib
import neuromta.system.software.common.kernel as common_kernel_lib
import neuromta.system.software.common.mapping as common_mapping_lib


__all__ = [
    "MCA_OP_LINEAR",
    "MCA_OP_RELU_INPLACE",
    "MCA_OP_LINEAR_RELU",
    "MCA_OP_CONV2D",
    "MCA_OP_MAXPOOL2D",
    "MCA_OP_AVGPOOL2D",
    "MCA_OP_FLATTEN",
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
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("wgt",  wgt,  is_input=True)
    op_sig.add_buffer("bias", bias, is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_LINEAR(op_sig)


@mca_operator_method    
def MCA_OP_RELU_INPLACE(
    ifm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="RELU_INPLACE",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU_INPLACE],
    )
    
    op_sig.add_buffer("ifm", ifm, is_input=True, is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_UNARY_INPLACE(op_sig)


@mca_operator_method  
def MCA_OP_LINEAR_RELU(
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="LINEAR_RELU",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("wgt",  wgt,  is_input=True)
    op_sig.add_buffer("bias", bias, is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_LINEAR(op_sig)


@mca_operator_method 
def MCA_OP_CONV2D(
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int=1,
    
    use_collective_tile_load: bool=False,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="CONV2D",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("wgt",  wgt,  is_input=True)
    op_sig.add_buffer("bias", bias, is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    op_sig.global_kwargs["groups"] = groups
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig, 
        is_conv2d=True, 
        use_collective_tile_load=use_collective_tile_load,
    )


@mca_operator_method 
def MCA_OP_MAXPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    
    use_collective_tile_load: bool=False,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="MAXPOOL2D",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
        use_collective_tile_load=use_collective_tile_load,
    )


@mca_operator_method 
def MCA_OP_AVGPOOL2D(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    
    use_collective_tile_load: bool=False,
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="AVGPOOL2D",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    op_sig.global_kwargs["window"] = (window, window) if isinstance(window, int) else window
    op_sig.global_kwargs["stride"] = (stride, stride) if isinstance(stride, int) else stride
    op_sig.global_kwargs["padding"] = (padding, padding) if isinstance(padding, int) else padding
    op_sig.global_kwargs["dilation"] = (dilation, dilation) if isinstance(dilation, int) else dilation
    
    return common_mapping_lib.MCA_MAPPER_CONV2D(
        op_sig,
        is_conv2d=False,
        use_collective_tile_load=use_collective_tile_load,
    )



@mca_operator_method 
def MCA_OP_FLATTEN(
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
) -> MCA_OperatorSignature:
    op_sig = MCA_OperatorSignature(
        op_type="FLATTEN",
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE,
        op_ex_kernels=[common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY],
    )
    
    op_sig.add_buffer("ifm",  ifm,  is_input=True)
    op_sig.add_buffer("ofm",  ofm,  is_output=True)
    
    return common_mapping_lib.MCA_MAPPER_FLATTEN(op_sig)