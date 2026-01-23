from typing import Sequence, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import MCA_Operator, mca_operator_method

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
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_LINEAR(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        wgt=wgt,
        bias=bias,
        ofm=ofm,
    )
    
    op_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    op_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR]
    
    return mapping, op_template, op_compute_methods


@mca_operator_method    
def MCA_OP_RELU_INPLACE(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_UNARY_INPLACE(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
    )
    
    op_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    op_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU_INPLACE]
    
    return mapping, op_template, op_compute_methods


@mca_operator_method  
def MCA_OP_LINEAR_RELU(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_LINEAR(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        wgt=wgt,
        bias=bias,
        ofm=ofm,
    )
    
    op_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    op_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU]
    
    return mapping, op_template, op_compute_methods


@mca_operator_method 
def MCA_OP_CONV2D(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int=1,
    
    use_collective_tile_load: bool=False,
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_CONV2D(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        ofm=ofm,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        
        # conv2d specific buffers
        wgt=wgt,
        bias=bias,
        use_collective_tile_load=use_collective_tile_load,
    )
    
    operator_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    operator_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D]
    
    return mapping, operator_template, operator_compute_methods


@mca_operator_method 
def MCA_OP_MAXPOOL2D(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    
    use_collective_tile_load: bool=False,
) -> tuple[MCA_OperatorMapper, Callable, list[Callable]]:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_CONV2D(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        ofm=ofm,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,   # TODO: groups should always be 1 for maxpooling
        
        # reuse conv2d mapper for maxpooling
        window=window,
        use_collective_tile_load=use_collective_tile_load,
    )
    
    operator_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    operator_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D]
    
    return mapping, operator_template, operator_compute_methods


@mca_operator_method 
def MCA_OP_AVGPOOL2D(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    
    use_collective_tile_load: bool=False,
) -> tuple[MCA_OperatorMapper, Callable, list[Callable]]:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_CONV2D(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        ofm=ofm,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,   # TODO: groups should always be 1 for avgpooling
        
        # reuse conv2d mapper for avgpooling
        window=window,
        use_collective_tile_load=use_collective_tile_load,
    )
    
    operator_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    operator_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D]
    
    return mapping, operator_template, operator_compute_methods 


@mca_operator_method 
def MCA_OP_FLATTEN(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
) -> tuple[MCA_OperatorMapper, Callable, list[Callable]]:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = common_mapping_lib.MCA_MAPPER_FLATTEN(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        ofm=ofm,
    )
    
    operator_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
    operator_compute_methods = [common_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY]
    
    return mapping, operator_template, operator_compute_methods