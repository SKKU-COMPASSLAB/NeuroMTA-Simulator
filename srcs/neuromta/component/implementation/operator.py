from typing import Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
import neuromta.component.implementation.kernel as mca_kernel_lib


__all__ = [
    "MCA_Operator",
    
    "MCA_OP_LINEAR",
    "MCA_OP_RELU_INPLACE",
    "MCA_OP_LINEAR_RELU",
    "MCA_OP_CONV2D",
    "MCA_OP_MAXPOOL2D",
]


class MCA_Operator:
    def __init__(self, device: MCA_DeviceBase, compiled_mapping: CompiledMapping, op_template: Callable, op_compute_methods: list[Callable]):
        self._device = device
        self._compiled_mapping = compiled_mapping
        self._op_template = op_template
        self._op_compute_methods = op_compute_methods
        
        if isinstance(op_template, str):
            self._op_template = getattr(mca_kernel_lib, op_template, None)
            if self._op_template is None:
                raise ValueError(f"Method '{op_template}' not found in MCA_Operator.")
            
        if not check_jit_prototype(self._op_template):
            raise ValueError(f"Method '{self._op_template.__name__}' is not a valid JIT prototype.")
        
        for method_idx, method in enumerate(self._op_compute_methods):
            if isinstance(method, str):
                method_func = getattr(mca_kernel_lib, method, None)
                if method_func is None:
                    raise ValueError(f"Compute method '{method}' not found in MCA_Operator.")
                self._op_compute_methods[method_idx] = method_func
                
            if not check_jit_prototype(self._op_compute_methods[method_idx]):
                raise ValueError(f"Compute method '{self._op_compute_methods[method_idx].__name__}' is not a valid JIT prototype.")
        
        self._is_dispatched = False
        
        self._pipelined_ops: list[MCA_Operator] = []
            
    def dispatch(self, slot_id: str="MAIN"):
        if self._is_dispatched:
            return self  # already dispatched
        
        try:
            for core_id, operator in self._compiled_mapping.operators.items():
                core = self._device.get_npu_core(core_id=core_id)
                
                for stage in operator.stages:
                    kernel: KernelPrototype = self._op_template(core, operator, stage, self._op_compute_methods)
                    kernel.dispatch(slot_id)
                
            self._is_dispatched = True
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch operator on device at slot '{slot_id}': {str(e)}") from e
        
        for op in self._pipelined_ops:
            if op.is_dispatched:
                continue
            
            op.dispatch(slot_id)
        
        return self
    
    def pipeline(self, dst_op: 'MCA_Operator', src_buf_name: str, dst_buf_name: str):
        self.compiled_mapping.apply_pipeline_optimization(
            dst_mapping=dst_op.compiled_mapping,
            src_buf_name=src_buf_name,
            dst_buf_name=dst_buf_name
        )
        
        self._pipelined_ops.append(dst_op)
        
        return self
    
    def summary(self) -> dict:
        return self.compiled_mapping.summary()
    
    @property
    def is_dispatched(self) -> bool:
        return self._is_dispatched
    
    @property
    def compiled_mapping(self) -> CompiledMapping:
        return self._compiled_mapping
        
        
def MCA_OP_LINEAR(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    broadcast_optimize: bool=True,
    broadcast_optimize_targets: list[str]=None,
    
    auto_dispatch: bool=False,
    
    mapping_strategy: str = MCA_OperatorMapper.OUTPUT_STATIONARY
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = MCA_OperatorMapper.LINEAR(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        wgt=wgt,
        bias=bias,
        ofm=ofm,
    ).compile(mapping_strategy=mapping_strategy)
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE, 
        op_compute_methods=[mca_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR]
    )
    
    if auto_dispatch:
        operator.dispatch()
    
    return operator


def MCA_OP_RELU_INPLACE(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    
    auto_dispatch: bool=False,
    
    mapping_strategy: Callable | str = TiledOperatorMapping.output_stationary
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    
    mapping = MCA_OperatorMapper.UNARY_INPLACE(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
    ).compile(mapping_strategy=mapping_strategy)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE, 
        op_compute_methods=[mca_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU_INPLACE]
    )
    
    if auto_dispatch:
        operator.dispatch()
        
    return operator


def MCA_OP_LINEAR_RELU(
    device: MCA_DeviceBase,
    core_group: MCA_CoreGroup,
    spad_ld_mem_space: MCA_L1MemorySpace,
    spad_st_mem_space: MCA_L1MemorySpace,
    
    ifm:  MCA_TensorBuffer,
    wgt:  MCA_TensorBuffer,
    bias: MCA_TensorBuffer,
    ofm:  MCA_TensorBuffer,
    
    broadcast_optimize: bool=True,
    broadcast_optimize_targets: list[str]=None,
    
    auto_dispatch: bool=False,
    
    mapping_strategy: Callable | str = MCA_OperatorMapper.CONTIGUOUS
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = MCA_OperatorMapper.LINEAR(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        wgt=wgt,
        bias=bias,
        ofm=ofm,
    ).compile(mapping_strategy=mapping_strategy)
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE, 
        op_compute_methods=[mca_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU]
    )
    
    if auto_dispatch:
        operator.dispatch()
    
    return operator


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
    
    broadcast_optimize: bool=True,
    broadcast_optimize_targets: list[str]=None,
    
    auto_dispatch: bool=False,
    
    mapping_strategy: str = MCA_OperatorMapper.OUTPUT_STATIONARY
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    wgt  = wgt.copy().tiling(tile_shape=device.mxu_config.wgt_tile_shape)
    bias = bias.copy().tiling(tile_shape=(1, device.mxu_config.wgt_tile_shape[0]))
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = MCA_OperatorMapper.CONV2D(
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
    ).compile(mapping_strategy=mapping_strategy)
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE, 
        op_compute_methods=[mca_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D]
    )
    
    if auto_dispatch:
        operator.dispatch()
    
    return operator


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
    
    broadcast_optimize: bool=True,
    broadcast_optimize_targets: list[str]=None,
    
    auto_dispatch: bool=False,
    
    mapping_strategy: str = MCA_OperatorMapper.OUTPUT_STATIONARY
) -> MCA_Operator:
    # copy before tiling (to avoid modifying the original buffers) -> other operators may use different tiling schemes
    ifm  = ifm.copy().tiling(tile_shape=device.mxu_config.ifm_tile_shape)
    ofm  = ofm.copy().tiling(tile_shape=device.mxu_config.ofm_tile_shape)
    
    mapping = MCA_OperatorMapper.CONV2D(
        core_group=core_group,
        spad_ld_mem_space=spad_ld_mem_space,
        spad_st_mem_space=spad_st_mem_space,
        ifm=ifm,
        ofm=ofm,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,   # groups should always be 1 for maxpooling
        
        # reuse conv2d mapper for maxpooling
        window=window,
    ).compile(mapping_strategy=mapping_strategy)
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        op_template=mca_kernel_lib.MCA_OP_CORE_TEMPLATE, 
        op_compute_methods=[mca_kernel_lib.MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D]
    )
    
    if auto_dispatch:
        operator.dispatch()
    
    return operator