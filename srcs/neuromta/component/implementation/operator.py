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
]


class MCA_Operator:
    def __init__(self, device: MCA_DeviceBase, compiled_mapping: CompiledMapping, method: Callable | str):
        self._device = device
        self._compiled_mapping = compiled_mapping
        self._method = method
        
        if isinstance(method, str):
            self._method = getattr(mca_kernel_lib, method, None)
            if self._method is None:
                raise ValueError(f"Method '{method}' not found in MCA_Operator.")
            
        if not check_jit_prototype(self._method):
            raise ValueError(f"Method '{self._method.__name__}' is not a valid JIT prototype.")
        
        self._is_dispatched = False
        
        self._pipelined_ops: list[MCA_Operator] = []
            
    def dispatch(self, slot_id: str="MAIN"):
        if self._is_dispatched:
            return self  # already dispatched
        
        try:
            for core_id, operator in self._compiled_mapping.operators.items():
                core = self._device.get_npu_core(core_id=core_id)
                
                for stage in operator.stages:
                    kernel: KernelPrototype = self._method(core, operator, stage)
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
    
    auto_dispatch: bool=False
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
    ).compile()
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        device=device, 
        compiled_mapping=mapping, 
        method=mca_kernel_lib.MCA_KERNEL_CORE_OP_LINEAR
    )
    
    if auto_dispatch:
        operator.dispatch()
    
    return operator