from typing import Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
import neuromta.component.implementation.kernel as mca_kernel_lib


__all__ = [
    "MCA_Operator",
    
    "MCA_LINEAR",
]


class MCA_Operator:
    def __init__(self, mapping: CompiledMapping, method: Callable | str):
        self._mapping = mapping
        self._method = method
        
        if isinstance(method, str):
            self._method = getattr(mca_kernel_lib, method, None)
            if self._method is None:
                raise ValueError(f"Method '{method}' not found in MCA_Operator.")
            
        if not check_jit_prototype(self._method):
            raise ValueError(f"Method '{self._method.__name__}' is not a valid JIT prototype.")
            
    def dispatch(self, device: MCA_DeviceBase, slot_id: str="MAIN"):
        try:
            for core_id, operator in self._mapping.operators.items():
                core = device.get_npu_core(core_id=core_id)
                kernel: KernelPrototype = self._method(core, operator)
                kernel.dispatch(slot_id)
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch operator on device at slot '{slot_id}': {str(e)}") from e
        
        return self
    
    def summary(self) -> dict:
        return self._mapping.summary()
        
        
def MCA_LINEAR(
    core_group: MCA_CoreGroup,
    spad_mem_space: MCA_L1MemorySpace,
    
    ifm_b:  MCA_TensorBuffer,
    wgt_b:  MCA_TensorBuffer,
    bias_b: MCA_TensorBuffer,
    ofm_b:  MCA_TensorBuffer,
    
    broadcast_optimize: bool=True,
    broadcast_optimize_targets: list[str]=None,
) -> MCA_Operator:
    mapping = MCA_OperatorMapper.LINEAR(
        core_group=core_group,
        spad_mem_space=spad_mem_space,
        ifm_b=ifm_b,
        wgt_b=wgt_b,
        bias_b=bias_b,
        ofm_b=ofm_b,
    ).compile()
    
    if broadcast_optimize:
        mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
    
    operator = MCA_Operator(
        mapping, 
        method="MCA_KERNEL_CORE_OP_LINEAR"
    )
    
    return operator