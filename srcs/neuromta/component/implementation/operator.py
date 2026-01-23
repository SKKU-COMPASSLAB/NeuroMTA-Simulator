from typing import Any, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
import neuromta.component.implementation.kernel as mca_kernel_lib


__all__ = [
    "MCA_Operator",
    "mca_operator_method",
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
    

def mca_operator_method(func: Callable[..., tuple[MCA_OperatorMapper, Callable, list[Callable]]]):
    def _mca_operator_method_wrapper(
        device: MCA_DeviceBase, 
        core_group: MCA_CoreGroup, 
        spad_ld_mem_space: MCA_L1MemorySpace, 
        spad_st_mem_space: MCA_L1MemorySpace,
        
        *args, 
        
        broadcast_optimize: bool=True,
        broadcast_optimize_targets: list[str]=None,
        auto_dispatch: bool=False,
        mapping_strategy: str = MCA_OperatorMapper.OUTPUT_STATIONARY,
        
        **kwargs,
    ) -> MCA_Operator:
        mapping, op_template, op_compute_methods = func(
            device, 
            core_group, 
            spad_ld_mem_space, 
            spad_st_mem_space,
            
            *args, 
            **kwargs,
        )
        
        if not isinstance(mapping, MCA_OperatorMapper):
            raise ValueError(f"The mapping algorithm function must return an MCA_OperatorMapper as the first element, but got {type(mapping)}.")
        
        if op_template is None:
            op_template = mca_kernel_lib.MCA_OP_CORE_TEMPLATE
        elif not isinstance(op_template, Callable):
            raise ValueError(f"The mapping algorithm function must return a Callable as the second element, but got {type(op_template)}.")
        
        if isinstance(op_compute_methods, Callable):
            if not check_jit_prototype(op_compute_methods):
                raise ValueError(f"The operation template '{op_compute_methods.__name__}' is not a valid JIT prototype.")
            op_compute_methods = [op_compute_methods]
        elif not isinstance(op_compute_methods, list) or not all(isinstance(m, Callable) for m in op_compute_methods):
            raise ValueError(f"The mapping algorithm function must return a list of Callables as the third element, but got {type(op_compute_methods)}.")
        
        compiled_mapping = mapping.compile(mapping_strategy=mapping_strategy)
        
        if broadcast_optimize:
            compiled_mapping.apply_broadcast_optimization(buf_targets=broadcast_optimize_targets)
            
        operator = MCA_Operator(
            device=device, 
            compiled_mapping=compiled_mapping, 
            op_template=op_template, 
            op_compute_methods=op_compute_methods
        )
        
        if auto_dispatch:
            operator.dispatch()
            
        return operator
    return _mca_operator_method_wrapper
