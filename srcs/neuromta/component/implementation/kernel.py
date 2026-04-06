from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import TileSignature
from neuromta.component.implementation.operator import *


__all__ = [
    "MCA_KERNEL_CORE_LD_THREAD",
    "MCA_KERNEL_CORE_EX_THREAD",
    "MCA_KERNEL_CORE_ST_THREAD",
]


def FIFO_IR_SORT_KEY(cmd: MCA_CompiledOperator.IR.MEM_LOAD_FROM_FIFO | MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO):
    return cmd.buf, cmd.entry_id
  
@jit_prototype
def MCA_KERNEL_CORE_LD_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    stage: MCA_CompiledOperator.Stage,
    ex_pp_cnt_var_name: str,
    ex_pr_cnt_var_name: str,
    gb_barrier: tuple[VariableHandle, VariableHandle, int] = None,
):
    
    core.var_atomic_compare_and_swap(env.variables[ex_pp_cnt_var_name], 0, 1)
    
    for group in stage.groups:
        mem_loads:   list[MCA_CompiledOperator.IR.MEM_LOAD_TILE]      = []
        fifo_loads:  list[MCA_CompiledOperator.IR.MEM_LOAD_FROM_FIFO] = []
        fifo_stores: list[MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO]  = []
        
        for ir in group.loads:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_LOAD_TILE):
                mem_loads.append(ir)
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_LOAD_FROM_FIFO):
                fifo_loads.append(ir)
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO):
                fifo_stores.append(ir)
            else:
                raise Exception(f"Unsupported IR type '{type(ir).__name__}' in load thread kernel.")
        
        with new_parallel_thread("MEM_LOADS"):
            for i, ir in enumerate(mem_loads):
                with new_parallel_thread(f"{i}"):
                    tile_sig = ir.tile_sig
                    buf = env.buffers[ir.tile_sig.buf_name]
                    src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad = buf.get_tile_ptr_read_args(*tile_sig.coords)
                    
                    core.mem_copy(ir.ptr, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad)
            
        core.parallel_merge()
        
        with new_parallel_thread("FIFO_LOADS"):
            for ir in fifo_loads:
                tile_sig = ir.tile_sig
                buf = env.fifo_buffers[ir.buf]
                tile_size = tile_sig.tile_size
            
                core.mem_copy_from_fifo(ir.ptr, buf, ir.entry_id, tile_size)
            core.var_atomic_increase(env.variables[ex_pr_cnt_var_name], 1)
                
        with new_parallel_thread("FIFO_STORES"):
            for ir in fifo_stores:
                tile_sig = ir.tile_sig
                buf = env.fifo_buffers[ir.buf]
                tile_size = tile_sig.tile_size
                
                core.mem_copy_to_fifo(ir.ptr, buf, ir.entry_id, tile_size, ir.ref_count)
                
    core.parallel_merge()
    
    if gb_barrier is not None:
        core.var_atomic_barrier(*gb_barrier)
    
@jit_prototype
def MCA_KERNEL_CORE_EX_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    stage: MCA_CompiledOperator.Stage,
    op_compute_methods: list[Callable],
    ex_pp_cnt_var_name: str,
    ex_pr_cnt_var_name: str,
    st_pp_cnt_var_name: str,
    st_pr_cnt_var_name: str,
    gb_barrier: tuple[VariableHandle, VariableHandle, int] = None,
):
    core.var_atomic_compare_and_swap(env.variables[ex_pp_cnt_var_name], 1, 0)
    core.var_atomic_compare_and_swap(env.variables[st_pp_cnt_var_name], 0, 1)
    
    for group in stage.groups:
        core.var_conditional_wait(env.variables[ex_pr_cnt_var_name], env.variables[ex_pr_cnt_var_name].greater_than(0))
        
        for ir in group.executes:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.EXE_UOP):
                for method in op_compute_methods:
                    method(core, env, ir)
            else:
                raise Exception(f"Unsupported IR type '{type(ir).__name__}' in execution thread kernel.")
        
        core.var_atomic_increase(env.variables[ex_pr_cnt_var_name], -1)
        core.var_atomic_increase(env.variables[st_pr_cnt_var_name], 1)
        
    if gb_barrier is not None:
        core.var_atomic_barrier(*gb_barrier)

@jit_prototype
def MCA_KERNEL_CORE_ST_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    stage: MCA_CompiledOperator.Stage,
    st_pp_cnt_var_name: str,
    st_pr_cnt_var_name: str,
    gb_barrier: tuple[VariableHandle, VariableHandle, int] = None,
):  
    core.var_atomic_compare_and_swap(env.variables[st_pp_cnt_var_name], 1, 0)
    
    for group in stage.groups:
        core.var_conditional_wait(env.variables[st_pr_cnt_var_name], env.variables[st_pr_cnt_var_name].greater_than(0))
        
        for ir in group.stores:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_STORE_TILE):
                tile_sig = ir.tile_sig
                buf = env.buffers[ir.tile_sig.buf_name]
                dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_write_args(*tile_sig.coords)
                
                core.mem_copy(dst_ptr, ir.ptr, row_size, row_num, src_row_stride, dst_row_stride)
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO):
                tile_sig = ir.tile_sig
                buf = env.fifo_buffers[ir.buf]
                tile_size = tile_sig.tile_size
                
                core.mem_copy_to_fifo(ir.ptr, buf, ir.entry_id, tile_size, ir.ref_count)
            else:
                raise Exception(f"Unsupported IR type '{type(ir).__name__}' in store thread kernel.")
        
        core.var_atomic_increase(env.variables[st_pr_cnt_var_name], -1)
        
    if gb_barrier is not None:
        core.var_atomic_barrier(*gb_barrier)