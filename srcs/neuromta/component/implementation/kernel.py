from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import TileSignature, CollectiveTileSignature
from neuromta.component.implementation.operator import *


__all__ = [
    "MCA_KERNEL_CORE_LD_THREAD",
    "MCA_KERNEL_CORE_ST_THREAD",
    "MCA_KERNEL_CORE_EX_THREAD",
]

    
@jit_prototype
def MCA_KERNEL_CORE_LD_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    preprocessing_cmds: list[MCA_CompiledOperator.Command.Base],
    mem_load_cmds: list[MCA_CompiledOperator.Command.Base],
    ld_ex_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
    
    for cmd in preprocessing_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_INIT):
            var = env.variables[cmd.var_name]
            core.var_atomic_init(var, cmd.initial_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_COMPARE_AND_SWAP):
            var = env.variables[cmd.var_name]
            core.var_atomic_compare_and_swap(var, cmd.expected_value, cmd.new_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
            vars = [env.variables[var_name] for var_name in cmd.var_names]
            core.var_conditional_wait(vars, cmd.condition)
        else:
            raise NotImplementedError(f"Preprocessing command {type(cmd)} is not implemented.")
    
    for cmd in mem_load_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            core.async_rpc_wait_all()  # Ensure all previous async operations are completed before the barrier
            
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
            core.async_rpc_wait_all()  # Ensure all previous async operations are completed before the barrier

            vars = [env.variables[var_name] for var_name in cmd.var_names]
            core.var_conditional_wait(vars, cmd.condition)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
                
            if isinstance(tile_sig, CollectiveTileSignature):
                for ptr in cmd.ptrs:
                    core.mem_init(ptr, buf.tile_size)
            
                for tile_sig, memcpy_pattern in zip(tile_sig.src_tiles, tile_sig.memcpy_patterns):
                    src_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_read_args(*tile_sig.coords)
                    
                    for dst_row_idx, src_row_idx in memcpy_pattern.items():
                        dst_row_ptr = cmd.ptrs[0] + dst_row_idx * dst_row_stride
                        src_row_ptr = src_ptr + src_row_idx * src_row_stride

                        core.local_mem_copy(dst_row_ptr, src_row_ptr, row_size, 1, src_row_stride, dst_row_stride, nowait=True)
            else:
                for ptr in cmd.ptrs:
                    core.mem_init(ptr, buf.tile_size)
                
                src_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_read_args(*tile_sig.coords)
                
                if len(cmd.ptrs) > 1:  # BROADCAST: broadcast optimization
                    core.local_mem_broadcast(cmd.ptrs, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
                else:
                    core.local_mem_copy(cmd.ptrs[0], src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_CPY_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            tile_size = buf.tile_size
            
            for ptr in cmd.dst_ptrs:
                core.mem_init(ptr, buf.tile_size)
            
            if len(cmd.dst_ptrs) > 1:  # BROADCAST: broadcast optimization
                core.local_mem_broadcast(cmd.dst_ptrs, cmd.src_ptr, tile_size, 1, nowait=True)
            else:
                core.local_mem_copy(cmd.dst_ptrs[0], cmd.src_ptr, tile_size, 1, nowait=True)
        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()
    
    var_arrived_count = env.variables[ld_ex_sync_barrier[0]]
    var_block_state = env.variables[ld_ex_sync_barrier[1]]
    core.var_atomic_barrier(var_arrived_count, var_block_state, ld_ex_sync_barrier[2])
    
    if post_sync_barrier is not None:
        var_arrived_count = env.variables[post_sync_barrier[0]]
        var_block_state = env.variables[post_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, post_sync_barrier[2])


@jit_prototype
def MCA_KERNEL_CORE_EX_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    execute_cmds: list[MCA_CompiledOperator.Command.Base],
    op_compute_methods: list[Callable],
    ld_ex_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    ex_st_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
    
    var_arrived_count = env.variables[ld_ex_sync_barrier[0]]
    var_block_state = env.variables[ld_ex_sync_barrier[1]]
    core.var_atomic_barrier(var_arrived_count, var_block_state, ld_ex_sync_barrier[2])
    
    for cmd in execute_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_INIT):
            var = env.variables[cmd.var_name]
            core.var_atomic_init(var, cmd.initial_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_COMPARE_AND_SWAP):
            var = env.variables[cmd.var_name]
            core.var_atomic_compare_and_swap(var, cmd.expected_value, cmd.new_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
            vars = [env.variables[var_name] for var_name in cmd.var_names]
            core.var_conditional_wait(vars, cmd.condition)
        else:
            for method in op_compute_methods:
                method(core, env, cmd)
    
    var_arrived_count = env.variables[ex_st_sync_barrier[0]]
    var_block_state = env.variables[ex_st_sync_barrier[1]]
    core.var_atomic_barrier(var_arrived_count, var_block_state, ex_st_sync_barrier[2])
    
    if post_sync_barrier is not None:
        var_arrived_count = env.variables[post_sync_barrier[0]]
        var_block_state = env.variables[post_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, post_sync_barrier[2])
    

@jit_prototype
def MCA_KERNEL_CORE_ST_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    mem_store_cmds: list[MCA_CompiledOperator.Command.Base],
    postprocessing_cmds: list[MCA_CompiledOperator.Command.Base],
    ex_st_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
    
    var_arrived_count = env.variables[ex_st_sync_barrier[0]]
    var_block_state = env.variables[ex_st_sync_barrier[1]]
    core.var_atomic_barrier(var_arrived_count, var_block_state, ex_st_sync_barrier[2])
    
    for cmd in mem_store_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.async_rpc_wait_all()  # Ensure all previous async operations are completed before the barrier
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_STORE_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_write_args(*tile_sig.coords)
            
            core.local_mem_copy(dst_ptr, cmd.ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_CPY_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            tile_size = buf.tile_size
            
            if len(cmd.dst_ptrs) > 0:  # BROADCAST: broadcast optimization
                core.local_mem_broadcast(cmd.dst_ptrs, cmd.src_ptr, tile_size, 1, nowait=True)
            else:
                core.local_mem_copy(cmd.dst_ptrs[0], cmd.src_ptr, tile_size, 1, nowait=True)
        else:
            raise NotImplementedError(f"DMA Store command {type(cmd)} is not implemented.")
    
    core.async_rpc_wait_all()
    
    for cmd in postprocessing_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_INIT):
            var = env.variables[cmd.var_name]
            core.var_atomic_init(var, cmd.initial_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_COMPARE_AND_SWAP):
            var = env.variables[cmd.var_name]
            core.var_atomic_compare_and_swap(var, cmd.expected_value, cmd.new_value)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
            vars = [env.variables[var_name] for var_name in cmd.var_names]
            core.var_conditional_wait(vars, cmd.condition)
        else:
            raise NotImplementedError(f"Postprocessing command {type(cmd)} is not implemented.")
        
    if post_sync_barrier is not None:
        var_arrived_count = env.variables[post_sync_barrier[0]]
        var_block_state = env.variables[post_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, post_sync_barrier[2])
