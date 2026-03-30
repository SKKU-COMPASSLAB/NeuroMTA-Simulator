from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import TileSignature
from neuromta.component.implementation.operator import *


__all__ = [
    # "MCA_KERNEL_CORE_MEM_THREAD",
    # "MCA_KERNEL_CORE_EXE_THREAD",
    "MCA_KERNEL_CORE_LD_THREAD",
    "MCA_KERNEL_CORE_EX_THREAD",
    "MCA_KERNEL_CORE_ST_THREAD",
]

    
@jit_prototype
def MCA_KERNEL_CORE_LD_THREAD(
    core: NPUCore, 
    env: MCA_OperatorGraphCompiler.Environment, 
    preprocessing_cmds: list[MCA_CompiledOperator.Command.Base],
    mem_load_cmds: list[MCA_CompiledOperator.Command.Base],
    stage_sync_barrier: tuple[str, str, int],
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
        
    if stage_sync_barrier is not None:
        var_arrived_count = env.variables[stage_sync_barrier[0]]
        var_block_state = env.variables[stage_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, stage_sync_barrier[2])
    
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
    
    load_from_fifo_cmds: list[MCA_CompiledOperator.Command.MEM_LOAD_FROM_FIFO] = []
    store_to_fifo_cmds:  list[MCA_CompiledOperator.Command.MEM_STORE_TO_FIFO]  = []
    
    for cmd in mem_load_cmds:
        if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, MCA_CompiledOperator.Command.BARRIER):
            core.async_rpc_wait_all()
            
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
            core.async_rpc_wait_all()

            vars = [env.variables[var_name] for var_name in cmd.var_names]
            core.var_conditional_wait(vars, cmd.condition)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            
            src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad = buf.get_tile_ptr_read_args(*tile_sig.coords)
            
            if len(cmd.ptrs) > 1:  # BROADCAST: broadcast optimization
                core.local_mem_broadcast(cmd.ptrs, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad, nowait=True)
            else:
                core.local_mem_copy(cmd.ptrs[0], src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad, nowait=True)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_FROM_FIFO):
            load_from_fifo_cmds.append(cmd)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_STORE_TO_FIFO):
            store_to_fifo_cmds.append(cmd)
        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()
    
    with new_parallel_thread("LD_FROM_FIFO"):
        for cmd in load_from_fifo_cmds:
            tile_sig = cmd.tile_sig
            buf = env.fifo_buffers[cmd.buf]
            tile_size = tile_sig.tile_size
            
            core.local_mem_copy_from_fifo(cmd.ptr, buf, cmd.entry_id, tile_size)
    
    with new_parallel_thread("ST_TO_FIFO"):
        for cmd in store_to_fifo_cmds:
            tile_sig = cmd.tile_sig
            buf = env.fifo_buffers[cmd.buf]
            tile_size = tile_sig.tile_size
            
            core.local_mem_copy_to_fifo(cmd.ptr, buf, cmd.entry_id, tile_size, cmd.ref_count)
            
    core.parallel_merge()
    
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
    stage_sync_barrier: tuple[str, str, int],
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
        
    if stage_sync_barrier is not None:
        var_arrived_count = env.variables[stage_sync_barrier[0]]
        var_block_state = env.variables[stage_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, stage_sync_barrier[2])
    
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
    stage_sync_barrier: tuple[str, str, int],
    pre_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
    post_sync_barrier: tuple[str, str, int],  # (var_arrived_count_name, var_block_state_name, total_arrivals)
):
    if pre_sync_barrier is not None:
        var_arrived_count = env.variables[pre_sync_barrier[0]]
        var_block_state = env.variables[pre_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, pre_sync_barrier[2])
        
    if stage_sync_barrier is not None:
        var_arrived_count = env.variables[stage_sync_barrier[0]]
        var_block_state = env.variables[stage_sync_barrier[1]]
        core.var_atomic_barrier(var_arrived_count, var_block_state, stage_sync_barrier[2])
    
    load_from_fifo_cmds: list[MCA_CompiledOperator.Command.MEM_LOAD_FROM_FIFO] = []
    store_to_fifo_cmds:  list[MCA_CompiledOperator.Command.MEM_STORE_TO_FIFO]  = []
    
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
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_FROM_FIFO):
            load_from_fifo_cmds.append(cmd)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_STORE_TO_FIFO):
            store_to_fifo_cmds.append(cmd)
        else:
            raise NotImplementedError(f"DMA Store command {type(cmd)} is not implemented.")
    
    core.async_rpc_wait_all()
    
    with new_parallel_thread("LD_FROM_FIFO"):
        for cmd in load_from_fifo_cmds:
            tile_sig = cmd.tile_sig
            buf = env.fifo_buffers[cmd.buf]
            tile_size = tile_sig.tile_size
            
            core.local_mem_copy_from_fifo(cmd.ptr, buf, cmd.entry_id, tile_size)
    
    with new_parallel_thread("ST_TO_FIFO"):
        for cmd in store_to_fifo_cmds:
            tile_sig = cmd.tile_sig
            buf = env.fifo_buffers[cmd.buf]
            tile_size = tile_sig.tile_size
            
            core.local_mem_copy_to_fifo(cmd.ptr, buf, cmd.entry_id, tile_size, cmd.ref_count)
    
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