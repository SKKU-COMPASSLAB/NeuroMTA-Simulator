from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import TileSignature, CollectiveTileSignature
from neuromta.component.implementation.operator import *


__all__ = [
    # KERNEL CORE STAGE
    "MCA_KERNEL_CORE_STAGE_PREPROCESSING",
    "MCA_KERNEL_CORE_STAGE_POSTPROCESSING",
    
    "MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST",
    "MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST",

    # KERNEL CORE OP
    "MCA_OP_CORE_TEMPLATE",
]

@jit_prototype
def MCA_KERNEL_CORE_STAGE_PREPROCESSING(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperator, commands: list[MCA_CompiledOperator.Command.Base]):                
    for cmd in commands:
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

@jit_prototype
def MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperator, commands: list[MCA_CompiledOperator.Command.Base]):
    for cmd in commands:
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

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperator, commands: list[MCA_CompiledOperator.Command.Base]):
    for cmd in commands:
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

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperator, commands: list[MCA_CompiledOperator.Command.Base]):
    for cmd in commands:
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

            for ptr in cmd.ptrs:
                core.mem_init(ptr, buf.tile_size)
                
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
                src_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_read_args(*tile_sig.coords)
                
                if len(cmd.ptrs) > 1:  # BROADCAST: broadcast optimization
                    core.local_mem_broadcast(cmd.ptrs, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
                else:
                    core.local_mem_copy(cmd.ptrs[0], src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_CPY_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            tile_size = buf.tile_size
            
            if len(cmd.dst_ptrs) > 1:  # BROADCAST: broadcast optimization
                core.local_mem_broadcast(cmd.dst_ptrs, cmd.src_ptr, tile_size, 1, nowait=True)
            else:
                core.local_mem_copy(cmd.dst_ptrs[0], cmd.src_ptr, tile_size, 1, nowait=True)
        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()


@jit_prototype
def MCA_OP_CORE_TEMPLATE(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperator, stage1_cursor: int, stage2_cursor: int, stage3_cursor: int, op_compute_methods: list[Callable]):
    stages = operator.mappings[core.core_id]
    
    if stage1_cursor >= 0 and stage1_cursor < len(stages):
        with new_parallel_thread("STAGE1"):
            MCA_KERNEL_CORE_STAGE_PREPROCESSING(core, env, operator, stages[stage1_cursor].preprocessing_commands)
            MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core, env, operator, stages[stage1_cursor].mem_load_commands)
    
    if stage2_cursor >= 0 and stage2_cursor < len(stages):
        with new_parallel_thread("STAGE2"):
            stage = stages[stage2_cursor]
            cached_cmds = []
            
            for cmd in stage.execute_commands:
                if isinstance(cmd, MCA_CompiledOperator.Command.NOP):
                    continue
                elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_INIT):
                    for method in op_compute_methods:
                        method(core, env, operator, cached_cmds)
                    cached_cmds = []
                    
                    var = env.variables[cmd.var_name]
                    core.var_atomic_init(var, cmd.initial_value)
                elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_COMPARE_AND_SWAP):
                    for method in op_compute_methods:
                        method(core, env, operator, cached_cmds)
                    cached_cmds = []
                    
                    var = env.variables[cmd.var_name]
                    core.var_atomic_compare_and_swap(var, cmd.expected_value, cmd.new_value)
                elif isinstance(cmd, MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT):
                    for method in op_compute_methods:
                        method(core, env, operator, cached_cmds)
                    cached_cmds = []
                    
                    vars = [env.variables[var_name] for var_name in cmd.var_names]
                    core.var_conditional_wait(vars, cmd.condition)
                else:
                    cached_cmds.append(cmd)
                    
            if len(cached_cmds) > 0:
                for method in op_compute_methods:
                    method(core, env, operator, cached_cmds)
                cached_cmds = []
            
            # for method in op_compute_methods:
            #     method(core, env, operator, stages[stage2_cursor])
            
    if stage3_cursor >= 0 and stage3_cursor < len(stages):
        with new_parallel_thread("STAGE3"):
            MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core, env, operator, stages[stage3_cursor].mem_store_commands)
            MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core, env, operator, stages[stage3_cursor].postprocessing_commands)

    core.parallel_merge()