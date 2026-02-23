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
def MCA_KERNEL_CORE_STAGE_PREPROCESSING(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperatorGraph, stage: MCA_CompiledOperatorGraph.Stage):                
    for cmd in stage.preprocessing_commands:
        with new_parallel_thread():
            if isinstance(cmd, MCA_CompiledOperatorGraph.Command.NOP):
                continue
            elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.BARRIER):
                var_arrived_count = env.variables[cmd.var_arrived_count]
                var_block_state = env.variables[cmd.var_block_state]
                core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
            else:
                raise NotImplementedError(f"Preprocessing command {type(cmd)} is not implemented.")
                
    core.parallel_merge()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperatorGraph, stage: MCA_CompiledOperatorGraph.Stage):
    for cmd in stage.postprocessing_commands:
        with new_parallel_thread():
            if isinstance(cmd, MCA_CompiledOperatorGraph.Command.NOP):
                continue
            elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.BARRIER):
                var_arrived_count = env.variables[cmd.var_arrived_count]
                var_block_state = env.variables[cmd.var_block_state]
                core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
            else:
                raise NotImplementedError(f"Postprocessing command {type(cmd)} is not implemented.")
    
    core.parallel_merge()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperatorGraph, stage: MCA_CompiledOperatorGraph.Stage):
    for cmd in stage.mem_store_commands:
        if isinstance(cmd, MCA_CompiledOperatorGraph.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.async_rpc_wait_all()  # Ensure all previous async operations are completed before the barrier
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_STORE_TILE):
            tile_sig = cmd.tile_sig
            buf = env.buffers[cmd.tile_sig.buf_name]
            dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_write_args(*tile_sig.coords)
            
            core.local_mem_copy(dst_ptr, cmd.ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
            core.mem_init(cmd.ptr, buf.tile_size)
        else:
            raise NotImplementedError(f"DMA Store command {type(cmd)} is not implemented.")
    
    core.async_rpc_wait_all()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperatorGraph, stage: MCA_CompiledOperatorGraph.Stage):
    for cmd in stage.mem_load_commands:
        if isinstance(cmd, MCA_CompiledOperatorGraph.Command.NOP):
            continue
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.BARRIER):
            var_arrived_count = env.variables[cmd.var_arrived_count]
            var_block_state = env.variables[cmd.var_block_state]
            core.async_rpc_wait_all()  # Ensure all previous async operations are completed before the barrier
            core.var_atomic_barrier(var_arrived_count, var_block_state, cmd.total_arrivals)
        elif isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE):
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
                
                if len(cmd.ptrs) > 0:  # BROADCAST: broadcast optimization
                    core.local_mem_broadcast(cmd.ptrs, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
                else:
                    core.local_mem_copy(cmd.ptrs[0], src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)

        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()


@jit_prototype
def MCA_OP_CORE_TEMPLATE(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, operator: MCA_CompiledOperatorGraph, stage1_cursor: int, stage2_cursor: int, stage3_cursor: int, op_compute_methods: list[Callable]):
    stages = operator.mappings[core.core_id]
    
    if stage1_cursor >= 0 and stage1_cursor < len(stages):
        with new_parallel_thread("STAGE1"):
            MCA_KERNEL_CORE_STAGE_PREPROCESSING(core, env, operator, stages[stage1_cursor])
            MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core, env, operator, stages[stage1_cursor])
    
    if stage2_cursor >= 0 and stage2_cursor < len(stages):
        with new_parallel_thread("STAGE2"):
            for method in op_compute_methods:
                method(core, env, operator, stages[stage2_cursor])
            
    if stage3_cursor >= 0 and stage3_cursor < len(stages):
        with new_parallel_thread("STAGE3"):
            MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core, env, operator, stages[stage3_cursor])
            MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core, env, operator, stages[stage3_cursor])
            
    core.parallel_merge()