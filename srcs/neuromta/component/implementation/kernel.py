from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *


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
def MCA_KERNEL_CORE_STAGE_PREPROCESSING(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):                
    for cmd in stage.preprocessings:
        with new_parallel_thread():
            if isinstance(cmd, CompiledCommand.NOP):
                continue
            elif isinstance(cmd, CompiledCommand.VAR_BARRIER):
                core.var_atomic_barrier(cmd.var_arrived_count, cmd.var_block_state, cmd.total_arrivals)
            else:
                raise NotImplementedError(f"Preprocessing command {type(cmd)} is not implemented.")
                
    core.parallel_merge()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.postprocessings:
        with new_parallel_thread():
            if isinstance(cmd, CompiledCommand.NOP):
                continue
            elif isinstance(cmd, CompiledCommand.VAR_BARRIER):
                core.var_atomic_barrier(cmd.var_arrived_count, cmd.var_block_state, cmd.total_arrivals)
            else:
                raise NotImplementedError(f"Postprocessing command {type(cmd)} is not implemented.")
    
    core.parallel_merge()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.dma_stores:
        if isinstance(cmd, CompiledCommand.NOP):
            continue
        elif isinstance(cmd, CompiledCommand.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, CompiledCommand.TILE_STORE):
            tile_sig = cmd.tile_sig
            dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_write_args(*tile_sig.coords)
            
            core.local_mem_copy(dst_ptr, tile_sig.spm_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
            core.mem_init(tile_sig.spm_ptr, tile_sig.buf.tile_size)
        else:
            raise NotImplementedError(f"DMA Store command {type(cmd)} is not implemented.")
    
    core.async_rpc_wait_all()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.dma_loads:
        if isinstance(cmd, CompiledCommand.NOP):
            continue
        elif isinstance(cmd, CompiledCommand.MEM_INIT):
            core.local_mem_init(cmd.ptr, cmd.size)
        elif isinstance(cmd, CompiledCommand.TILE_LOAD):
            tile_sig = cmd.tile_sig
            src_ptr, row_size, row_num, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_read_args(*tile_sig.coords)
            
            if len(cmd.broadcast_dst_ptrs) > 0:  # BROADCAST: broadcast optimization
                target_ptrs = cmd.broadcast_dst_ptrs + [tile_sig.spm_ptr,]
                for ptr in target_ptrs:
                    core.mem_init(ptr, tile_sig.buf.tile_size)
                core.local_mem_broadcast(target_ptrs, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
            else:
                core.mem_init(tile_sig.spm_ptr, tile_sig.buf.tile_size)
                core.local_mem_copy(tile_sig.spm_ptr, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=True)
        elif isinstance(cmd, CompiledCommand.COLLECTIVE_TILE_LOAD):
            collective_tile_sig = cmd.collective_tile_sig
            core.mem_init(collective_tile_sig.spm_ptr, collective_tile_sig.buf.tile_size)
            
            for tile_sig, memcpy_pattern in zip(collective_tile_sig.src_tiles, collective_tile_sig.memcpy_patterns):
                src_ptr, row_size, row_num, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_read_args(*tile_sig.coords)
                
                for dst_row_idx, src_row_idx in memcpy_pattern.items():
                    dst_row_ptr = collective_tile_sig.spm_ptr + dst_row_idx * dst_row_stride
                    src_row_ptr = src_ptr + src_row_idx * src_row_stride

                    core.local_mem_copy(dst_row_ptr, src_row_ptr, row_size, 1, src_row_stride, dst_row_stride, nowait=True)
        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()


@jit_prototype
def MCA_OP_CORE_TEMPLATE(core: NPUCore, operator: CompiledOperator, stage: CompiledStage, op_compute_methods: list[Callable]):
    MCA_KERNEL_CORE_STAGE_PREPROCESSING(core, operator, stage)
    
    with new_parallel_thread("DMA_LOAD"):
        MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core, operator, stage)
        
    with new_parallel_thread("DMA_STORE"):
        MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core, operator, stage)
            
    with new_parallel_thread("COMPUTE"):
        for method in op_compute_methods:
            method(core, operator, stage)
            
    core.parallel_merge()
    
    MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core, operator, stage)
