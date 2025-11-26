from typing import Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *


__all__ = [
    # KERNEL CORE OP
    "MCA_KERNEL_CORE_OP_LINEAR",
]

@jit_prototype
def MCA_KERNEL_CORE_STAGE_PREPROCESSING(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):                
    for thread in stage.preprocessings:
        with new_parallel_thread():
            for cmd in thread:
                if isinstance(cmd, CompiledCommand.NOP):
                    continue
                elif isinstance(cmd, CompiledCommand.VAR_BARRIER):
                    core.var_atomic_barrier(cmd.var_arrived_count, cmd.var_block_state, cmd.total_arrivals)
                else:
                    raise NotImplementedError(f"Preprocessing command {type(cmd)} is not implemented.")
                
    core.parallel_merge()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for thread in stage.postprocessings:
        with new_parallel_thread():
            for cmd in thread:
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
        else:
            raise NotImplementedError(f"DMA Load command {type(cmd)} is not implemented.")
        
    core.async_rpc_wait_all()

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        ifm_sig = op_sig.i_tiles[inner_op_idx][0]
        wgt_sig = op_sig.i_tiles[inner_op_idx][1]
        bias_sig = op_sig.i_tiles[inner_op_idx][2]
        ofm_sig = op_sig.o_tile
        
        ifm  = DataContainer(shape=ifm_sig.buf.tile_shape, dtype=ifm_sig.buf.dtype)
        wgt  = DataContainer(shape=wgt_sig.buf.tile_shape, dtype=wgt_sig.buf.dtype)
        bias = DataContainer(shape=bias_sig.buf.tile_shape, dtype=bias_sig.buf.dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape, dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig.buf.dtype, acc_dtype=ofm_sig.buf.dtype)
        
        core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_sig.buf.tile_size)
        core.local_mem_page_read(wgt_sig.spm_ptr, wgt, wgt_sig.buf.tile_size)
        if preload_psum:
            core.local_mem_page_read(bias_sig.spm_ptr, bias, bias_sig.buf.tile_size)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)

@jit_prototype
def MCA_KERNEL_CORE_OP_LINEAR(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    # for stage in operator.stages:
    MCA_KERNEL_CORE_STAGE_PREPROCESSING(core, operator, stage)
    
    with new_parallel_thread("DMA_LOAD"):
        MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core, operator, stage)
        
    with new_parallel_thread("DMA_STORE"):
        MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core, operator, stage)
            
    with new_parallel_thread("COMPUTE"):
        MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core, operator, stage)
            
    core.parallel_merge()
    
    MCA_KERNEL_CORE_STAGE_POSTPROCESSING(core, operator, stage)
