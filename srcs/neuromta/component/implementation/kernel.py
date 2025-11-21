from typing import Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *


__all__ = [
    # KERNEL CORE STAGE
    "MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST",
    "MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR",
    
    # KERNEL CORE OP
    "MCA_KERNEL_CORE_OP_LINEAR",
]


@jit_prototype
def MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core: NPUCore, stage: CompiledStage):
    for store_cmd in stage.dma_stores:
        tile_sig = store_cmd.tile_sig
        dst_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_write_args(*tile_sig.coords)
        core.local_mem_copy(dst_ptr, tile_sig.spm_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
    
    core.async_rpc_wait_all()

@jit_prototype 
def MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core: NPUCore, stage: CompiledStage, stage_lock: VariableHandle):
    for load_cmd in stage.dma_loads:
        if isinstance(load_cmd, CompiledCommand.OP_LOCK_INCR):
            core.var_atomic_increase(stage_lock, load_cmd._increment)  # BROADCAST: increase the stage lock and wait for the response from the source core
            continue
        
        tile_sig = load_cmd.tile_sig
        src_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_read_args(*tile_sig.coords)
        
        if len(load_cmd.broadcast_dst_ptrs) > 0:  # BROADCAST: broadcast optimization
            core.local_mem_broadcast(load_cmd.broadcast_dst_ptrs + [tile_sig.spm_ptr,], src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
        else:
            core.local_mem_copy(tile_sig.spm_ptr, src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
        
    core.async_rpc_wait_all()
    
    for load_cmd in stage.dma_loads:
        if isinstance(load_cmd, CompiledCommand.TILE_LOAD):
            for lock in load_cmd.broadcast_locks:  # BROADCAST: notify dst cores that the broadcast load is complete
                core.var_atomic_increase(lock, -1)
                
    core.var_atomic_wait(stage_lock, 0)  # BORADCAST: wait for all broadcast loads to complete

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core: NPUCore, stage: CompiledStage):
    for compute_cmd in stage.compute_ops:
        op_sig = compute_cmd.op_sig
        inner_op_idx = compute_cmd.inner_op_idx
        
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
        
        core.local_mem_page_read(ifm_sig.spm_ptr, ifm_sig.buf.tile_size, ifm)
        core.local_mem_page_read(wgt_sig.spm_ptr, wgt_sig.buf.tile_size, wgt)
        if preload_psum:
            core.local_mem_page_read(bias_sig.spm_ptr, bias_sig.buf.tile_size, bias)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_sig.spm_ptr, ofm_sig.buf.tile_size, ofm)

@jit_prototype
def MCA_KERNEL_CORE_OP_LINEAR(core: NPUCore, operator: CompiledOperator):
    for stage in operator.stages:
        with new_parallel_thread("DMA"):
            # DMA STORE
            MCA_KERNEL_CORE_STAGE_DMA_STORE_BURST(core, stage)
            # DMA LOAD
            MCA_KERNEL_CORE_STAGE_DMA_LOAD_BURST(core, stage, stage_lock=operator.stage_lock)
                
        with new_parallel_thread("COMPUTE"):
            # COMPUTE (LINEAR TILED)
            MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core, stage)
                
        core.parallel_merge()
