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
    
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D",

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
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU_INPLACE(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        ifm_sig = op_sig.i_tiles[inner_op_idx][0]
        ofm_sig = op_sig.o_tile
        
        ifm = DataContainer(shape=ifm_sig.buf.tile_shape, dtype=ifm_sig.buf.dtype)
        ofm = DataContainer(shape=ofm_sig.buf.tile_shape, dtype=ofm_sig.buf.dtype)
        
        core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_sig.buf.tile_size)
        core.local_mem_page_read(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)
        
        vlen        = ifm_sig.buf.tile_shape[1]
        burst_len   = ifm_sig.buf.tile_shape[0]
        vdtype      = ifm_sig.buf.dtype
        n_vreg_num  = core.vpu_context.get_vreg_num_with_config(vlen=vlen, vdtype=vdtype)
        
        if n_vreg_num < burst_len:
            raise Exception(f"VPU register number ({n_vreg_num}) is insufficient for burst length ({burst_len}).")  # TODO: implement split burst if insufficient vreg
        
        if inner_op_idx == 0:
            core.vpu_reconfigure(vlen=vlen, vdtype=ifm_sig.buf.dtype)
        
        core.vpu_load_reg(ifm, 0, burst_len=burst_len, offset=0)
        core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
        core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)
        
        core.local_mem_page_write(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)
        
        
@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
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
            core.vpu_reconfigure(vlen=ofm_sig.buf.tile_shape[1], vdtype=ofm_sig.buf.dtype)
        
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
            burst_len = ofm_sig.buf.tile_shape[0]
            
            core.vpu_load_reg(ofm, 0, burst_len=burst_len, offset=0)
            core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
            core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)

            core.local_mem_page_write(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)
            

@jit_prototype    
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        ifm_sig_arr = op_sig.i_tiles[inner_op_idx][2:]
        ifm_memcpy_pattern_arr: list[dict[int, int]] = op_sig.op_metadata[inner_op_idx][2:]
        wgt_sig  = op_sig.i_tiles[inner_op_idx][0]
        bias_sig = op_sig.i_tiles[inner_op_idx][1]
        ofm_sig = op_sig.o_tile
        
        ifm_tile_shape = ofm_sig.buf.tile_shape  # TODO: infer IFM tile shape from OFM tile shape and Conv2d params
        ifm_tile_dtype = ifm_sig_arr[0].buf.dtype
        
        ifm  = DataContainer(shape=ifm_tile_shape,          dtype=ifm_tile_dtype)
        wgt  = DataContainer(shape=wgt_sig.buf.tile_shape,  dtype=wgt_sig.buf.dtype)
        bias = DataContainer(shape=bias_sig.buf.tile_shape, dtype=bias_sig.buf.dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape,  dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_tile_dtype, acc_dtype=ofm_sig.buf.dtype)
        
        for ifm_sig, ifm_memcpy_pattern in zip(ifm_sig_arr, ifm_memcpy_pattern_arr):
            core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_tile_shape[-1] * ifm_tile_dtype.itemsize, row_pattern=ifm_memcpy_pattern)
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
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        ifm_sig_arr = op_sig.i_tiles[inner_op_idx]
        ifm_memcpy_pattern_arr: list[dict[int, int]] = op_sig.op_metadata[inner_op_idx]
        ofm_sig = op_sig.o_tile
        
        ifm_tile_shape = ofm_sig.buf.tile_shape  # TODO: infer IFM tile shape from OFM tile shape and Conv2d params
        ifm_tile_dtype = ifm_sig_arr[0].buf.dtype
        
        ifm  = DataContainer(shape=ifm_tile_shape,          dtype=ifm_tile_dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape,  dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_tile_dtype, acc_dtype=ofm_sig.buf.dtype)
        
        for ifm_sig, ifm_memcpy_pattern in zip(ifm_sig_arr, ifm_memcpy_pattern_arr):
            core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_tile_shape[-1] * ifm_tile_dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.mxu_tiled_maxpool(
            ifm, ifm, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)
            

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

