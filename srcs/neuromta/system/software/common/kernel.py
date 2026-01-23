from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *


__all__ = [
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY",
]


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
        
        uop_kwargs = op_sig.op_kwargs["ifm_load_kwargs"][inner_op_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = op_sig.i_tiles[inner_op_idx][0]
            ifm_tile_shape = ifm_sig.buf.tile_shape
            ifm_tile_dtype = ifm_sig.buf.dtype
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = op_sig.i_tiles[inner_op_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_tile_shape = ifm_sig_arr[0].buf.tile_shape  # TODO: infer IFM tile shape from OFM tile shape and Conv2d params
            ifm_tile_dtype = ifm_sig_arr[0].buf.dtype
            
        wgt_sig  = op_sig.i_tiles[inner_op_idx][ifm_tile_count]
        bias_sig = op_sig.i_tiles[inner_op_idx][ifm_tile_count + 1]
        ofm_sig = op_sig.o_tile
        
        ifm  = DataContainer(shape=ifm_tile_shape,          dtype=ifm_tile_dtype)
        wgt  = DataContainer(shape=wgt_sig.buf.tile_shape,  dtype=wgt_sig.buf.dtype)
        bias = DataContainer(shape=bias_sig.buf.tile_shape, dtype=bias_sig.buf.dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape,  dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_tile_dtype, acc_dtype=ofm_sig.buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_sig.buf.tile_size)
        else:
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
        
        uop_kwargs = op_sig.op_kwargs["ifm_load_kwargs"][inner_op_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = op_sig.i_tiles[inner_op_idx][0]
            ifm_tile_shape = ifm_sig.buf.tile_shape
            ifm_tile_dtype = ifm_sig.buf.dtype
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = op_sig.i_tiles[inner_op_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_tile_shape = ifm_sig_arr[0].buf.tile_shape  # TODO: infer IFM tile shape from OFM tile shape and Conv2d params
            ifm_tile_dtype = ifm_sig_arr[0].buf.dtype
            
        ofm_sig = op_sig.o_tile
        
        ifm  = DataContainer(shape=ifm_tile_shape,          dtype=ifm_tile_dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape,  dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_tile_dtype, acc_dtype=ofm_sig.buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_sig.buf.tile_size)
        else:
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
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        uop_kwargs = op_sig.op_kwargs["ifm_load_kwargs"][inner_op_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = op_sig.i_tiles[inner_op_idx][0]
            ifm_tile_shape = ifm_sig.buf.tile_shape
            ifm_tile_dtype = ifm_sig.buf.dtype
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = op_sig.i_tiles[inner_op_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_tile_shape = ifm_sig_arr[0].buf.tile_shape  # TODO: infer IFM tile shape from OFM tile shape and Conv2d params
            ifm_tile_dtype = ifm_sig_arr[0].buf.dtype
            
        ofm_sig = op_sig.o_tile
        
        ifm  = DataContainer(shape=ifm_tile_shape,          dtype=ifm_tile_dtype)
        ofm  = DataContainer(shape=ofm_sig.buf.tile_shape,  dtype=ofm_sig.buf.dtype)
        
        preload_psum = (inner_op_idx == 0)
        flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
        
        if inner_op_idx == 0:
            core.mxu_reconfigure(dtype=ifm_tile_dtype, acc_dtype=ofm_sig.buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_sig.buf.tile_size)
        else:
            for ifm_sig, ifm_memcpy_pattern in zip(ifm_sig_arr, ifm_memcpy_pattern_arr):
                core.local_mem_page_read(ifm_sig.spm_ptr, ifm, ifm_tile_shape[-1] * ifm_tile_dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.mxu_tiled_elemwise(
            op=MXUElementwiseOp.ADD,
            src=ifm,
            dst=ofm,
            preload_psum=preload_psum,
            flush_ofm=False,  # flush later after division
        )
        
        if flush_ofm:
            core.mxu_tiled_elemwise_imm(
                op=MXUElementwiseOp.DIV,
                imm=len(op_sig.i_tiles),
                dst=ofm,
                flush_ofm=True
            )
            
            core.local_mem_page_write(ofm_sig.spm_ptr, ofm, ofm_sig.buf.tile_size)
            

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY(core: NPUCore, operator: CompiledOperator, stage: CompiledStage):
    for cmd in stage.compute_ops:
        if not isinstance(cmd, CompiledCommand.TILED_OP):
            raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        op_sig = cmd.op_sig
        inner_op_idx = cmd.inner_op_idx
        
        src_sig = op_sig.i_tiles[inner_op_idx][0]
        dst_sig = op_sig.o_tile
        
        container = DataContainer(shape=src_sig.buf.tile_shape, dtype=src_sig.buf.dtype)
        
        core.local_mem_page_read(src_sig.spm_ptr, container, src_sig.buf.tile_size)
        core.local_mem_page_write(dst_sig.spm_ptr, container, dst_sig.buf.tile_size)