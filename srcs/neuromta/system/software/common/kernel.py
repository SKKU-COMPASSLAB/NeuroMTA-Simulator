from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
# from neuromta.component.implementation.kernel import MCA_OP_CORE_TEMPLATE
from neuromta.component.implementation.operator import *
# import torch


__all__ = [
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D",
    "MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY",
]


@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):      
        ifm_sig, ifm_ptr = cmd.i_tiles[0]
        wgt_sig, wgt_ptr = cmd.i_tiles[1]
        bias_sig, bias_ptr = cmd.i_tiles[2]
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm  = DataContainer(shape=ifm_sig.tile_shape, dtype=ifm_sig.dtype)
        wgt  = DataContainer(shape=wgt_sig.tile_shape, dtype=wgt_sig.dtype)
        bias = DataContainer(shape=bias_sig.tile_shape, dtype=bias_sig.dtype)
        ofm  = DataContainer(shape=ofm_sig.tile_shape, dtype=ofm_sig.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == cmd.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig.dtype, acc_dtype=ofm_sig.dtype)
        
        core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_size)
        core.local_mem_page_read(wgt_ptr, wgt, wgt_sig.tile_size)
        if preload_psum:
            core.local_mem_page_read(bias_ptr, bias, bias_sig.tile_size)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            
@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        ifm_sig, ifm_ptr = cmd.i_tiles[0]
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm = DataContainer(shape=ifm_sig.tile_shape, dtype=ifm_sig.dtype)
        ofm = DataContainer(shape=ofm_sig.tile_shape, dtype=ofm_sig.dtype)
        
        core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_size)
        
        vlen        = ifm_sig.tile_shape[1]
        burst_len   = ifm_sig.tile_shape[0]
        vdtype      = ifm_sig.dtype
        n_vreg_num  = core.vpu_context.get_vreg_num_with_config(vlen=vlen, vdtype=vdtype)
        
        if n_vreg_num < burst_len:
            raise Exception(f"VPU register number ({n_vreg_num}) is insufficient for burst length ({burst_len}).")  # TODO: implement split burst if insufficient vreg
        
        if cmd.uop_idx == 0:
            core.vpu_reconfigure(vlen=vlen, vdtype=ifm_sig.dtype)
        
        core.vpu_load_reg(ifm, 0, burst_len=burst_len, offset=0)
        core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
        core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)
        
        core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        
@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        ifm_sig, ifm_ptr = cmd.i_tiles[0]
        wgt_sig, wgt_ptr = cmd.i_tiles[1]
        bias_sig, bias_ptr = cmd.i_tiles[2]
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm  = DataContainer(shape=ifm_sig.tile_shape, dtype=ifm_sig.dtype)
        wgt  = DataContainer(shape=wgt_sig.tile_shape, dtype=wgt_sig.dtype)
        bias = DataContainer(shape=bias_sig.tile_shape, dtype=bias_sig.dtype)
        ofm  = DataContainer(shape=ofm_sig.tile_shape, dtype=ofm_sig.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == cmd.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig.dtype, acc_dtype=ofm_sig.dtype)
            core.vpu_reconfigure(vlen=ofm_sig.tile_shape[1], vdtype=ofm_sig.dtype)
        
        core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_size)
        core.local_mem_page_read(wgt_ptr, wgt, wgt_sig.tile_size)
        if preload_psum:
            core.local_mem_page_read(bias_ptr, bias, bias_sig.tile_size)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            burst_len = ofm_sig.tile_shape[0]
            
            core.vpu_load_reg(ofm, 0, burst_len=burst_len, offset=0)
            core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
            core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)

            core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype    
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        ifm_tile_count: int = cmd.uop_kwargs["ifm_tile_count"]
        ifm_memcpy_pattern_arr: list[dict] = cmd.uop_kwargs["memcpy_pattern"]
        
        ifm_sig_arr = [i_tile_sig for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        ifm_ptr_arr = [i_tile_ptr for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        
        wgt_sig, wgt_ptr = cmd.i_tiles[ifm_tile_count]
        bias_sig, bias_ptr = cmd.i_tiles[ifm_tile_count + 1]
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm  = DataContainer(shape=ifm_sig_arr[0].tile_shape,  dtype=ifm_sig_arr[0].dtype)
        wgt  = DataContainer(shape=wgt_sig.tile_shape,  dtype=wgt_sig.dtype)
        bias = DataContainer(shape=bias_sig.tile_shape, dtype=bias_sig.dtype)
        ofm  = DataContainer(shape=ofm_sig.tile_shape,  dtype=ofm_sig.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == cmd.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig_arr[0].dtype, acc_dtype=ofm_sig.dtype)
        
        for i, (ifm_sig, ifm_ptr, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_ptr_arr, ifm_memcpy_pattern_arr)):
            core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_shape[-1] * ifm_sig.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.local_mem_page_read(wgt_ptr, wgt, wgt_sig.tile_size)
        if preload_psum:
            core.local_mem_page_read(bias_ptr, bias, bias_sig.tile_size)
        
        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        ifm_tile_count: int = cmd.uop_kwargs["ifm_tile_count"]
        ifm_memcpy_pattern_arr: list[dict] = cmd.uop_kwargs["memcpy_pattern"]
        
        ifm_sig_arr = [i_tile_sig for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        ifm_ptr_arr = [i_tile_ptr for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm  = DataContainer(shape=ifm_sig_arr[0].tile_shape,  dtype=ifm_sig_arr[0].dtype)
        ofm  = DataContainer(shape=ofm_sig.tile_shape,  dtype=ofm_sig.dtype)

        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == cmd.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig_arr[0].dtype, acc_dtype=ofm_sig.dtype)
        
        for i, (ifm_sig, ifm_ptr, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_ptr_arr, ifm_memcpy_pattern_arr)):
            core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_shape[-1] * ifm_sig.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.mxu_tiled_maxpool(
            ifm, ifm, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
        )
        
        if flush_ofm:
            core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        ifm_tile_count = cmd.uop_kwargs["ifm_tile_count"]
        ifm_memcpy_pattern_arr = cmd.uop_kwargs["memcpy_pattern"]
        ifm_sig_arr = [i_tile_sig for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        ifm_ptr_arr = [i_tile_ptr for i_tile_sig, i_tile_ptr in cmd.i_tiles[:ifm_tile_count]]
        
        ofm_sig, ofm_ptr = cmd.o_tile
        
        ifm  = DataContainer(shape=ifm_sig_arr[0].tile_shape,  dtype=ifm_sig_arr[0].dtype)
        ofm  = DataContainer(shape=ofm_sig.tile_shape,  dtype=ofm_sig.dtype)

        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == cmd.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_sig_arr[0].dtype, acc_dtype=ofm_sig.dtype)
        
        for i, (ifm_sig, ifm_ptr, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_ptr_arr, ifm_memcpy_pattern_arr)):
            core.local_mem_page_read(ifm_ptr, ifm, ifm_sig.tile_shape[-1] * ifm_sig.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
    
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
                imm=cmd.n_uops,
                dst=ofm,
                flush_ofm=True
            )
            
            core.local_mem_page_write(ofm_ptr, ofm, ofm_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY(core: NPUCore, variables: dict[str, VariableHandle], buffers: dict[str, MCA_TensorBuffer], cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        i_tile_sig, i_tile_ptr = cmd.i_tiles[0]
        o_tile_sig, o_tile_ptr = cmd.o_tile
        
        container = DataContainer(shape=i_tile_sig.tile_shape, dtype=i_tile_sig.dtype)
        
        core.local_mem_page_read(i_tile_ptr, container, i_tile_sig.tile_size)
        core.local_mem_page_write(o_tile_ptr, container, o_tile_sig.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")