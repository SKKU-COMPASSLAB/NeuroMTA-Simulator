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
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_LINEAR(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
        wgt_sig = tiled_op.i_tiles[cmd.uop_idx][1]
        bias_sig = tiled_op.i_tiles[cmd.uop_idx][2]
        ofm_sig = tiled_op.o_tile
        
        ifm_buf = env.buffers[ifm_sig.buf_name]
        wgt_buf = env.buffers[wgt_sig.buf_name]
        bias_buf = env.buffers[bias_sig.buf_name]
        ofm_buf = env.buffers[ofm_sig.buf_name]
        
        ifm  = DataContainer(shape=ifm_buf.tile_shape, dtype=ifm_buf.dtype)
        wgt  = DataContainer(shape=wgt_buf.tile_shape, dtype=wgt_buf.dtype)
        bias = DataContainer(shape=bias_buf.tile_shape, dtype=bias_buf.dtype)
        ofm  = DataContainer(shape=ofm_buf.tile_shape, dtype=ofm_buf.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == tiled_op.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_buf.dtype, acc_dtype=ofm_buf.dtype)
        
        core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        core.local_mem_page_read(cmd.i_tile_ptrs[1], wgt, wgt_buf.tile_size)
        if preload_psum:
            core.local_mem_page_read(cmd.i_tile_ptrs[2], bias, bias_buf.tile_size)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
            
            # def debug_func(ofm_sig: TileSignature, container: DataContainer):
            #     t: torch.Tensor = container.data
            #     print(ofm_sig.signature)
            #     print(t.view(torch.int16).reshape(ofm_buf.tile_shape))
            # core.debug_core_with_ambiguous_func(debug_func, ofm_sig, ofm)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            
@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_RELU(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
        ofm_sig = tiled_op.o_tile
        
        ifm_buf = env.buffers[ifm_sig.buf_name]
        ofm_buf = env.buffers[ofm_sig.buf_name]
        
        ifm = DataContainer(shape=ifm_buf.tile_shape, dtype=ifm_buf.dtype)
        ofm = DataContainer(shape=ofm_buf.tile_shape, dtype=ofm_buf.dtype)
        
        core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        
        vlen        = ifm_buf.tile_shape[1]
        burst_len   = ifm_buf.tile_shape[0]
        vdtype      = ifm_buf.dtype
        n_vreg_num  = core.vpu_context.get_vreg_num_with_config(vlen=vlen, vdtype=vdtype)
        
        if n_vreg_num < burst_len:
            raise Exception(f"VPU register number ({n_vreg_num}) is insufficient for burst length ({burst_len}).")  # TODO: implement split burst if insufficient vreg
        
        if cmd.uop_idx == 0:
            core.vpu_reconfigure(vlen=vlen, vdtype=ifm_buf.dtype)
        
        core.vpu_load_reg(ifm, 0, burst_len=burst_len, offset=0)
        core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
        core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)
        
        core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
        
        
@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MERGED_LINEAR_RELU(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
        wgt_sig = tiled_op.i_tiles[cmd.uop_idx][1]
        bias_sig = tiled_op.i_tiles[cmd.uop_idx][2]
        ofm_sig = tiled_op.o_tile
        
        ifm_buf = env.buffers[ifm_sig.buf_name]
        wgt_buf = env.buffers[wgt_sig.buf_name]
        bias_buf = env.buffers[bias_sig.buf_name]
        ofm_buf = env.buffers[ofm_sig.buf_name]
        
        ifm  = DataContainer(shape=ifm_buf.tile_shape, dtype=ifm_buf.dtype)
        wgt  = DataContainer(shape=wgt_buf.tile_shape, dtype=wgt_buf.dtype)
        bias = DataContainer(shape=bias_buf.tile_shape, dtype=bias_buf.dtype)
        ofm  = DataContainer(shape=ofm_buf.tile_shape, dtype=ofm_buf.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == tiled_op.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_buf.dtype, acc_dtype=ofm_buf.dtype)
            core.vpu_reconfigure(vlen=ofm_buf.tile_shape[1], vdtype=ofm_buf.dtype)
        
        core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        core.local_mem_page_read(cmd.i_tile_ptrs[1], wgt, wgt_buf.tile_size)
        if preload_psum:
            core.local_mem_page_read(cmd.i_tile_ptrs[2], bias, bias_buf.tile_size)

        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            burst_len = ofm_buf.tile_shape[0]
            
            core.vpu_load_reg(ofm, 0, burst_len=burst_len, offset=0)
            core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
            core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)

            core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype    
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_CONV2D(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        uop_kwargs = tiled_op.op_kwargs[cmd.uop_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
            ifm_buf  = env.buffers[ifm_sig.buf_name]
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = tiled_op.i_tiles[cmd.uop_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_buf  = env.buffers[ifm_sig_arr[0].buf_name]
            
        wgt_sig  = tiled_op.i_tiles[cmd.uop_idx][ifm_tile_count]
        bias_sig = tiled_op.i_tiles[cmd.uop_idx][ifm_tile_count + 1]
        ofm_sig  = tiled_op.o_tile
        
        wgt_buf  = env.buffers[wgt_sig.buf_name]
        bias_buf = env.buffers[bias_sig.buf_name]
        ofm_buf  = env.buffers[ofm_sig.buf_name]
        
        ifm  = DataContainer(shape=ifm_buf.tile_shape,  dtype=ifm_buf.dtype)
        wgt  = DataContainer(shape=wgt_buf.tile_shape,  dtype=wgt_buf.dtype)
        bias = DataContainer(shape=bias_buf.tile_shape, dtype=bias_buf.dtype)
        ofm  = DataContainer(shape=ofm_buf.tile_shape,  dtype=ofm_buf.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == tiled_op.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_buf.dtype, acc_dtype=ofm_buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        else:
            for i, (ifm_sig, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_memcpy_pattern_arr)):
                core.local_mem_page_read(cmd.i_tile_ptrs[i], ifm, ifm_buf.tile_shape[-1] * ifm_buf.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.local_mem_page_read(cmd.i_tile_ptrs[ifm_tile_count], wgt, wgt_buf.tile_size)
        if preload_psum:
            core.local_mem_page_read(cmd.i_tile_ptrs[ifm_tile_count + 1], bias, bias_buf.tile_size)
        
        core.mxu_tiled_gemm(
            ifm, wgt, bias, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_MAXPOOL2D(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        uop_kwargs = tiled_op.op_kwargs[cmd.uop_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
            ifm_buf  = env.buffers[ifm_sig.buf_name]
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = tiled_op.i_tiles[cmd.uop_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_buf  = env.buffers[ifm_sig_arr[0].buf_name]
            
        ofm_sig = tiled_op.o_tile
        ofm_buf = env.buffers[ofm_sig.buf_name]
        
        ifm  = DataContainer(shape=ifm_buf.tile_shape,  dtype=ifm_buf.dtype)
        ofm  = DataContainer(shape=ofm_buf.tile_shape,  dtype=ofm_buf.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == tiled_op.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_buf.dtype, acc_dtype=ofm_buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        else:
            for i, (ifm_sig, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_memcpy_pattern_arr)):
                core.local_mem_page_read(cmd.i_tile_ptrs[i], ifm, ifm_buf.tile_shape[-1] * ifm_buf.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
        
        core.mxu_tiled_maxpool(
            ifm, ifm, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
        )
        
        if flush_ofm:
            core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_TILED_AVGPOOL2D(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.local_mem_page_read(cmd.ptr, cont, buf.tile_size)
        core.mxu_load_context(cont)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        buf = env.buffers[cmd.tile_sig.buf_name]
        cont  = DataContainer(shape=buf.tile_shape, dtype=buf.dtype)
        
        core.mxu_store_context(cont)
        core.local_mem_page_write(cmd.ptr, cont, buf.tile_size)
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        uop_kwargs = tiled_op.op_kwargs[cmd.uop_idx]
        use_collective_tile_load = uop_kwargs.get("use_collective_tile_load", False)
        
        if use_collective_tile_load:
            ifm_tile_count = 1
            ifm_sig = tiled_op.i_tiles[cmd.uop_idx][0]
            ifm_buf  = env.buffers[ifm_sig.buf_name]
        else:
            ifm_tile_count = uop_kwargs["ifm_tile_count"]
            ifm_sig_arr = tiled_op.i_tiles[cmd.uop_idx][:ifm_tile_count]
            ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
            ifm_buf  = env.buffers[ifm_sig_arr[0].buf_name]
            
        ofm_sig = tiled_op.o_tile
        ofm_buf = env.buffers[ofm_sig.buf_name]
        
        ifm  = DataContainer(shape=ifm_buf.tile_shape,  dtype=ifm_buf.dtype)
        ofm  = DataContainer(shape=ofm_buf.tile_shape,  dtype=ofm_buf.dtype)
        
        preload_psum = (cmd.uop_idx == 0)
        flush_ofm    = (cmd.uop_idx == tiled_op.n_uops - 1)
        
        if cmd.uop_idx == 0:
            core.mxu_reconfigure(dtype=ifm_buf.dtype, acc_dtype=ofm_buf.dtype)
        
        if use_collective_tile_load:
            core.local_mem_page_read(cmd.i_tile_ptrs[0], ifm, ifm_buf.tile_size)
        else:
            for i, (ifm_sig, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_memcpy_pattern_arr)):
                core.local_mem_page_read(cmd.i_tile_ptrs[i], ifm, ifm_buf.tile_shape[-1] * ifm_buf.dtype.itemsize, row_pattern=ifm_memcpy_pattern)
    
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
                imm=tiled_op.n_uops,
                dst=ofm,
                flush_ofm=True
            )
            
            core.local_mem_page_write(cmd.o_tile_ptr, ofm, ofm_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")
            

@jit_prototype
def MCA_KERNEL_CORE_STAGE_COMPUTE_DIRECT_COPY(core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, cmd: MCA_CompiledOperator.Command.Base):
    if isinstance(cmd, MCA_CompiledOperator.Command.EXE_LOAD_CONTEXT):
        raise NotImplementedError("STORE_CONTEXT is not supported in DIRECT_COPY kernel.")
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_STORE_CONTEXT):
        raise NotImplementedError("STORE_CONTEXT is not supported in DIRECT_COPY kernel.")
    elif isinstance(cmd, MCA_CompiledOperator.Command.EXE_UOP):
        op_sig = env.op_meta[cmd.op_id].op_sig
        tiled_op = op_sig.tiled_ops[cmd.tiled_op_idx]
        
        src_buf = env.buffers[tiled_op.i_tiles[cmd.uop_idx][0].buf_name]
        dst_buf = env.buffers[tiled_op.o_tile.buf_name]
        
        container = DataContainer(shape=src_buf.tile_shape, dtype=src_buf.dtype)
        
        core.local_mem_page_read(cmd.i_tile_ptrs[0], container, src_buf.tile_size)
        core.local_mem_page_write(cmd.o_tile_ptr, container, dst_buf.tile_size)
    else:
        raise NotImplementedError(f"Compute command {type(cmd)} is not implemented.")