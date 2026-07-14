import torch
from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *
from neuromta.component.implementation.kernel import *


__all__ = [
    "MCA_KERNEL_TILED_LINEAR",
    "MCA_KERNEL_TILED_RELU",
    "MCA_KERNEL_MERGED_LINEAR_RELU",
    "MCA_KERNEL_TILED_CONV2D",
    "MCA_KERNEL_TILED_MAXPOOL2D",
    "MCA_KERNEL_TILED_AVGPOOL2D",
    "MCA_KERNEL_DIRECT_COPY",
]


class _MCA_KERNEL_BASE(MCA_KernelTemplate):
    @classmethod
    def EXE_CTX_LOAD(
        cls,
        core: NPUCore, 
        env: MCA_OperatorGraphCompiler.Environment, 
        ir: MCA_CompiledOperator.IR.EXE_CTX_LOAD,
    ):
        container = cls.read_from_ref(core, env, ir.ref)
        core.mxu_load_context(container)
    
    @classmethod
    def EXE_CTX_STORE(
        cls,
        core: NPUCore, 
        env: MCA_OperatorGraphCompiler.Environment, 
        ir: MCA_CompiledOperator.IR.EXE_CTX_STORE,
    ):    
        container = DataContainer()
        core.mxu_store_context(container)
        cls.write_to_ref(core, env, container, ir.ref)


class MCA_KERNEL_TILED_LINEAR(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        op_sig = env.op_meta[ir.op_id].op_sig
        tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]
        
        uop_kwargs = tiled_op.op_kwargs[ir.uop_idx]
        use_bias = uop_kwargs.get("use_bias", False)
        preload_psum = (ir.uop_idx == 0)
        flush_ofm    = (ir.uop_idx == tiled_op.n_uops - 1)
        
        dtype = ir.dtype
        acc_dtype = ir.acc_dtype
        
        if ir.uop_idx == 0:
            core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
        
        ifm = cls.read_from_ref(core, env, ir.i_tile_refs[0])
        wgt = cls.read_from_ref(core, env, ir.i_tile_refs[1])
        psum = cls.read_from_ref(core, env, ir.i_tile_refs[2]) if preload_psum and use_bias else None
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype) if flush_ofm else None

        core.mxu_tiled_gemm(
            ifm, wgt, psum, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            cls.write_to_ref(core, env, ofm, ir.o_tile_ref)


class MCA_KERNEL_TILED_RELU(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        ifm = cls.read_from_ref(core, env, ir.i_tile_refs[0])
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype)
        
        vlen        = ir.i_tile_refs[0].tile_sig.tile_shape[1]
        burst_len   = ir.i_tile_refs[0].tile_sig.tile_shape[0]
        vdtype      = ir.i_tile_refs[0].tile_sig.dtype
        n_vreg_num  = core.vpu_context.get_vreg_num_with_config(vlen=vlen, vdtype=vdtype)
        
        if n_vreg_num < burst_len:
            raise Exception(f"VPU register number ({n_vreg_num}) is insufficient for burst length ({burst_len}).")  # TODO: implement split burst if insufficient vreg
        
        if ir.uop_idx == 0:
            core.vpu_reconfigure(vlen=vlen, vdtype=vdtype)
        
        core.vpu_load_reg(ifm, 0, burst_len=burst_len, offset=0)
        core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
        core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)
        
        cls.write_to_ref(core, env, ofm, ir.o_tile_ref)
        
        
class MCA_KERNEL_MERGED_LINEAR_RELU(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        op_sig = env.op_meta[ir.op_id].op_sig
        tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]
        
        preload_psum = (ir.uop_idx == 0)
        flush_ofm    = (ir.uop_idx == tiled_op.n_uops - 1)
        
        dtype = ir.dtype
        acc_dtype = ir.acc_dtype
        
        vlen        = ir.i_tile_refs[1].tile_sig.tile_shape[0]  # wgt height
        burst_len   = ir.i_tile_refs[0].tile_sig.tile_shape[0]  # ifm height
        vdtype      = acc_dtype
        
        if ir.uop_idx == 0:
            core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
            core.vpu_reconfigure(vlen=vlen, vdtype=vdtype)
        
        ifm = cls.read_from_ref(core, env, ir.i_tile_refs[0])
        wgt = cls.read_from_ref(core, env, ir.i_tile_refs[1])
        psum = cls.read_from_ref(core, env, ir.i_tile_refs[2]) if preload_psum else None
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype) if flush_ofm else None
        
        core.mxu_tiled_gemm(
            ifm, wgt, psum, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            burst_len = ir.o_tile_ref.tile_sig.tile_shape[0]
            
            core.vpu_load_reg(ofm, 0, burst_len=burst_len, offset=0)
            core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
            core.vpu_store_reg(ofm, 0, burst_len=burst_len, offset=0)
            
            cls.write_to_ref(core, env, ofm, ir.o_tile_ref)
            
            
class MCA_KERNEL_TILED_CONV2D(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        op_sig = env.op_meta[ir.op_id].op_sig
        tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]

        uop_kwargs = tiled_op.op_kwargs[ir.uop_idx]
        
        use_bias = uop_kwargs.get("use_bias", False)
        ifm_tile_count: int = uop_kwargs["ifm_tile_count"]
        ifm_sig_arr = tiled_op.i_tiles[ir.uop_idx][:ifm_tile_count]
        ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
        
        preload_psum = (ir.uop_idx == 0) and use_bias
        flush_ofm    = (ir.uop_idx == tiled_op.n_uops - 1)
        
        dtype = ir.dtype
        acc_dtype = ir.acc_dtype
        
        if ir.uop_idx == 0:
            core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
        
        ifm = DataContainer(shape=ir.i_tile_refs[0].tile_sig.tile_shape, dtype=ir.i_tile_refs[0].tile_sig.dtype)
        for i, (ifm_sig, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_memcpy_pattern_arr)):
            cls.read_from_ref(core, env, ir.i_tile_refs[i], row_pattern=ifm_memcpy_pattern, inplace_container=ifm)  # Reuse the same container for each IFM tile read to save memory
        wgt = cls.read_from_ref(core, env, ir.i_tile_refs[ifm_tile_count])    
        psum = cls.read_from_ref(core, env, ir.i_tile_refs[ifm_tile_count + 1]) if preload_psum else None
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype) if flush_ofm else None
            
        core.mxu_tiled_gemm(
            ifm, wgt, psum, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,
            psum_vectored=True,
        )
        
        if flush_ofm:
            cls.write_to_ref(core, env, ofm, ir.o_tile_ref)
            
            
class MCA_KERNEL_TILED_MAXPOOL2D(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        op_sig = env.op_meta[ir.op_id].op_sig
        tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]
        
        preload_psum = (ir.uop_idx == 0)
        flush_ofm    = (ir.uop_idx == tiled_op.n_uops - 1)
        
        uop_kwargs = tiled_op.op_kwargs[ir.uop_idx]
        
        ifm_tile_count = uop_kwargs["ifm_tile_count"]
        ifm_sig_arr = tiled_op.i_tiles[ir.uop_idx][:ifm_tile_count]
        ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
        
        dtype = ir.dtype
        acc_dtype = ir.acc_dtype
        
        ifm = DataContainer(shape=ir.i_tile_refs[0].tile_sig.tile_shape, dtype=ir.i_tile_refs[0].tile_sig.dtype)
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype) if flush_ofm else None
        
        if ir.uop_idx == 0:
            core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
            
        for i, (ifm_sig, ifm_memcpy_pattern) in enumerate(zip(ifm_sig_arr, ifm_memcpy_pattern_arr)):
            cls.read_from_ref(core, env, ir.i_tile_refs[i], row_pattern=ifm_memcpy_pattern, inplace_container=ifm)
            
        core.mxu_tiled_maxpool(
            ifm, ifm, ofm,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
        )
        
        if flush_ofm:
            cls.write_to_ref(core, env, ofm, ir.o_tile_ref)


class MCA_KERNEL_TILED_AVGPOOL2D(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        op_sig = env.op_meta[ir.op_id].op_sig
        tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]
        
        preload_psum = (ir.uop_idx == 0)
        flush_ofm    = (ir.uop_idx == tiled_op.n_uops - 1)
        
        uop_kwargs = tiled_op.op_kwargs[ir.uop_idx]
        
        ifm_memcpy_pattern_arr = uop_kwargs["memcpy_pattern"]
        
        ifm = DataContainer(shape=ir.i_tile_refs[0].tile_sig.tile_shape, dtype=ir.i_tile_refs[0].tile_sig.dtype)
        ofm = DataContainer(shape=ir.o_tile_ref.tile_sig.tile_shape, dtype=ir.o_tile_ref.tile_sig.dtype) if flush_ofm else None
        
        dtype = ir.dtype
        acc_dtype = ir.acc_dtype
        
        if ir.uop_idx == 0:
            core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
            
        for i, (ref, ifm_memcpy_pattern) in enumerate(zip(ir.i_tile_refs, ifm_memcpy_pattern_arr)):
            cls.read_from_ref(core, env, ref, row_pattern=ifm_memcpy_pattern, inplace_container=ifm)
            
        core.mxu_tiled_elemwise(
            op=MXUElementwiseOp.ADD,
            src=ifm,
            dst=ofm,
            preload_psum=preload_psum,
            flush_ofm=False,
        )
        
        if flush_ofm:
            core.mxu_tiled_elemwise_imm(
                op=MXUElementwiseOp.DIV,
                imm=tiled_op.n_uops,
                dst=ofm,
                flush_ofm=True
            )
            
            cls.write_to_ref(core, env, ofm, ir.o_tile_ref)
            
            
class MCA_KERNEL_DIRECT_COPY(_MCA_KERNEL_BASE):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        container = cls.read_from_ref(core, env, ir.i_tile_refs[0])
        cls.write_to_ref(core, env, container, ir.o_tile_ref)
