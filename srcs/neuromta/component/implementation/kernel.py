from typing import Callable

from matplotlib import container
from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.mapping import TileSignature
from neuromta.component.implementation.operator import *


__all__ = [
    "MCA_KernelTemplate",
]


class MCA_KernelTemplate:
    def get_ld_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.LD_THREAD,
            args=(env, stage),
            kwargs={}
        )
        
    def get_ex_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.EX_THREAD,
            args=(env, stage),
            kwargs={}
        )
        
    def get_st_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.ST_THREAD,
            args=(env, stage),
            kwargs={}
        )
        
    def get_barrier_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, barrier: tuple[str, str, int]) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.BARRIER,
            args=(env, barrier),
            kwargs={}
        )
    
    @classmethod
    def read_from_ref(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ref: MCA_CompiledOperator.IR.Reference, row_pattern: dict[int, int]=None, inplace_container: DataContainer=None) -> DataContainer:
        tile_sig = ref.tile_sig
                    
        if tile_sig is None:
            raise Exception("Tile signature is required for MEM_COPY_TILE IR in load thread kernel.")
        
        tensor = env.buffers[tile_sig.buf_name]
        row_size = tensor.tile_shape[1] * tensor.dtype.itemsize
        row_num = tensor.tile_shape[0]
        
        container = inplace_container if inplace_container is not None else DataContainer(shape=tensor.tile_shape, dtype=tensor.dtype)  # Allocate container with the same shape and dtype as the tile
        
        if ref.is_fifo():
            fifo_handle = env.fifo_buffers[ref.ref_type.buf_name]
            slot_id = ref.slot_id
            core.mem_read_from_fifo(container, fifo_handle, slot_id, row_size, row_num, row_pattern=row_pattern)
        elif ref.is_spm():
            tile_sig = ref.tile_sig
            ptr = ref.ptr
            core.mem_read(ptr, container, row_size, row_num, row_pattern=row_pattern)
        elif ref.is_tensor():
            tile_sig = ref.tile_sig
            buf = env.buffers[ref.ref_type.buf_name]
            src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad = buf.get_tile_ptr_read_args(*tile_sig.coords)
            core.mem_read(src_ptr, container, row_size, row_num, src_row_stride, dst_row_stride, row_pattern=row_pattern, cont_row_zero_pad=dst_row_zero_pad)
        else:
            raise Exception(f"Unsupported source type '{type(ref).__name__}' in MEM_COPY_TILE IR in load thread kernel.")
        
        return container
        
    @classmethod
    def write_to_ref(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, container: DataContainer, ref: MCA_CompiledOperator.IR.Reference, row_pattern: dict[int, int]=None):
        tile_sig = ref.tile_sig
                    
        if tile_sig is None:
            raise Exception("Tile signature is required for MEM_COPY_TILE IR in load thread kernel.")
        
        tensor = env.buffers[tile_sig.buf_name]
        row_size = tensor.tile_shape[1] * tensor.dtype.itemsize
        row_num = tensor.tile_shape[0]
        
        if ref.is_fifo():
            fifo_handle = env.fifo_buffers[ref.ref_type.buf_name]
            slot_id = ref.slot_id
            core.mem_write_to_fifo(container, fifo_handle, slot_id, row_size, row_num, row_pattern=row_pattern, ref_count=ref.ref_cnt)
        elif ref.is_spm():
            tile_sig = ref.tile_sig
            ptr = ref.ptr
            core.mem_write(ptr, container, row_size, row_num, row_pattern=row_pattern)
        elif ref.is_tensor():
            tile_sig = ref.tile_sig
            buf = env.buffers[ref.ref_type.buf_name]
            dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = buf.get_tile_ptr_write_args(*tile_sig.coords)
            core.mem_write(dst_ptr, container, row_size, row_num, dst_row_stride, src_row_stride, row_pattern=row_pattern)
        else:
            raise Exception(f"Unsupported destination type '{type(ref).__name__}' in MEM_COPY_TILE IR in load thread kernel.")
    
    @classmethod
    def LD_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage):
        thread_lock = VariableHandle.tmp(initial_value=0)
        
        for thread_id, ir in enumerate(stage.loads):
            with new_parallel_thread(f"{thread_id}"):
                if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                    continue
                elif isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                    container = cls.read_from_ref(core, env, ir.src)
                    core.var_atomic_wait(thread_lock, thread_id)
                    for dst in ir.dsts:
                        cls.write_to_ref(core, env, container, dst)
                    core.var_atomic_increase(thread_lock)
                else:
                    raise Exception(f"Unsupported IR type '{type(ir).__name__}' in LD_THREAD kernel.")
                
        core.parallel_merge()

    @classmethod
    def EX_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage):
        for ir in stage.executes:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.EXE_UOP):
                cls.EXE_UOP(core, env, ir)
            elif isinstance(ir, MCA_CompiledOperator.IR.EXE_CTX_LOAD):
                cls.EXE_CTX_LOAD(core, env, ir)
            elif isinstance(ir, MCA_CompiledOperator.IR.EXE_CTX_STORE):
                cls.EXE_CTX_STORE(core, env, ir)
            else:
                raise Exception(f"Unsupported IR type '{type(ir).__name__}' in EX_THREAD kernel.")
            
    @classmethod
    def ST_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, stage: MCA_CompiledOperator.Stage):
        for ir in stage.stores:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):                
                container = cls.read_from_ref(core, env, ir.src)
                for dst in ir.dsts:
                    cls.write_to_ref(core, env, container, dst)
            else:
                raise Exception(f"Unsupported IR type '{type(ir).__name__}' in ST_THREAD kernel.")
    
    @classmethod
    def BARRIER(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, barrier: tuple[str, str, int]):
        arrival_cnt, blocking, total_cnt = barrier
        core.var_atomic_barrier(env.variables[arrival_cnt], env.variables[blocking], total_cnt)
    
    @classmethod
    def EXE_UOP(
        cls,
        core: NPUCore, 
        env: MCA_OperatorGraphCompiler.Environment, 
        ir: MCA_CompiledOperator.IR.EXE_UOP,
    ):
        raise NotImplementedError("EXE_UOP execution is not implemented yet.")
    
    @classmethod
    def EXE_CTX_LOAD(
        cls,
        core: NPUCore, 
        env: MCA_OperatorGraphCompiler.Environment, 
        ir: MCA_CompiledOperator.IR.EXE_CTX_LOAD,
    ):
        raise NotImplementedError("EXE_CTX_LOAD execution is not implemented yet.")
    
    @classmethod
    def EXE_CTX_STORE(
        cls,
        core: NPUCore, 
        env: MCA_OperatorGraphCompiler.Environment, 
        ir: MCA_CompiledOperator.IR.EXE_CTX_STORE,
    ):
        raise NotImplementedError("EXE_CTX_STORE execution is not implemented yet.")
