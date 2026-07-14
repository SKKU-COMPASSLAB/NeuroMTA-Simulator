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
    def get_ld_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.LD_THREAD,
            args=(env, ir_seq),
            kwargs={}
        )

    def get_ex_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.EX_THREAD,
            args=(env, ir_seq),
            kwargs={}
        )

    def get_st_thread_kernel(self, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]) -> KernelPrototype:
        return KernelPrototype(
            core=core,
            func=self.ST_THREAD,
            args=(env, ir_seq),
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
    def read_from_ref(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ref: MCA_CompiledOperator.IR.Reference, row_pattern: dict[int, int]=None, inplace_container: DataContainer=None, fifo_sync: bool=True) -> DataContainer:
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
            src_ptr = fifo_handle.get_ptr(slot_id)
            if fifo_sync:
                core.fifo_wait_until_valid(fifo_handle, slot_id)
            core.mem_read(src_ptr, container, row_size, row_num, row_pattern=row_pattern)
            if fifo_sync:
                core.fifo_pop(fifo_handle, slot_id)

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
    def write_to_ref(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, container: DataContainer, ref: MCA_CompiledOperator.IR.Reference, row_pattern: dict[int, int]=None, fifo_sync: bool=True):
        tile_sig = ref.tile_sig

        if tile_sig is None:
            raise Exception("Tile signature is required for MEM_COPY_TILE IR in load thread kernel.")

        tensor = env.buffers[tile_sig.buf_name]
        row_size = tensor.tile_shape[1] * tensor.dtype.itemsize
        row_num = tensor.tile_shape[0]

        if ref.is_fifo():
            fifo_handle = env.fifo_buffers[ref.ref_type.buf_name]
            slot_id = ref.slot_id
            dst_ptr = fifo_handle.get_ptr(slot_id)
            if fifo_sync:
                core.fifo_wait_until_vacant(fifo_handle, slot_id)
            core.mem_write(dst_ptr, container, row_size, row_num, row_pattern=row_pattern)
            if fifo_sync:
                core.fifo_push(fifo_handle, slot_id)
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
    def LD_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]):
        ir_locks = {ir.ir_idx: VariableHandle.tmp(initial_value=0) for ir in ir_seq if isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE) and ir.ir_idx is not None}
        fifo_srcs: dict[str, list[tuple[int, int]]] = {}
        fifo_dsts: dict[str, list[tuple[int, int]]] = {}

        for ir in ir_seq:
            if isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                if ir.src.is_fifo():
                    fifo_srcs.setdefault(ir.src.ref_type.buf_name, []).append((ir.src.slot_id, ir.src.ref_cnt))
                for dst in ir.dsts:
                    if dst.is_fifo():
                        fifo_dsts.setdefault(dst.ref_type.buf_name, []).append((dst.slot_id, dst.ref_cnt))

        for fifo_name in fifo_srcs.keys():
            fifo_srcs[fifo_name].sort(key=lambda x: x[0])  # Sort by slot_id
        for fifo_name in fifo_dsts.keys():
            fifo_dsts[fifo_name].sort(key=lambda x: x[0])  # Sort by slot_id

        for fifo_name, slot_refs in fifo_dsts.items():
            fifo_handle = env.fifo_buffers[fifo_name]
            slot_ids = sorted(set(slot_id for slot_id, _ in slot_refs))
            core.fifo_wait_until_vacant(fifo_handle, slot_ids)

        for fifo_name, slot_refs in fifo_srcs.items():
            fifo_handle = env.fifo_buffers[fifo_name]
            slot_ids = sorted(set(slot_id for slot_id, _ in slot_refs))
            core.fifo_wait_until_valid(fifo_handle, slot_ids)

        for ir in ir_seq:
            # with new_parallel_thread():
            if isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                if ir.ir_idx is None:
                    raise Exception("IR index is required for MEM_COPY_TILE IR in load thread kernel.")

                for wait_ir_idx in ir.wait_ir_idx:
                    if wait_ir_idx not in ir_locks:
                        raise Exception(f"IR index '{wait_ir_idx}' is not found in the load thread kernel locks.")
                    wait_lock = ir_locks[wait_ir_idx]
                    core.var_conditional_wait(wait_lock, wait_lock.equals_to(1))

                container = cls.read_from_ref(core, env, ir.src, fifo_sync=False)

                for dst in ir.dsts:
                    cls.write_to_ref(core, env, container, dst, fifo_sync=False)

                core.var_atomic_init(ir_locks[ir.ir_idx], 1)  # Mark this IR as completed

        core.parallel_merge()

        for fifo_name, slot_refs in fifo_dsts.items():
            fifo_handle = env.fifo_buffers[fifo_name]
            entry_refs = {}
            for slot_id, ref_cnt in slot_refs:
                entry_refs[slot_id] = max(ref_cnt, entry_refs.get(slot_id, 0))
            core.fifo_push(fifo_handle, sorted(entry_refs.items()))

        for fifo_name, slot_refs in fifo_srcs.items():
            fifo_handle = env.fifo_buffers[fifo_name]
            core.fifo_pop(fifo_handle, [slot_id for slot_id, _ in slot_refs])

    @classmethod
    def EX_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]):
        for ir in ir_seq:
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
    def ST_THREAD(cls, core: NPUCore, env: MCA_OperatorGraphCompiler.Environment, ir_seq: list[MCA_CompiledOperator.IR.Base]):
        for ir in ir_seq:
            if isinstance(ir, MCA_CompiledOperator.IR.NOP):
                continue
            elif isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                container = cls.read_from_ref(core, env, ir.src, fifo_sync=True)
                for dst in ir.dsts:
                    cls.write_to_ref(core, env, container, dst, fifo_sync=True)
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
