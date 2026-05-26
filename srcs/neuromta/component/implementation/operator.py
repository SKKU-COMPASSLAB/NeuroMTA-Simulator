import abc
import copy
import enum
import functools
import math
import pprint
import torch
import tqdm
from typing import Any, Sequence, Dict, List, Callable
from collections import deque, defaultdict, Counter
from bisect import bisect_left

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context.global_context import GlobalContextMemInfo
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.mapping import *


__all__ = [
    "MCA_CompiledProgram",
    "MCA_OperatorSignature",
    "MCA_CompiledOperator",
    "MCA_OperatorGraphCompiler",
    "mca_operator_method",
    "mca_operator_method_check",
]


def mca_operator_method_check(func: Callable) -> bool:
    if not hasattr(func, "_is_mca_operator_method"):
        return False
    return func._is_mca_operator_method

def mca_operator_method(func: Callable):
    @functools.wraps(func)
    def _mca_mapper_method_wrapper(*args, **kwargs) -> 'MCA_OperatorSignature':
        op_sig = func(*args, **kwargs)
        if not isinstance(op_sig, MCA_OperatorSignature):
            raise TypeError("The decorated function must return an instance of MCA_OperatorSignature.")
        return op_sig
    _mca_mapper_method_wrapper._is_mca_operator_method = True
    return _mca_mapper_method_wrapper


class MCA_KernelTemplate:
    def get_ld_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', ir_seq: 'list[MCA_CompiledOperator.IR.Base]', load_ir_lock: str) -> KernelPrototype: ...
    def get_ex_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', ir_seq: 'list[MCA_CompiledOperator.IR.Base]') -> KernelPrototype: ...
    def get_st_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', ir_seq: 'list[MCA_CompiledOperator.IR.Base]') -> KernelPrototype: ...
    def get_barrier_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', barrier: tuple[str, str, int]) -> KernelPrototype: ...


class MCA_OperatorSignature:
    def __init__(
        self, 
        op_type: str, 
        kernel_template: MCA_KernelTemplate,
    ):
        self._op_type = op_type
        self.op_id = op_type    # will be initialized by MCA_OperatorGraphCompiler (initially set to op_type) 
        
        self.kernel_template = kernel_template
        
        self._buffers: dict[str, MCA_TensorBuffer] = {}
        self._tiles: dict[str, dict[tuple[int, ...], TileSignature]] = {}
        self._tiled_ops: list[TiledOperatorSignature] = []
        self.global_kwargs: dict[str, Any] = {}
        
        self.buffer_names: list[str] = []
        self.input_buffer_names: list[str] = []
        self.param_buffer_names: list[str] = []
        self.output_buffer_name: str = None
        
        self.core_group: MCA_CoreGroup = None
        
    def add_buffer(self, buf_name: str, buffer: MCA_TensorBuffer, is_input: bool=False, is_output: bool=False, is_param: bool=False):
        if is_param:
            is_input = True
        if (not is_input) and (not is_output):
            raise ValueError("Buffer must be marked as input or output.")
        
        self._buffers[buf_name] = buffer
        self._tiles[buf_name] = {}
        
        if buffer is not None:
            for y_s in range(buffer.shard_grid[0]):
                for x_s in range(buffer.shard_grid[1]):
                    for y_t in range(buffer.tile_grid_per_shard[0]):
                        for x_t in range(buffer.tile_grid_per_shard[1]):
                            self._tiles[buf_name][(y_s, x_s, y_t, x_t)] = TileSignature(buf_name, buffer.tile_shape, buffer.dtype, y_s, x_s, y_t, x_t)
        
        self.buffer_names.append(buf_name)
        if is_param:
            self.param_buffer_names.append(buf_name)
        if is_input:
            self.input_buffer_names.append(buf_name)
        if is_output:
            if self.output_buffer_name is not None:
                raise ValueError("Multiple output buffers are not supported.")
            self.output_buffer_name = buf_name
            
        return self
    
    def new_tiled_op(self) -> TiledOperatorSignature:
        tiled_op = TiledOperatorSignature(tiled_op_id=len(self._tiled_ops))
        self._tiled_ops.append(tiled_op)
        return tiled_op
    
    def get_tiled_op(self, tiled_op_id: int) -> TiledOperatorSignature:
        if tiled_op_id < 0 or tiled_op_id >= len(self._tiled_ops):
            raise ValueError(f"Tiled operator ID {tiled_op_id} is out of range.")
        return self._tiled_ops[tiled_op_id]
        
    def update_global_kwargs(self, op_kwargs: dict[str, Any]):
        self.global_kwargs.update(op_kwargs)
        
    def initialize_core_group(self, core_group: MCA_CoreGroup):
        self.core_group = core_group
        return self
        
    @property
    def is_core_group_initialized(self) -> bool:
        return self.core_group is not None
        
    def rename_buffers(self, rename_map: dict[str, str]):
        for old_name, new_name in rename_map.items():
            if old_name not in self._buffers:
                raise ValueError(f"Buffer {old_name} does not exist.")
            if new_name in self._buffers:
                raise ValueError(f"Buffer {new_name} already exists.")
            
            # STEP 1: rename buffers
            self._buffers[new_name] = self._buffers.pop(old_name)
            
            # STEP 2: rename tiles
            self._tiles[new_name] = self._tiles.pop(old_name)
            for _, tile in self._tiles[new_name].items():
                tile.buf_name = new_name
            
            # STEP 3: update tiled ops
            for tiled_op in self._tiled_ops:
                for uop_idx in range(tiled_op.n_uops):
                    for tile in tiled_op.i_tiles[uop_idx]:
                        if tile.buf_name == old_name:
                            tile.buf_name = new_name
                    if tiled_op.o_tile.buf_name == old_name:
                        tiled_op.o_tile.buf_name = new_name
            
            # STEP 4: update output buffer names
            if old_name == self.output_buffer_name:
                self.output_buffer_name = new_name
            if old_name in self.input_buffer_names:
                idx = self.input_buffer_names.index(old_name)
                self.input_buffer_names[idx] = new_name
            if old_name in self.buffer_names:
                idx = self.buffer_names.index(old_name)
                self.buffer_names[idx] = new_name
            if old_name in self.param_buffer_names:
                idx = self.param_buffer_names.index(old_name)
                self.param_buffer_names[idx] = new_name
                
    @property
    def op_type(self):      return self._op_type
    @property
    def buffers(self):      return self._buffers
    @property
    def tiles(self):        return self._tiles
    @property
    def tiled_ops(self):    return self._tiled_ops
    @property
    def total_buffer_size(self) -> int:
        return sum(b.total_size for b in self.buffers.values())
    @property
    def total_n_uops(self) -> int:
        return sum(tiled_op.n_uops for tiled_op in self.tiled_ops)
    @property
    def total_arithmetic_intensity(self) -> float:
        if self.total_buffer_size == 0:
            return float('inf')
        return self.total_n_uops / self.total_buffer_size

    
class MCA_CompiledOperator:
    class IR:
        class DescriptorBase(metaclass=abc.ABCMeta):
            def __init__(self, buf_name: str):
                self.buf_name = buf_name
                
            @abc.abstractmethod
            def ref(self, *args, **kwargs) -> 'MCA_CompiledOperator.IR.Reference':
                raise NotImplementedError("Reference method must be implemented by subclasses.")
            
            def compare_with(self, other: 'MCA_CompiledOperator.IR.DescriptorBase') -> bool:
                return isinstance(other, self.__class__) and self.buf_name == other.buf_name
            
            def __eq__(self, value):
                return self.compare_with(value)
        
        class FIFODescriptor(DescriptorBase):
            def __init__(self, buf_name: str, ptr: Pointer, slot_size: int, slot_num: int):
                super().__init__(buf_name)
                
                self.ptr = ptr
                self.slot_size = slot_size
                self.slot_num = slot_num
                
            def ref(self, tile_sig: TileSignature, slot_id: int, ref_cnt: int):
                return MCA_CompiledOperator.IR.Reference(self, tile_sig=tile_sig, ptr=self.ptr + slot_id * self.slot_size, slot_id=slot_id, ref_cnt=ref_cnt)
            
            def compare_with(self, other: 'MCA_CompiledOperator.IR.DescriptorBase') -> bool:
                if not isinstance(other, MCA_CompiledOperator.IR.FIFODescriptor):
                    return False
                return self.buf_name == other.buf_name and self.slot_size == other.slot_size and self.slot_num == other.slot_num
                
        class SPMDescriptor(DescriptorBase):
            def __init__(self, buf_name: str, ptr: Pointer):
                super().__init__(buf_name)
                self.ptr = ptr
                
            def ref(self, tile_sig: TileSignature, offset: int=0):
                return MCA_CompiledOperator.IR.Reference(self, tile_sig=tile_sig, ptr=self.ptr + offset)
            
            def compare_with(self, other: 'MCA_CompiledOperator.IR.DescriptorBase') -> bool:
                if not isinstance(other, MCA_CompiledOperator.IR.SPMDescriptor):
                    return False
                return self.buf_name == other.buf_name and self.ptr == other.ptr
            
        class TensorBufferDescriptor(DescriptorBase):
            def __init__(self, buf_name: str):
                super().__init__(buf_name)
                
            def ref(self, tile_sig: TileSignature):
                if tile_sig.buf_name != self.buf_name:
                    raise ValueError(f"Tile signature buffer name {tile_sig.buf_name} does not match descriptor buffer name {self.buf_name}.")
                return MCA_CompiledOperator.IR.Reference(self, tile_sig=tile_sig)
            
            def compare_with(self, other: 'MCA_CompiledOperator.IR.DescriptorBase') -> bool:
                if not isinstance(other, MCA_CompiledOperator.IR.TensorBufferDescriptor):
                    return False
                return self.buf_name == other.buf_name

        class Reference:
            def __init__(self, ref_type: 'MCA_CompiledOperator.IR.DescriptorBase', **kwargs):
                self.ref_type = ref_type
                self.kwargs = kwargs
                
            def is_fifo(self) -> bool:
                return isinstance(self.ref_type, MCA_CompiledOperator.IR.FIFODescriptor)
            
            def is_spm(self) -> bool:
                return isinstance(self.ref_type, MCA_CompiledOperator.IR.SPMDescriptor)
            
            def is_tensor(self) -> bool:
                return isinstance(self.ref_type, MCA_CompiledOperator.IR.TensorBufferDescriptor)
            
            @property
            def ptr(self) -> Pointer:
                return self.kwargs.get("ptr", None)
            
            @property
            def slot_id(self) -> int:
                return self.kwargs.get("slot_id", None)
            
            @property
            def tile_sig(self) -> TileSignature:
                return self.kwargs.get("tile_sig", None)
            
            @property
            def ref_cnt(self) -> int:
                return self.kwargs.get("ref_cnt", None)
            
            def compare_with(self, other: 'MCA_CompiledOperator.IR.Reference') -> bool:
                if not isinstance(other, MCA_CompiledOperator.IR.Reference):
                    return False
                
                if self.ref_type != other.ref_type:
                    return False
                if self.kwargs.keys() != other.kwargs.keys():
                    return False
                for key in self.kwargs.keys():
                    if self.kwargs[key] != other.kwargs[key]:
                        return False
            
                return True

            def __repr__(self):
                if self.is_fifo():
                    return f"FIFO({self.ref_type.buf_name}, slot_id={self.slot_id}, ref_cnt={self.ref_cnt})"
                elif self.is_spm():
                    return f"SPM({self.ref_type.buf_name}, ptr={self.ptr.addr})"
                elif self.is_tensor():
                    return f"TENSOR({self.ref_type.buf_name}, tile_sig={self.tile_sig})"
                else:
                    return f"UNKNOWN_REF({self.ref_type}, {self.kwargs})"
                
            def __eq__(self, value):
                return self.compare_with(value)
        
        class Base(metaclass=abc.ABCMeta):
            @abc.abstractmethod
            def signature(self) -> str:
                raise NotImplementedError("Command signature method must be implemented by subclasses.")
            
            def __repr__(self):
                return self.signature()
            
        class NOP(Base):
            def signature(self):
                return "NOP"
            
        class MEM_COPY_TILE(Base):
            def __init__(self, src: 'MCA_CompiledOperator.IR.Reference', dsts: 'list[MCA_CompiledOperator.IR.Reference]'=None):
                self.src = src
                self.dsts = dsts
                self.ir_idx = None  # will be set by MCA_CompiledOperator when added to stage
                self.wait_ir_idx = []  # will be set by MCA_CompiledOperator when added to stage
                
                if dsts is None:
                    self.dsts = []
                if isinstance(dsts, MCA_CompiledOperator.IR.Reference):
                    self.dsts = [dsts]
                
            def signature(self):
                dsts_str = ", ".join([str(dst) for dst in self.dsts])
                wait_repr = f" (wait until IR {', '.join(str(idx) for idx in self.wait_ir_idx)})" if self.wait_ir_idx else ""
                return f"MEM_COPY_TILE [ir_idx={self.ir_idx}] {self.src.tile_sig} {self.src} -> [{dsts_str}]{wait_repr}"
            
        class EXE_UOP(Base):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, i_tile_refs: list['MCA_CompiledOperator.IR.Reference'], o_tile_ref: 'MCA_CompiledOperator.IR.Reference', dtype: torch.dtype, acc_dtype: torch.dtype):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.i_tile_refs = i_tile_refs
                self.o_tile_ref = o_tile_ref
                self.dtype = dtype
                self.acc_dtype = acc_dtype
                
            def signature(self):
                i_tiles_str = ", ".join([str(ref) for ref in self.i_tile_refs])
                o_tile_str = str(self.o_tile_ref) if self.o_tile_ref is not None else "UNDEFINED"
                return f"EXE_UOP {self.op_id} tiled_op_idx={self.tiled_op_idx} uop_idx={self.uop_idx} [{i_tiles_str}] -> {o_tile_str}"
            
        class EXE_CTX_LOAD(Base):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, tile_sig: TileSignature, ref: 'MCA_CompiledOperator.IR.Reference'):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.tile_sig = tile_sig
                self.ref = ref
                
            def signature(self):
                return f"EXE_CTX_LOAD {self.op_id} tiled_op_idx={self.tiled_op_idx} uop_idx={self.uop_idx} {self.ref} -> {self.tile_sig.signature}"
            
        class EXE_CTX_STORE(Base):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, tile_sig: TileSignature, ref: 'MCA_CompiledOperator.IR.Reference'):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.tile_sig = tile_sig
                self.ref = ref
                
            def signature(self):
                return f"EXE_CTX_STORE {self.op_id} tiled_op_idx={self.tiled_op_idx} uop_idx={self.uop_idx} {self.tile_sig.signature} -> {self.ref}"

    class Stage:
        def __init__(self, n_load_threads: int):
            self.n_load_threads = n_load_threads
            
            # self.loads:    list[MCA_CompiledOperator.IR.Base] = []
            self.loads: list[list[MCA_CompiledOperator.IR.Base]] = [[] for _ in range(n_load_threads)]
            self.executes: list[MCA_CompiledOperator.IR.Base] = []
            self.stores:   list[MCA_CompiledOperator.IR.Base] = []
            
        def add_load_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            # if isinstance(cmd, MCA_CompiledOperator.IR.MEM_COPY_TILE):
            #     cmd.ir_idx = len(self.loads)
            if isinstance(cmd, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                thread_id = cmd.ir_idx % self.n_load_threads
                self.loads[thread_id].append(cmd)
            else:
                self.loads[0].append(cmd)
            
        def add_execute_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.executes.append(cmd)
            
        def add_store_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.stores.append(cmd)
            
        def freeze(self):
            self.loads    = tuple(ir for ir in self.loads    if not isinstance(ir, MCA_CompiledOperator.IR.NOP))
            self.executes = tuple(ir for ir in self.executes if not isinstance(ir, MCA_CompiledOperator.IR.NOP))
            self.stores   = tuple(ir for ir in self.stores   if not isinstance(ir, MCA_CompiledOperator.IR.NOP))

        def summary(self) -> dict:
            return {
                # "loads":    [ir.signature() for ir in self.loads    if not isinstance(ir, MCA_CompiledOperator.IR.NOP)],
                "loads":    [[ir.signature() for ir in thread_ir] for thread_ir in self.loads],
                "executes": [ir.signature() for ir in self.executes if not isinstance(ir, MCA_CompiledOperator.IR.NOP)],
                "stores":   [ir.signature() for ir in self.stores   if not isinstance(ir, MCA_CompiledOperator.IR.NOP)],
            }
            
        @property
        def n_uops(self) -> int:
            return len(self.executes)
        
        @property
        def n_tiled_ops(self) -> int:
            return sum(1 for ir in self.executes if isinstance(ir, MCA_CompiledOperator.IR.EXE_UOP) and ir.o_tile_ref is not None)
            
        @property
        def is_bubble(self) -> bool:
            return all(len(thread_ir) == 0 for thread_ir in self.loads) and len(self.executes) == 0 and len(self.stores) == 0

    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata'):
        self._env = env
        self._op_id = op_meta.op_sig.op_id
        self._kernel_template = op_meta.op_sig.kernel_template
        
        self._concurrent_load_num = self._env.recipe.concurrent_load_num
        
        self._ld_ir_counters: dict[int, int] = {core_id: 0 for core_id in op_meta.op_sig.core_group.core_ids}  # {core_id: counter}
        
        self._ld_ir_locks: dict[int, str] = {
            core_id: self._env.add_variable(f"{self._op_id}_core_{core_id}_ld_ir_lock").handle_name 
            for core_id in op_meta.op_sig.core_group.core_ids
        }
        
        self._mappings: dict[int, list[MCA_CompiledOperator.Stage]] = {
            core_id: [MCA_CompiledOperator.Stage(n_load_threads=self._concurrent_load_num)] 
            for core_id in op_meta.op_sig.core_group.core_ids
        }  # {core_id: [stage1, stage2, ...]}
        
    def new_stage(self, core_id: int) -> 'MCA_CompiledOperator.Stage':
        stage = MCA_CompiledOperator.Stage(n_load_threads=self._concurrent_load_num)
        self._mappings[core_id].append(stage)
        return stage
    
    def current_stage(self, core_id: int) -> 'MCA_CompiledOperator.Stage':
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        if len(self._mappings[core_id]) == 0:
            raise ValueError(f"No stage exists for core ID {core_id}.")
        return self._mappings[core_id][-1]
    
    def add_load_ir(self, core_id: int, cmd: 'MCA_CompiledOperator.IR.Base'):
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        if not self._mappings[core_id]:
            self.new_stage(core_id)
        if isinstance(cmd, MCA_CompiledOperator.IR.MEM_COPY_TILE):
            cmd.ir_idx = self._ld_ir_counters[core_id]
            self._ld_ir_counters[core_id] += 1
        self._mappings[core_id][-1].add_load_ir(cmd)
        
    def add_execute_ir(self, core_id: int, cmd: 'MCA_CompiledOperator.IR.Base'):
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        if not self._mappings[core_id]:
            self.new_stage(core_id)
        self._mappings[core_id][-1].add_execute_ir(cmd)
        
    def add_store_ir(self, core_id: int, cmd: 'MCA_CompiledOperator.IR.Base'):
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        if not self._mappings[core_id]:
            self.new_stage(core_id)
        self._mappings[core_id][-1].add_store_ir(cmd)
        
    @property
    def n_stages(self) -> int:
        return max(len(stages) for stages in self._mappings.values())
        
    def freeze(self):
        _mappings = {}
        
        for core_id, stages in self._mappings.items():
            _stages = []
            
            for stage in stages:
                stage.freeze()
                
                if not stage.is_bubble:
                    _stages.append(stage)
                    
            if len(_stages) > 0:    
                _mappings[core_id] = _stages
        
        self._mappings = _mappings
        
    def dispatch(self, device: MCA_DeviceBase, gb_barrier: tuple[str, str, int]=None, gb_core_ids: list[int]=None, postsync_globals: bool=False):
        self.freeze()
        
        op_barrier = (
            self._env.add_variable(f"{self._op_id}_barrier_arrival_cnt", 0).handle_name,
            self._env.add_variable(f"{self._op_id}_barrier_blocking", 0).handle_name,
            len(self.mappings.keys()) * 3,
        )
        
        # PRESYNC BARRIER
        # if presync_globals:
        #     for core_id in gb_core_ids:
        #         core = device.get_npu_core(core_id)
                
        #         self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("LD")
        #         self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("EX")
        #         self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("ST")
        # else:
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            
            self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("LD")
            self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("EX")
            self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("ST")
            
        # LD/EX/ST THREAD
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            
            for stage in self.mappings[core_id]:
                if stage.is_bubble:
                    continue
                
                for thread_id, load_ir in enumerate(stage.loads):
                    if len(load_ir) > 0:
                        self._kernel_template.get_ld_thread_kernel(core, self._env, load_ir, self._ld_ir_locks[core_id]).dispatch(f"LD{thread_id}")
                self._kernel_template.get_ex_thread_kernel(core, self._env, stage.executes).dispatch("EX")
                self._kernel_template.get_st_thread_kernel(core, self._env, stage.stores).dispatch("ST")
        
        # POSTSYNC BARRIER
        if postsync_globals:
            for core_id in gb_core_ids:
                core = device.get_npu_core(core_id)
                
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("LD")
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("EX")
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("ST")
        else:
            for core_id in self.mappings.keys():
                core = device.get_npu_core(core_id)
                
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("LD")
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("EX")
                self._kernel_template.get_barrier_kernel(core, self._env, barrier=op_barrier).dispatch("ST")
        
    @property
    def mappings(self):
        return self._mappings
    
    def summary(self) -> dict:
        return {
            core_id: [stage.summary() for stage in stages]
            for core_id, stages in self._mappings.items()
        }


class MCA_CompiledProgram:
    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', compiled_ops: dict[str, MCA_CompiledOperator]):
        self._env = env
        self._compiled_ops = compiled_ops

    def dispatch(self):
        global_core_ids = self._env.recipe.global_core_group.core_ids
        global_barrier = self._env.global_barrier
                
        for i, op_id in enumerate(self._env.target_op_order):
            compiled_op = self._compiled_ops[op_id]
            compiled_op.dispatch(self._env.recipe.device, gb_barrier=global_barrier, gb_core_ids=global_core_ids, postsync_globals=(i==len(self._env.target_op_order)-1))
        return self
    
    @property
    def device(self) -> MCA_DeviceBase:
        return self._env.recipe.device
            
    def summary(self) -> dict:
        return {op_id: compiled_op.summary() for op_id, compiled_op in self._compiled_ops.items()}
    
    
class MCA_OperatorGraphCompiler:
    ALL="ALL"
    DEFAULT="DEFAULT"
    
    class CompileRecipe:
        class ReuseType(enum.Enum):
            IGNORE      = "IGNORE"
            ALL         = "ALL"
            SINGLE      = "SINGLE"
            ALL_MAIN    = "ALL_MAIN"
            ALL_L1      = "ALL_L1"
            SINGLE_MAIN = "SINGLE_MAIN"
            SINGLE_L1   = "SINGLE_L1"

        def __init__(
            self, 
            device: MCA_DeviceBase,
            core_groups: list[MCA_CoreGroup],
            spad_space_size_per_core: int,
            broadcast_optimize_queue_depth: int=8,
            broadcast_optimize_max_ref_cnt: int=4,
            context_buffer_slot_num: int=16,
            ld_ex_buffer_slot_num: int=16,
            ex_st_buffer_slot_num: int=8,
            concurrent_load_num: int=1,
            temporal_reuse_type: ReuseType=ReuseType.ALL,
            spatial_reuse_type: ReuseType=ReuseType.SINGLE_MAIN,
            greedy_temporal_reuse: bool=True,
        ):
            if len(core_groups) == 0:
                raise ValueError("At least one core group must be provided.")
            if not isinstance(core_groups[0], (MCA_CoreGroup, list)):
                core_groups = [core_groups]

            self.device                         = device
            self.core_groups                    = core_groups
            self.spad_space_size_per_core       = spad_space_size_per_core
            self.broadcast_optimize_queue_depth = broadcast_optimize_queue_depth
            self.broadcast_optimize_max_ref_cnt = broadcast_optimize_max_ref_cnt
            self.context_buffer_slot_num        = context_buffer_slot_num
            self.ld_ex_buffer_slot_num          = ld_ex_buffer_slot_num
            self.ex_st_buffer_slot_num          = ex_st_buffer_slot_num
            self.concurrent_load_num            = concurrent_load_num
            self.temporal_reuse_type    = temporal_reuse_type if isinstance(temporal_reuse_type, self.ReuseType) else self.ReuseType(temporal_reuse_type)
            self.spatial_reuse_type     = spatial_reuse_type  if isinstance(spatial_reuse_type,  self.ReuseType) else self.ReuseType(spatial_reuse_type)
            self.greedy_temporal_reuse  = greedy_temporal_reuse

        @property
        def global_core_group(self) -> MCA_CoreGroup:
            return MCA_CoreGroup.merge_core_groups(self.core_groups)
        
        @property
        def broadcast_optimize(self) -> bool:
            return self.broadcast_optimize_queue_depth > 0
            
    class Thread:
        class _NodeBase(metaclass=abc.ABCMeta):
            pass
        
        class UopNode(_NodeBase):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False, bcast_schedule: dict[TileSignature, tuple[int, int, int]]=None, cache_schedule: dict[TileSignature, tuple[bool, int]]=None):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.output = output
                self.bcast_schedule = bcast_schedule if bcast_schedule is not None else {}  # {tile_sig: (bcast_core_id, bcast_slot_id, bcast_ref_cnt)}
                self.cache_schedule = cache_schedule if cache_schedule is not None else {}  # {tile_sig: (is_cache_hit, cache_slot_id)}
                
        class ContextStoreNode(_NodeBase):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, slot_id: int):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.slot_id = slot_id
                
        class ContextLoadNode(_NodeBase):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, slot_id: int):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.slot_id = slot_id
                
        def __init__(self, core_id: int):
            self.core_id = core_id
            self.uop_nodes: list[MCA_OperatorGraphCompiler.Thread._NodeBase] = []
            
        # def add_uop_node(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False, bcast_schedule: dict[TileSignature, tuple[int, int, int]]=None):
        def add_uop_node(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False, bcast_schedule: dict[TileSignature, tuple[int, int, int]]=None, cache_schedule: dict[TileSignature, tuple[bool, int]]=None):
            uop_node = self.UopNode(op_id, tiled_op_idx, uop_idx, output=output, bcast_schedule=bcast_schedule, cache_schedule=cache_schedule)
            self.uop_nodes.append(uop_node)
            
        def add_context_store(self, op_id: str, tiled_op_idx: int, uop_idx: int, slot_id: int):
            context_store_node = self.ContextStoreNode(op_id, tiled_op_idx, uop_idx, slot_id)
            self.uop_nodes.append(context_store_node)
            
        def add_context_load(self, op_id: str, tiled_op_idx: int, uop_idx: int, slot_id: int):
            context_load_node = self.ContextLoadNode(op_id, tiled_op_idx, uop_idx, slot_id)
            self.uop_nodes.append(context_load_node)

        @property 
        def n_uop_nodes(self) -> int:
            return len(self.uop_nodes)
            
    class OperatorMetadata:
        class SrcType:
            def __init__(self, t: str, k: Any):
                self.t = t
                self.k = k
            
            @classmethod
            def BUFFER(cls):
                return cls("BUFFER", None)
            
            @classmethod
            def TILE_SHARED(cls, src_op_id: str):
                return cls("TILE_SHARED", src_op_id)
            
            @property
            def is_buffer(self):
                return self.t == "BUFFER"
            
            @property
            def is_tile_shared(self):
                return self.t == "TILE_SHARED"
            
            def __repr__(self):
                if self.is_buffer:
                    return f"SrcType(BUFFER)"
                elif self.is_tile_shared:
                    return f"SrcType(TILE_SHARED from {self.k})"
        
        def __init__(self, op_sig: 'MCA_OperatorSignature', recipe: 'MCA_OperatorGraphCompiler.CompileRecipe'):
            self.op_sig = op_sig
            self.spad_space_size_per_core = recipe.spad_space_size_per_core
            self.broadcast_optimize_max_ref_cnt = recipe.broadcast_optimize_max_ref_cnt
            
            self.i_buf_src: dict[str, MCA_OperatorGraphCompiler.OperatorMetadata.SrcType] = {
                buf_name: MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.BUFFER() 
                for buf_name in op_sig.input_buffer_names
            }
            
            # Initialize min LD/ST area based on the operator signature
            self.min_ld_area_per_pp = 0
            self.min_st_area_per_pp = 0
            _tmp_total_ld_area = 0
            _tmp_total_st_area = 0
            
            for tiled_op_sig in op_sig.tiled_ops:
                for uop_idx in range(tiled_op_sig.n_uops):
                    _tmp_ld_area = sum(op_sig.buffers[tile.buf_name].tile_size for tile in tiled_op_sig.i_tiles[uop_idx])
                    _tmp_st_area = op_sig.buffers[tiled_op_sig.o_tile.buf_name].tile_size
                    self.min_ld_area_per_pp = max(self.min_ld_area_per_pp, _tmp_ld_area)
                    self.min_st_area_per_pp = max(self.min_st_area_per_pp, _tmp_st_area)
                    _tmp_total_ld_area += _tmp_ld_area
                    _tmp_total_st_area += _tmp_st_area
                    
            self.ld_ratio = (self.min_ld_area_per_pp / (self.min_ld_area_per_pp + self.min_st_area_per_pp)) if (self.min_ld_area_per_pp + self.min_st_area_per_pp) > 0 else 0.8
            
            # Constant buffer sizes
            self.bcast_fifo_slot_size = max(buf.tile_size for buf_name, buf in op_sig.buffers.items() if buf_name in op_sig.input_buffer_names and buf is not None)
            self.bcast_fifo_size      = recipe.broadcast_optimize_queue_depth * self.bcast_fifo_slot_size
            
            self.ctx_buffer_slot_size = self.op_sig.buffers[op_sig.output_buffer_name].tile_size  # conservatively reserve the same size as output tile buffer for context store (for tile-level pipelining)
            self.ctx_buffer_size      = recipe.context_buffer_slot_num * self.ctx_buffer_slot_size
            
            self.ld_ex_fifo_slot_size = max(buf.tile_size for buf_name, buf in op_sig.buffers.items() if buf_name in op_sig.input_buffer_names and buf is not None)
            self.ld_ex_fifo_size      = recipe.ld_ex_buffer_slot_num * self.ld_ex_fifo_slot_size
            
            self.ex_st_fifo_slot_size = op_sig.buffers[op_sig.output_buffer_name].tile_size
            self.ex_st_fifo_size      = recipe.ex_st_buffer_slot_num * self.ex_st_fifo_slot_size
            
            self.cache_buffer_slot_size = self.ld_ex_fifo_slot_size  # conservatively reserve the same size as LD->EX FIFO slot for cache buffer (for tile-level reuse)
            
            # Reuse targets
            reuse_targets = [buf_name for buf_name in op_sig.input_buffer_names if op_sig.buffers[buf_name] is not None]
            reuse_targets = sorted(reuse_targets, key=lambda buf_name: (op_sig.buffers[buf_name].total_size, 1 if op_sig.buffers[buf_name].mem_space.is_main else 0), reverse=True)
            main_reuse_targets = [n for n in reuse_targets if op_sig.buffers[n].mem_space.is_main]
            l1_reuse_targets = [n for n in reuse_targets if op_sig.buffers[n].mem_space.is_l1]
            
            if recipe.temporal_reuse_type == recipe.ReuseType.ALL:
                self.temporal_reuse_targets = reuse_targets if len(reuse_targets) > 0 else op_sig.input_buffer_names
            elif recipe.temporal_reuse_type == recipe.ReuseType.SINGLE:
                self.temporal_reuse_targets = [reuse_targets[0]] if len(reuse_targets) > 0 else [op_sig.input_buffer_names[0]]
            elif recipe.temporal_reuse_type == recipe.ReuseType.ALL_MAIN:
                self.temporal_reuse_targets = main_reuse_targets if len(main_reuse_targets) > 0 else op_sig.input_buffer_names
            elif recipe.temporal_reuse_type == recipe.ReuseType.ALL_L1:
                self.temporal_reuse_targets = l1_reuse_targets if len(l1_reuse_targets) > 0 else op_sig.input_buffer_names
            elif recipe.temporal_reuse_type == recipe.ReuseType.SINGLE_MAIN:
                self.temporal_reuse_targets = [main_reuse_targets[0]] if len(main_reuse_targets) > 0 else [op_sig.input_buffer_names[0]]
            elif recipe.temporal_reuse_type == recipe.ReuseType.SINGLE_L1:
                self.temporal_reuse_targets = [l1_reuse_targets[0]] if len(l1_reuse_targets) > 0 else [op_sig.input_buffer_names[0]]
            elif recipe.temporal_reuse_type == recipe.ReuseType.IGNORE:
                self.temporal_reuse_targets = []
            else:
                raise ValueError(f"Invalid temporal reuse type: {recipe.temporal_reuse_type}")
            
            if recipe.spatial_reuse_type == recipe.ReuseType.ALL:
                raise Exception(f"Invalid spatial reuse type: {recipe.spatial_reuse_type}. Spatial reuse typically targets a single buffer, so ALL is not supported.")
            elif recipe.spatial_reuse_type == recipe.ReuseType.SINGLE:
                self.spatial_reuse_target = reuse_targets[0] if len(reuse_targets) > 0 else op_sig.input_buffer_names[0]
            elif recipe.spatial_reuse_type == recipe.ReuseType.ALL_MAIN:
                raise Exception(f"Invalid spatial reuse type: {recipe.spatial_reuse_type}. Spatial reuse typically targets a single buffer, so ALL_MAIN is not supported.")
            elif recipe.spatial_reuse_type == recipe.ReuseType.ALL_L1:
                raise Exception(f"Invalid spatial reuse type: {recipe.spatial_reuse_type}. Spatial reuse typically targets a single buffer, so ALL_L1 is not supported.")
            elif recipe.spatial_reuse_type == recipe.ReuseType.SINGLE_MAIN:
                self.spatial_reuse_target = main_reuse_targets[0] if len(main_reuse_targets) > 0 else op_sig.input_buffer_names[0]
            elif recipe.spatial_reuse_type == recipe.ReuseType.SINGLE_L1:
                self.spatial_reuse_target = l1_reuse_targets[0] if len(l1_reuse_targets) > 0 else op_sig.input_buffer_names[0]
            elif recipe.spatial_reuse_type == recipe.ReuseType.IGNORE:
                self.spatial_reuse_target = None
            else:
                raise ValueError(f"Invalid spatial reuse type: {recipe.spatial_reuse_type}")
            
            self.greedy_temporal_reuse = recipe.greedy_temporal_reuse

            # Initialized after freezing the operator metadata
            self.cache_buffer_size = 0 # cache buffer size for tile-level reuse (cache buffer)
            self.thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            self.o_tile_store = op_sig.buffers[op_sig.output_buffer_name].is_allocated  # if the output buffer is allocated, the computation result should be updated to the buffer
            self.o_tile_sharers: set[str] = set()  # set of op_ids that directly consume this operator's output tiles (tile-level sharers via SHARED area)
            
            self._is_frozen = False

        @staticmethod
        def optimal_grid_placement(n_queues: int, items: list[int], row_sigs: dict[int, tuple[int]], col_sigs: dict[int, tuple[int]]) -> dict:
            """
            Distributes items with list-based signatures, optimizing for LCS-based col_score and row_score.
            """
            def get_lcs(seq1: list[int], seq2: list[int]) -> list[int]:
                """Computes the Longest Common Subsequence (LCS) of two sequences."""
                m, n = len(seq1), len(seq2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if seq1[i-1] == seq2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                            
                # Reconstruct the LCS sequence
                lcs = []
                i, j = m, n
                while i > 0 and j > 0:
                    if seq1[i-1] == seq2[j-1]:
                        lcs.append(seq1[i-1])
                        i -= 1
                        j -= 1
                    elif dp[i-1][j] > dp[i][j-1]:
                        i -= 1
                    else:
                        j -= 1
                return lcs[::-1]
            
            def get_multi_lcs(seqs: list[list[int]]) -> list[int]:
                """Approximates the multi-sequence LCS using a progressive approach."""
                if not seqs:
                    return []
                current_lcs = seqs[0]
                for i in range(1, len(seqs)):
                    current_lcs = get_lcs(current_lcs, seqs[i])
                    if not current_lcs:
                        break
                return current_lcs
            
            queued_items = {queue_id: [] for queue_id in range(n_queues)}
            row_cluster_info = {}
            
            if not items:
                return queued_items, row_cluster_info

            # 1. Strict Balance (Priority 1)
            total_items = len(items)
            base_size = total_items // n_queues
            remainder = total_items % n_queues
            target_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_queues)]

            # 2. Group by col_sig (Priority 2: Maximize col_score)
            col_groups = defaultdict(list)
            for item in items:
                col_groups[col_sigs[item]].append(item)
                
            # Within each col_sig group, sort by row_sig to improve row alignment (Priority 3)
            for sig in col_groups:
                col_groups[sig].sort(key=lambda x: row_sigs[x])
                
            # 3. Order groups to form a stream
            # Sorting by sig tuples helps keep identical or similar sequences close to each other.
            sorted_col_sigs = sorted(col_groups.keys())
            ordered_items_stream = []
            for sig in sorted_col_sigs:
                ordered_items_stream.extend(col_groups[sig])
                
            # 4. Sequential Placement to maintain balance
            current_idx = 0
            for q_id in range(n_queues):
                size = target_sizes[q_id]
                queued_items[q_id] = ordered_items_stream[current_idx : current_idx + size]
                current_idx += size
            
            return queued_items
            
        def _create_tiled_op_mapping(self) -> tuple[dict[int, list[TiledOperatorSignature]], dict]:
            # STEP 1: Create spatial and temporal clusters based on the tile access patterns of each tiled op
            _tiled_op_ids = [tiled_op.tiled_op_id for tiled_op in self.op_sig.tiled_ops]
            _temporal_sigs = {}
            _spatial_sigs = {}
            _tile_sig_to_id = {}
            _tile_id_to_sig = {}
            
            for tiled_op_id in _tiled_op_ids:
                tiled_op = self.op_sig.get_tiled_op(tiled_op_id)
                
                ss_order: list[TileSignature] = []
                ts_order: list[TileSignature] = []
                
                for uop_idx in range(tiled_op.n_uops):
                    for tile in tiled_op.i_tiles[uop_idx]:
                        if tile not in _tile_sig_to_id:
                            _tile_sig_to_id[tile] = len(_tile_sig_to_id)
                            _tile_id_to_sig[_tile_sig_to_id[tile]] = tile
                        
                        if tile.buf_name == self.spatial_reuse_target:
                            ss_order.append(_tile_sig_to_id[tile])
                        if len(self.temporal_reuse_targets) > 0:
                            if tile.buf_name == self.temporal_reuse_targets[0]:
                                ts_order.append(_tile_sig_to_id[tile])
                            
                _spatial_sigs[tiled_op_id] = tuple(ss_order)
                _temporal_sigs[tiled_op_id] = tuple(ts_order)

            queued_tiled_op_ids = self.optimal_grid_placement(
                n_queues=len(self.op_sig.core_group.core_ids),
                items=_tiled_op_ids,
                row_sigs=_spatial_sigs,
                col_sigs=_temporal_sigs,
            )

            tiled_op_mapping = {
                core_id: [
                    self.op_sig.get_tiled_op(tiled_op_id)
                    for tiled_op_id in queued_tiled_op_ids[i]
                ]
                for i, core_id in enumerate(self.op_sig.core_group.core_ids)
            }

            return tiled_op_mapping
        
        @staticmethod
        def optimal_clustering(queued_items: dict, max_cluster_size: int) -> list[list]:
            """
            Groups items with the same signature into clusters across different queues,
            respecting ordering constraints and a maximum cluster size.
            """
            current_pos = {q_id: 0 for q_id in queued_items.keys()}
            clustered_items = []

            # 1. Precompute indices for each signature in each queue
            sig_indices = {q_id: defaultdict(list) for q_id in queued_items.keys()}
            all_signatures = set()
            for q_id, items in queued_items.items():
                for idx, (sig, item_id) in enumerate(items):
                    sig_indices[q_id][sig].append(idx)
                    all_signatures.add(sig)

            while True:
                best_candidate = None

                # 2. Search for the best next cluster across all signatures
                for sig in all_signatures:
                    first_indices = []
                    for q_id in queued_items.keys():
                        pos_list = sig_indices[q_id][sig]
                        start_search_idx = bisect_left(pos_list, current_pos[q_id])
                        if start_search_idx < len(pos_list):
                            first_indices.append((q_id, pos_list[start_search_idx]))

                    if len(first_indices) < 2:
                        continue

                    first_indices.sort(key=lambda x: x[1])

                    limit = min(len(first_indices), max_cluster_size)
                    for r in range(2, limit + 1):
                        subset = first_indices[:r]
                        max_idx = subset[-1][1]
                        gain = len(subset) - 1
                        sum_idx = sum(p[1] for p in subset)

                        candidate = {
                            'sig': sig,
                            'positions': subset,
                            'max_idx': max_idx,
                            'gain': gain,
                            'sum_idx': sum_idx
                        }

                        if best_candidate is None or \
                        (candidate['max_idx'], -candidate['gain'], candidate['sum_idx']) < \
                        (best_candidate['max_idx'], -best_candidate['gain'], best_candidate['sum_idx']):
                            best_candidate = candidate

                if best_candidate is None:
                    break

                # 4. Finalize the chosen cluster and update queue pointers safely
                new_cluster = []
                max_idx_of_cluster = best_candidate['max_idx']
                
                for q_id, idx in best_candidate['positions']:
                    new_cluster.append(queued_items[q_id][idx])
                
                for q_id in queued_items.keys():
                    current_pos[q_id] = max(current_pos[q_id], max_idx_of_cluster + 1)
                
                clustered_items.append(new_cluster)

            return clustered_items
        
        def _create_thread_mapping(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]) -> 'dict[int, MCA_OperatorGraphCompiler.Thread]':
            thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            if self.greedy_temporal_reuse:
                actual_temporal_reuse_targets = self.temporal_reuse_targets + [buf_name for buf_name in self.op_sig.input_buffer_names 
                                                                               if buf_name not in self.temporal_reuse_targets and self.op_sig.buffers[buf_name] is not None]  # If greedy temporal reuse is enabled, we can also consider other input buffers as temporal reuse targets (in case the specified temporal reuse targets do not provide enough reuse opportunities), but only consider those that are from buffers (not tile-shared sources) since tile-shared sources cannot be guaranteed to be reused across different tiled ops
            else:
                actual_temporal_reuse_targets = self.temporal_reuse_targets
            actual_temporal_reuse_targets = [buf_name for buf_name in actual_temporal_reuse_targets if self.i_buf_src[buf_name].is_buffer]  # Only consider buffers as temporal reuse targets for thread mapping, since tile-shared sources cannot be guaranteed to be reused across different tiled ops
            
            tile_temporal_reuse_counts = {core_id: defaultdict(int) for core_id in tiled_op_mapping.keys()}  # {core_id: {tile_sig: list of (tiled_op_idx, uop_idx) that access this tile}}
            for core_id, tiled_ops in tiled_op_mapping.items():
                for tiled_op in tiled_ops:
                    for uop_idx in range(tiled_op.n_uops):
                        for tile in tiled_op.i_tiles[uop_idx]:
                            if tile.buf_name in actual_temporal_reuse_targets:
                                tile_temporal_reuse_counts[core_id][tile] += 1
            
            def max_reuse_distance(core_id: int, tiled_ops: list[TiledOperatorSignature]) -> int:
                temporal_tiles = [
                    tile.signature
                    for tiled_op in tiled_ops
                    for uop_idx in range(tiled_op.n_uops)
                    for tile in tiled_op.i_tiles[uop_idx]
                    if tile.buf_name in actual_temporal_reuse_targets and tile_temporal_reuse_counts[core_id][tile] > 1
                ]
                
                tile_positions = defaultdict(int)
                max_distance = 0
                
                for idx, tile_sig in enumerate(temporal_tiles):
                    if tile_sig in tile_positions:
                        distance = idx - tile_positions[tile_sig]
                        max_distance = max(max_distance, distance)
                    tile_positions[tile_sig] = idx
                    
                return max_distance
            
            def max_n_temporal_tiles_per_uop(tiled_ops: list[TiledOperatorSignature]) -> int:
                max_count = 0
                for tiled_op in tiled_ops:
                    for uop_idx in range(tiled_op.n_uops):
                        count = sum(1 for tile in tiled_op.i_tiles[uop_idx] if tile.buf_name in actual_temporal_reuse_targets)
                        max_count = max(max_count, count)
                return max_count
            
            def max_n_temporal_tiles_per_tiled_op(tiled_ops: list[TiledOperatorSignature]) -> int:
                max_count = 0
                for tiled_op in tiled_ops:
                    count = sum(1 for uop_idx in range(tiled_op.n_uops) for tile in tiled_op.i_tiles[uop_idx] if tile.buf_name in actual_temporal_reuse_targets)
                    max_count = max(max_count, count)
                return max_count
            
            _max_sequential_ctx_length = max(max([tiled_op.n_uops for tiled_op in core_tiled_ops], default=0) for core_tiled_ops in tiled_op_mapping.values())
            
            def compute_sequential_ctx_length() -> int | None:
                _max_reuse_distance = max(max_reuse_distance(core_id, core_tiled_ops) for core_id, core_tiled_ops in tiled_op_mapping.items())
                _max_n_temporal_tiles_per_uop = max(max_n_temporal_tiles_per_uop(core_tiled_ops) for core_tiled_ops in tiled_op_mapping.values())
                _max_n_temporal_tiles_per_tiled_op = max(max_n_temporal_tiles_per_tiled_op(core_tiled_ops) for core_tiled_ops in tiled_op_mapping.values())
                
                if _max_reuse_distance <= self.cache_buffer_slot_num:
                    sequential_ctx_length = _max_sequential_ctx_length
                    logger.debug(f"Reuse distance {_max_reuse_distance} fits in cache buffer slots {self.cache_buffer_slot_num}. Using sequential context with length {_max_sequential_ctx_length}.")
                else:
                    _n_tiled_op_per_reuse_distance = math.ceil(_max_reuse_distance / _max_n_temporal_tiles_per_tiled_op)
                    _ctx_temporal_reuse_degree = math.floor(self.ctx_buffer_slot_num / _n_tiled_op_per_reuse_distance)
                    
                    if _ctx_temporal_reuse_degree >= 2:
                        sequential_ctx_length = math.ceil(self.cache_buffer_slot_num / self.ctx_buffer_slot_num / _max_n_temporal_tiles_per_uop)
                        logger.debug(f"Using temporal reuse pattern with sequential context length {sequential_ctx_length}. Reuse distance {_max_reuse_distance} exceeds cache buffer slots {self.cache_buffer_slot_num}, but temporal reuse degree {_ctx_temporal_reuse_degree} allows for some reuse. Max temporal tiles per tiled op: {_max_n_temporal_tiles_per_tiled_op}, Max temporal tiles per uop: {_max_n_temporal_tiles_per_uop}.")
                    else:
                        sequential_ctx_length = None
                return sequential_ctx_length
            
            while True:
                sequential_ctx_length = compute_sequential_ctx_length()
                
                if sequential_ctx_length is not None:
                    logger.info(f"Chosen sequential context length: {sequential_ctx_length} | Current temporal reuse targets: {actual_temporal_reuse_targets}")
                    break
                
                if len(actual_temporal_reuse_targets) <= 1:
                    sequential_ctx_length = _max_sequential_ctx_length
                    logger.warning(f"Cannot utilize temporal reuse. Falling back to sequential context with length {_max_sequential_ctx_length}.")
                    break
                
                # If we cannot fit the reuse distance in the cache buffer, we can try to reduce the temporal reuse targets to decrease the reuse distance
                logger.warning(f"Reuse distance exceeds cache buffer slots {self.cache_buffer_slot_num}. Attempting to reduce temporal reuse targets to decrease reuse distance. Current temporal reuse targets: {actual_temporal_reuse_targets}")
                actual_temporal_reuse_targets = actual_temporal_reuse_targets[:-1]  # Remove the least prioritized temporal reuse target and recompute
                
            self.temporal_reuse_targets = actual_temporal_reuse_targets  # Update the temporal reuse targets based on the compute sequential context length decision
                  
            cache_context: dict[int, dict[int, tuple[TileSignature, int]]] = {
                core_id: {
                    slot_id: (None, slot_id)  # (tile_sig, lru_counter)
                    for slot_id in range(self.cache_buffer_slot_num)
                }
                for core_id in self.op_sig.core_group.core_ids    
            }  # {core_id: {slot_id: {tile_sig: (cache_slot_id, lru_counter)}}}
            cache_schedules: dict[int, dict[tuple[int, int], dict[TileSignature, tuple[bool, int]]]] = {}  # {core_id: {(tiled_op_idx, uop_idx): {tile_sig: (is_hit, cache_slot_id)}}}
            
            logger.debug(f"Creating thread mapping with {len(tiled_op_mapping)} cores.")
            
            for iii, (core_id, core_tiled_ops) in enumerate(tiled_op_mapping.items()):
                logger.debug(f"thread mapping with core {iii}/{len(tiled_op_mapping)}", end="\r")
                thread = MCA_OperatorGraphCompiler.Thread(core_id)
                thread_mapping[core_id] = thread
                
                n_concurrent_tiled_ops = self.ctx_buffer_slot_num
                grouped_tiled_ops = [core_tiled_ops[i:i + n_concurrent_tiled_ops] for i in range(0, len(core_tiled_ops), n_concurrent_tiled_ops)]
                collected_uops: dict[int, list[int]] = {}
                tiled_op_slot_map: dict[int, int] = {}
                
                def fill_out_cache_schedule():
                    def read_cache(core_id: int, tile: TileSignature) -> bool:
                        target_slot_id, target_lru_counter = None, 0
                        for slot_id, (cached_tile_sig, lru_counter) in cache_context[core_id].items():
                            if cached_tile_sig == tile:
                                target_slot_id = slot_id
                                target_lru_counter = lru_counter

                        if target_slot_id is not None:
                            for slot_id, (cached_tile_sig, lru_counter) in cache_context[core_id].items():
                                if lru_counter < target_lru_counter:
                                    cache_context[core_id][slot_id] = (cached_tile_sig, lru_counter + 1)
                            cache_context[core_id][target_slot_id] = (tile, 0)
                            return True
                        return False
                    
                    def write_cache(core_id: int, tile: TileSignature) -> int:
                        target_slot_id, (_, target_lru_counter) = max(cache_context[core_id].items(), key=lambda item: item[1][1])
                        
                        for slot_id, (cached_tile_sig, lru_counter) in cache_context[core_id].items():
                            if lru_counter < target_lru_counter:
                                cache_context[core_id][slot_id] = (cached_tile_sig, lru_counter + 1)
                        
                        cache_context[core_id][target_slot_id] = (tile, 0)
                        return target_slot_id
                    
                    for tiled_op_idx, uop_indices in collected_uops.items():
                        tiled_op = self.op_sig.tiled_ops[tiled_op_idx]
                        for uop_idx in uop_indices:
                            for tile in tiled_op.i_tiles[uop_idx]:
                                if tile.buf_name in self.temporal_reuse_targets:
                                    # CASE: Cache hit
                                    if read_cache(core_id, tile):
                                        cache_slot_id = next(slot_id for slot_id, (cached_tile_sig, _) in cache_context[core_id].items() if cached_tile_sig == tile)
                                        cache_schedules.setdefault(core_id, {}).setdefault((tiled_op_idx, uop_idx), {})[tile] = (True, cache_slot_id)
                                                
                                    # CASE: Cache miss
                                    elif tile_temporal_reuse_counts[core_id][tile] > 1:  # Only cache if this tile will be reused again in the future
                                        cache_slot_id = write_cache(core_id, tile)
                                        cache_schedules.setdefault(core_id, {}).setdefault((tiled_op_idx, uop_idx), {})[tile] = (False, cache_slot_id)
                                        tile_temporal_reuse_counts[core_id][tile] -= 1
                                        
                def fill_out_thread(thread: MCA_OperatorGraphCompiler.Thread, collected_uops: dict[int, list[int]], tiled_op_slot_map: dict[int, int]):
                    for tiled_op_idx, uop_indices in collected_uops.items():
                        if uop_indices[0] > 0:
                            thread.add_context_load(self.op_sig.op_id, tiled_op_idx, uop_indices[0] - 1, slot_id=tiled_op_slot_map[tiled_op_idx])
                        for uop_idx in uop_indices:
                            thread.add_uop_node(self.op_sig.op_id, tiled_op_idx, uop_idx, output=uop_idx == self.op_sig.tiled_ops[tiled_op_idx].n_uops - 1)
                        if uop_indices[-1] < self.op_sig.tiled_ops[tiled_op_idx].n_uops - 1:
                            thread.add_context_store(self.op_sig.op_id, tiled_op_idx, uop_indices[-1], slot_id=tiled_op_slot_map[tiled_op_idx])
                
                for group in grouped_tiled_ops:
                    tiled_op_slot_map = {tiled_op.tiled_op_id: idx for idx, tiled_op in enumerate(group)}
                    n_uop_per_tiled_op = max(tiled_op.n_uops for tiled_op in group)
                    uop_cursor = 0
                    
                    while uop_cursor < n_uop_per_tiled_op:
                        if uop_cursor % sequential_ctx_length == 0 and len(collected_uops) > 0:
                            fill_out_cache_schedule()
                            fill_out_thread(thread, collected_uops, tiled_op_slot_map)
                            collected_uops.clear()
                            
                        for tiled_op in group:
                            tiled_op_idx = tiled_op.tiled_op_id
                            
                            if uop_cursor < tiled_op.n_uops:
                                collected_uops.setdefault(tiled_op_idx, []).append(uop_cursor)

                        uop_cursor += 1
                          
                    fill_out_cache_schedule()  
                    fill_out_thread(thread, collected_uops, tiled_op_slot_map)
                    collected_uops.clear()
            
            for core_id in self.op_sig.core_group.core_ids:
                for uop_node in thread_mapping[core_id].uop_nodes:
                    if isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                        tiled_op_idx = uop_node.tiled_op_idx
                        uop_idx = uop_node.uop_idx
                        
                        if core_id in cache_schedules:
                            if (tiled_op_idx, uop_idx) in cache_schedules[core_id]:
                                uop_node.cache_schedule = cache_schedules[core_id][(tiled_op_idx, uop_idx)]

            if self.bcast_fifo_depth > 0:
                _core_to_spatial_access_pattern = {core_id: [] for core_id in self.op_sig.core_group.core_ids}
                for core_id, core_uop_nodes in thread_mapping.items():
                    for uop_node in core_uop_nodes.uop_nodes:
                        if isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                            tiled_op_idx = uop_node.tiled_op_idx
                            uop_idx = uop_node.uop_idx
                            
                            for tile_idx, tile in enumerate(self.op_sig.tiled_ops[tiled_op_idx].i_tiles[uop_idx]):
                                _is_cache_hit, _ = uop_node.cache_schedule.get(tile, (False, None))
                                
                                if _is_cache_hit:
                                    continue    # skip if the given tile is cache hit
                                if not self.i_buf_src[tile.buf_name].is_buffer:
                                    continue    # skip if the given tile is not sourced from a buffer (i.e., inter operator pipelining)

                                if tile.buf_name == self.spatial_reuse_target:
                                    _core_to_spatial_access_pattern[core_id].append(
                                        (tile, (core_id, tiled_op_idx, uop_idx, tile_idx))
                                    )
                
                _clustered_spatial_access_pattern = self.optimal_clustering(_core_to_spatial_access_pattern, max_cluster_size=self.broadcast_optimize_max_ref_cnt)
                _bcast_schedules: dict[int, dict[TileSignature, tuple[int, int, int]]] = {}
                _bcast_slot_usages: dict[int, int] = {core_id: 0 for core_id in self.op_sig.core_group.core_ids}
                _uop_tile_sig_to_cluster_id: dict[tuple[int, int, int], list[int]] = {}
                
                for cluster_id, clustered_pattern in enumerate(_clustered_spatial_access_pattern):
                    consumers = []
                    for tile, (core_id, tiled_op_id, uop_idx, tile_idx) in clustered_pattern:
                        consumers.append(core_id)
                        _uop_tile_sig_to_cluster_id.setdefault((core_id, tiled_op_id, uop_idx), []).append(cluster_id)
                        
                    bcast_core_id = min(consumers, key=lambda c: _bcast_slot_usages[c])
                    bcast_slot_id = _bcast_slot_usages[bcast_core_id]
                    bcast_total_ref_count = len(consumers)
                                    
                    _bcast_slot_usages[bcast_core_id] += 1
                    
                    _bcast_schedules.setdefault(cluster_id, {}).setdefault(tile, (bcast_core_id, bcast_slot_id, bcast_total_ref_count))

                for core_id in self.op_sig.core_group.core_ids:
                    for uop_node in thread_mapping[core_id].uop_nodes:
                        if isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                            tiled_op_idx = uop_node.tiled_op_idx
                            uop_idx = uop_node.uop_idx
                            cluster_ids = _uop_tile_sig_to_cluster_id.get((core_id, tiled_op_idx, uop_idx), [])
                            
                            for cluster_id in cluster_ids:
                                uop_node.bcast_schedule.update(_bcast_schedules[cluster_id])
            
            return thread_mapping
        
        def unfreeze(self):
            self.cache_buffer_size = 0
            self.thread_mapping = {}
            
            self.o_tile_store = self.op_sig.buffers[self.op_sig.output_buffer_name].is_allocated
            self.o_tile_sharers = set()
            
            self._is_frozen = False
            
        def freeze(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]=None) -> bool:
            self.cache_buffer_size = self.spad_space_size_per_core - (self.ctx_buffer_size + self.bcast_fifo_size + self.ld_ex_fifo_size + self.ex_st_fifo_size)
            if self.cache_buffer_size <= 0:
                self.unfreeze()
                return False
            
            if tiled_op_mapping is None:
                tiled_op_mapping = self._create_tiled_op_mapping()
                
            thread_mapping = self._create_thread_mapping(tiled_op_mapping)
            if len(thread_mapping) == 0 and len(tiled_op_mapping) > 0:
                self.unfreeze()
                return False

            # If any core with assigned tiled-ops has no schedulable thread, fail freeze.
            for core_id, mapped_tiled_ops in tiled_op_mapping.items():
                if len(mapped_tiled_ops) == 0:
                    continue
                if core_id not in thread_mapping:
                    self.unfreeze()
                    return False
                if thread_mapping[core_id].n_uop_nodes == 0:
                    self.unfreeze()
                    return False

            self.thread_mapping = thread_mapping
            
            self._is_frozen = True
            return True
            
        @property
        def is_frozen(self):
            return self._is_frozen
        
        @property
        def bcast_fifo_depth(self):
            if self.bcast_fifo_size == 0:
                return 0
            return self.bcast_fifo_size // self.bcast_fifo_slot_size
        
        @property
        def ld_ex_fifo_depth(self):
            if self.ld_ex_fifo_slot_size == 0:
                return 0
            return self.ld_ex_fifo_size // self.ld_ex_fifo_slot_size
        
        @property
        def ex_st_fifo_depth(self):
            if self.ex_st_fifo_slot_size == 0:
                return 0
            return self.ex_st_fifo_size // self.ex_st_fifo_slot_size
        
        @property
        def ctx_buffer_slot_num(self):
            return self.ctx_buffer_size // self.ctx_buffer_slot_size
        
        @property
        def cache_buffer_slot_num(self):
            return self.cache_buffer_size // self.cache_buffer_slot_size  
    
    class Environment:
        def __init__(self, recipe: 'MCA_OperatorGraphCompiler.CompileRecipe'):
            self.recipe = recipe
            
            self.op_meta:      dict[str, MCA_OperatorGraphCompiler.OperatorMetadata] = {}
            self.buffers:      dict[str, MCA_TensorBuffer]   = {}
            self.variables:    dict[str, VariableHandle]     = {}
            self.fifo_buffers: dict[str, FIFOBufferHandle]   = {}
            
            self.target_op_order: list[str] = []  # order of operator addition (for debugging and visualization purposes)
            self.grouped_compile_targets: list[list[str]] = []
            
            self.add_variable("global_barrier_arrival_cnt")
            self.add_variable("global_barrier_blocking")
        
        @property
        def global_barrier(self):
            return (
                "global_barrier_arrival_cnt",
                "global_barrier_blocking",
                len(self.recipe.global_core_group.core_ids)
            )
            
        def add_op_sig(self, op_sig: MCA_OperatorSignature):
            buf_names = list(op_sig.buffers.keys())
            
            for buf_name in buf_names:
                buffer = op_sig.buffers[buf_name]
                
                # initially, search for existing buffer handle
                new_buf_name = None
                for existing_buf_name in self.buffers.keys():
                    if buffer is self.buffers[existing_buf_name]:
                        new_buf_name = existing_buf_name
                        break
                
                # if not found, rename buffer to unique name
                if new_buf_name is None:
                    new_buf_name = f"{op_sig.op_id}_{buf_name}"
                
                op_sig.rename_buffers({buf_name: new_buf_name})
            
            self.buffers.update(op_sig.buffers)
            
            op_meta = MCA_OperatorGraphCompiler.OperatorMetadata(op_sig, self.recipe)
            self.op_meta[op_sig.op_id] = op_meta
            self.target_op_order.append(op_sig.op_id)
            
            logger.debug(f"added operator {op_sig.op_id}({', '.join(op_sig.input_buffer_names)}) -> {op_sig.output_buffer_name} to the environment.")
                    
        def add_variable(self, name: str, initial_value: int=0) -> VariableHandle:
            if name in self.variables:
                raise ValueError(f"Variable with name {name} already exists.")
            var_handle = VariableHandle(name, initial_value=initial_value)
            self.variables[name] = var_handle
            return var_handle
        
        def add_fifo_buffer(self, name: str, depth: int, entry_size: int, ptr: Pointer=None) -> FIFOBufferHandle:
            if name in self.fifo_buffers:
                raise ValueError(f"FIFO buffer with name {name} already exists.")
            fifo_handle = FIFOBufferHandle(name, depth, entry_size)
            if ptr is not None:
                if isinstance(ptr, int):
                    ptr = Pointer(addr=ptr)
                fifo_handle.mem_ptr.addr = ptr.addr
            self.fifo_buffers[name] = fifo_handle
            return fifo_handle
            
        def freeze(self):
            if len(self.op_meta) == 0:
                raise ValueError("No operators to compile.")
            
            # TODO: implement multiple operator compilation based on the AI (Arithmetic Intensity) and operator pipelining
            core_group = self.recipe.global_core_group
            for op_meta in self.op_meta.values():
                op_meta.op_sig.initialize_core_group(core_group)
            
            for op_meta in self.op_meta.values():
                if not op_meta.freeze():
                    raise ValueError(f"Failed to freeze operator metadata for operator {op_meta.op_sig.op_id}.")
                logger.debug(f"Successfully froze operator metadata for operator {op_meta.op_sig.op_id}.")
                self.grouped_compile_targets.append([op_meta.op_sig.op_id])
            
            return self
        
    class MemoryState:
        BCAST = "bcast"
        LD_EX = "ld_ex"
        EX_ST = "ex_st"
        
        def __init__(
            self,
            op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata',
            recipe: 'MCA_OperatorGraphCompiler.CompileRecipe',
        ):
            op_sig = op_meta.op_sig
            core_group = op_sig.core_group
            device = recipe.device

            l1_space = device.create_l1_mem_space(op_meta.spad_space_size_per_core, core_group.core_ids)
           
            # SPMs
            self.ctx_descriptors    = {core_id:  MCA_CompiledOperator.IR.SPMDescriptor (f"CORE{core_id}_{op_meta.op_sig.op_id}_CTX",   l1_space.allocate(core_id, op_meta.ctx_buffer_size))   for core_id in core_group.core_ids}
            self.cache_descriptors  = {core_id:  MCA_CompiledOperator.IR.SPMDescriptor (f"CORE{core_id}_{op_meta.op_sig.op_id}_CACHE", l1_space.allocate(core_id, op_meta.cache_buffer_size)) for core_id in core_group.core_ids}
            
            # FIFOs
            self.bcast_descriptors  = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_BCAST", l1_space.allocate(core_id, op_meta.bcast_fifo_size), op_meta.bcast_fifo_slot_size, op_meta.bcast_fifo_depth) for core_id in core_group.core_ids}
            self.ld_ex_descriptors  = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_LD_EX", l1_space.allocate(core_id, op_meta.ld_ex_fifo_size), op_meta.ld_ex_fifo_slot_size, op_meta.ld_ex_fifo_depth) for core_id in core_group.core_ids}
            self.ex_st_descriptors  = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_EX_ST", l1_space.allocate(core_id, op_meta.ex_st_fifo_size), op_meta.ex_st_fifo_slot_size, op_meta.ex_st_fifo_depth) for core_id in core_group.core_ids}
            
            # Off-chip Buffers
            self.tensor_descriptors = {buf_name: MCA_CompiledOperator.IR.TensorBufferDescriptor(buf_name) for buf_name in op_sig.buffer_names}
            
            # States
            self._ctx_states: dict[int, dict[int, TileSignature]] = {core_id: {slot_id: None for slot_id in range(op_meta.ctx_buffer_slot_num)} for core_id in core_group.core_ids}
            
            self._cache_slot_size = op_meta.cache_buffer_slot_size
            self._cache_states: dict[int, dict[int, tuple[TileSignature, int, MCA_CompiledOperator.IR.MEM_COPY_TILE]]] = {core_id: {slot_id: [None, slot_id, None] for slot_id in range(op_meta.cache_buffer_slot_num)} for core_id in core_group.core_ids}  # {core_id: {slot_id: [tile_signature, lru_cnt, last_used_ir]}}
            
            self._ld_ex_states: dict[int, int] = {core_id: 0 for core_id in core_group.core_ids}  # {core_id: slot_num}
            self._ld_ex_tile_to_slot_id: dict[int, dict[TileSignature, int]] = {core_id: {} for core_id in core_group.core_ids}  # {core_id: {tile_signature: slot_id}}
            
            self._ex_st_states: dict[int, int] = {core_id: 0 for core_id in core_group.core_ids}  # {core_id: slot_num}
            self._ex_st_tile_to_slot_id: dict[int, dict[TileSignature, int]] = {core_id: {} for core_id in core_group.core_ids}  # {core_id: {tile_signature: slot_id}}
            
            l1_space.remove()
            
        def ctx_push(self, core_id: int, slot_id: int, tile: TileSignature):
            if self._ctx_states[core_id][slot_id] is not None:
                raise Exception(f"Context slot {slot_id} in core {core_id} is already occupied. Cannot store tile {tile}.")
            self._ctx_states[core_id][slot_id] = tile
        
        def ctx_pop(self, core_id: int, slot_id: int) -> TileSignature:
            tile = self._ctx_states[core_id][slot_id]
            if tile is None:
                raise Exception(f"Context slot {slot_id} in core {core_id} is empty. Cannot load context.")
            self._ctx_states[core_id][slot_id] = None
            return tile
        
        def cache_write(self, core_id: int, tile_sig: TileSignature, slot_id: int, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE):
            evicted_tile_sig, _, evicted_ir = self._cache_states[core_id][slot_id]
            
            if evicted_ir is not None:
                ir.wait_ir_idx.append(evicted_ir.ir_idx)
            
            self._cache_states[core_id][slot_id][0] = tile_sig
            self._cache_states[core_id][slot_id][2] = ir
            
            ir.dsts.append(self.cache_descriptors[core_id].ref(tile_sig=tile_sig, offset=slot_id * self._cache_slot_size))
            
        def cache_read(self, core_id: int, tile_sig: TileSignature, slot_id: int, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE) -> bool:
            cached_tile_sig, _, last_used_ir = self._cache_states[core_id][slot_id]
            if cached_tile_sig == tile_sig:
                if last_used_ir is not None:
                    ir.wait_ir_idx.append(last_used_ir.ir_idx)
                    
                ir.src = self.cache_descriptors[core_id].ref(tile_sig=tile_sig, offset=slot_id * self._cache_slot_size)
                return True
            return False
        
        def ld_ex_push(self, core_id: int, tile_sig: TileSignature) -> int:
            slot_id = self._ld_ex_states[core_id]
            self._ld_ex_tile_to_slot_id[core_id][tile_sig] = slot_id
            self._ld_ex_states[core_id] += 1
            return slot_id
        
        def ld_ex_pop(self, core_id: int, tile_sig: TileSignature) -> int:
            if tile_sig not in self._ld_ex_tile_to_slot_id[core_id]:
                raise Exception(f"Tile {tile_sig} is not found in the load-execute buffer of core {core_id}.")
            slot_id = self._ld_ex_tile_to_slot_id[core_id][tile_sig]
            return slot_id
        
        def ex_st_push(self, core_id: int, tile_sig: TileSignature) -> int:
            slot_id = self._ex_st_states[core_id]
            self._ex_st_tile_to_slot_id[core_id][tile_sig] = slot_id
            self._ex_st_states[core_id] += 1
            return slot_id
        
        def ex_st_pop(self, core_id: int, tile_sig: TileSignature) -> int:
            if tile_sig not in self._ex_st_tile_to_slot_id[core_id]:
                raise Exception(f"Tile {tile_sig} is not found in the execute-store buffer of core {core_id}.")
            slot_id = self._ex_st_tile_to_slot_id[core_id][tile_sig]
            return slot_id

    def __init__(self):
        self._op_sigs: dict[str, MCA_OperatorSignature] = {}
        self._op_order: list[str] = []  # order of operator addition (for debugging and visualization purposes)
    
    def add_op(self, op_sig: MCA_OperatorSignature) -> str:
        suffix = 1
        while f"{op_sig.op_id}_{suffix}" in self._op_sigs:
            suffix += 1
        op_sig.op_id = f"{op_sig.op_id}_{suffix}"
        self._op_sigs[op_sig.op_id] = op_sig
        self._op_order.append(op_sig.op_id)
        return op_sig.op_id
    
    def clear_ops(self):
        self._op_sigs = {}
        self._op_order = []
        
    def compile_grouped_target_ops(self, env: 'MCA_OperatorGraphCompiler.Environment', op_ids: list[str]) -> dict[str, MCA_CompiledOperator]:
        op_metas = {target_op_id: env.op_meta[target_op_id] for target_op_id in op_ids}
        
        compiled_ops    = {op_id: MCA_CompiledOperator(env, op_meta) for op_id, op_meta in op_metas.items()}
        mem_states      = {op_id: MCA_OperatorGraphCompiler.MemoryState(op_meta, env.recipe) for op_id, op_meta in op_metas.items()} 
        thread_mappings = {op_id: op_meta.thread_mapping for op_id, op_meta in op_metas.items()}
        
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            op_sig = op_meta.op_sig
            core_group = op_sig.core_group
            mem_state = mem_states[op_id]
            
            for core_id in core_group.core_ids:
                env.add_fifo_buffer(mem_state.bcast_descriptors[core_id].buf_name, op_meta.bcast_fifo_depth, op_meta.bcast_fifo_slot_size, mem_state.bcast_descriptors[core_id].ptr)
                env.add_fifo_buffer(mem_state.ld_ex_descriptors[core_id].buf_name, op_meta.ld_ex_fifo_depth, op_meta.ld_ex_fifo_slot_size, mem_state.ld_ex_descriptors[core_id].ptr)
                env.add_fifo_buffer(mem_state.ex_st_descriptors[core_id].buf_name, op_meta.ex_st_fifo_depth, op_meta.ex_st_fifo_slot_size, mem_state.ex_st_descriptors[core_id].ptr)
        
        # STAGE 2: Create compiled ops and update memory states while iteratively resolving tile-level dependencies
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            thread_mapping = thread_mappings[op_id]
            mem_state = mem_states[op_id]
            
            for core_id, thread in tqdm.tqdm(thread_mapping.items(), desc=f"Compiling operator {op_id} on cores", leave=False, disable=not logger.is_current_debug_log_level()):           
                for iii, uop_node in enumerate(thread.uop_nodes):
                    if isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.ContextLoadNode):
                        slot_id = uop_node.slot_id
                        tiled_op_idx = uop_node.tiled_op_idx
                        uop_idx = uop_node.uop_idx
                        tile = mem_state.ctx_pop(core_id, slot_id)

                        if tile != op_meta.op_sig.tiled_ops[tiled_op_idx].o_tile or tile is None:
                            raise Exception(f"Context slot {slot_id} in core {core_id} does not contain the expected tile for operator {op_id}. tile: {tile}, expected: {op_meta.op_sig.tiled_ops[tiled_op_idx].o_tile}.")
                        
                        compiled_ops[op_id].add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_CTX_LOAD(
                            op_id, tiled_op_idx, uop_idx, tile, mem_state.ctx_descriptors[core_id].ref(tile, offset=slot_id * op_meta.ctx_buffer_slot_size)
                        ))
                        
                    elif isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.ContextStoreNode):
                        slot_id = uop_node.slot_id
                        tiled_op_idx = uop_node.tiled_op_idx
                        uop_idx = uop_node.uop_idx
                        tile = op_meta.op_sig.tiled_ops[tiled_op_idx].o_tile
                        
                        mem_state.ctx_push(core_id, slot_id, tile)
                        
                        compiled_ops[op_id].add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_CTX_STORE(
                            op_id, tiled_op_idx, uop_idx, tile, mem_state.ctx_descriptors[core_id].ref(tile, offset=slot_id * op_meta.ctx_buffer_slot_size)
                        ))
                        
                        compiled_ops[op_id].new_stage(core_id)  # add new stage to reduce the size of kernel object (compile time overhead issue)
                    
                    elif isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                        tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        o_tile = tiled_op_sig.o_tile
                        
                        for i_tile in i_tiles:
                            dst = mem_state.ld_ex_descriptors[core_id].ref(i_tile, slot_id=mem_state.ld_ex_push(core_id, i_tile), ref_cnt=1)
                            ir = MCA_CompiledOperator.IR.MEM_COPY_TILE(None, dst)   # src undefined until the cache hit is verified
                            
                            _is_cache_schedule_valid, _is_cache_hit, _cache_slot_id = False, None, None
                            _is_bcast_schedule_valid, _bcast_core_id, _bcast_slot_id, _bcast_ref_cnt = False, None, None, None
                                
                            if i_tile in uop_node.cache_schedule:
                                _is_cache_hit, _cache_slot_id = uop_node.cache_schedule[i_tile]
                                _is_cache_schedule_valid = True
                            
                            if i_tile in uop_node.bcast_schedule:
                                _bcast_core_id, _bcast_slot_id, _bcast_ref_cnt = uop_node.bcast_schedule[i_tile]
                                _is_bcast_schedule_valid = True
                            
                            # CASE: cache hit
                            if _is_cache_schedule_valid and _is_cache_hit:
                                mem_state.cache_read(core_id, i_tile, _cache_slot_id, ir)
                            # CASE: broadcast target (spatial reuse)
                            elif _is_bcast_schedule_valid:
                                if _bcast_core_id == core_id:
                                    ir.src = mem_state.tensor_descriptors[i_tile.buf_name].ref(i_tile)
                                    ir.dsts.append(mem_state.bcast_descriptors[_bcast_core_id].ref(i_tile, _bcast_slot_id, _bcast_ref_cnt - 1))
                                else:
                                    ir.src = mem_state.bcast_descriptors[_bcast_core_id].ref(i_tile, _bcast_slot_id, 1)
                            else:
                                ir.src = mem_state.tensor_descriptors[i_tile.buf_name].ref(i_tile)
                            
                            # Write to cache if scheduled, regardless of hit or miss, to update the cache state and enable subsequent hits
                            if _is_cache_schedule_valid and not _is_cache_hit:
                                mem_state.cache_write(core_id, i_tile, _cache_slot_id, ir)
                                
                            compiled_ops[op_id].add_load_ir(core_id, ir)
                        
                        compiled_ops[op_id].add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_UOP(
                            op_id, uop_node.tiled_op_idx, uop_node.uop_idx,
                            i_tile_refs=[mem_state.ld_ex_descriptors[core_id].ref(i_tile, slot_id=mem_state.ld_ex_pop(core_id, i_tile), ref_cnt=1) for i_tile in i_tiles],
                            o_tile_ref=mem_state.ex_st_descriptors[core_id].ref(o_tile, slot_id=mem_state.ex_st_push(core_id, o_tile), ref_cnt=1) if uop_node.output else None,
                            dtype=i_tiles[0].dtype if len(i_tiles) > 0 else None,
                            acc_dtype=o_tile.dtype
                        ))
                        
                        if uop_node.output:
                            src = mem_state.ex_st_descriptors[core_id].ref(o_tile, slot_id=mem_state.ex_st_pop(core_id, o_tile), ref_cnt=1)
                            ir = MCA_CompiledOperator.IR.MEM_COPY_TILE(src)
                            
                            if op_meta.o_tile_store:
                                ir.dsts.append(mem_state.tensor_descriptors[o_tile.buf_name].ref(o_tile))

                            compiled_ops[op_id].add_store_ir(core_id, ir)
                            compiled_ops[op_id].new_stage(core_id)  # add new stage to reduce the size of kernel object (compile time overhead issue)
        
        return compiled_ops
        
    def compile(self, recipe: 'MCA_OperatorGraphCompiler.CompileRecipe') -> MCA_CompiledProgram:  
        # Initialize environment
        env = MCA_OperatorGraphCompiler.Environment(recipe)
            
        for op_id in self._op_order:
            op_sig = self._op_sigs[op_id]
            env.add_op_sig(op_sig)
            
        env.freeze()  # freeze the environment to create L1 memory space with pipeline pattern
        
        compiled_ops: dict[str, MCA_CompiledOperator] = {}
        
        for grouped_op_ids in env.grouped_compile_targets:
            compiled_ops.update(self.compile_grouped_target_ops(env, grouped_op_ids))
        
        return MCA_CompiledProgram(env, compiled_ops)
