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
    def get_ld_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', ir_seq: 'list[MCA_CompiledOperator.IR.Base]') -> KernelPrototype: ...
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
        def __init__(self):
            self.loads:    list[MCA_CompiledOperator.IR.Base] = []
            self.executes: list[MCA_CompiledOperator.IR.Base] = []
            self.stores:   list[MCA_CompiledOperator.IR.Base] = []
            
        def add_load_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.loads.append(cmd)
            
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
                "loads":    [ir.signature() for ir in self.loads    if not isinstance(ir, MCA_CompiledOperator.IR.NOP)],
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
            return len(self.loads) == 0 and len(self.executes) == 0 and len(self.stores) == 0

    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata'):
        self._env = env
        self._op_id = op_meta.op_sig.op_id
        self._kernel_template = op_meta.op_sig.kernel_template
        
        self._ld_ir_counters: dict[int, int] = {core_id: 0 for core_id in op_meta.op_sig.core_group.core_ids}  # {core_id: counter}
        
        # self._ld_ir_locks: dict[int, str] = {
        #     core_id: self._env.add_variable(f"{self._op_id}_core_{core_id}_ld_ir_lock").handle_name 
        #     for core_id in op_meta.op_sig.core_group.core_ids
        # }
        
        self._mappings: dict[int, list[MCA_CompiledOperator.Stage]] = {
            core_id: [MCA_CompiledOperator.Stage()] 
            for core_id in op_meta.op_sig.core_group.core_ids
        }  # {core_id: [stage1, stage2, ...]}
        
    def new_stage(self, core_id: int) -> 'MCA_CompiledOperator.Stage':
        stage = MCA_CompiledOperator.Stage()
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
        
        # OPERATOR BARRIER
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
                
                self._kernel_template.get_ld_thread_kernel(core, self._env, stage.loads).dispatch(f"LD")
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
        class ReuseTarget(enum.Enum):
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
            core_groups: list[MCA_CoreGroup | MTA_CoreGrid],
            spad_space_size_per_core: int,
            context_buffer_slot_num: int=16,
            fifo_buffer_slot_num: int=16,
            temporal_reuse_target: ReuseTarget=ReuseTarget.ALL,
            spatial_reuse_target: ReuseTarget=ReuseTarget.SINGLE_MAIN,
        ):
            if len(core_groups) == 0:
                raise ValueError("At least one core group must be provided.")
            if not isinstance(core_groups[0], (MCA_CoreGroup, MTA_CoreGrid, list)):
                core_groups = [core_groups]

            self.device                         = device
            self.core_groups                    = core_groups
            self.spad_space_size_per_core       = spad_space_size_per_core
            self.context_buffer_slot_num        = context_buffer_slot_num
            self.fifo_buffer_slot_num           = fifo_buffer_slot_num
            self.temporal_reuse_target          = temporal_reuse_target if isinstance(temporal_reuse_target, self.ReuseTarget) else self.ReuseTarget(temporal_reuse_target)
            self.spatial_reuse_target           = spatial_reuse_target  if isinstance(spatial_reuse_target,  self.ReuseTarget) else self.ReuseTarget(spatial_reuse_target)
            
        @property
        def global_core_group(self) -> MCA_CoreGroup:
            return MCA_CoreGroup.merge_core_groups(self.core_groups)
            
    class Thread:
        class _NodeBase(metaclass=abc.ABCMeta):
            pass
        
        class UopNode(_NodeBase):
            def __init__(
                self, 
                op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False, 
                cache_schedule: dict[TileSignature, tuple[bool, int]]=None, 
                spatial_x_schedule: dict[TileSignature, dict[str, tuple[int, int, int]]]=None,
                spatial_y_schedule: dict[TileSignature, dict[str, tuple[int, int, int]]]=None,
            ):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.output = output
                self.cache_schedule = cache_schedule if cache_schedule is not None else {}  # {tile_sig: (is_cache_hit, cache_slot_id)}
                self.spatial_x_schedule = spatial_x_schedule if spatial_x_schedule is not None else {}  # {tile_sig: {"src": (src_core_id, src_fifo_entry_id, src_fifo_ref_cnt), "dst": (dst_core_id, dst_fifo_entry_id, dst_fifo_ref_cnt)}}
                self.spatial_y_schedule = spatial_y_schedule if spatial_y_schedule is not None else {}  # {tile_sig: {"src": (src_core_id, src_fifo_entry_id, src_fifo_ref_cnt), "dst": (dst_core_id, dst_fifo_entry_id, dst_fifo_ref_cnt)}}

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
            
        def add_uop_node(
            self,
            op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False, 
            cache_schedule: dict[TileSignature, tuple[bool, int]]=None, 
            spatial_x_schedule: dict[TileSignature, dict[str, tuple[int, int, int]]]=None,
            spatial_y_schedule: dict[TileSignature, dict[str, tuple[int, int, int]]]=None,
        ):
            uop_node = self.UopNode(op_id, tiled_op_idx, uop_idx, output=output, cache_schedule=cache_schedule, spatial_x_schedule=spatial_x_schedule, spatial_y_schedule=spatial_y_schedule)
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
    
    class MemoryState:
        def __init__(
            self,
            env: 'MCA_OperatorGraphCompiler.Environment',
            op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata',
        ):
            self.env = env
            self.op_meta = op_meta
            op_sig = op_meta.op_sig
            core_group = op_sig.core_group
            device = env.recipe.device
            l1_space = device.create_l1_mem_space(op_meta.spad_space_size_per_core, core_group.core_ids)
            
            self.ctx_descriptors = {}
            self.cache_descriptors = {}
            self.spatial_x_descriptors = {}
            self.spatial_y_descriptors = {}
            self.ld_ex_descriptors = {}
            self.ex_st_descriptors = {}
            
            for core_id in core_group.core_ids:
                self.ctx_descriptors[core_id] = MCA_CompiledOperator.IR.SPMDescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_CTX",
                    l1_space.allocate(core_id, op_meta.ctx_buffer_size),
                )
                self.cache_descriptors[core_id] = MCA_CompiledOperator.IR.SPMDescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_CACHE",
                    l1_space.allocate(core_id, op_meta.cache_buffer_size),
                )
                self.spatial_x_descriptors[core_id] = MCA_CompiledOperator.IR.FIFODescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_SPATIAL_X",
                    l1_space.allocate(core_id, op_meta.bcast_fifo_size),
                    op_meta.bcast_fifo_slot_size,
                    op_meta.bcast_fifo_depth,
                )
                self.spatial_y_descriptors[core_id] = MCA_CompiledOperator.IR.FIFODescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_SPATIAL_Y",
                    l1_space.allocate(core_id, op_meta.bcast_fifo_size),
                    op_meta.bcast_fifo_slot_size,
                    op_meta.bcast_fifo_depth,
                )
                self.ld_ex_descriptors[core_id] = MCA_CompiledOperator.IR.FIFODescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_LD_EX",
                    l1_space.allocate(core_id, op_meta.ld_ex_fifo_size),
                    op_meta.ld_ex_fifo_slot_size,
                    op_meta.ld_ex_fifo_depth,
                )
                self.ex_st_descriptors[core_id] = MCA_CompiledOperator.IR.FIFODescriptor(
                    f"CORE{core_id}_{op_sig.op_id}_EX_ST",
                    l1_space.allocate(core_id, op_meta.ex_st_fifo_size),
                    op_meta.ex_st_fifo_slot_size,
                    op_meta.ex_st_fifo_depth,
                )
            
            self.tensor_descriptors = {
                buf_name: MCA_CompiledOperator.IR.TensorBufferDescriptor(buf_name)
                for buf_name in op_sig.buffer_names
            }
            
            self.ctx_states = {
                core_id: {slot_id: None for slot_id in range(op_meta.ctx_buffer_slot_num)}
                for core_id in core_group.core_ids
            }
            self.cache_states = {
                core_id: {slot_id: [None, None, None] for slot_id in range(op_meta.cache_buffer_slot_num)}
                for core_id in core_group.core_ids
            }
            
            # Spatial FIFO entry IDs and cache slot IDs are already fixed in
            # Thread.UopNode schedules. LD_EX and EX_ST are transient pipeline
            # FIFOs, so the compiler still assigns their logical entries while
            # creating load/execute/store IRs.
            self.ld_ex_entry_counters = defaultdict(int)
            self.ex_st_entry_counters = defaultdict(int)
            self.ld_ex_tile_entries = {core_id: {} for core_id in core_group.core_ids}
            self.ex_st_tile_entries = {core_id: {} for core_id in core_group.core_ids}
            
            l1_space.remove()
        
        def register_fifos(self):
            for core_id in self.op_meta.op_sig.core_group.core_ids:
                for desc in (
                    self.spatial_x_descriptors[core_id],
                    self.spatial_y_descriptors[core_id],
                    self.ld_ex_descriptors[core_id],
                    self.ex_st_descriptors[core_id],
                ):
                    self.env.add_fifo_buffer(desc.buf_name, desc.slot_num, desc.slot_size, desc.ptr)
        
        def ctx_push(self, core_id: int, slot_id: int, tile: TileSignature):
            if self.ctx_states[core_id][slot_id] is not None:
                raise Exception(f"Context slot {slot_id} in core {core_id} is already occupied.")
            self.ctx_states[core_id][slot_id] = tile
        
        def ctx_pop(self, core_id: int, slot_id: int) -> TileSignature:
            tile = self.ctx_states[core_id][slot_id]
            if tile is None:
                raise Exception(f"Context slot {slot_id} in core {core_id} is empty.")
            self.ctx_states[core_id][slot_id] = None
            return tile
        
        def cache_ref(self, core_id: int, tile: TileSignature, slot_id: int):
            return self.cache_descriptors[core_id].ref(tile, offset=slot_id * self.op_meta.cache_buffer_slot_size)
        
        def cache_read(
            self,
            core_id: int,
            tile: TileSignature,
            slot_id: int,
            stage: MCA_CompiledOperator.Stage,
            ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
        ):
            cached_tile, last_write_ir, last_write_stage = self.cache_states[core_id][slot_id]
            if cached_tile != tile:
                raise Exception(f"Cache miss while compiling cache-hit schedule: core={core_id}, slot={slot_id}, tile={tile}, cached={cached_tile}")
            if last_write_stage is stage and last_write_ir is not None and last_write_ir.ir_idx is not None:
                ir.wait_ir_idx.append(last_write_ir.ir_idx)
            ir.src = self.cache_ref(core_id, tile, slot_id)
        
        def cache_write(
            self,
            core_id: int,
            tile: TileSignature,
            slot_id: int,
            stage: MCA_CompiledOperator.Stage,
            ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
        ):
            _, evicted_last_write_ir, evicted_last_write_stage = self.cache_states[core_id][slot_id]
            if evicted_last_write_stage is stage and evicted_last_write_ir is not None and evicted_last_write_ir.ir_idx is not None:
                ir.wait_ir_idx.append(evicted_last_write_ir.ir_idx)
            self.cache_states[core_id][slot_id] = [tile, ir, stage]
            ir.dsts.append(self.cache_ref(core_id, tile, slot_id))
        
        def replace_cache_writer(
            self,
            core_id: int,
            tile: TileSignature,
            slot_id: int,
            old_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
            new_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
            stage: MCA_CompiledOperator.Stage,
        ):
            cached_tile, last_write_ir, last_write_stage = self.cache_states[core_id][slot_id]
            if cached_tile == tile and last_write_ir is old_ir and last_write_stage is stage:
                self.cache_states[core_id][slot_id] = [tile, new_ir, stage]
        
        def ld_ex_push(self, core_id: int, tile: TileSignature) -> int:
            entry_id = self.ld_ex_entry_counters[core_id]
            self.ld_ex_entry_counters[core_id] += 1
            self.ld_ex_tile_entries[core_id].setdefault(tile, deque()).append(entry_id)
            return entry_id
        
        def ld_ex_pop(self, core_id: int, tile: TileSignature) -> int:
            entries = self.ld_ex_tile_entries[core_id].get(tile)
            if not entries:
                raise Exception(f"Tile {tile} is not available in LD_EX FIFO state for core {core_id}.")
            return entries.popleft()
        
        def ex_st_push(self, core_id: int, tile: TileSignature) -> int:
            entry_id = self.ex_st_entry_counters[core_id]
            self.ex_st_entry_counters[core_id] += 1
            self.ex_st_tile_entries[core_id].setdefault(tile, deque()).append(entry_id)
            return entry_id
        
        def ex_st_pop(self, core_id: int, tile: TileSignature) -> int:
            entries = self.ex_st_tile_entries[core_id].get(tile)
            if not entries:
                raise Exception(f"Tile {tile} is not available in EX_ST FIFO state for core {core_id}.")
            return entries.popleft()
            
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
            
            self.i_buf_src: dict[str, MCA_OperatorGraphCompiler.OperatorMetadata.SrcType] = {
                buf_name: MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.BUFFER() 
                for buf_name in op_sig.input_buffer_names
            }
            
            _n_i_tiles_per_uop = 0
            _i_tile_size = 0
            _o_tile_size = 0
            
            for tiled_op in op_sig.tiled_ops:
                for uop_idx in range(tiled_op.n_uops):
                    _n_i_tiles_per_uop = max(_n_i_tiles_per_uop, len(tiled_op.i_tiles[uop_idx]))
                    _i_tile_size = max(_i_tile_size, max(tile.tile_size for tile in tiled_op.i_tiles[uop_idx]))
                _o_tile_size = max(_o_tile_size, tiled_op.o_tile.tile_size)

            self._ld_ex_fifo_slot_num = recipe.fifo_buffer_slot_num * _n_i_tiles_per_uop
            self._ex_st_fifo_slot_num = recipe.fifo_buffer_slot_num
            self._bcast_fifo_slot_num = recipe.fifo_buffer_slot_num
            self._context_buffer_slot_num = recipe.context_buffer_slot_num
            
            self._ld_ex_fifo_entry_size = _i_tile_size
            self._ex_st_fifo_entry_size = _o_tile_size
            self._bcast_fifo_entry_size = _i_tile_size
            self._context_buffer_slot_size = _o_tile_size
            
            self._ld_ex_fifo_size = self._ld_ex_fifo_slot_num * self._ld_ex_fifo_entry_size
            self._ex_st_fifo_size = self._ex_st_fifo_slot_num * self._ex_st_fifo_entry_size
            self._bcast_fifo_size = self._bcast_fifo_slot_num * self._bcast_fifo_entry_size
            self._context_buffer_size = self._context_buffer_slot_num * self._context_buffer_slot_size
            
            self.spatial_reuse_target = recipe.spatial_reuse_target
            self.temporal_reuse_target = recipe.temporal_reuse_target
            
            # Initialized after freezing the operator metadata
            self.cache_buffer_slot_size = _i_tile_size
            self.cache_buffer_size = 0  # self.spad_space_size_per_core - (self._ld_ex_fifo_size + self._ex_st_fifo_size + self._bcast_fifo_size + self._context_buffer_size)
            self.cache_buffer_slot_num = 0  # self.cache_buffer_size // self.cache_buffer_slot_size
            
            self.thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            self.o_tile_store = op_sig.buffers[op_sig.output_buffer_name].is_allocated  # if the output buffer is allocated, the computation result should be updated to the buffer
            self.o_tile_sharers: set[str] = set()  # set of op_ids that directly consume this operator's output tiles (tile-level sharers via SHARED area)
            
            self._is_frozen = False
            
        def create_tiled_op_mapping(self) -> 'dict[int, list[TiledOperatorSignature]]':
            # STEP 1: Create tiled op mapping
            _tiled_op_mapping: dict[int, list[TiledOperatorSignature]] = None
            
            if isinstance(self.op_sig.core_group, MTA_CoreGrid):
                _tiled_op_mapping = {}
                output_tile_grid = self.op_sig.buffers[self.op_sig.output_buffer_name].tile_grid
                output_tile_grid_per_shard = self.op_sig.buffers[self.op_sig.output_buffer_name].tile_grid_per_shard
                core_grid = self.op_sig.core_group.shape
                _is_x_grid_available = output_tile_grid[-1] >= (core_grid[-1] * 0.5)
                _is_y_grid_available = output_tile_grid[-2] >= (core_grid[-2] * 0.5)
                
                if _is_x_grid_available or _is_y_grid_available:
                    for tiled_op in self.op_sig.tiled_ops:
                        _o_tile_coords = tiled_op.o_tile.coords
                        _o_tile_x = _o_tile_coords[1] * output_tile_grid_per_shard[1] + _o_tile_coords[3]
                        _o_tile_y = _o_tile_coords[0] * output_tile_grid_per_shard[0] + _o_tile_coords[2]
                        
                        _core_x = _o_tile_x % core_grid[1]
                        _core_y = _o_tile_y % core_grid[0]
                        _core_id = self.op_sig.core_group.get_core_id(_core_y, _core_x)
                        _tiled_op_mapping.setdefault(_core_id, []).append(tiled_op)
                elif len(self.op_sig.tiled_ops) > 0:
                    _tiled_op_mapping = None
            
            if _tiled_op_mapping is None:
                _tiled_op_mapping = {}
                core_ids = self.op_sig.core_group.core_ids
                for idx, tiled_op in enumerate(self.op_sig.tiled_ops):
                    _core_id = core_ids[idx % len(core_ids)]
                    _tiled_op_mapping.setdefault(_core_id, []).append(tiled_op)
            return _tiled_op_mapping
            
        def create_thread_mapping(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]) -> 'dict[int, MCA_OperatorGraphCompiler.Thread]':
            # STEP 2: Determine the reuse strategy 
            SPATIAL_X = "SPATIAL_X"
            SPATIAL_Y = "SPATIAL_Y"
            TEMPORAL = "TEMPORAL"
            
            _reuse_strategy: dict[str, str] = {SPATIAL_X: None, SPATIAL_Y: None, TEMPORAL: None}
            _input_buffer_names = [buf_name for buf_name in self.op_sig.input_buffer_names if self.op_sig.buffers[buf_name] is not None]
            _spatial_reuse_target_order = sorted(_input_buffer_names, key=lambda buf_name: (self.op_sig.buffers[buf_name].mem_space.is_main, self.op_sig.buffers[buf_name].total_size), reverse=True)
            _temporal_reuse_target_order = sorted(_input_buffer_names, key=lambda buf_name: (self.op_sig.buffers[buf_name].mem_space.is_main, self.op_sig.buffers[buf_name].total_size), reverse=True)
            
            if self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL_MAIN:
                _spatial_reuse_target_order = [buf_name for buf_name in _spatial_reuse_target_order if self.op_sig.buffers[buf_name].mem_space.is_main]
            elif self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN:
                _spatial_reuse_target_order = [buf_name for buf_name in _spatial_reuse_target_order if self.op_sig.buffers[buf_name].mem_space.is_main][:1]
            elif self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL_L1:
                _spatial_reuse_target_order = [buf_name for buf_name in _spatial_reuse_target_order if not self.op_sig.buffers[buf_name].mem_space.is_main]
            elif self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_L1:
                _spatial_reuse_target_order = [buf_name for buf_name in _spatial_reuse_target_order if not self.op_sig.buffers[buf_name].mem_space.is_main][:1]
            elif self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE:
                _spatial_reuse_target_order = _spatial_reuse_target_order[:1]
                
            if self.temporal_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL_MAIN:
                _temporal_reuse_target_order = [buf_name for buf_name in _temporal_reuse_target_order if self.op_sig.buffers[buf_name].mem_space.is_main]
            elif self.temporal_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN:
                _temporal_reuse_target_order = [buf_name for buf_name in _temporal_reuse_target_order if self.op_sig.buffers[buf_name].mem_space.is_main][:1]
            elif self.temporal_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL_L1:
                _temporal_reuse_target_order = [buf_name for buf_name in _temporal_reuse_target_order if not self.op_sig.buffers[buf_name].mem_space.is_main]
            elif self.temporal_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_L1:
                _temporal_reuse_target_order = [buf_name for buf_name in _temporal_reuse_target_order if not self.op_sig.buffers[buf_name].mem_space.is_main][:1]
            elif self.temporal_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE:
                _temporal_reuse_target_order = _temporal_reuse_target_order[:1]
            
            def _core_position(core_id: int) -> tuple[int, int]:
                if isinstance(self.op_sig.core_group, MTA_CoreGrid):
                    idx = self.op_sig.core_group.core_ids.index(core_id)
                    return idx // self.op_sig.core_group.shape[1], idx % self.op_sig.core_group.shape[1]
                return 0, self.op_sig.core_group.core_ids.index(core_id)
            
            def _axis_key(core_id: int, axis: str) -> tuple[int, int]:
                y, x = _core_position(core_id)
                return (y, x) if axis == SPATIAL_X else (x, y)
            
            def _tile_sequence(core_id: int, buf_name: str) -> list[TileSignature]:
                seq = []
                for tiled_op in tiled_op_mapping.get(core_id, []):
                    for uop_idx in range(tiled_op.n_uops):
                        for tile in tiled_op.i_tiles[uop_idx]:
                            if tile.buf_name == buf_name:
                                seq.append(tile)
                return seq
            
            def _has_spatial_reuse(buf_name: str, axis: str) -> bool:
                grouped_core_ids: dict[int, list[int]] = defaultdict(list)
                for core_id in self.op_sig.core_group.core_ids:
                    fixed_coord, moving_coord = _axis_key(core_id, axis)
                    grouped_core_ids[fixed_coord].append(core_id)
                
                for core_ids in grouped_core_ids.values():
                    core_ids.sort(key=lambda core_id: _axis_key(core_id, axis)[1])
                    seqs = [_tile_sequence(core_id, buf_name) for core_id in core_ids]
                    max_seq_len = max((len(seq) for seq in seqs), default=0)
                    for seq_idx in range(max_seq_len):
                        tiles = [seq[seq_idx] for seq in seqs if seq_idx < len(seq)]
                        if len(tiles) < 2:
                            continue
                        if len(set(tiles)) < len(tiles):
                            return True
                return False
            
            for buf_name in _spatial_reuse_target_order:
                if self.spatial_reuse_target == MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.IGNORE:
                    break
                if _reuse_strategy[SPATIAL_X] is None and _has_spatial_reuse(buf_name, SPATIAL_X):
                    _reuse_strategy[SPATIAL_X] = buf_name
                    continue
                if _reuse_strategy[SPATIAL_Y] is None and _has_spatial_reuse(buf_name, SPATIAL_Y):
                    _reuse_strategy[SPATIAL_Y] = buf_name
            
            _temporal_reuse_targets = []
            if self.temporal_reuse_target != MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.IGNORE:
                _temporal_reuse_targets = list(_temporal_reuse_target_order)
                if len(_temporal_reuse_targets) > 0:
                    _reuse_strategy[TEMPORAL] = _temporal_reuse_targets[0]
            
            # STEP 3: Create the final per-core execution order first. Spatial FIFO
            # schedules must be derived from this order so producer and consumer
            # access the FIFO entries in the same sequence even with context switching.
            thread_mapping = {
                core_id: MCA_OperatorGraphCompiler.Thread(core_id)
                for core_id in self.op_sig.core_group.core_ids
            }
            
            execution_plans: dict[int, list[tuple[str, TiledOperatorSignature, int, int | None]]] = {
                core_id: []
                for core_id in self.op_sig.core_group.core_ids
            }
            
            def _append_uop(core_id: int, tiled_op: TiledOperatorSignature, uop_idx: int):
                execution_plans[core_id].append(("uop", tiled_op, uop_idx, None))
            
            def _append_context_load(core_id: int, tiled_op: TiledOperatorSignature, uop_idx: int, slot_id: int):
                execution_plans[core_id].append(("ctx_load", tiled_op, uop_idx, slot_id))
            
            def _append_context_store(core_id: int, tiled_op: TiledOperatorSignature, uop_idx: int, slot_id: int):
                execution_plans[core_id].append(("ctx_store", tiled_op, uop_idx, slot_id))
            
            def _has_temporal_cache_pressure(core_id: int, tiled_ops: list[TiledOperatorSignature]) -> bool:
                if self.cache_buffer_slot_num <= 0 or len(_temporal_reuse_targets) == 0:
                    return False
                
                temporal_tiles = set()
                for tiled_op in tiled_ops:
                    for uop_idx in range(tiled_op.n_uops):
                        for tile in tiled_op.i_tiles[uop_idx]:
                            if tile.buf_name in _temporal_reuse_targets:
                                temporal_tiles.add(tile)
                return len(temporal_tiles) > self.cache_buffer_slot_num
            
            for core_id in self.op_sig.core_group.core_ids:
                core_tiled_ops = tiled_op_mapping.get(core_id, [])
                
                if _has_temporal_cache_pressure(core_id, core_tiled_ops) and self._context_buffer_slot_num > 1:
                    for group_start in range(0, len(core_tiled_ops), self._context_buffer_slot_num):
                        tiled_op_group = core_tiled_ops[group_start:group_start + self._context_buffer_slot_num]
                        tiled_op_slot_map = {tiled_op.tiled_op_id: slot_id for slot_id, tiled_op in enumerate(tiled_op_group)}
                        max_uops = max((tiled_op.n_uops for tiled_op in tiled_op_group), default=0)
                        
                        for uop_idx in range(max_uops):
                            for tiled_op in tiled_op_group:
                                if uop_idx >= tiled_op.n_uops:
                                    continue
                                
                                if uop_idx > 0:
                                    _append_context_load(core_id, tiled_op, uop_idx - 1, tiled_op_slot_map[tiled_op.tiled_op_id])
                                
                                _append_uop(core_id, tiled_op, uop_idx)
                                
                                if uop_idx < tiled_op.n_uops - 1:
                                    _append_context_store(core_id, tiled_op, uop_idx, tiled_op_slot_map[tiled_op.tiled_op_id])
                else:
                    for tiled_op in core_tiled_ops:
                        for uop_idx in range(tiled_op.n_uops):
                            _append_uop(core_id, tiled_op, uop_idx)
            
            spatial_schedule_maps: dict[str, dict[tuple[int, int, int, TileSignature], dict[str, tuple[int, int, int]]]] = {
                SPATIAL_X: {},
                SPATIAL_Y: {},
            }
            spatial_fifo_entry_counters: dict[int, int] = defaultdict(int)
            spatial_links: list[
                tuple[
                    str,
                    int,
                    int,
                    tuple[int, int, int, TileSignature],
                    tuple[int, int, int, TileSignature],
                ]
            ] = []
            
            def _apply_spatial_run(axis: str, records: list[tuple[int, int, int, int, int, TileSignature]]):
                if len(records) < 2:
                    return
                
                for record_idx in range(len(records) - 1):
                    _, producer_plan_idx, producer_core_id, producer_tiled_op_idx, producer_uop_idx, producer_tile = records[record_idx]
                    _, _, consumer_core_id, consumer_tiled_op_idx, consumer_uop_idx, consumer_tile = records[record_idx + 1]
                    producer_key = (producer_core_id, producer_tiled_op_idx, producer_uop_idx, producer_tile)
                    consumer_key = (consumer_core_id, consumer_tiled_op_idx, consumer_uop_idx, consumer_tile)
                    spatial_links.append((axis, producer_core_id, producer_plan_idx, producer_key, consumer_key))
            
            for axis in (SPATIAL_X, SPATIAL_Y):
                buf_name = _reuse_strategy[axis]
                if buf_name is None or self._bcast_fifo_slot_num <= 0:
                    continue
                
                grouped_records: dict[tuple[int, int, TileSignature], list[tuple[int, int, int, int, int, TileSignature]]] = defaultdict(list)
                for core_id in self.op_sig.core_group.core_ids:
                    fixed_coord, moving_coord = _axis_key(core_id, axis)
                    seq_idx = 0
                    for plan_idx, (node_type, tiled_op, uop_idx, _) in enumerate(execution_plans[core_id]):
                        if node_type != "uop":
                            continue
                        for tile in tiled_op.i_tiles[uop_idx]:
                            if tile.buf_name != buf_name:
                                continue
                            grouped_records[(fixed_coord, seq_idx, tile)].append((moving_coord, plan_idx, core_id, tiled_op.tiled_op_id, uop_idx, tile))
                            seq_idx += 1
                
                for records in grouped_records.values():
                    if len(records) < 2:
                        continue
                    records.sort(key=lambda record: record[0])
                    
                    spatial_run = [records[0]]
                    for record in records[1:]:
                        if record[0] == spatial_run[-1][0] + 1:
                            spatial_run.append(record)
                        else:
                            _apply_spatial_run(axis, spatial_run)
                            spatial_run = [record]
                    _apply_spatial_run(axis, spatial_run)
            
            spatial_links.sort(key=lambda link: (link[1], link[2]))
            for axis, producer_core_id, _, producer_key, consumer_key in spatial_links:
                slot_id = spatial_fifo_entry_counters[producer_core_id]
                spatial_fifo_entry_counters[producer_core_id] += 1
                
                producer_schedule = spatial_schedule_maps[axis].setdefault(producer_key, {"src": None, "dst": None})
                consumer_schedule = spatial_schedule_maps[axis].setdefault(consumer_key, {"src": None, "dst": None})
                producer_schedule["dst"] = (producer_core_id, slot_id, 1)
                consumer_schedule["src"] = (producer_core_id, slot_id, 1)
            
            cache_states: dict[int, dict[TileSignature, int]] = {core_id: {} for core_id in self.op_sig.core_group.core_ids}
            cache_lru: dict[int, list[TileSignature]] = {core_id: [] for core_id in self.op_sig.core_group.core_ids}
            
            def _cache_schedule(core_id: int, tile: TileSignature) -> tuple[bool, int] | None:
                if tile.buf_name not in _temporal_reuse_targets:
                    return None
                if self.cache_buffer_slot_num <= 0:
                    return None
                
                states = cache_states[core_id]
                lru = cache_lru[core_id]
                
                if tile in states:
                    lru.remove(tile)
                    lru.append(tile)
                    return True, states[tile]
                
                if len(states) < self.cache_buffer_slot_num:
                    used_slots = set(states.values())
                    slot_id = next(slot_id for slot_id in range(self.cache_buffer_slot_num) if slot_id not in used_slots)
                else:
                    evicted_tile = lru.pop(0)
                    slot_id = states.pop(evicted_tile)
                
                states[tile] = slot_id
                lru.append(tile)
                return False, slot_id
            
            def _add_scheduled_uop(thread: MCA_OperatorGraphCompiler.Thread, core_id: int, tiled_op: TiledOperatorSignature, uop_idx: int):
                cache_schedule = {}
                spatial_x_schedule = {}
                spatial_y_schedule = {}
                
                for tile in tiled_op.i_tiles[uop_idx]:
                    spatial_key = (core_id, tiled_op.tiled_op_id, uop_idx, tile)
                    if spatial_key in spatial_schedule_maps[SPATIAL_X]:
                        spatial_x_schedule[tile] = spatial_schedule_maps[SPATIAL_X][spatial_key]
                    if spatial_key in spatial_schedule_maps[SPATIAL_Y]:
                        spatial_y_schedule[tile] = spatial_schedule_maps[SPATIAL_Y][spatial_key]
                    
                    scheduled_cache = _cache_schedule(core_id, tile)
                    if scheduled_cache is not None:
                        cache_schedule[tile] = scheduled_cache
                
                thread.add_uop_node(
                    self.op_sig.op_id,
                    tiled_op.tiled_op_id,
                    uop_idx,
                    output=(uop_idx == tiled_op.n_uops - 1),
                    cache_schedule=cache_schedule,
                    spatial_x_schedule=spatial_x_schedule,
                    spatial_y_schedule=spatial_y_schedule,
                )
            
            for core_id in self.op_sig.core_group.core_ids:
                thread = thread_mapping[core_id]
                for node_type, tiled_op, uop_idx, slot_id in execution_plans[core_id]:
                    if node_type == "ctx_load":
                        thread.add_context_load(self.op_sig.op_id, tiled_op.tiled_op_id, uop_idx, slot_id)
                    elif node_type == "ctx_store":
                        thread.add_context_store(self.op_sig.op_id, tiled_op.tiled_op_id, uop_idx, slot_id)
                    elif node_type == "uop":
                        _add_scheduled_uop(thread, core_id, tiled_op, uop_idx)
                    
            return thread_mapping
        
        def unfreeze(self):
            self.cache_buffer_size = 0
            self.cache_buffer_slot_num = 0
            
            self.thread_mapping = {}
            
            self.o_tile_store = self.op_sig.buffers[self.op_sig.output_buffer_name].is_allocated
            self.o_tile_sharers = set()
            
            self._is_frozen = False
            
        def freeze(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]=None) -> bool:
            self.cache_buffer_size = self.spad_space_size_per_core - (self.ctx_buffer_size + self.bcast_fifo_size * 2 + self.ld_ex_fifo_size + self.ex_st_fifo_size)
            self.cache_buffer_slot_num = self.cache_buffer_size // self.cache_buffer_slot_size
            
            if self.cache_buffer_size < 0:
                self.unfreeze()
                return False
            
            if tiled_op_mapping is None:
                tiled_op_mapping = self.create_tiled_op_mapping()
            
            thread_mapping = self.create_thread_mapping(tiled_op_mapping)
            if len(thread_mapping) == 0 and len(self.op_sig.tiled_ops) > 0:
                self.unfreeze()
                return False

            for thread in thread_mapping.values():
                if thread.n_uop_nodes > 0:
                    self.thread_mapping = thread_mapping
                    break
            else:
                self.unfreeze()
                return False
            
            self._is_frozen = True
            return True
            
        @property
        def is_frozen(self):
            return self._is_frozen 
        
        @property
        def ctx_buffer_size(self):
            return self._context_buffer_size
        
        @property
        def bcast_fifo_size(self):
            return self._bcast_fifo_size
        
        @property
        def ld_ex_fifo_size(self):
            return self._ld_ex_fifo_size
        
        @property
        def ex_st_fifo_size(self):
            return self._ex_st_fifo_size
        
        @property
        def bcast_fifo_depth(self):
            return self._bcast_fifo_slot_num
        
        @property
        def ld_ex_fifo_depth(self):
            return self._ld_ex_fifo_slot_num
        
        @property
        def ex_st_fifo_depth(self):
            return self._ex_st_fifo_slot_num
        
        @property
        def bcast_fifo_slot_size(self):
            return self._bcast_fifo_entry_size
        
        @property
        def ld_ex_fifo_slot_size(self):
            return self._ld_ex_fifo_entry_size
        
        @property
        def ex_st_fifo_slot_size(self):
            return self._ex_st_fifo_entry_size
        
        @property
        def ctx_buffer_slot_size(self):
            return self._context_buffer_slot_size
        
        @property
        def ctx_buffer_slot_num(self):
            return self._context_buffer_slot_num
    
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
        thread_mappings = {op_id: op_meta.thread_mapping for op_id, op_meta in op_metas.items()}
        
        mem_states = {
            op_id: MCA_OperatorGraphCompiler.MemoryState(env, op_meta)
            for op_id, op_meta in op_metas.items()
        }
        for mem_state in mem_states.values():
            mem_state.register_fifos()
        
        max_uop_nodes_per_stage = max(1, env.recipe.fifo_buffer_slot_num // 2)
        
        def _spatial_src(schedule: dict[TileSignature, dict[str, tuple[int, int, int]]], tile: TileSignature):
            if tile not in schedule:
                return None
            return schedule[tile].get("src", None)
        
        def _spatial_dst(schedule: dict[TileSignature, dict[str, tuple[int, int, int]]], tile: TileSignature):
            if tile not in schedule:
                return None
            return schedule[tile].get("dst", None)
        
        def _add_or_merge_load_ir(
            compiled_op: MCA_CompiledOperator,
            core_id: int,
            stage: MCA_CompiledOperator.Stage,
            load_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
        ) -> MCA_CompiledOperator.IR.MEM_COPY_TILE:
            def _can_merge_src(existing_src: MCA_CompiledOperator.IR.Reference, new_src: MCA_CompiledOperator.IR.Reference) -> bool:
                if existing_src == new_src:
                    return True
                if not existing_src.is_fifo() and not new_src.is_fifo():
                    return True
                if existing_src.is_fifo() and not new_src.is_fifo():
                    return True
                return False
            
            def _stage_load_ir_indices() -> set[int]:
                return {
                    ir.ir_idx
                    for ir in stage.loads
                    if isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE) and ir.ir_idx is not None
                }
            
            def _would_create_wait_cycle(
                existing_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
                merged_wait_ir_idx: set[int],
            ) -> bool:
                if existing_ir.ir_idx is None:
                    return False
                
                stage_ir_indices = _stage_load_ir_indices()
                if existing_ir.ir_idx not in stage_ir_indices:
                    return False
                
                graph: dict[int, set[int]] = {ir_idx: set() for ir_idx in stage_ir_indices}
                for ir in stage.loads:
                    if not isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                        continue
                    if ir.ir_idx is None or ir.ir_idx not in stage_ir_indices:
                        continue
                    wait_ir_idx = merged_wait_ir_idx if ir is existing_ir else set(ir.wait_ir_idx)
                    graph[ir.ir_idx].update(wait_ir_idx & stage_ir_indices)
                
                visited: set[int] = set()
                visiting: set[int] = set()
                
                def _visit(ir_idx: int) -> bool:
                    if ir_idx in visiting:
                        return True
                    if ir_idx in visited:
                        return False
                    visiting.add(ir_idx)
                    for wait_ir_idx in graph.get(ir_idx, ()):
                        if _visit(wait_ir_idx):
                            return True
                    visiting.remove(ir_idx)
                    visited.add(ir_idx)
                    return False
                
                return any(_visit(ir_idx) for ir_idx in stage_ir_indices)
            
            def _would_create_forward_wait(
                existing_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE,
                merged_wait_ir_idx: set[int],
            ) -> bool:
                if existing_ir.ir_idx is None:
                    return False
                stage_ir_indices = _stage_load_ir_indices()
                return any(
                    wait_ir_idx in stage_ir_indices and wait_ir_idx > existing_ir.ir_idx
                    for wait_ir_idx in merged_wait_ir_idx
                )
            
            if load_ir.src is not None:
                for existing_ir in stage.loads:
                    if not isinstance(existing_ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                        continue
                    if existing_ir.src is None:
                        continue
                    if existing_ir.src.tile_sig != load_ir.src.tile_sig:
                        continue
                    if not _can_merge_src(existing_ir.src, load_ir.src):
                        continue
                    
                    wait_ir_idx = set(existing_ir.wait_ir_idx)
                    wait_ir_idx.update(load_ir.wait_ir_idx)
                    if existing_ir.ir_idx is not None:
                        wait_ir_idx.discard(existing_ir.ir_idx)
                    if _would_create_forward_wait(existing_ir, wait_ir_idx):
                        continue
                    if _would_create_wait_cycle(existing_ir, wait_ir_idx):
                        continue
                    
                    for dst in load_ir.dsts:
                        if not any(dst == existing_dst for existing_dst in existing_ir.dsts):
                            existing_ir.dsts.append(dst)
                    existing_ir.wait_ir_idx = sorted(wait_ir_idx)
                    return existing_ir
            
            compiled_op.add_load_ir(core_id, load_ir)
            return load_ir
        
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            op_sig = op_meta.op_sig
            compiled_op = compiled_ops[op_id]
            mem_state = mem_states[op_id]
            thread_mapping = thread_mappings[op_id]
            
            for core_id, thread in thread_mapping.items():
                uop_nodes_in_stage = 0
                
                for node in thread.uop_nodes:
                    if isinstance(node, MCA_OperatorGraphCompiler.Thread.ContextLoadNode):
                        if uop_nodes_in_stage >= max_uop_nodes_per_stage:
                            compiled_op.new_stage(core_id)
                            uop_nodes_in_stage = 0
                        
                        tile = mem_state.ctx_pop(core_id, node.slot_id)
                        expected_tile = op_sig.tiled_ops[node.tiled_op_idx].o_tile
                        if tile != expected_tile:
                            raise Exception(f"Context load mismatch on core {core_id}: expected {expected_tile}, got {tile}.")
                        
                        compiled_op.add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_CTX_LOAD(
                            op_id,
                            node.tiled_op_idx,
                            node.uop_idx,
                            tile,
                            mem_state.ctx_descriptors[core_id].ref(tile, offset=node.slot_id * op_meta.ctx_buffer_slot_size),
                        ))
                        continue
                    
                    if isinstance(node, MCA_OperatorGraphCompiler.Thread.ContextStoreNode):
                        tile = op_sig.tiled_ops[node.tiled_op_idx].o_tile
                        mem_state.ctx_push(core_id, node.slot_id, tile)
                        compiled_op.add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_CTX_STORE(
                            op_id,
                            node.tiled_op_idx,
                            node.uop_idx,
                            tile,
                            mem_state.ctx_descriptors[core_id].ref(tile, offset=node.slot_id * op_meta.ctx_buffer_slot_size),
                        ))
                        continue
                    
                    if not isinstance(node, MCA_OperatorGraphCompiler.Thread.UopNode):
                        raise Exception(f"Unsupported thread node type: {type(node).__name__}")
                    
                    if uop_nodes_in_stage >= max_uop_nodes_per_stage:
                        compiled_op.new_stage(core_id)
                        uop_nodes_in_stage = 0
                    
                    tiled_op = op_sig.tiled_ops[node.tiled_op_idx]
                    i_tiles = tiled_op.i_tiles[node.uop_idx]
                    o_tile = tiled_op.o_tile
                    current_stage = compiled_op.current_stage(core_id)
                    
                    for i_tile in i_tiles:
                        ld_ex_entry_id = mem_state.ld_ex_push(core_id, i_tile)
                        ld_ex_ref = mem_state.ld_ex_descriptors[core_id].ref(i_tile, slot_id=ld_ex_entry_id, ref_cnt=1)
                        load_ir = MCA_CompiledOperator.IR.MEM_COPY_TILE(None, ld_ex_ref)
                        cache_write_slot_id = None
                        
                        spatial_x_src = _spatial_src(node.spatial_x_schedule, i_tile)
                        spatial_y_src = _spatial_src(node.spatial_y_schedule, i_tile)
                        if spatial_x_src is not None and spatial_y_src is not None:
                            raise Exception(f"Tile {i_tile} has both SPATIAL_X and SPATIAL_Y sources on core {core_id}.")
                        
                        if spatial_x_src is not None:
                            src_core_id, src_entry_id, src_ref_cnt = spatial_x_src
                            load_ir.src = mem_state.spatial_x_descriptors[src_core_id].ref(i_tile, src_entry_id, src_ref_cnt)
                        elif spatial_y_src is not None:
                            src_core_id, src_entry_id, src_ref_cnt = spatial_y_src
                            load_ir.src = mem_state.spatial_y_descriptors[src_core_id].ref(i_tile, src_entry_id, src_ref_cnt)
                        elif i_tile in node.cache_schedule and node.cache_schedule[i_tile][0]:
                            _, cache_slot_id = node.cache_schedule[i_tile]
                            mem_state.cache_read(core_id, i_tile, cache_slot_id, current_stage, load_ir)
                        else:
                            load_ir.src = mem_state.tensor_descriptors[i_tile.buf_name].ref(i_tile)
                        
                        spatial_x_dst = _spatial_dst(node.spatial_x_schedule, i_tile)
                        spatial_y_dst = _spatial_dst(node.spatial_y_schedule, i_tile)
                        if spatial_x_dst is not None:
                            dst_core_id, dst_entry_id, dst_ref_cnt = spatial_x_dst
                            if dst_core_id != core_id:
                                raise Exception(f"SPATIAL_X destination core mismatch: scheduled={dst_core_id}, current={core_id}.")
                            load_ir.dsts.append(mem_state.spatial_x_descriptors[dst_core_id].ref(i_tile, dst_entry_id, dst_ref_cnt))
                        if spatial_y_dst is not None:
                            dst_core_id, dst_entry_id, dst_ref_cnt = spatial_y_dst
                            if dst_core_id != core_id:
                                raise Exception(f"SPATIAL_Y destination core mismatch: scheduled={dst_core_id}, current={core_id}.")
                            load_ir.dsts.append(mem_state.spatial_y_descriptors[dst_core_id].ref(i_tile, dst_entry_id, dst_ref_cnt))
                        
                        if i_tile in node.cache_schedule and not node.cache_schedule[i_tile][0]:
                            _, cache_slot_id = node.cache_schedule[i_tile]
                            mem_state.cache_write(core_id, i_tile, cache_slot_id, current_stage, load_ir)
                            cache_write_slot_id = cache_slot_id
                        
                        merged_load_ir = _add_or_merge_load_ir(compiled_op, core_id, current_stage, load_ir)
                        if cache_write_slot_id is not None and merged_load_ir is not load_ir:
                            mem_state.replace_cache_writer(core_id, i_tile, cache_write_slot_id, load_ir, merged_load_ir, current_stage)
                    
                    execute_ir = MCA_CompiledOperator.IR.EXE_UOP(
                        op_id,
                        node.tiled_op_idx,
                        node.uop_idx,
                        i_tile_refs=[
                            mem_state.ld_ex_descriptors[core_id].ref(
                                i_tile,
                                slot_id=mem_state.ld_ex_pop(core_id, i_tile),
                                ref_cnt=1,
                            )
                            for i_tile in i_tiles
                        ],
                        o_tile_ref=mem_state.ex_st_descriptors[core_id].ref(
                            o_tile,
                            slot_id=mem_state.ex_st_push(core_id, o_tile),
                            ref_cnt=1,
                        ) if node.output else None,
                        dtype=i_tiles[0].dtype if len(i_tiles) > 0 else None,
                        acc_dtype=o_tile.dtype,
                    )
                    compiled_op.add_execute_ir(core_id, execute_ir)
                    
                    if node.output:
                        store_src = mem_state.ex_st_descriptors[core_id].ref(
                            o_tile,
                            slot_id=mem_state.ex_st_pop(core_id, o_tile),
                            ref_cnt=1,
                        )
                        store_ir = MCA_CompiledOperator.IR.MEM_COPY_TILE(store_src)
                        if op_meta.o_tile_store:
                            store_ir.dsts.append(mem_state.tensor_descriptors[o_tile.buf_name].ref(o_tile))
                        compiled_op.add_store_ir(core_id, store_ir)
                    
                    uop_nodes_in_stage += 1
        
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
