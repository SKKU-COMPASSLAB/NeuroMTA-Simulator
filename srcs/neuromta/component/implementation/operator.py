import abc
import enum
import functools
import math
import pprint
import torch
import tqdm
from typing import Any, Sequence, Dict, List, Callable
from collections import deque, defaultdict

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
]


def mca_operator_method(func: Callable):
    @functools.wraps(func)
    def _mca_mapper_method_wrapper(*args, **kwargs) -> 'MCA_OperatorSignature':
        op_sig = func(*args, **kwargs)
        if not isinstance(op_sig, MCA_OperatorSignature):
            raise TypeError("The decorated function must return an instance of MCA_OperatorSignature.")
        return op_sig
    return _mca_mapper_method_wrapper


class MCA_KernelTemplate:
    def get_ld_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', stage: 'MCA_CompiledOperator.Stage', concurrent_load_num: int) -> KernelPrototype: ...
    def get_ex_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', stage: 'MCA_CompiledOperator.Stage') -> KernelPrototype: ...
    def get_st_thread_kernel(self, core: NPUCore, env: 'MCA_OperatorGraphCompiler.Environment', stage: 'MCA_CompiledOperator.Stage') -> KernelPrototype: ...
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
        tiled_op = TiledOperatorSignature()
        self._tiled_ops.append(tiled_op)
        return tiled_op
        
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
            if isinstance(cmd, MCA_CompiledOperator.IR.MEM_COPY_TILE):
                cmd.ir_idx = len(self.loads)
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
        def is_bubble(self) -> bool:
            return len(self.loads) == 0 and len(self.executes) == 0 and len(self.stores) == 0

    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata'):
        self._env = env
        self._op_id = op_meta.op_sig.op_id
        self._kernel_template = op_meta.op_sig.kernel_template
        
        self._mappings: dict[int, list[MCA_CompiledOperator.Stage]] = {
            core_id: [MCA_CompiledOperator.Stage()] 
            for core_id in op_meta.op_sig.core_group.core_ids
        }  # {core_id: [stage1, stage2, ...]}
        
    def new_stage(self, core_id: int) -> 'MCA_CompiledOperator.Stage':
        stage = MCA_CompiledOperator.Stage()
        self._mappings[core_id].append(stage)
        return stage
    
    def current_stage_idx(self, core_id: int) -> int:
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        return len(self._mappings[core_id]) - 1
    
    def add_load_ir(self, core_id: int, cmd: 'MCA_CompiledOperator.IR.Base'):
        if core_id not in self._mappings:
            raise ValueError(f"Core ID {core_id} is not in the operator's core group.")
        if not self._mappings[core_id]:
            self.new_stage(core_id)
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
        
    def dispatch(self, device: MCA_DeviceBase):
        self.freeze()
        
        # gb_barrier = (
        #     self._env.add_variable(f"{self._op_id}_barrier_arrival_cnt", 0).handle_name,
        #     self._env.add_variable(f"{self._op_id}_barrier_blocking", 0).handle_name,
        #     len(self.mappings.keys()) * 3,
        # )
        
        # # PRESYNC BARRIER
        # for core_id in self.mappings.keys():
        #     core = device.get_npu_core(core_id)
            
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("LD")
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("EX")
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("ST")
            
        # LD/EX/ST THREAD
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            
            for stage in self.mappings[core_id]:
                self._kernel_template.get_ld_thread_kernel(core, self._env, stage, self._env.recipe.concurrent_load_num).dispatch("LD")
                self._kernel_template.get_ex_thread_kernel(core, self._env, stage).dispatch("EX")
                self._kernel_template.get_st_thread_kernel(core, self._env, stage).dispatch("ST")
        
        # # POSTSYNC BARRIER
        # for core_id in self.mappings.keys():
        #     core = device.get_npu_core(core_id)
            
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("LD")
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("EX")
        #     self._kernel_template.get_barrier_kernel(core, self._env, barrier=gb_barrier).dispatch("ST")
        
    @property
    def mappings(self):
        return self._mappings
    
    def summary(self) -> dict:
        return {
            core_id: [stage.summary() for stage in stages]
            for core_id, stages in self._mappings.items()
        }


class MCA_CompiledProgram:
    def __init__(self, device: MCA_DeviceBase, compiled_ops: dict[str, MCA_CompiledOperator]):
        self._device = device
        self._compiled_ops = compiled_ops
        
    def dispatch(self):
        for op_id, compiled_op in self._compiled_ops.items():
            compiled_op.dispatch(self._device)
        return self
            
    def summary(self) -> dict:
        return {op_id: compiled_op.summary() for op_id, compiled_op in self._compiled_ops.items()}
    
    
class MCA_OperatorGraphCompiler:
    ALL="ALL"
    DEFAULT="DEFAULT"
    
    class CompileRecipe:
        class ReusePriority(enum.Enum):
            TEMPORAL = "TEMPORAL"
            SPATIAL = "SPATIAL"

        def __init__(
            self, 
            device: MCA_DeviceBase,
            core_groups: list[MCA_CoreGroup],
            spad_space_size_per_core: int,
            broadcast_optimize_queue_depth: int=32,
            operator_pipelining: bool=False,
            context_buffer_slot_num: int=4,
            ld_ex_buffer_slot_num: int=16,
            ex_st_buffer_slot_num: int=16,
            concurrent_load_num: int=8,
            reuse_priority: ReusePriority=ReusePriority.TEMPORAL,
        ):
            if len(core_groups) == 0:
                raise ValueError("At least one core group must be provided.")
            if not isinstance(core_groups[0], (MCA_CoreGroup, list)):
                core_groups = [core_groups]

            self.device                         = device
            self.core_groups                    = core_groups
            self.spad_space_size_per_core       = spad_space_size_per_core
            self.broadcast_optimize_queue_depth = broadcast_optimize_queue_depth
            self.operator_pipelining            = operator_pipelining
            self.context_buffer_slot_num        = context_buffer_slot_num
            self.ld_ex_buffer_slot_num          = ld_ex_buffer_slot_num
            self.ex_st_buffer_slot_num          = ex_st_buffer_slot_num
            self.concurrent_load_num            = concurrent_load_num
            self.reuse_priority                 = reuse_priority if isinstance(reuse_priority, self.ReusePriority) else self.ReusePriority(reuse_priority)

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
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.output = output
            
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
            
        def add_uop_node(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False):
            uop_node = self.UopNode(op_id, tiled_op_idx, uop_idx, output=output)
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
            self.bcast_fifo_slot_size = max(buf.tile_size for buf_name, buf in op_sig.buffers.items() if buf_name in op_sig.input_buffer_names)
            self.bcast_fifo_size      = recipe.broadcast_optimize_queue_depth * self.bcast_fifo_slot_size
            
            self.opp_fifo_slot_size   = op_sig.buffers[op_sig.output_buffer_name].tile_size
            
            self.ctx_buffer_slot_size = self.op_sig.buffers[op_sig.output_buffer_name].tile_size  # conservatively reserve the same size as output tile buffer for context store (for tile-level pipelining)
            self.ctx_buffer_size      = recipe.context_buffer_slot_num * self.ctx_buffer_slot_size
            
            self.ld_ex_fifo_slot_size = max(buf.tile_size for buf_name, buf in op_sig.buffers.items() if buf_name in op_sig.input_buffer_names)
            self.ld_ex_fifo_size      = recipe.ld_ex_buffer_slot_num * self.ld_ex_fifo_slot_size
            
            self.ex_st_fifo_slot_size = op_sig.buffers[op_sig.output_buffer_name].tile_size
            self.ex_st_fifo_size      = recipe.ex_st_buffer_slot_num * self.ex_st_fifo_slot_size
            
            self.cache_buffer_slot_size = self.ld_ex_fifo_slot_size  # conservatively reserve the same size as LD->EX FIFO slot for cache buffer (for tile-level reuse)
            
            # Reuse targets
            reuse_targets = sorted(self.op_sig.input_buffer_names, key=lambda buf_name: self.op_sig.buffers[buf_name].total_size, reverse=True)
            
            if recipe.reuse_priority == recipe.ReusePriority.TEMPORAL:
                self.temporal_reuse_target = reuse_targets[0] if len(reuse_targets) > 0 else op_sig.input_buffer_names[0]
                self.spatial_reuse_target  = reuse_targets[1] if len(reuse_targets) > 1 else self.temporal_reuse_target
            else:
                self.spatial_reuse_target  = reuse_targets[0] if len(reuse_targets) > 0 else op_sig.input_buffer_names[0]
                self.temporal_reuse_target = reuse_targets[1] if len(reuse_targets) > 1 else self.spatial_reuse_target
                
            # Initialized after freezing the operator metadata
            self.opp_fifo_size = 0    # operator pipelining buffer size (operator pipelining FIFO)
            self.cache_buffer_size = 0 # cache buffer size for tile-level reuse (cache buffer)
            
            self.thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            self.o_tile_store = op_sig.buffers[op_sig.output_buffer_name].is_allocated  # if the output buffer is allocated, the computation result should be updated to the buffer
            self.o_tile_sharers: set[str] = set()  # set of op_ids that directly consume this operator's output tiles (tile-level sharers via SHARED area)
            
            self._is_frozen = False
        
        def _create_tiled_op_mapping(self) -> dict[int, list[TiledOperatorSignature]]:
            tiled_op_mapping = {core_id: [] for core_id in self.op_sig.core_group.core_ids}
            core_ids = list(tiled_op_mapping.keys())
            n_cores = len(core_ids)

            if not self.op_sig.tiled_ops:
                return tiled_op_mapping

            # 1. Preliminary Assignment & Heuristic Analysis
            # Assign ops to cores based on physical topology if available to preserve spatial locality.
            op_to_core = {}
            if isinstance(self.op_sig.core_group, MTA_CoreGrid):
                grid = self.op_sig.core_group
                gh, gw = grid.shape
                for op in self.op_sig.tiled_ops:
                    ys, xs, _, _ = op.o_tile.coords
                    op_to_core[id(op)] = grid[ys % gh, xs % gw]
            else:
                for i, op in enumerate(self.op_sig.tiled_ops):
                    op_to_core[id(op)] = core_ids[i % n_cores]

            # Analyze each buffer's reuse potential across the assigned cores.
            from collections import defaultdict
            buf_metrics = defaultdict(lambda: {"cores": set(), "unique_tiles": set(), "total_access": 0})
            for op in self.op_sig.tiled_ops:
                cid = op_to_core[id(op)]
                # Collect all unique input buffers and tiles used by this operator.
                for uop_tiles in op.i_tiles:
                    for tile in uop_tiles:
                        bn = tile.buf_name
                        buf_metrics[bn]["cores"].add(cid)
                        buf_metrics[bn]["unique_tiles"].add(tile.signature)
                        buf_metrics[bn]["total_access"] += 1

            # Rank buffers for sorting priority: 
            # Primary: Highest spatial sharing (Bcast efficiency).
            # Secondary: Highest temporal reuse intensity (total_access / unique_tiles).
            ranked_bufs = sorted(
                buf_metrics.keys(),
                key=lambda bn: (
                    len(buf_metrics[bn]["cores"]), 
                    buf_metrics[bn]["total_access"] / max(1, len(buf_metrics[bn]["unique_tiles"]))
                ),
                reverse=True
            )

            # 2. Global Synchronized Sorting
            # Establish a deterministic execution sequence that all cores follow.
            def get_global_sort_key(op: TiledOperatorSignature):
                sort_key = []
                for bn in ranked_bufs:
                    # Find all tiles belonging to this buffer in the operator's input set.
                    matching_coords = []
                    for uop_tiles in op.i_tiles:
                        for tile in uop_tiles:
                            if tile.buf_name == bn:
                                matching_coords.append(tile.coords)
                    
                    # Use the lexicographical minimum coordinate as the stable anchor for this buffer.
                    # This handles multi-tile references (like Conv2d windows) gracefully.
                    if matching_coords:
                        sort_key.append(min(matching_coords))
                    else:
                        sort_key.append((math.inf, math.inf, math.inf, math.inf))
                
                # Append output tile coordinates as final tie-breaker.
                sort_key.append(op.o_tile.coords)
                return tuple(sort_key)

            all_ops_sorted = sorted(self.op_sig.tiled_ops, key=get_global_sort_key)

            # 3. Final Distribution
            # Map ops to their designated cores while maintaining the global sorted order.
            for op in all_ops_sorted:
                cid = op_to_core[id(op)]
                tiled_op_mapping[cid].append(op)

            return tiled_op_mapping
            
        def _create_thread_mapping(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]) -> 'dict[int, MCA_OperatorGraphCompiler.Thread]':
            thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}

            tiled_op_idx_map = {tiled_op: idx for idx, tiled_op in enumerate(self.op_sig.tiled_ops)}

            current_cache_buffer_usage = 0
            current_cache_buffer_allocated: set[TileSignature] = set()

            for core_id, core_tiled_ops in tiled_op_mapping.items():
                thread = MCA_OperatorGraphCompiler.Thread(core_id)
                thread_mapping[core_id] = thread
                
                n_concurrent_tiled_ops = self.ctx_buffer_slot_num
                grouped_tiled_ops = [core_tiled_ops[i:i + n_concurrent_tiled_ops] for i in range(0, len(core_tiled_ops), n_concurrent_tiled_ops)]
                collected_uops: dict[int, list[int]] = {}
                tiled_op_slot_map: dict[int, int] = {}
                
                def fill_out_thread(thread: MCA_OperatorGraphCompiler.Thread, collected_uops: dict[int, list[int]], tiled_op_slot_map: dict[int, int]):
                    for tiled_op_idx, uop_indices in collected_uops.items():
                        if uop_indices[0] > 0:
                            thread.add_context_load(self.op_sig.op_id, tiled_op_idx, uop_indices[0] - 1, slot_id=tiled_op_slot_map[tiled_op_idx])
                        for uop_idx in uop_indices:
                            thread.add_uop_node(self.op_sig.op_id, tiled_op_idx, uop_idx, output=uop_idx == self.op_sig.tiled_ops[tiled_op_idx].n_uops - 1)
                        if uop_indices[-1] < self.op_sig.tiled_ops[tiled_op_idx].n_uops - 1:
                            thread.add_context_store(self.op_sig.op_id, tiled_op_idx, uop_indices[-1], slot_id=tiled_op_slot_map[tiled_op_idx])
                
                for group in grouped_tiled_ops:
                    tiled_op_slot_map = {tiled_op_idx_map[tiled_op]: idx for idx, tiled_op in enumerate(group)}
                    n_uop_per_tiled_op = max(tiled_op.n_uops for tiled_op in group)
                    uop_cursor = 0
                    
                    while uop_cursor < n_uop_per_tiled_op:
                        _tmp_i_tile_size = 0
                        _tmp_o_tile_size = 0
                        
                        for tiled_op in group:
                            if uop_cursor < tiled_op.n_uops:
                                for i_tile in tiled_op.i_tiles[uop_cursor]:
                                    if i_tile not in current_cache_buffer_allocated:
                                        _tmp_i_tile_size += self.op_sig.buffers[i_tile.buf_name].tile_size
                                if uop_cursor == tiled_op.n_uops - 1 and tiled_op.o_tile not in current_cache_buffer_allocated:
                                    _tmp_o_tile_size += self.op_sig.buffers[tiled_op.o_tile.buf_name].tile_size
                                    
                        if current_cache_buffer_usage + _tmp_i_tile_size + _tmp_o_tile_size > self.cache_buffer_size:
                            fill_out_thread(thread, collected_uops, tiled_op_slot_map)
                            collected_uops.clear()
                            current_cache_buffer_usage = 0
                            current_cache_buffer_allocated.clear()
                            
                        for tiled_op in group:
                            tiled_op_idx = tiled_op_idx_map[tiled_op]
                            
                            if uop_cursor < tiled_op.n_uops:
                                collected_uops.setdefault(tiled_op_idx, []).append(uop_cursor)
                                for i_tile in tiled_op.i_tiles[uop_cursor]:
                                    if i_tile not in current_cache_buffer_allocated:
                                        current_cache_buffer_allocated.add(i_tile)
                                        current_cache_buffer_usage += i_tile.tile_size
                                if tiled_op.o_tile not in current_cache_buffer_allocated:
                                    current_cache_buffer_allocated.add(tiled_op.o_tile)
                                    current_cache_buffer_usage += tiled_op.o_tile.tile_size
                                    
                        uop_cursor += 1
                                
                    fill_out_thread(thread, collected_uops, tiled_op_slot_map)
                    collected_uops.clear()
                    current_cache_buffer_usage = 0
                    current_cache_buffer_allocated.clear()
                
            return thread_mapping
        
        def unfreeze(self):
            self.opp_fifo_size = 0
            self.cache_buffer_size = 0
            self.thread_mapping = {}
            
            self.o_tile_store = self.op_sig.buffers[self.op_sig.output_buffer_name].is_allocated
            self.o_tile_sharers = set()
            
            self._is_frozen = False
            
        def freeze(self, tiled_op_mapping: dict[int, list[TiledOperatorSignature]]=None) -> bool:
            self.cache_buffer_size = self.spad_space_size_per_core - (self.ctx_buffer_size + self.bcast_fifo_size + self.opp_fifo_size + self.ld_ex_fifo_size + self.ex_st_fifo_size)
            
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
        
        @staticmethod
        def establish_dependency(env: 'MCA_OperatorGraphCompiler.Environment', op_ids: list[str]) -> list[str]:
            for op_id in op_ids:
                op_meta = env.op_meta[op_id]
                
                for dst_op_id, dst_op_meta in env.op_meta.items():
                    if dst_op_id == op_id:
                        continue
                    if op_meta.op_sig.output_buffer_name in dst_op_meta.op_sig.input_buffer_names:
                        op_meta.o_tile_sharers.add(dst_op_id)
                        dst_op_meta.i_buf_src[op_meta.op_sig.output_buffer_name] = MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.TILE_SHARED(op_id)
                        
            return env.topological_sort_grouped_target_ops(op_ids)

        @staticmethod
        def check_dependency_and_freeze(env: 'MCA_OperatorGraphCompiler.Environment', op_ids: list[str]) -> bool:
            dept_candidates: list[str] = []
            
            for op_id in reversed(op_ids):
                op_meta = env.op_meta[op_id]
                
                tiled_op_mapping = op_meta._create_tiled_op_mapping()
                max_shared_area_per_core: dict[int, int] = {core_id: 0 for core_id in tiled_op_mapping.keys()}  # {core_id: shared_area_size}
                
                for dst_op_id in dept_candidates:
                    dst_op_meta  = env.op_meta[dst_op_id]
                    
                    if op_meta.op_sig.output_buffer_name not in dst_op_meta.op_sig.input_buffer_names:
                        continue
                    
                    # op_meta.o_tile_sharers.add(dst_op_id)
                    # dst_op_meta.i_buf_src[op_meta.op_sig.output_buffer_name] = MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.TILE_SHARED(op_id)
                    
                    dst_tile_access_orders: dict[int, list[TileSignature]] = {core_id: [] for core_id in dst_op_meta.op_sig.core_group.core_ids}
                    
                    for dst_core_id, dst_thread in dst_op_meta.thread_mapping.items():
                        for dst_uop_node in dst_thread.uop_nodes:
                            if isinstance(dst_uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                                dst_tiled_op_sig = dst_op_meta.op_sig.tiled_ops[dst_uop_node.tiled_op_idx]
                                dst_i_tiles = dst_tiled_op_sig.i_tiles[dst_uop_node.uop_idx]
                                
                                for i_tile in dst_i_tiles:
                                    if i_tile.buf_name == op_meta.op_sig.output_buffer_name:
                                        dst_tile_access_orders[dst_core_id].append(i_tile)
                
                    for src_core_id, src_tiled_ops in tiled_op_mapping.items():
                        _prev_src_o_tiles = set()

                        for src_tiled_op_sig in src_tiled_ops:
                            src_o_tile = src_tiled_op_sig.o_tile
                            
                            for dst_core_id, dst_tile_access_order in dst_tile_access_orders.items():
                                _searched_indices = [i for i, tile in enumerate(dst_tile_access_order) if tile.signature == src_o_tile.signature]
                                if len(_searched_indices) == 0:
                                    continue
                                
                                _first_search = _searched_indices[0]
                                _final_search = _searched_indices[-1]
                                _tiles_required = set(dst_tile_access_order[i] for i in range(_first_search, _final_search, 1))
                                _n_tiles_cached = len(_tiles_required.intersection(_prev_src_o_tiles))
                                
                                max_shared_area_per_core[src_core_id] = max(max_shared_area_per_core[src_core_id], ((_n_tiles_cached + 1) * src_o_tile.tile_size))
                            
                            _prev_src_o_tiles.add(src_o_tile)
                            
                max_shared_area = max(max_shared_area_per_core.values())
                op_meta.opp_fifo_size = max_shared_area * 2
                
                if max_shared_area > (op_meta.spad_space_size_per_core - op_meta.min_ld_area_per_pp - op_meta.min_st_area_per_pp):
                    logger.debug(f"Operator {op_id} cannot be pipelined with its dependencies due to insufficient shared area per core. Required: {max_shared_area} bytes, available: {op_meta.spad_space_size_per_core - op_meta.min_ld_area_per_pp - op_meta.min_st_area_per_pp} bytes.")
                    op_meta.unfreeze()
                    for dst_op_id in dept_candidates:
                        env.op_meta[dst_op_id].unfreeze()
                    return False
                
                if not op_meta.freeze(tiled_op_mapping):
                    logger.debug(f"Operator {op_id} cannot be frozen due to unsatisfiable scheduling constraints with the current thread mapping and shared tile-to-slot mapping.")
                    op_meta.unfreeze()
                    for dst_op_id in dept_candidates:
                        env.op_meta[dst_op_id].unfreeze()
                    return False

                dept_candidates.append(op_id)
           
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
        def opp_fifo_depth(self):
            if self.opp_fifo_slot_size == 0:
                return 0
            return self.opp_fifo_size // self.opp_fifo_slot_size
        
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
        
        def allocate_core_groups(self):
            core_groups = self.recipe.core_groups
            n_core_groups = len(core_groups)
            
            if (n_core_groups < len(self.op_meta)) or (not self.recipe.operator_pipelining):
                for i, op_id in enumerate(self.op_meta.keys()):
                    op_sig = self.op_meta[op_id].op_sig
                    op_sig.initialize_core_group(self.recipe.global_core_group)
                return
            
            remaining_core_groups = n_core_groups - len(self.op_meta)
            core_group_allocation_map = {i: 1 for i in self.op_meta.keys()}  # {op_id: n_core_groups_allocated}  
            total_arithmetic_intensity = sum(op_meta.op_sig.total_arithmetic_intensity for op_meta in self.op_meta.values())
            
            if remaining_core_groups > 0 and total_arithmetic_intensity > 0:
                allocation_data = []
                for op_id, op_meta in self.op_meta.items():
                    exact_allocation = (op_meta.op_sig.total_arithmetic_intensity / total_arithmetic_intensity) * remaining_core_groups
                    floor_allocation = math.floor(exact_allocation)
                    remainder = exact_allocation - floor_allocation
                    
                    core_group_allocation_map[op_id] += floor_allocation
                    allocation_data.append((op_id, remainder))
                    
                current_allocated = sum(core_group_allocation_map.values())
                leftover_core_groups = n_core_groups - current_allocated
                
                allocation_data.sort(key=lambda x: x[1], reverse=True)
                
                for i in range(leftover_core_groups):
                    op_id, _ = allocation_data[i]
                    core_group_allocation_map[op_id] += 1
                
            for op_id in self.target_op_order:
                op_sig = self.op_meta[op_id].op_sig
                n_allocated_core_groups = core_group_allocation_map[op_id]
                allocated_core_groups = core_groups[:n_allocated_core_groups]
                core_groups = core_groups[n_allocated_core_groups:]
                
                merged_core_group = MCA_CoreGroup.merge_core_groups(allocated_core_groups)
                op_sig.initialize_core_group(merged_core_group)
                
                logger.debug(f"allocated core group {merged_core_group} for operator {op_id} (allocated {n_allocated_core_groups} core groups).")
        
        def topological_sort_grouped_target_ops(self, op_ids: set[str]) -> list[str]:
            graph = {
                op_id: [
                    dep for dep in self.op_meta[op_id].o_tile_sharers
                ] 
                for op_id in op_ids
            }
            
            in_degree = {u: 0 for u in graph}
            
            for u in graph:
                for v in graph[u]:
                    if v not in in_degree:
                        in_degree[v] = 0
                    in_degree[v] += 1

            queue = deque([u for u in in_degree if in_degree[u] == 0])
            result = []

            while queue:
                u = queue.popleft()
                result.append(u)

                if u in graph:
                    for v in graph[u]:
                        in_degree[v] -= 1

                        if in_degree[v] == 0:
                            queue.append(v)

            if len(result) != len(in_degree):
                raise ValueError("Cyclic dependency detected in the operator graph.")

            return result
            
        def freeze(self):
            if any(not op_meta.op_sig.is_core_group_initialized for op_meta in self.op_meta.values()):
                self.allocate_core_groups()
                
            pipeline_targets: list[set[str]] = [set()]
            merge_targets: list[set[str]] = []  # set of operator IDs that have been merged into the current pipeline target (to avoid merging the same operator multiple times)
            
            for op_id in self.target_op_order:
                op_meta = self.op_meta[op_id]
                is_core_group_overlapped = False
                
                for prev_target_id in pipeline_targets[-1]:
                    prev_target_meta = self.op_meta[prev_target_id]
                    if len(op_meta.op_sig.core_group.intersection(prev_target_meta.op_sig.core_group)) != 0:
                        is_core_group_overlapped = True
                        break
                    
                if is_core_group_overlapped:
                    pipeline_targets.append(set())
                    
                pipeline_targets[-1].add(op_id)
                
            for op_ids in pipeline_targets:
                if len(op_ids) == 0:
                    continue
                elif len(op_ids) == 1:
                    op_id = next(iter(op_ids))
                    op_meta = self.op_meta[op_id]
                    op_meta.freeze()
                    logger.debug(f"Successfully froze operator metadata for pipeline target with single operator {op_id}.")
                    self.grouped_compile_targets.append([op_id])
                else:
                    op_ids = MCA_OperatorGraphCompiler.OperatorMetadata.establish_dependency(self, op_ids)
                    if not MCA_OperatorGraphCompiler.OperatorMetadata.check_dependency_and_freeze(self, op_ids):
                        merge_targets.append(op_ids)
                    else:
                        logger.debug(f"Successfully froze operator metadata for pipeline target with operators {op_ids}.")
                        self.grouped_compile_targets.append(sorted(list(op_ids), key=lambda x: self.target_op_order.index(x)))
            
            for op_ids in merge_targets:
                core_group = MCA_CoreGroup.merge_core_groups([self.op_meta[op_id].op_sig.core_group for op_id in op_ids])
                for op_id in op_ids:
                    op_meta = self.op_meta[op_id]
                    op_meta.op_sig.initialize_core_group(core_group)
                    op_meta.freeze()
                    self.grouped_compile_targets.append([op_id])
                
                    logger.debug(f"Successfully froze operator metadata for merge target {op_id} (merged in {core_group}).")
                
            return self
        
    class MemoryState:
        BCAST = "bcast"
        OPP = "opp"
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
            self.opp_descriptors    = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_OPP",   l1_space.allocate(core_id, op_meta.opp_fifo_size),   op_meta.opp_fifo_slot_size,   op_meta.opp_fifo_depth)   for core_id in core_group.core_ids}
            self.ld_ex_descriptors  = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_LD_EX", l1_space.allocate(core_id, op_meta.ld_ex_fifo_size), op_meta.ld_ex_fifo_slot_size, op_meta.ld_ex_fifo_depth) for core_id in core_group.core_ids}
            self.ex_st_descriptors  = {core_id:  MCA_CompiledOperator.IR.FIFODescriptor(f"CORE{core_id}_{op_meta.op_sig.op_id}_EX_ST", l1_space.allocate(core_id, op_meta.ex_st_fifo_size), op_meta.ex_st_fifo_slot_size, op_meta.ex_st_fifo_depth) for core_id in core_group.core_ids}
            
            # Off-chip Buffers
            self.tensor_descriptors = {buf_name: MCA_CompiledOperator.IR.TensorBufferDescriptor(buf_name) for buf_name in op_sig.buffer_names}
            
            # States
            self._ctx_states: dict[int, dict[int, TileSignature]] = {core_id: {slot_id: None for slot_id in range(op_meta.ctx_buffer_slot_num)} for core_id in core_group.core_ids}
            
            self._cache_slot_size = op_meta.cache_buffer_slot_size
            self._cache_states: dict[int, dict[int, tuple[TileSignature, int, MCA_CompiledOperator.IR.MEM_COPY_TILE]]] = {core_id: {slot_id: [None, slot_id, None] for slot_id in range(op_meta.cache_buffer_slot_num)} for core_id in core_group.core_ids}  # {core_id: {slot_id: [tile_signature, lru_cnt, last_used_ir]}}
            self._cache_suspended: dict[int, dict[TileSignature, MCA_CompiledOperator.IR.MEM_COPY_TILE]] = {core_id: {} for core_id in core_group.core_ids}  # {tile_signature: (suspended mem copy IR, stage_idx)}
            
            self._bcast_states: dict[int, list[tuple[TileSignature, MCA_CompiledOperator.IR.MEM_COPY_TILE, int]]] = {core_id: [] for core_id in core_group.core_ids}  # {core_id: {slot_id: [tile_sig, mem_copy_ir, ref_count]}}
            self._bcast_tile_to_slot_id: dict[int, dict[TileSignature, int]] = {core_id: {} for core_id in core_group.core_ids}  # {core_id: {tile_signature: slot_id}}
            
            self._opp_states:   dict[int, list[tuple[TileSignature, MCA_CompiledOperator.IR.MEM_COPY_TILE, int]]] = {core_id: [] for core_id in core_group.core_ids}  # {core_id: {slot_id: [tile_sig, mem_copy_ir, ref_count]}}
            self._opp_tile_to_slot_id: dict[int, dict[TileSignature, int]] = {core_id: {} for core_id in core_group.core_ids}  # {core_id: {tile_signature: slot_id}}
            
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
        
        def cache_write(self, core_id: int, tile_sig: TileSignature, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE):
            self._cache_suspended[core_id][tile_sig] = ir
            
        def cache_read(self, core_id: int, tile_sig: TileSignature, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE) -> bool:
            # Condition: If the tile is already cached in a slot, return the slot ID and update LRU counts.
            for slot_id, (cached_tile, lru_cnt, last_used_ir) in self._cache_states[core_id].items():
                if cached_tile == tile_sig:
                    # Serialize cache readers for the same slot to avoid consuming a cache line
                    # before its producer/load path has fully committed.
                    if last_used_ir is not None and last_used_ir.ir_idx is not None:
                        ir.wait_ir_idx.append(last_used_ir.ir_idx)

                    for other_slot_id, (other_cached_tile, other_lru_cnt, other_last_used_ir) in self._cache_states[core_id].items():
                        if other_slot_id == slot_id:
                            continue
                        if other_lru_cnt < lru_cnt:
                            self._cache_states[core_id][other_slot_id][1] += 1
                    
                    self._cache_states[core_id][slot_id][1] = 0
                    self._cache_states[core_id][slot_id][2] = ir
                    # return slot_id, None, None
                    ir.src = self.cache_descriptors[core_id].ref(tile_sig=tile_sig, offset=slot_id * self._cache_slot_size)
                    return True
            
            # Condition: If the tile is not cached but there is an predecessor cmd suspended for the tile, allocate a slot to the tile
            if tile_sig in self._cache_suspended[core_id]:
                target_slot_id = max(self._cache_states[core_id].keys(), key=lambda slot_id: self._cache_states[core_id][slot_id][1])
                
                suspended_ir = self._cache_suspended[core_id][tile_sig]
                suspended_ir.dsts.append(self.cache_descriptors[core_id].ref(tile_sig=tile_sig, offset=target_slot_id * self._cache_slot_size))
                
                evicted_ir = self._cache_states[core_id][target_slot_id][2]
                
                for slot_id, (cached_tile, lru_cnt, last_used_ir) in self._cache_states[core_id].items():
                    if slot_id == target_slot_id:
                        continue
                    if lru_cnt < self._cache_states[core_id][target_slot_id][1]:
                        self._cache_states[core_id][slot_id][1] += 1
                
                self._cache_states[core_id][target_slot_id][0] = tile_sig  # evict the tile currently in the target slot
                self._cache_states[core_id][target_slot_id][1] = 0
                self._cache_states[core_id][target_slot_id][2] = ir
                del self._cache_suspended[core_id][tile_sig]
                
                if suspended_ir.ir_idx is None:
                    raise Exception(f"Suspended IR for tile {tile_sig} in core {core_id} does not have a valid IR index.")
                
                if evicted_ir is not None:
                    ir.wait_ir_idx.append(evicted_ir.ir_idx)
                ir.wait_ir_idx.append(suspended_ir.ir_idx)
                
                # return target_slot_id, suspended_ir.ir_idx, evicted_ir.ir_idx
                ir.src = self.cache_descriptors[core_id].ref(tile_sig=tile_sig, offset=target_slot_id * self._cache_slot_size)
                return True

            # Condition: If the tile is not cached and there is no predecessor cmd suspended for the tile, return None
            return False
        
        def bcast_push(self, core_id: int, tile_sig: TileSignature, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE) -> int:
            slot_id = len(self._bcast_states[core_id])
            self._bcast_tile_to_slot_id[core_id][tile_sig] = slot_id
            self._bcast_states[core_id].append([tile_sig, ir, 0])
            return slot_id
            
        def bcast_pop(self, core_id: int, slot_id: int) -> int:
            # if tile_sig not in self._bcast_tile_to_slot_id[core_id]:
            #     raise Exception(f"Tile {tile_sig} is not found in the broadcast buffer of core {core_id}.")
            # slot_id = self._bcast_tile_to_slot_id[core_id][tile_sig]
            if slot_id >= len(self._bcast_states[core_id]):
                raise Exception(f"Slot ID {slot_id} is out of range for the broadcast buffer of core {core_id}.")
            self._bcast_states[core_id][slot_id][2] += 1
            return slot_id
        
        def bcast_cleanup(self):
            for core_id in self._bcast_states.keys():
                for slot_id, (tile_sig, ir, ref_cnt) in enumerate(self._bcast_states[core_id]):
                    if ref_cnt == 0:
                        continue
                    
                    ir.dsts.append(self.bcast_descriptors[core_id].ref(tile_sig, slot_id, ref_cnt))
            
                self._bcast_states[core_id].clear()
                self._bcast_tile_to_slot_id[core_id].clear()
        
        def opp_push(self, core_id: int, tile_sig: TileSignature, ir: MCA_CompiledOperator.IR.MEM_COPY_TILE):
            slot_id = len(self._opp_states[core_id])
            self._opp_tile_to_slot_id[core_id][tile_sig] = slot_id
            self._opp_states[core_id].append([tile_sig, ir, 0])
            
        def opp_pop(self, tile_sig: TileSignature) -> tuple[int, int]:
            for core_id, tile_to_slot_id in self._opp_tile_to_slot_id.items():
                if tile_sig in tile_to_slot_id:
                    slot_id = tile_to_slot_id[tile_sig]
                    self._opp_states[core_id][slot_id][2] += 1
                    return core_id, slot_id
            
            raise Exception(f"Tile {tile_sig} is not found in the operand buffer of any core.")
        
        def opp_cleanup(self):
            for core_id in self._opp_states.keys():
                for slot_id, (tile_sig, ir, ref_cnt) in enumerate(self._opp_states[core_id]):
                    if ref_cnt == 0:
                        continue
                    
                    ir.dsts.append(self.opp_descriptors[core_id].ref(tile_sig, slot_id, ref_cnt))
                
                self._opp_states[core_id].clear()
                self._opp_tile_to_slot_id[core_id].clear()
        
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
    
    @staticmethod
    def _resolve_bcast_deadlocks(
        event_keys: dict[str, tuple[int, int, int]], 
        event_locations: dict[str, dict[int, tuple[int, int]]], 
        core_event_order: dict[int, list[str]]
    ) -> None:
        """
        Hyper-optimized deadlock resolution using incremental contracted graphs.
        """
        def get_sccs(nodes, adj):
            indices, lowlink = {}, {}
            visited_stack, on_stack = [], set()
            index_counter = [0]
            sccs = []
            for node in nodes:
                if node not in indices:
                    call_stack = [(node, iter(adj[node]))]
                    while call_stack:
                        u, neighbors = call_stack[-1]
                        if u not in indices:
                            indices[u] = lowlink[u] = index_counter[0]
                            index_counter[0] += 1
                            visited_stack.append(u)
                            on_stack.add(u)
                        try:
                            v = next(neighbors)
                            if v not in indices:
                                call_stack.append((v, iter(adj[v])))
                            elif v in on_stack:
                                lowlink[u] = min(lowlink[u], indices[v])
                        except StopIteration:
                            call_stack.pop()
                            if lowlink[u] == indices[u]:
                                new_scc = set()
                                while True:
                                    w = visited_stack.pop()
                                    on_stack.remove(w)
                                    new_scc.add(w)
                                    if w == u: break
                                if len(new_scc) > 1: sccs.append(new_scc)
                            if call_stack:
                                parent, _ = call_stack[-1]
                                lowlink[parent] = min(lowlink[parent], lowlink[u])
            return sccs

        # 1. INITIALIZATION: Build indexing for O(1) full program updates
        event_to_cores = defaultdict(set)
        core_full_pos = {c: {ev: i for i, ev in enumerate(order)} for c, order in core_event_order.items()}
        for c, order in core_event_order.items():
            for ev in order: event_to_cores[ev].add(c)
        
        shared_set = {ev for ev, cores in event_to_cores.items() if len(cores) > 1}
        # core_shared_order: Contains only events that are globally shared (graph nodes)
        core_shared_order = {c: [ev for ev in order if ev in shared_set] for c, order in core_event_order.items()}
        
        iter_cnt = 0
        while True:
            # 2. ADJ CONSTRUCTION: Build graph using contracted lists (very fast)
            # with print_log_execution_time("ADJ CONSTRUCTION"):
            adj = defaultdict(set)
            for order in core_shared_order.values():
                for i in range(len(order) - 1):
                    adj[order[i]].add(order[i+1])
            
            # 3. SSC DETERMINATION: Detect all cycles in current topology
            # with print_log_execution_time("SCC DETECTION"):
            sccs = get_sccs(list(shared_set), adj)
            if not sccs: break
            
            # 4. DEADLOCK RESOLUTION: Batch process SCCs
            # Maps for SHARED nodes only (small dictionary)
            # with print_log_execution_time("DEADLOCK RESOLUTION"):
            shared_pos_map = {c: {ev: i for i, ev in enumerate(order)} for c, order in core_shared_order.items()}
            
            for scc in sccs:
                target_event = max(scc, key=lambda e: len(event_to_cores[e]))
                orig_data = event_keys[target_event]
                
                core_groups = defaultdict(list)
                for core in event_to_cores[target_event]:
                    c_pos = shared_pos_map[core]
                    curr_idx = c_pos[target_event]
                    # Signature based on positions of SCC members appearing before target_event
                    sig = tuple(sorted(c_pos[m] for m in scc if m in c_pos and c_pos[m] < curr_idx))
                    core_groups[sig].append(core)
                
                if len(core_groups) <= 1:
                    cores = list(event_to_cores[target_event])
                    core_groups = {0: [cores[0]], 1: cores[1:]}

                sorted_sigs = sorted(core_groups.keys(), key=str)
                for i, sig in enumerate(sorted_sigs, 1):
                    new_ev = f"{target_event}_{i}"
                    event_keys[new_ev] = (orig_data[0], orig_data[1], i)
                    event_locations[new_ev] = {}
                    
                    for core in core_groups[sig]:
                        event_locations[new_ev][core] = event_locations[target_event][core]
                        # Update physical program (O(1))
                        f_idx = core_full_pos[core][target_event]
                        core_event_order[core][f_idx] = new_ev
                        core_full_pos[core][new_ev] = f_idx
                        # Update contracted shared order (O(1))
                        s_idx = shared_pos_map[core][target_event]
                        core_shared_order[core][s_idx] = new_ev
                        event_to_cores[new_ev].add(core)
                
                event_to_cores.pop(target_event)
                event_keys.pop(target_event)
                event_locations.pop(target_event)
            
            # 5. REFRESH: Re-contract lists for next pass
            # with print_log_execution_time("REFRESH"):
            shared_set = {ev for ev, cores in event_to_cores.items() if len(cores) > 1}
            for c in core_shared_order:
                core_shared_order[c] = [ev for ev in core_shared_order[c] if ev in shared_set]
                
            logger.debug(f"Deadlock resolution iteration {iter_cnt} completed with {len(sccs)} SCCs resolved.")
            iter_cnt += 1

    @staticmethod
    def _apply_bcast(compiled_op: MCA_CompiledOperator, op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata', mem_state: 'MCA_OperatorGraphCompiler.MemoryState'):
        logger.debug("Applying broadcast optimization...")
        
        # STEP 1: Tagging access and occurrences to build a global event registry
        event_keys = {}
        event_locations = defaultdict(dict)
        core_event_order = {core_id: [] for core_id in compiled_op.mappings.keys()}
        
        for core_id, stages in compiled_op.mappings.items():
            ocurrence_tracker = defaultdict(int)
            for stage_idx, stage in enumerate(stages):
                for ir_idx, ir in enumerate(stage.loads):
                    if not isinstance(ir, MCA_CompiledOperator.IR.MEM_COPY_TILE): continue
                    if not ir.src.is_tensor(): continue
                    if not op_meta.op_sig.buffers[ir.src.tile_sig.buf_name].mem_space.is_main: continue
                    occ_num = ocurrence_tracker[ir.src.tile_sig]
                    ocurrence_tracker[ir.src.tile_sig] += 1
                    event_key = f"{ir.src.tile_sig.signature}_{occ_num}"
                    event_keys[event_key] = (ir.src.tile_sig, occ_num, None)
                    event_locations[event_key][core_id] = (stage_idx, ir_idx)
                    core_event_order[core_id].append(event_key)
        
        # STEP 2: Check inconsistent broadcast event order and resolve the inconsistency
        MCA_OperatorGraphCompiler._resolve_bcast_deadlocks(event_keys, event_locations, core_event_order)
        
        # STEP 3: Producer Selection with Load Balancing
        # Filter events that have multiple cores participating (qualified for broadcast)
        qualified_events = {ev for ev, locs in event_locations.items() if len(locs) > 1}
        _producer_cnt = {core_id: 0 for core_id in compiled_op.mappings.keys()}
        final_producers = {}
        final_consumers = defaultdict(set)
        
        # Sort by earliest arrival time to keep load balancing stable
        sorted_qualified = sorted(list(qualified_events), key=lambda ev: min(event_locations[ev].values()))
        for event_key in sorted_qualified:
            locations = event_locations[event_key]
            min_time = min(locations.values())
            candidates = [cid for cid, t in locations.items() if t == min_time]
            candidates.sort(key=lambda cid: (_producer_cnt[cid], cid))
            producer = candidates[0]
            final_producers[event_key] = producer
            _producer_cnt[producer] += 1
            for cid in locations:
                if cid != producer: final_consumers[event_key].add(cid)

        # STEP 4: Apply the finalized broadcast plan in exact Core Program Order
        event_slot_map = {} # { event_key: slot_id }
        
        # Producers must allocate slots and push in their physical program order
        for prod_core in compiled_op.mappings.keys():
            # Get all events where this core is the chosen producer and there are consumers
            ordered_productions = [ev for ev in core_event_order[prod_core] if final_producers.get(ev) == prod_core and final_consumers[ev]]
            for event_key in ordered_productions:
                s_idx, i_idx = event_locations[event_key][prod_core]
                ir = compiled_op.mappings[prod_core][s_idx].loads[i_idx]
                tile_sig = event_keys[event_key][0]
                slot_id = mem_state.bcast_push(prod_core, tile_sig, ir)
                event_slot_map[event_key] = slot_id

        # Consumers must pop according to the producer-specific slot sequence
        for cons_core in compiled_op.mappings.keys():
            for event_key in core_event_order[cons_core]:
                if cons_core in final_consumers[event_key]:
                    prod_core = final_producers[event_key]
                    slot_id = event_slot_map[event_key]
                    tile_sig = event_keys[event_key][0]
                    mem_state.bcast_pop(prod_core, slot_id=slot_id)
                    s_idx, i_idx = event_locations[event_key][cons_core]
                    ir = compiled_op.mappings[cons_core][s_idx].loads[i_idx]
                    ir.src = mem_state.bcast_descriptors[prod_core].ref(tile_sig, slot_id, ref_cnt=1)
                
        mem_state.bcast_cleanup()
        
    def compile_grouped_target_ops(self, env: 'MCA_OperatorGraphCompiler.Environment', op_ids: set[str]) -> dict[str, MCA_CompiledOperator]:
        op_ids: list[str] = env.topological_sort_grouped_target_ops(op_ids)
        
        op_metas = {target_op_id: env.op_meta[target_op_id] for target_op_id in op_ids}
        
        compiled_ops    = {op_id: MCA_CompiledOperator(env, op_meta) for op_id, op_meta in op_metas.items()}
        mem_states      = {op_id: MCA_OperatorGraphCompiler.MemoryState(op_meta, env.recipe) for op_id, op_meta in op_metas.items()} 
        thread_mappings = {op_id: op_meta.thread_mapping for op_id, op_meta in op_metas.items()}
        
        tile_producers: dict[str, dict[TileSignature, int]] = {op_id: {} for op_id in op_ids}
        
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            op_sig = op_meta.op_sig
            core_group = op_sig.core_group
            mem_state = mem_states[op_id]
            
            for core_id in core_group.core_ids:
                env.add_fifo_buffer(mem_state.bcast_descriptors[core_id].buf_name, op_meta.bcast_fifo_depth, op_meta.bcast_fifo_slot_size, mem_state.bcast_descriptors[core_id].ptr)
                env.add_fifo_buffer(mem_state.opp_descriptors[core_id].buf_name,   op_meta.opp_fifo_depth,   op_meta.opp_fifo_slot_size,   mem_state.opp_descriptors[core_id].ptr)
                env.add_fifo_buffer(mem_state.ld_ex_descriptors[core_id].buf_name, op_meta.ld_ex_fifo_depth, op_meta.ld_ex_fifo_slot_size, mem_state.ld_ex_descriptors[core_id].ptr)
                env.add_fifo_buffer(mem_state.ex_st_descriptors[core_id].buf_name, op_meta.ex_st_fifo_depth, op_meta.ex_st_fifo_slot_size, mem_state.ex_st_descriptors[core_id].ptr)
        
        # STAGE 1: Analyze dependencies and determine tile-level sharing relationships (for pipelining)
        for src_op_id in op_ids:
            src_meta = op_metas[src_op_id]
            
            for src_core_id, src_thread in thread_mappings[src_op_id].items():
                for uop_node_idx, uop_node in enumerate(src_thread.uop_nodes):
                    if isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                        tiled_op_sig = src_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        src_o_tile = tiled_op_sig.o_tile
                        
                        tile_producers[src_op_id][src_o_tile] = src_core_id
        
        # STAGE 2: Create compiled ops and update memory states while iteratively resolving tile-level dependencies
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            thread_mapping = thread_mappings[op_id]
            mem_state = mem_states[op_id]
            
            for core_id, thread in thread_mapping.items():     
                logger.debug(f"Compiling operator {op_id} on core {core_id}...")           
                for iii, uop_node in enumerate(thread.uop_nodes):
                    logger.debug(f"Processing uop node {iii}/{len(thread.uop_nodes)}", end="\r")
                    
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
                    
                    elif isinstance(uop_node, MCA_OperatorGraphCompiler.Thread.UopNode):
                        tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        o_tile = tiled_op_sig.o_tile
                        
                        for i_tile in i_tiles:
                            dst = mem_state.ld_ex_descriptors[core_id].ref(i_tile, slot_id=mem_state.ld_ex_push(core_id, i_tile), ref_cnt=1)
                            ir = MCA_CompiledOperator.IR.MEM_COPY_TILE(None, dst)   # src undefined until the cache hit is verified
                            is_cache_hit = mem_state.cache_read(core_id, i_tile, ir)
                            
                            # CASE: cache hit
                            if is_cache_hit:
                                pass  # the cache_read method already updates the src of the IR
                            # CASE: cache miss / from buffer
                            elif op_meta.i_buf_src[i_tile.buf_name].is_buffer:  
                                ir.src = mem_state.tensor_descriptors[i_tile.buf_name].ref(i_tile)
                            # CASE: cache miss / from opp fifo
                            else:
                                opp_src_op_id = op_meta.i_buf_src[i_tile.buf_name].k
                                opp_src_core_id, opp_src_slot_id = mem_states[opp_src_op_id].opp_pop(i_tile)
                                ir.src = mem_states[opp_src_op_id].opp_descriptors[opp_src_core_id].ref(i_tile, slot_id=opp_src_slot_id, ref_cnt=1)
                            
                            if not is_cache_hit:
                                mem_state.cache_write(core_id, i_tile, ir)
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
                            if len(op_meta.o_tile_sharers) > 0:
                                mem_state.opp_push(core_id, o_tile, ir)
                                
                            compiled_ops[op_id].add_store_ir(core_id, ir)
        
        for op_id in op_ids:
            mem_states[op_id].opp_cleanup()

        # STAGE 4: Apply bcast
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            mem_state = mem_states[op_id]
            compiled_op = compiled_ops[op_id]
            
            if op_meta.bcast_fifo_depth == 0:
                continue
            
            self._apply_bcast(compiled_op, op_meta, mem_state)
        
        # # STAGE 4: Apply bcast
        # for op_id in op_ids:
        #     op_meta = op_metas[op_id]
        #     core_group = op_meta.op_sig.core_group
        #     mem_state = mem_states[op_id]
        #     compiled_op = compiled_ops[op_id]
            
        #     if op_meta.bcast_fifo_depth == 0:
        #         continue
            
        #     core_ids = sorted(core_group.core_ids)
            
        #     for src_core_id, dst_core_id in zip(core_ids[:-1], core_ids[1:]):
        #         for stage_idx in range(compiled_op.n_stages):
        #             src_stage = compiled_op.mappings[src_core_id][stage_idx] if stage_idx < len(compiled_op.mappings.get(src_core_id, [])) else None
        #             dst_stage = compiled_op.mappings[dst_core_id][stage_idx] if stage_idx < len(compiled_op.mappings.get(dst_core_id, [])) else None
                    
        #             if src_stage is None or dst_stage is None:
        #                 continue
                    
        #             ir_pairs: list[tuple[int, int]] = []
                    
        #             for src_ir_idx, src_ir in enumerate(src_stage.loads):
        #                 if not isinstance(src_ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
        #                     continue
        #                 if (src_ir.src is None) or (not src_ir.src.is_tensor()):
        #                     continue    # ignore non-tensor sources
        #                 if env.buffers[src_ir.src.tile_sig.buf_name].mem_space.is_l1:
        #                     continue    # ignore L1 buffers
                        
        #                 for dst_ir_idx, dst_ir in enumerate(dst_stage.loads):
        #                     if not isinstance(dst_ir, MCA_CompiledOperator.IR.MEM_COPY_TILE):
        #                         continue
        #                     if (dst_ir.src is None) or (not dst_ir.src.is_tensor()):
        #                         continue    # ignore non-tensor sources
        #                     if env.buffers[dst_ir.src.tile_sig.buf_name].mem_space.is_l1:
        #                         continue    # ignore L1 buffers
                            
        #                     if src_ir.src.tile_sig == dst_ir.src.tile_sig:
        #                         ir_pairs.append((src_ir_idx, dst_ir_idx))
        #                         break
                    
        #             for src_ir_idx, dst_ir_idx in ir_pairs:
        #                 src_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE = src_stage.loads[src_ir_idx]
        #                 dst_ir: MCA_CompiledOperator.IR.MEM_COPY_TILE = dst_stage.loads[dst_ir_idx]
                        
        #                 mem_state.bcast_push(src_core_id, src_ir.src.tile_sig, src_ir)
        #                 bcast_slot_id = mem_state._bcast_tile_to_slot_id[src_core_id][src_ir.src.tile_sig]
                        
        #                 mem_state.bcast_pop(src_core_id, dst_ir.src.tile_sig)
        #                 dst_ir.src = mem_state.bcast_descriptors[src_core_id].ref(
        #                     dst_ir.src.tile_sig,
        #                     slot_id=bcast_slot_id,
        #                     ref_cnt=1
        #                 )
                
        # for op_id in op_ids:
        #     mem_states[op_id].bcast_cleanup()
                
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
        
        return MCA_CompiledProgram(recipe.device, compiled_ops)
