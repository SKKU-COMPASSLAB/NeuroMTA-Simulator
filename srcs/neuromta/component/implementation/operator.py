import abc
import enum
import functools
import math
import time
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


class MCA_OperatorSignature:
    class ReorderType(enum.Enum):
        ROW_MAJOR = "ROW_MAJOR"
        COL_MAJOR = "COL_MAJOR"
    
    def __init__(
        self, 
        op_type: str, 
        ld_thread_template: Callable,
        ex_thread_template: Callable,
        st_thread_template: Callable,
        op_ex_kernels: list[Callable]
    ):
        self._op_type = op_type
        self.op_id = op_type    # will be initialized by MCA_OperatorGraphCompiler (initially set to op_type) 
        
        self.ld_thread_template = ld_thread_template
        self.ex_thread_template = ex_thread_template
        self.st_thread_template = st_thread_template
        self.op_ex_kernels = op_ex_kernels
        
        self._buffers: dict[str, MCA_TensorBuffer] = {}
        self._tiles: dict[str, dict[tuple[int, ...], TileSignature]] = {}
        self._tiled_ops: list[TiledOperatorSignature] = []
        self.global_kwargs: dict[str, Any] = {}
        
        self.buffer_names: list[str] = []
        self.input_buffer_names: list[str] = []
        self.output_buffer_name: str = None
        
        self.core_group: MCA_CoreGroup = None
        self.reorder_type = MCA_OperatorSignature.ReorderType.ROW_MAJOR
        
    def add_buffer(self, buf_name: str, buffer: MCA_TensorBuffer, is_input: bool=False, is_output: bool=False):
        if (not is_input) and (not is_output):
            raise ValueError("Buffer must be marked as input or output.")
        
        self._buffers[buf_name] = buffer
        self._tiles[buf_name] = {}
        
        for y_s in range(buffer.shard_grid[0]):
            for x_s in range(buffer.shard_grid[1]):
                for y_t in range(buffer.tile_grid_per_shard[0]):
                    for x_t in range(buffer.tile_grid_per_shard[1]):
                        self._tiles[buf_name][(y_s, x_s, y_t, x_t)] = TileSignature(buf_name, buffer.tile_size, y_s, x_s, y_t, x_t)
        
        self.buffer_names.append(buf_name)
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
    
    def reorder_tiled_ops(self):
        def tile_sort_key(tiled_op: TiledOperatorSignature):
            tile = tiled_op.o_tile
            y_s, x_s, y_t, x_t = tile.coords
            if self.reorder_type == MCA_OperatorSignature.ReorderType.ROW_MAJOR:
                return (y_s, y_t, x_s, x_t)
            else:
                return (x_s, x_t, y_s, y_t)
            
        self._tiled_ops.sort(key=tile_sort_key)
            
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
        class Base(metaclass=abc.ABCMeta):
            @abc.abstractmethod
            def signature(self) -> str:
                raise NotImplementedError("Command signature method must be implemented by subclasses.")
            
            def __repr__(self):
                return self.signature()
            
        class NOP(Base):
            def signature(self):
                return "NOP"
        
        class MEM_INIT(Base):
            def __init__(self, ptr: Pointer, size: int):
                self.ptr = ptr
                self.size = size
                
            def signature(self):
                return f"MEM_INIT MEM@{self.ptr.addr} size={self.size}"
            
        class MEM_SYNC(Base):
            def signature(self):
                return f"MEM_SYNC # async_rpc_wait_all()"
        
        class MEM_LOAD_TILE(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer):
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                    
            def signature(self):
                return f"MEM_LOAD_TILE {self.tile_sig.signature} -> SPM@{self.ptr.addr}"
                
        class MEM_STORE_TILE(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer, is_partial: bool=False):
                self.tile_sig = tile_sig
                self.ptr = ptr
                self.is_partial = is_partial
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                    
            def signature(self):
                sig = f"MEM_STORE_TILE SPM@{self.ptr.addr} -> {self.tile_sig.signature}"
                if self.is_partial:
                    sig += " (partial)"
                return sig
            
        class MEM_LOAD_FROM_FIFO(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer, buf: str, entry_id: int):
                self.tile_sig = tile_sig
                self.ptr = ptr
                self.buf = buf
                self.entry_id = entry_id

                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                if isinstance(self.buf, FIFOBufferHandle):
                    self.buf = self.buf.handle_name
                    
            def signature(self):
                return f"MEM_LOAD_FROM_FIFO {self.tile_sig.signature} FIFO@{self.buf}[entry_id={self.entry_id}] -> SPM@{self.ptr.addr}"
            
        class MEM_STORE_TO_FIFO(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer, buf: str, entry_id: int, ref_count: int):
                self.tile_sig = tile_sig
                self.ptr = ptr
                self.buf = buf
                self.entry_id = entry_id
                self.ref_count = ref_count

                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                if isinstance(self.buf, FIFOBufferHandle):
                    self.buf = self.buf.handle_name

            def signature(self):
                return f"MEM_STORE_TO_FIFO {self.tile_sig.signature} SPM@{self.ptr.addr} -> FIFO@{self.buf}[entry_id={self.entry_id}] (ref_count={self.ref_count})"
            
        class EXE_UOP(Base):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, i_tile_ptrs: list[Pointer], o_tile_ptr: Pointer, o_tile_sig: TileSignature):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.i_tile_ptrs = i_tile_ptrs
                self.o_tile_ptr = o_tile_ptr
                self.o_tile_sig = o_tile_sig
                
            def signature(self):
                i_ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.i_tile_ptrs])
                if self.o_tile_ptr is None:
                    o_ptr_str = f"{self.o_tile_sig.signature} SPM@UNDEFINED"
                else:
                    o_ptr_str = f"{self.o_tile_sig.signature} SPM@{self.o_tile_ptr.addr}"
                return f"EXE_UOP {self.op_id} tiled_op_idx={self.tiled_op_idx} uop_idx={self.uop_idx} ({i_ptrs_str}) -> {o_ptr_str}"

    class Group:
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
        def is_bubble(self) -> bool:
            return len(self.loads) == 0 and len(self.executes) == 0 and len(self.stores) == 0

    class Stage:
        def __init__(self):
            self.groups: list[MCA_CompiledOperator.Group] = [MCA_CompiledOperator.Group()]
            
        def new_group(self) -> 'MCA_CompiledOperator.Group':
            group = MCA_CompiledOperator.Group()
            self.groups.append(group)
            return group
        
        def add_load_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.groups[-1].add_load_ir(cmd)
            
        def add_execute_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.groups[-1].add_execute_ir(cmd)
            
        def add_store_ir(self, cmd: 'MCA_CompiledOperator.IR.Base'):
            self.groups[-1].add_store_ir(cmd)
            
        def freeze(self):
            _groups = []
            
            for group in self.groups:
                group.freeze()
                
                if not group.is_bubble:
                    _groups.append(group)
            
            self.groups = _groups

        def summary(self) -> list:
            return [group.summary() for group in self.groups]
        
        @property
        def is_bubble(self) -> bool:
            return len(self.groups) == 0 or all(group.is_bubble for group in self.groups)

    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata'):
        self._env = env
        self._op_id = op_meta.op_sig.op_id
        self._ld_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.ld_thread_template
        self._ex_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.ex_thread_template
        self._st_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.st_thread_template
        self._op_ex_kernels = op_meta.op_sig.op_ex_kernels
        
        self._mappings: dict[int, list[MCA_CompiledOperator.Stage]] = {
            core_id: [MCA_CompiledOperator.Stage()] 
            for core_id in op_meta.op_sig.core_group.core_ids
        }  # {core_id: [stage1, stage2, ...]}
        
    def new_stage(self, core_id: int) -> 'MCA_CompiledOperator.Stage':
        stage = MCA_CompiledOperator.Stage()
        self._mappings[core_id].append(stage)
        return stage
    
    def new_group(self, core_id: int) -> 'MCA_CompiledOperator.Group':
        if not self._mappings[core_id]:
            raise ValueError(f"No stage exists for core ID {core_id}. Create a stage before creating a group.")
        return self._mappings[core_id][-1].new_group()
    
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
        
        barrier = (
            self._env.add_variable(f"op_{self._op_id}_barrier_arrival_cnt", initial_value=0),
            self._env.add_variable(f"op_{self._op_id}_barrier_blocking_state", initial_value=0),
            len(self.mappings.keys()) * 3
        )
        
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            
            ex_pp_cnt_var = self._env.add_variable(f"op_{self._op_id}_core_{core_id}_ex_pp_cnt", initial_value=0)
            ex_pr_cnt_var = self._env.add_variable(f"op_{self._op_id}_core_{core_id}_ex_pr_cnt", initial_value=0)
            st_pp_cnt_var = self._env.add_variable(f"op_{self._op_id}_core_{core_id}_st_pp_cnt", initial_value=0)
            st_pr_cnt_var = self._env.add_variable(f"op_{self._op_id}_core_{core_id}_st_pr_cnt", initial_value=0)
            
            for i, stage in enumerate(self.mappings[core_id]):
                is_last = (i == len(self.mappings[core_id]) - 1)
                
                ld_thread = self._ld_thread_template(core, self._env, stage, ex_pp_cnt_var.handle_name, ex_pr_cnt_var.handle_name, barrier if is_last else None)
                ex_thread = self._ex_thread_template(core, self._env, stage, self._op_ex_kernels, ex_pp_cnt_var.handle_name, ex_pr_cnt_var.handle_name, st_pp_cnt_var.handle_name, st_pr_cnt_var.handle_name, barrier if is_last else None)
                st_thread = self._st_thread_template(core, self._env, stage, st_pp_cnt_var.handle_name, st_pr_cnt_var.handle_name, barrier if is_last else None)
                
                ld_thread.dispatch("LD")
                ex_thread.dispatch("EX")
                st_thread.dispatch("ST")
        
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
        def __init__(
            self, 
            device: MCA_DeviceBase,
            core_groups: list[MCA_CoreGroup],
            spad_space_size_per_core: int,
            pipeline_granularity: int=8,
            broadcast_optimize_queue_depth: int=16,
            operator_pipelining: bool=False,
        ):
            if len(core_groups) == 0:
                raise ValueError("At least one core group must be provided.")
            if not isinstance(core_groups[0], (MCA_CoreGroup, list)):
                core_groups = [core_groups]

            self.device                         = device
            self.core_groups = core_groups
            self.spad_space_size_per_core       = spad_space_size_per_core
            self.pipeline_granularity           = pipeline_granularity
            self.broadcast_optimize_queue_depth = broadcast_optimize_queue_depth
            self.operator_pipelining            = operator_pipelining
        
        @property
        def global_core_group(self) -> MCA_CoreGroup:
            return MCA_CoreGroup.merge_core_groups(self.core_groups)
        
        @property
        def broadcast_optimize(self) -> bool:
            return self.broadcast_optimize_queue_depth > 0
            
    class Thread:
        class UopNode:
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.output = output
                
            @classmethod
            def bubble(cls, op_id: str) -> 'MCA_OperatorGraphCompiler.Thread.UopNode':
                return cls(op_id=op_id, tiled_op_idx=None, uop_idx=None, output=False)
            
            @property
            def is_bubble(self) -> bool:
                return self.tiled_op_idx is None and self.uop_idx is None
                
        def __init__(self, core_id: int, uop_nodes: list['MCA_OperatorGraphCompiler.Thread.UopNode']=None):
            self.core_id = core_id
            self.uop_nodes = uop_nodes if uop_nodes is not None else []
            
        def add_uop_node(self, op_id: str, tiled_op_idx: int, uop_idx: int, output: bool=False):
            uop_node = self.UopNode(op_id, tiled_op_idx, uop_idx, output=output)
            self.uop_nodes.append(uop_node)
            
        def add_bubble(self, op_id: str):
            bubble_node = self.UopNode.bubble(op_id)
            self.uop_nodes.append(bubble_node)
        
        @property 
        def n_uop_nodes(self) -> int:
            return len(self.uop_nodes)
        
        @staticmethod
        def _mapping(op_sig: 'MCA_OperatorSignature', tiled_op_idx: int) -> int:
            n_cores = len(op_sig.core_group.core_ids)
            core_id = op_sig.core_group.core_ids[tiled_op_idx % n_cores]
            return core_id
            
        @classmethod
        def from_op_sig(cls, op_sig: 'MCA_OperatorSignature') -> 'dict[int, MCA_OperatorGraphCompiler.Thread]':
            mappings: dict[int, MCA_OperatorGraphCompiler.Thread] = {core_id: MCA_OperatorGraphCompiler.Thread(core_id) for core_id in op_sig.core_group.core_ids}
            
            for tiled_op_idx, tiled_op_sig in enumerate(op_sig.tiled_ops):
                core_id = cls._mapping(op_sig, tiled_op_idx)
                
                for uop_idx in range(tiled_op_sig.n_uops):
                    output = (uop_idx == (tiled_op_sig.n_uops - 1))  # mark the last uop of each tiled op as output uop (for store scheduling)
                    mappings[core_id].add_uop_node(op_sig.op_id, tiled_op_idx, uop_idx, output=output)
                    
            max_n_uop_nodes_per_thread = max(thread.n_uop_nodes for thread in mappings.values())
            
            for thread in mappings.values():
                while thread.n_uop_nodes < max_n_uop_nodes_per_thread:
                    thread.add_bubble(op_id=op_sig.op_id)
            
            return mappings
            
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
            # self.spad_space_size_per_pp = recipe.spad_space_size_per_core // 2
            self.spad_space_size_per_core = recipe.spad_space_size_per_core
            
            self.i_buf_src: dict[str, MCA_OperatorGraphCompiler.OperatorMetadata.SrcType] = {
                buf_name: MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.BUFFER() 
                for buf_name in op_sig.input_buffer_names
            }
            
            # Dependencies related to output buffer sharing (a.k.a tile-level pipelining)
            self.o_tile_store = op_sig.buffers[op_sig.output_buffer_name].is_allocated  # if the output buffer is allocated, the computation result should be updated to the buffer
            self.o_tile_sharers: set[str] = set()  # set of op_ids that directly consume this operator's output tiles (tile-level sharers via SHARED area)

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
                    
            self.ld_ratio = self.min_ld_area_per_pp / (self.min_ld_area_per_pp + self.min_st_area_per_pp) if (self.min_ld_area_per_pp + self.min_st_area_per_pp) > 0 else 0.8
            
            # FIFO buffer space
            self.spad_space_size_per_pp = self.min_ld_area_per_pp + self.min_st_area_per_pp
            
            self.bcast_fifo_slot_size = max(buf.tile_size for buf_name, buf in op_sig.buffers.items() if buf_name in op_sig.input_buffer_names)
            self.bcast_buffer_size    = recipe.broadcast_optimize_queue_depth * self.bcast_fifo_slot_size
            self.opp_fifo_slot_size   = op_sig.buffers[op_sig.output_buffer_name].tile_size
            self.max_opp_buffer_size  = self.spad_space_size_per_core - self.bcast_buffer_size - 2 * self.spad_space_size_per_pp
            
            if self.max_opp_buffer_size < 0:
                raise ValueError(f"Insufficient shared area per core for operator {op_sig.op_id}. Required: at least {self.min_ld_area_per_pp + self.min_st_area_per_pp + self.bcast_buffer_size} bytes, but only {recipe.spad_space_size_per_core} bytes available per core.")
            
            # Actual LD/ST/SHARED area per core to be determined based on the dependencies with consumer operators (initially set to 0, will be updated when analyzing dependencies)
            self.min_opp_buffer_size = 0
            self.thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            self._is_frozen = False
            
        def freeze(self, thread_mapping: 'dict[int, MCA_OperatorGraphCompiler.Thread]' = None):
            self.thread_mapping = MCA_OperatorGraphCompiler.Thread.from_op_sig(self.op_sig) if thread_mapping is None else thread_mapping
            self.spad_space_size_per_pp = math.floor((self.spad_space_size_per_core - self.bcast_buffer_size - self.min_opp_buffer_size) / 2)
            
            if self.spad_space_size_per_pp < 0:
                raise ValueError(f"Insufficient shared area per core for operator {self.op_sig.op_id} after accounting for reserved space. Required: at least {self.min_opp_buffer_size + self.bcast_buffer_size} bytes, but only {self.spad_space_size_per_core} bytes available per core.")
            if self.min_st_area_per_pp + self.min_ld_area_per_pp > self.spad_space_size_per_pp:
                raise ValueError(f"Insufficient shared area per core for operator {self.op_sig.op_id}. Required: {self.min_st_area_per_pp + self.min_ld_area_per_pp} bytes, maximum allowed: {self.spad_space_size_per_pp} bytes.")
            
            if self.o_tile_store and not self.op_sig.buffers[self.op_sig.output_buffer_name].is_allocated:
                self.op_sig.buffers[self.op_sig.output_buffer_name].allocate()  # TODO: out-of-memory situation?
            
            self._is_frozen = True
            
        @staticmethod
        def check_dependency_and_freeze(env: 'MCA_OperatorGraphCompiler.Environment', op_ids: list[str]) -> bool:
            op_metas = {target_op_id: env.op_meta[target_op_id] for target_op_id in op_ids}
            thread_mappings = {op_id: MCA_OperatorGraphCompiler.Thread.from_op_sig(op_metas[op_id].op_sig) for op_id in op_ids}
            max_shared_area_per_core: dict[str, dict[int, int]] = {op_id: {core_id: 0 for core_id in op_metas[op_id].op_sig.core_group} for op_id in op_ids}  # {op_id: {core_id: shared_area_size}}
            
            for srd_op_id in op_ids:
                src_op_meta = op_metas[srd_op_id]
                
                for dst_op_id in op_ids:
                    if srd_op_id == dst_op_id:
                        continue
                    
                    dst_op_meta = op_metas[dst_op_id]
                    
                    if src_op_meta.op_sig.output_buffer_name not in dst_op_meta.op_sig.input_buffer_names:
                        continue
                    
                    dst_tile_access_orders: dict[int, list[TileSignature]] = {core_id: [] for core_id in dst_op_meta.op_sig.core_group.core_ids}
                    
                    for dst_core_id, dst_thread in thread_mappings[dst_op_id].items():
                        for dst_uop_node in dst_thread.uop_nodes:
                            if dst_uop_node.is_bubble:
                                continue
                            
                            dst_tiled_op_sig = dst_op_meta.op_sig.tiled_ops[dst_uop_node.tiled_op_idx]
                            dst_i_tiles = dst_tiled_op_sig.i_tiles[dst_uop_node.uop_idx]
                            
                            for i_tile in dst_i_tiles:
                                if i_tile.buf_name == src_op_meta.op_sig.output_buffer_name:
                                    dst_tile_access_orders[dst_core_id].append(i_tile)
                
                    for src_core_id, src_thread in tqdm.tqdm(thread_mappings[srd_op_id].items(), desc=f"Analyzing dependencies from {srd_op_id} to {dst_op_id}", leave=False):
                        _prev_src_o_tiles = set()

                        for uop_node in src_thread.uop_nodes:
                            if uop_node.is_bubble:
                                continue
                            if not uop_node.output:
                                continue
                            
                            src_tiled_op_sig = src_op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                            src_o_tile = src_tiled_op_sig.o_tile
                            
                            for dst_core_id, dst_tile_access_order in dst_tile_access_orders.items():
                                _searched_indices = [i for i, tile in enumerate(dst_tile_access_order) if tile.signature == src_o_tile.signature]
                                if len(_searched_indices) == 0:
                                    continue
                                
                                _first_search = _searched_indices[0]
                                _final_search = _searched_indices[-1]
                                _tiles_required = set(dst_tile_access_order[i] for i in range(_first_search, _final_search, 1))
                                _n_tiles_cached = len(_tiles_required.intersection(_prev_src_o_tiles))
                                
                                max_shared_area_per_core[srd_op_id][src_core_id] = max(max_shared_area_per_core[srd_op_id][src_core_id], ((_n_tiles_cached + 1) * src_o_tile.tile_size))
                            
                            _prev_src_o_tiles.add(src_o_tile)

            # STAGE 2: Freeze the operator metadata with the determined thread mapping and shared tile-to-slot mapping
            is_freeze_possible = True
            
            for op_id in op_ids:
                op_meta = op_metas[op_id]
                shared_area_required = max(max_shared_area_per_core[op_id].values()) * 2  # factor of 2 for double buffering
                
                if shared_area_required > op_meta.max_opp_buffer_size:
                    logger.debug(f"Cannot freeze operator {op_id} due to insufficient shared area per pp for pipelining. Required: {shared_area_required} bytes, maximum allowed: {op_meta.max_opp_buffer_size} bytes.")
                    is_freeze_possible = False
                    
            if not is_freeze_possible:
                for op_id in op_ids:
                    op_meta = op_metas[op_id]
                    op_meta.min_opp_buffer_size = 0
                    for i_buf_name, i_buf_src in op_meta.i_buf_src.items():
                        if i_buf_src.is_tile_shared and i_buf_src.k in op_ids:
                            op_meta.i_buf_src[i_buf_name] = MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.BUFFER()  # fallback to buffer-level pipelining (i.e., no tile-level sharing)
                            src_op_meta = op_metas[i_buf_src.k]
                            src_op_meta.o_tile_sharers.remove(op_id)
                            
                return False
            else:
                for op_id in op_ids:
                    op_meta = op_metas[op_id]
                    
                    for dst_op_id in op_ids:
                        if op_id == dst_op_id:
                            continue
                        
                        dst_op_meta = op_metas[dst_op_id]
                        
                        if op_meta.op_sig.output_buffer_name not in dst_op_meta.op_sig.input_buffer_names:
                            continue
                        
                        op_meta.o_tile_sharers.add(dst_op_id)
                        dst_op_meta.i_buf_src[op_meta.op_sig.output_buffer_name] = MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.TILE_SHARED(op_id)
                    
                    op_meta.min_opp_buffer_size = max(max_shared_area_per_core[op_id].values()) * 2
                    op_meta.freeze(thread_mapping=thread_mappings[op_id])
                
                return True
            
        @property
        def is_frozen(self):
            return self._is_frozen
        
        @property
        def bcast_fifo_depth(self):
            if self.bcast_buffer_size == 0:
                return 0
            return self.bcast_buffer_size // self.bcast_fifo_slot_size
        
        @property
        def opp_fifo_depth(self):
            if self.opp_fifo_slot_size == 0:
                return 0
            return self.min_opp_buffer_size // self.opp_fifo_slot_size
    
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
                if not MCA_OperatorGraphCompiler.OperatorMetadata.check_dependency_and_freeze(self, list(op_ids)):
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
        def __init__(
            self,
            op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata',
            recipe: 'MCA_OperatorGraphCompiler.CompileRecipe',
        ):
            op_sig = op_meta.op_sig
            core_group = op_sig.core_group
            device = recipe.device

            l1_space = device.create_l1_mem_space(op_meta.spad_space_size_per_core, core_group.core_ids)
            self.spad_space_size_per_pp = op_meta.spad_space_size_per_pp
            
            # L1 Memory Layout
            #     
            #                                                    (boundary direction)                                 (boundary direction)
            #     (bcast_offset)     (opp_offset)           (pp_offset)   -->                                    (pp_offset)   -->         
            #     |--- BCAST FIFO ---|------ OPP FIFO ------|-- ST Area -->|<------------- LD Area --------------|-- ST Area -->|<------------- LD Area --------------|
            #                                               |---------------- Ping-Pong Buffer 0 ----------------|---------------- Ping-Pong Buffer 1 ----------------|
            
            self.bcast_offsets = {core_id: l1_space.allocate(core_id, op_meta.bcast_buffer_size) for core_id in core_group.core_ids}
            self.opp_offsets   = {core_id: l1_space.allocate(core_id, op_meta.min_opp_buffer_size) for core_id in core_group.core_ids}
            self.pp_offsets    = {core_id: [l1_space.allocate(core_id, op_meta.spad_space_size_per_pp), l1_space.allocate(core_id, op_meta.spad_space_size_per_pp)] for core_id in core_group.core_ids}
            self.pp_flags      = {core_id: 0 for core_id in core_group.core_ids}
            
            self.tile_to_bcast_fifo_slot: dict[TileSignature, tuple[int, int]] = {}  # {tile_signature: (core_id, slot_index)} (for broadcast optimization)
            self.tile_to_opp_fifo_slot:   dict[TileSignature, tuple[int, int]] = {}  # {tile_signature: (core_id, slot_index)} (for pipelining optimization)
            self.bcast_fifo_tile_cnt: dict[int, int] = {core_id: 0 for core_id in core_group.core_ids}  # {core_id: number of tiles in the broadcast FIFO}
            self.opp_fifo_tile_cnt: dict[int, int] = {core_id: 0 for core_id in core_group.core_ids}  # {core_id: number of tiles in the OPP FIFO}
            
            self.cached_ld_tiles: dict[int, dict[int, dict[TileSignature, Pointer]]] = {core_id: {pp_idx: {} for pp_idx in [0, 1]} for core_id in core_group.core_ids}  # {core_id: {pp_idx: {tile_signature: pointer}}}
            self.cached_st_tiles: dict[int, dict[int, dict[TileSignature, Pointer]]] = {core_id: {pp_idx: {} for pp_idx in [0, 1]} for core_id in core_group.core_ids}  # {core_id: {pp_idx: {tile_signature: pointer}}}
            
            self.st_ld_boundaries: dict[int, dict[int, Pointer]] = {
                core_id: {
                    pp_idx: self.pp_offsets[core_id][pp_idx] + max(op_meta.min_st_area_per_pp, op_meta.min_opp_buffer_size)
                    for pp_idx in [0, 1]
                } 
            for core_id in core_group.core_ids}  # {core_id: {pp_idx: boundary_pointer}}
            
            self.ld_cursors: dict[int, dict[int, Pointer]] = {
                core_id: {
                    pp_idx: self.pp_offsets[core_id][pp_idx] + op_meta.spad_space_size_per_pp 
                    for pp_idx in [0, 1]
                } 
            for core_id in core_group.core_ids}  # {core_id: {pp_idx: current_offset}}
            
            l1_space.remove()
            
        def bcast_add_tile(self, tile: TileSignature, core_id: int) -> int:
            self.tile_to_bcast_fifo_slot[tile] = (core_id, self.bcast_fifo_tile_cnt[core_id])
            self.bcast_fifo_tile_cnt[core_id] += 1
            return self.tile_to_bcast_fifo_slot[tile][1]
        
        def bcast_get_tile_slot(self, tile: TileSignature) -> tuple[int, int] | None:
            if tile not in self.tile_to_bcast_fifo_slot:
                return None
            return self.tile_to_bcast_fifo_slot[tile]
        
        def opp_add_tile(self, tile: TileSignature, core_id: int) -> int:
            self.tile_to_opp_fifo_slot[tile] = (core_id, self.opp_fifo_tile_cnt[core_id])
            self.opp_fifo_tile_cnt[core_id] += 1
            return self.tile_to_opp_fifo_slot[tile][1]
        
        def opp_get_tile_slot(self, tile: TileSignature) -> tuple[int, int] | None:
            if tile not in self.tile_to_opp_fifo_slot:
                return None
            return self.tile_to_opp_fifo_slot[tile]

        def _get_st_tiles_to_be_stored_in_fragmented_st_area(self, core_id: int, new_st_tiles: list[TileSignature]) -> dict[TileSignature, Pointer]:
            pp_idx = self.pp_flags[core_id]
            overlapped_tiles = {}
            
            occupied_spaces: list[tuple[Pointer, int]] = []  # list of (start_pointer, size) tuples for occupied spaces in the current ST area (including both shared and non-shared tiles)
            for tile, pointer in self.cached_st_tiles[core_id][pp_idx].items():
                occupied_spaces.append((pointer, tile.tile_size))
                
            occupied_spaces.sort(key=lambda x: x[0].addr)  # sort by start pointer
            
            merged_occupied_spaces: list[tuple[Pointer, int]] = []  # list of (start_pointer, size) tuples for merged occupied spaces (after merging overlapping or contiguous spaces)
            current_start, current_end = None, None
            for start, size in occupied_spaces:
                if current_start is None:
                    current_start, current_end = start, Pointer(start.addr + size)
                elif start.addr <= current_end.addr:  # overlapping or contiguous
                    current_end = Pointer(max(current_end.addr, start.addr + size))
                else:
                    merged_occupied_spaces.append((current_start, current_end.addr - current_start.addr))
                    current_start, current_end = start, Pointer(start.addr + size)
            if current_start is not None:
                merged_occupied_spaces.append((current_start, current_end.addr - current_start.addr))
            
            # Find available spaces in the ST area by identifying gaps between merged_occupied_spaces
            st_area_start = self.pp_offsets[core_id][pp_idx]
            st_area_end = self.st_ld_boundaries[core_id][pp_idx]
            
            available_spaces: list[tuple[Pointer, int]] = []  # list of (start_pointer, size) tuples for available spaces
            
            if len(merged_occupied_spaces) == 0:
                # No occupied spaces, entire ST area is available
                available_spaces.append((st_area_start, st_area_end.addr - st_area_start.addr))
            else:
                # First gap: from st_area_start to first occupied space
                first_occupied_start, _ = merged_occupied_spaces[0]
                if st_area_start.addr < first_occupied_start.addr:
                    available_spaces.append((st_area_start, first_occupied_start.addr - st_area_start.addr))
                
                # Middle gaps: between occupied spaces
                for i in range(len(merged_occupied_spaces) - 1):
                    current_occupied_start, current_occupied_size = merged_occupied_spaces[i]
                    current_occupied_end = current_occupied_start.addr + current_occupied_size
                    next_occupied_start, _ = merged_occupied_spaces[i + 1]
                    
                    if current_occupied_end < next_occupied_start.addr:
                        available_spaces.append((Pointer(current_occupied_end), next_occupied_start.addr - current_occupied_end))
                
                # Last gap: from last occupied space to st_area_end
                last_occupied_start, last_occupied_size = merged_occupied_spaces[-1]
                last_occupied_end = last_occupied_start.addr + last_occupied_size
                if last_occupied_end < st_area_end.addr:
                    available_spaces.append((Pointer(last_occupied_end), st_area_end.addr - last_occupied_end))
            
            # Allocate tiles to available spaces using first-fit strategy
            available_space_idx = 0
            for tile in new_st_tiles:
                # Try to find an available space for this tile
                while available_space_idx < len(available_spaces):
                    space_start, space_size = available_spaces[available_space_idx]
                    
                    if tile.tile_size <= space_size:
                        # Allocate this tile to this space
                        overlapped_tiles[tile] = space_start
                        
                        # Update available space
                        new_space_start = Pointer(space_start.addr + tile.tile_size)
                        new_space_size = space_size - tile.tile_size
                        available_spaces[available_space_idx] = (new_space_start, new_space_size)
                        
                        break
                    else:
                        # This space cannot fit the tile, move to next space
                        available_space_idx += 1
                   
            return overlapped_tiles
            
        def pp_check_space_available(self, core_id: int, new_ld_tiles: list[TileSignature], new_st_tiles: list[TileSignature]=None) -> bool:
            if new_st_tiles is None:
                new_st_tiles = []
                
            pp_idx = self.pp_flags[core_id]
            
            overlapped_st_tiles = self._get_st_tiles_to_be_stored_in_fragmented_st_area(core_id, new_st_tiles)
            new_st_area = sum(tile.tile_size for tile in new_st_tiles if tile not in overlapped_st_tiles)
            new_ld_area = sum(tile.tile_size for tile in new_ld_tiles)
            
            if self.ld_cursors[core_id][pp_idx].addr - new_ld_area < self.st_ld_boundaries[core_id][pp_idx].addr + new_st_area:
                return False
            return True
        
        def pp_allocate_new_tiles(self, core_id: int, new_ld_tiles: list[TileSignature], new_st_tiles: list[TileSignature]=None) -> dict[TileSignature, Pointer]:
            if new_st_tiles is None:
                new_st_tiles = []
            
            pp_idx = self.pp_flags[core_id]
            
            overlapped_st_tiles = self._get_st_tiles_to_be_stored_in_fragmented_st_area(core_id, new_st_tiles)
            
            for tile in new_st_tiles:
                for i in [0, 1]:
                    if tile in self.cached_st_tiles[core_id][i]:
                        raise Exception(f"Trying to allocate tile {tile.signature} for ST in core {core_id} which is already cached. This should never happen since the caller should have already checked the cache before calling this method.")
                    
                if tile in overlapped_st_tiles:
                    new_pointer = overlapped_st_tiles[tile]
                else:
                    new_pointer = self.st_ld_boundaries[core_id][pp_idx]
                    self.st_ld_boundaries[core_id][pp_idx] = self.st_ld_boundaries[core_id][pp_idx] + tile.tile_size
                    
                self.cached_st_tiles[core_id][pp_idx][tile] = new_pointer
            
            _cached_ld_ptrs = {}
            
            for tile in new_ld_tiles:
                if tile in self.cached_ld_tiles[core_id][pp_idx]:
                    shared_ptr = self.cached_ld_tiles[core_id][pp_idx][tile]
                    if shared_ptr < self.ld_cursors[core_id][pp_idx]:  
                        raise Exception("Debug")
                    if shared_ptr < self.st_ld_boundaries[core_id][pp_idx]:
                        raise Exception("Debug")
                    else:
                        _cached_ld_ptrs[tile] = self.cached_ld_tiles[core_id][pp_idx][tile]
                        continue
                
                new_pointer = self.ld_cursors[core_id][pp_idx] - tile.tile_size
                self.cached_ld_tiles[core_id][pp_idx][tile] = new_pointer
                self.ld_cursors[core_id][pp_idx] = new_pointer
                
            if self.ld_cursors[core_id][pp_idx].addr < self.st_ld_boundaries[core_id][pp_idx].addr:
                return None    # this implies that there is not enough space for new LD tiles
                
            return _cached_ld_ptrs
        
        def pp_clear(self, core_id: int):
            self.pp_flags[core_id] = 1 - self.pp_flags[core_id]  # switch ping-pong buffer
            
            pp_idx = self.pp_flags[core_id]
            self.ld_cursors[core_id][pp_idx] = self.pp_offsets[core_id][pp_idx] + self.spad_space_size_per_pp  # reset LD cursor to the end of the LD area
            self.cached_ld_tiles[core_id][pp_idx] = {}  # clear cached LD tiles for the new ping-pong buffer
            self.cached_st_tiles[core_id][pp_idx] = {}  # clear cached ST tiles for the new ping-pong buffer
                    
        def pp_get_cached_ld_tile_ptr(self, core_id: int, tile: TileSignature) -> Pointer:
            pp_idx = self.pp_flags[core_id]
            if tile not in self.cached_ld_tiles[core_id][pp_idx]:
                return None
            return self.cached_ld_tiles[core_id][pp_idx][tile]
        
        def pp_get_cached_st_tile_ptr(self, core_id: int, tile: TileSignature) -> Pointer:
            pp_idx = self.pp_flags[core_id]
            if tile not in self.cached_st_tiles[core_id][pp_idx]:
                return None
            return self.cached_st_tiles[core_id][pp_idx][tile]

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
    def topological_sort_grouped_target_ops(env: 'MCA_OperatorGraphCompiler.Environment', op_ids: set[str]) -> list[str]:
        graph = {
            op_id: [
                dep for dep in env.op_meta[op_id].o_tile_sharers
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
        
    def compile_grouped_target_ops(self, env: 'MCA_OperatorGraphCompiler.Environment', op_ids: set[str]) -> dict[str, MCA_CompiledOperator]:
        op_ids: list[str] = self.topological_sort_grouped_target_ops(env, op_ids)
        
        op_metas = {target_op_id: env.op_meta[target_op_id] for target_op_id in op_ids}
        
        compiled_ops   = {op_id: MCA_CompiledOperator(env, op_meta) for op_id, op_meta in op_metas.items()}
        mem_states     = {op_id: MCA_OperatorGraphCompiler.MemoryState(op_meta, env.recipe) for op_id, op_meta in op_metas.items()} 
        
        thread_mappings = {op_id: MCA_OperatorGraphCompiler.Thread.from_op_sig(op_meta.op_sig) for op_id, op_meta in op_metas.items()}
        
        tile_ref_counts: dict[str, dict[TileSignature, int]] = {
            op_id: {
                tile: 0
                for tile in op_metas[op_id].op_sig.tiles[op_metas[op_id].op_sig.output_buffer_name].values()
            }
            for op_id in op_ids
        }
        
        tile_producers: dict[str, dict[TileSignature, int]] = {
            op_id: {}   # {tile_signature: core_id}
            for op_id in op_ids
        }
        
        opp_fifo_buffers: dict[str, dict[int, FIFOBufferHandle]] = {
            op_id: {
                core_id: env.add_fifo_buffer(f"opp_fifo_{op_id}_{core_id}", op_metas[op_id].opp_fifo_depth, op_metas[op_id].opp_fifo_slot_size, ptr=mem_states[op_id].opp_offsets[core_id])
                for core_id in op_metas[op_id].op_sig.core_group.core_ids
            }
            for op_id in op_ids
        }
        
        # STAGE 1: Analyze dependencies and determine tile-level sharing relationships (for pipelining)
        for src_op_id in op_ids:
            src_meta = op_metas[src_op_id]
            
            for src_core_id, src_thread in thread_mappings[src_op_id].items():
                for uop_node_idx, uop_node in enumerate(src_thread.uop_nodes):
                    if uop_node.is_bubble:
                        continue
                    
                    tiled_op_sig = src_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                    src_o_tile = tiled_op_sig.o_tile
                    
                    tile_producers[src_op_id][src_o_tile] = src_core_id
        
        # STAGE 2: Create compiled ops and update memory states while iteratively resolving tile-level dependencies
        for op_id in op_ids:
            op_meta = op_metas[op_id]
            thread_mapping = thread_mappings[op_id]
            mem_state = mem_states[op_id]
            
            for core_id, thread in thread_mapping.items():
                group_max_cnt = env.recipe.pipeline_granularity
                group_uop_cnt = 0
                
                for uop_node in thread.uop_nodes:    
                    if uop_node.is_bubble:
                        continue
                    
                    tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                    i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                    o_tile = tiled_op_sig.o_tile
                    
                    if not mem_state.pp_check_space_available(core_id, i_tiles, [o_tile] if uop_node.output else None):
                        mem_state.pp_clear(core_id)
                        compiled_ops[op_id].new_stage(core_id)
                        group_uop_cnt = 0
                    
                    _cached_ld_ptrs = mem_state.pp_allocate_new_tiles(core_id, i_tiles, [o_tile] if uop_node.output else None)
                    
                    if _cached_ld_ptrs is None: 
                        return None  # OOM situation (not enough space in the ping-pong buffer for new LD tiles even after clearing)
                    
                    for i_tile in i_tiles:
                        if i_tile not in _cached_ld_ptrs:
                            if op_meta.i_buf_src[i_tile.buf_name].is_buffer:
                                compiled_ops[op_id].add_load_ir(core_id, MCA_CompiledOperator.IR.MEM_LOAD_TILE(
                                    i_tile, mem_states[op_id].pp_get_cached_ld_tile_ptr(core_id, i_tile),
                                ))
                            else:
                                opp_src_op_id = op_meta.i_buf_src[i_tile.buf_name].k
                                opp_src_info = mem_states[opp_src_op_id].opp_get_tile_slot(i_tile)
                                
                                if opp_src_info is None:
                                    raise Exception(f"Tile {i_tile.signature} for operator {op_id} in core {core_id} is not cached in any valid slot.")
                                    
                                opp_src_core_id, opp_src_slot_idx = opp_src_info
                                tile_ref_counts[opp_src_op_id][i_tile] += 1
                                
                                compiled_ops[op_id].add_load_ir(core_id, MCA_CompiledOperator.IR.MEM_LOAD_FROM_FIFO(
                                    i_tile, mem_state.pp_get_cached_ld_tile_ptr(core_id, i_tile), opp_fifo_buffers[opp_src_op_id][opp_src_core_id], opp_src_slot_idx
                                ))
                            
                    compiled_ops[op_id].add_execute_ir(core_id, MCA_CompiledOperator.IR.EXE_UOP(
                        op_id, uop_node.tiled_op_idx, uop_node.uop_idx, 
                        i_tile_ptrs=[mem_states[op_id].pp_get_cached_ld_tile_ptr(core_id, i_tile) for i_tile in i_tiles],
                        o_tile_ptr=mem_states[op_id].pp_get_cached_st_tile_ptr(core_id, o_tile),
                        o_tile_sig=o_tile
                    ))
                    
                    if uop_node.output:
                        if op_meta.o_tile_store:
                            compiled_ops[op_id].add_store_ir(core_id, MCA_CompiledOperator.IR.MEM_STORE_TILE(
                                o_tile, mem_states[op_id].pp_get_cached_st_tile_ptr(core_id, o_tile)
                            ))

                        opp_slot_id = mem_state.opp_add_tile(o_tile, core_id)            
                        compiled_ops[op_id].add_store_ir(core_id, MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO(
                            o_tile, mem_state.pp_get_cached_st_tile_ptr(core_id, o_tile), opp_fifo_buffers[op_id][core_id], opp_slot_id, 0
                        ))
                        
                    group_uop_cnt += 1
                    if group_uop_cnt >= group_max_cnt:
                        group_uop_cnt = 0
                        compiled_ops[op_id].new_group(core_id)
                
        # STAGE 3: Initialize FIFO reference count
        for op_id in op_ids:
            compiled_op = compiled_ops[op_id]
            
            for core_id, stages in compiled_op._mappings.items():
                for stage in stages:
                    for group in stage.groups:
                        for cmd_idx, cmd in enumerate(group.stores):
                            if isinstance(cmd, MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO):
                                o_tile = cmd.tile_sig
                                ref_count = tile_ref_counts[op_id][o_tile]
                                
                                if ref_count > 0:
                                    cmd.ref_count = ref_count
                                else:
                                    group.stores[cmd_idx] = MCA_CompiledOperator.IR.NOP()
                            
        # STAGE 4: Apply broadcasting optimization
        if env.recipe.broadcast_optimize:
            bcast_fifo_buffers: dict[str, dict[int, FIFOBufferHandle]] = {
                op_id: {
                    core_id: env.add_fifo_buffer(f"bcast_fifo_{op_id}_{core_id}", op_metas[op_id].bcast_fifo_depth, op_metas[op_id].bcast_fifo_slot_size, ptr=mem_states[op_id].bcast_offsets[core_id])
                    for core_id in op_metas[op_id].op_sig.core_group.core_ids
                }
                for op_id in op_ids
            }
            
            for op_id, compiled_op in compiled_ops.items():
                op_meta = op_metas[op_id]
                
                stage_cursor_limit = max(len(stages) for stages in compiled_op._mappings.values())
                
                for stage_cursor in range(stage_cursor_limit):
                    bcast_mem_load_cmds: dict[tuple[TileSignature, int], list[tuple[int, int, int]]] = {}
                    
                    for core_id, stages in compiled_op._mappings.items():
                        if stage_cursor >= len(stages):
                            continue
                        
                        current_stage = stages[stage_cursor]
                        
                        for group_idx, group in enumerate(current_stage.groups):
                            for cmd_idx, cmd in enumerate(group.loads):
                                if isinstance(cmd, MCA_CompiledOperator.IR.MEM_LOAD_TILE):
                                    key = (cmd.tile_sig, group_idx)
                                    if key not in bcast_mem_load_cmds:
                                        bcast_mem_load_cmds[key] = []
                                    bcast_mem_load_cmds[key].append((core_id, cmd_idx))
                    
                    # _tmp_bcast_request_traffic: dict[int, int] = {core_id: 0 for core_id in op_meta.op_sig.core_group.core_ids}
                    _tmp_bcast_request_fifocnt: dict[int, int] = {core_id: 0 for core_id in op_meta.op_sig.core_group.core_ids}
                    
                    if len(bcast_mem_load_cmds) == 0:
                        continue
                    
                    for (tile_sig, group_idx), cmd_locs in bcast_mem_load_cmds.items():
                        if len(cmd_locs) <= 1:
                            continue
                        
                        bcast_core_id, bcast_cmd_idx = min(cmd_locs, key=lambda x: _tmp_bcast_request_fifocnt[x[0]])
                        # _tmp_bcast_request_traffic[bcast_core_id] += env.buffers[tile_sig.buf_name].tile_size
                        if _tmp_bcast_request_fifocnt[bcast_core_id] >= env.recipe.pipeline_granularity:
                            break
                        _tmp_bcast_request_fifocnt[bcast_core_id] += 1
                        
                        bcast_stage = compiled_op._mappings[bcast_core_id][stage_cursor]
                        bcast_cmd: MCA_CompiledOperator.IR.MEM_LOAD_TILE = bcast_stage.groups[group_idx].loads[bcast_cmd_idx]
                        bcast_ref_count = len(cmd_locs) - 1
                        
                        bcast_tile_slot = mem_states[op_id].bcast_add_tile(tile_sig, bcast_core_id)
                        
                        bcast_stage.groups[group_idx].add_load_ir(MCA_CompiledOperator.IR.MEM_STORE_TO_FIFO(
                            tile_sig, bcast_cmd.ptr, bcast_fifo_buffers[op_id][bcast_core_id], bcast_tile_slot, bcast_ref_count
                        ))
                        
                        for core_id, cmd_idx in cmd_locs:
                            if core_id == bcast_core_id:
                                continue
                            
                            consum_stage = compiled_op._mappings[core_id][stage_cursor]
                            consum_cmd: MCA_CompiledOperator.IR.MEM_LOAD_TILE = consum_stage.groups[group_idx].loads[cmd_idx]
                            
                            consum_stage.groups[group_idx].loads[cmd_idx] = MCA_CompiledOperator.IR.NOP()
                            consum_stage.groups[group_idx].add_load_ir(MCA_CompiledOperator.IR.MEM_LOAD_FROM_FIFO(
                                tile_sig, consum_cmd.ptr, bcast_fifo_buffers[op_id][bcast_core_id], bcast_tile_slot
                            ))
                
        return compiled_ops
        
    def compile(self, recipe: 'MCA_OperatorGraphCompiler.CompileRecipe') -> MCA_CompiledProgram:    
        # Initialize environment
        env = MCA_OperatorGraphCompiler.Environment(recipe)
            
        for op_id in self._op_order:
            op_sig = self._op_sigs[op_id]
            op_sig.reorder_tiled_ops()  # reorder tiled ops in ROW/COLUMN major order (it will be predetermined by the global enviroment)
            env.add_op_sig(op_sig)
            
        env.freeze()  # freeze the environment to create L1 memory space with pipeline pattern
        
        compiled_ops: dict[str, MCA_CompiledOperator] = {}
        
        for grouped_op_ids in env.grouped_compile_targets:
            compiled_ops.update(self.compile_grouped_target_ops(env, grouped_op_ids))
        
        return MCA_CompiledProgram(recipe.device, compiled_ops)
