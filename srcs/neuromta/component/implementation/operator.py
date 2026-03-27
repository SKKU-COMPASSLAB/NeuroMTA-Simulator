import abc
import enum
import functools
import math
import time
import tqdm
from typing import Any, Sequence, Dict, List, Callable

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
        # mem_thread_template: Callable,
        ld_thread_template: Callable,
        ex_thread_template: Callable,
        st_thread_template: Callable,
        op_ex_kernels: list[Callable]
    ):
        self._op_type = op_type
        self.op_id = op_type    # will be initialized by MCA_OperatorGraphCompiler (initially set to op_type) 
        
        # self.op_template = op_template
        # self.mem_thread_template = mem_thread_template
        # self.exe_thread_template = exe_thread_template
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

    
class MCA_CompiledOperator:
    class Command:
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
        
        class MEM_LOAD_TILE(Base):
            def __init__(self, tile_sig: TileSignature, ptrs: list[Pointer]):
                self.tile_sig = tile_sig
                self.ptrs = ptrs
                
                if isinstance(self.ptrs, Pointer):
                    self.ptrs = [self.ptrs]
                if isinstance(self.ptrs, int):
                    self.ptrs = [Pointer(addr=self.ptrs)]
                
                for i in range(len(self.ptrs)):
                    if isinstance(self.ptrs[i], int):
                        self.ptrs[i] = Pointer(addr=self.ptrs[i])
                    
            def signature(self):
                ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.ptrs])
                return f"MEM_LOAD_TILE {self.tile_sig.signature} -> [{ptrs_str}]"
                
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
            
        class MEM_CPY_TILE(Base):
            def __init__(self, tile_sig: TileSignature, src_ptr: Pointer, dst_ptrs: list[Pointer]):
                self.tile_sig = tile_sig
                self.src_ptr = src_ptr
                self.dst_ptrs = dst_ptrs
                
                if isinstance(self.src_ptr, int):
                    self.src_ptr = Pointer(addr=self.src_ptr)
                if isinstance(self.dst_ptrs, Pointer):
                    self.dst_ptrs = [self.dst_ptrs]
                if isinstance(self.dst_ptrs, int):
                    self.dst_ptrs = [Pointer(addr=self.dst_ptrs)]
                for i in range(len(self.dst_ptrs)):
                    if isinstance(self.dst_ptrs[i], int):
                        self.dst_ptrs[i] = Pointer(addr=self.dst_ptrs[i])
                    
            def signature(self):
                dst_ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.dst_ptrs])
                return f"MEM_CPY_TILE {self.tile_sig.signature} SPM@{self.src_ptr.addr} -> [{dst_ptrs_str}]"
                
        class EXE_LOAD_CONTEXT(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer):
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                    
            def signature(self):
                return f"EXE_LOAD_CONTEXT SPM@{self.ptr.addr} -> {self.tile_sig.signature}"

        class EXE_STORE_CONTEXT(Base):
            def __init__(self, tile_sig: TileSignature, ptr: Pointer):
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
        
            def signature(self):
                return f"EXE_STORE_CONTEXT {self.tile_sig.signature} -> SPM@{self.ptr.addr}"

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
                
        class BARRIER(Base):
            def __init__(self, var_arrived_count: str, var_block_state: str, total_arrivals: int):
                self.var_arrived_count = var_arrived_count
                self.var_block_state = var_block_state
                self.total_arrivals = total_arrivals
                
                if isinstance(self.var_arrived_count, VariableHandle):
                    self.var_arrived_count = self.var_arrived_count.handle_name
                if isinstance(self.var_block_state, VariableHandle):
                    self.var_block_state = self.var_block_state.handle_name
                
            def signature(self):
                return f"BARRIER arrived_count={self.var_arrived_count} block_state={self.var_block_state} total_arrivals={self.total_arrivals}"
            
        class VAR_INIT(Base):
            def __init__(self, var_name: str, initial_value: int=0):
                self.var_name = var_name
                self.initial_value = initial_value
                
                if isinstance(self.var_name, VariableHandle):
                    self.var_name = self.var_name.handle_name
                    
            def signature(self):
                return f"VAR_INIT {self.var_name}={self.initial_value}"
            
        class VAR_COMPARE_AND_SWAP(Base):
            def __init__(self, var_name: str, expected_value: int, new_value: int):
                self.var_name = var_name
                self.expected_value = expected_value
                self.new_value = new_value
                
                if isinstance(self.var_name, VariableHandle):
                    self.var_name = self.var_name.handle_name
                    
            def signature(self):
                return f"VAR_COMPARE_AND_SWAP {self.var_name} expected={self.expected_value} new={self.new_value}"
            
        class VAR_CONDITIONAL_WAIT(Base):
            def __init__(self, var_names: list[str], condition: Callable[[int], bool]):
                super().__init__()
            
                self.var_names = var_names
                self.condition = condition

                for i in range(len(self.var_names)):
                    if isinstance(self.var_names[i], VariableHandle):
                        self.var_names[i] = self.var_names[i].handle_name
                    
            def signature(self):
                return f"VAR_CONDITIONAL_WAIT condition={self.condition.__name__} vars={self.var_names}"
            
            @staticmethod
            def _GE(threshold, value):
                return value >= threshold
            
            @classmethod
            def greater_equal(cls, threshold: int):
                func = functools.partial(cls._GE, threshold)
                func.__name__ = f"GE({threshold})"
                return func

    class Stage:
        def __init__(self):
            # STAGE 1: Preprocessing & Memory Load
            #   - Preprocessing commands cannot be executed simultaneously
            #   - Memory load commands can be executed simultaneously
            #   - However, preprocessing commands should always be executed before memory load commands
            #
            # STAGE 2: Execute
            #   - All execute commands are executed sequentially
            #
            # STAGE 3: Memory Store & Postprocessing
            #   - Memory store commands can be executed simultaneously
            #   - Postprocessing commands cannot be executed simultaneously
            #   - However, memory store commands should always be executed before postprocessing commands

            self.preprocessing_commands:    list[MCA_CompiledOperator.Command.Base] = []
            self.mem_load_commands:         list[MCA_CompiledOperator.Command.Base] = []
            self.execute_commands:          list[MCA_CompiledOperator.Command.Base] = []
            self.mem_store_commands:        list[MCA_CompiledOperator.Command.Base] = []
            self.postprocessing_commands:   list[MCA_CompiledOperator.Command.Base] = []
        
        @property
        def is_bubble(self) -> bool:
            return len(self.preprocessing_commands) == 0 and len(self.mem_load_commands) == 0 and len(self.execute_commands) == 0 and len(self.mem_store_commands) == 0 and len(self.postprocessing_commands) == 0
            
    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_meta: 'MCA_OperatorGraphCompiler.OperatorMetadata'):
        self._env = env
        # self._mem_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.mem_thread_template
        # self._exe_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.exe_thread_template
        self._ld_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.ld_thread_template
        self._ex_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.ex_thread_template
        self._st_thread_template: Callable[..., KernelPrototype] = op_meta.op_sig.st_thread_template
        self._op_ex_kernels = op_meta.op_sig.op_ex_kernels
        self._mappings: dict[int, list[MCA_CompiledOperator.Stage]] = {core_id: [] for core_id in op_meta.op_sig.core_group.core_ids}  # {core_id: [stage1, stage2, ...]}
        
        self._stage_barriers = {
            core_id: (
                env.add_variable(f"op_{op_meta.op_sig.op_id}_core_{core_id}_stage_sync_barrier_arrived_count", initial_value=0).handle_name,
                env.add_variable(f"op_{op_meta.op_sig.op_id}_core_{core_id}_stage_sync_barrier_block_state", initial_value=0).handle_name,
                3,
            )
            for core_id in op_meta.op_sig.core_group.core_ids
        }
        
        self._global_sync_barrier = (
            env.add_variable(f"op_{op_meta.op_sig.op_id}_global_sync_barrier_arrived_count", initial_value=0).handle_name,
            env.add_variable(f"op_{op_meta.op_sig.op_id}_global_sync_barrier_block_state", initial_value=0).handle_name,
            len(op_meta.op_sig.core_group.core_ids) * 3,
        )
    
    def add_stage(self, core_id: int, stage: 'MCA_CompiledOperator.Stage'):
        self._mappings[core_id].append(stage)
        
    def dispatch(self, device: MCA_DeviceBase):
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            n_stages = len(self.mappings[core_id])
            
            for i in range(n_stages + 2):
                presync_b = self._global_sync_barrier if i == 0 else None
                postsync_b = self._global_sync_barrier if i == (n_stages + 1) else None
                stage_b = self._stage_barriers[core_id]
                
                ld_preprocessing_commands = []
                ld_mem_load_commands = []
                ex_execute_commands = []
                st_mem_store_commands = []
                st_postprocessing_commands = []    
                
                if 0 <= i < n_stages:
                    ld_preprocessing_commands = self.mappings[core_id][i].preprocessing_commands
                    ld_mem_load_commands = self.mappings[core_id][i].mem_load_commands
                if 1 <= i < n_stages + 1:
                    ex_execute_commands = self.mappings[core_id][i-1].execute_commands
                if 2 <= i < n_stages + 2:
                    st_mem_store_commands = self.mappings[core_id][i-2].mem_store_commands
                    st_postprocessing_commands = self.mappings[core_id][i-2].postprocessing_commands
                
                ld_thread = self._ld_thread_template(core, self._env, ld_preprocessing_commands, ld_mem_load_commands, stage_b, presync_b, postsync_b)
                ex_thread = self._ex_thread_template(core, self._env, ex_execute_commands, self._op_ex_kernels, stage_b, presync_b, postsync_b)
                st_thread = self._st_thread_template(core, self._env, st_mem_store_commands, st_postprocessing_commands, stage_b, presync_b, postsync_b)
                
                ld_thread.dispatch("LD")
                ex_thread.dispatch("EX")
                st_thread.dispatch("ST")
        
    @property
    def mappings(self):
        return self._mappings
    
    def summary(self) -> dict:
        summary = {}
        for core_id, stages in self._mappings.items():
            summary[core_id] = []
            for stage in stages:
                stage_summary = {
                    "preprocessing":  [cmd.signature() for cmd in stage.preprocessing_commands  if not isinstance(cmd, MCA_CompiledOperator.Command.NOP)],
                    "mem_load":       [cmd.signature() for cmd in stage.mem_load_commands       if not isinstance(cmd, MCA_CompiledOperator.Command.NOP)],
                    "execute":        [cmd.signature() for cmd in stage.execute_commands        if not isinstance(cmd, MCA_CompiledOperator.Command.NOP)],
                    "mem_store":      [cmd.signature() for cmd in stage.mem_store_commands      if not isinstance(cmd, MCA_CompiledOperator.Command.NOP)],
                    "postprocessing": [cmd.signature() for cmd in stage.postprocessing_commands if not isinstance(cmd, MCA_CompiledOperator.Command.NOP)],
                }
                summary[core_id].append(stage_summary)
        return summary


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
            spad_space_size_per_core: int,
            broadcast_optimize: bool=True,
        ):
            self.device = device
            self.spad_space_size_per_core = spad_space_size_per_core
            self.broadcast_optimize = broadcast_optimize
            
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
            if not op_sig.is_core_group_initialized:
                raise ValueError(f"Core group must be initialized for operator {op_sig.op_id} before creating metadata.")
            
            self.op_sig = op_sig
            self.spad_space_size_per_pp = recipe.spad_space_size_per_core // 2
            
            self.i_buf_src: dict[str, MCA_OperatorGraphCompiler.OperatorMetadata.SrcType] = {
                buf_name: MCA_OperatorGraphCompiler.OperatorMetadata.SrcType.BUFFER() 
                for buf_name in op_sig.input_buffer_names
            }
            
            # Dependencies related to output buffer sharing (a.k.a tile-level pipelining)
            self.o_tile_store = op_sig.buffers[op_sig.output_buffer_name].is_allocated  # if the output buffer is allocated, the computation result should be updated to the buffer
            self.o_tile_sharers: set[str] = set()  # set of op_ids that directly consume this operator's output tiles (tile-level sharers via SHARED area)
            self.op2op_barrier_arrived_count_var: str = f"{op_sig.op_id}_op2op_barrier_arrived_count"
            self.op2op_barrier_block_state_var: str = f"{op_sig.op_id}_op2op_barrier_block_state"
                        
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
                    
            self.max_shared_area_per_core = self.spad_space_size_per_pp - (self.min_ld_area_per_pp + self.min_st_area_per_pp)
            
            # Actual LD/ST/SHARED area per core to be determined based on the dependencies with consumer operators (initially set to 0, will be updated when analyzing dependencies)
            self.min_shared_area_per_pp = 0
            self.thread_mapping: dict[int, MCA_OperatorGraphCompiler.Thread] = {}
            
            self._is_frozen = False
            
            
        def freeze(self, thread_mapping: 'dict[int, MCA_OperatorGraphCompiler.Thread]' = None):
            self.thread_mapping = MCA_OperatorGraphCompiler.Thread.from_op_sig(self.op_sig) if thread_mapping is None else thread_mapping
            
            if self.min_shared_area_per_pp + self.min_ld_area_per_pp > self.spad_space_size_per_pp:
                raise ValueError(f"Insufficient shared area per core for operator {self.op_sig.op_id}. Required: {self.min_shared_area_per_pp} bytes, maximum allowed: {self.max_shared_area_per_core} bytes.")   
            if self.min_st_area_per_pp + self.min_ld_area_per_pp > self.spad_space_size_per_pp:
                raise ValueError(f"Insufficient shared area per core for operator {self.op_sig.op_id}. Required: {self.min_st_area_per_pp} bytes, maximum allowed: {self.max_shared_area_per_core} bytes.")
            
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
                shared_area_required = max(max_shared_area_per_core[op_id].values())
                
                if shared_area_required > op_meta.max_shared_area_per_core:
                    logger.debug(f"Cannot freeze operator {op_id} due to insufficient shared area per pp for pipelining. Required: {shared_area_required} bytes, maximum allowed: {op_meta.max_shared_area_per_core} bytes.")
                    is_freeze_possible = False
                    
            if not is_freeze_possible:
                for op_id in op_ids:
                    op_meta = op_metas[op_id]
                    op_meta.min_shared_area_per_pp = 0
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
                    
                    op_meta.min_shared_area_per_pp = max(max_shared_area_per_core[op_id].values())
                    op_meta.freeze(thread_mapping=thread_mappings[op_id])
                    
                    env.add_variable(op_meta.op2op_barrier_arrived_count_var, initial_value=0)
                    env.add_variable(op_meta.op2op_barrier_block_state_var, initial_value=0)
                
                return True
            
        @property
        def is_frozen(self):
            return self._is_frozen
    
    class Environment:
        def __init__(self, recipe: 'MCA_OperatorGraphCompiler.CompileRecipe'):
            self.recipe = recipe
            
            self.op_meta:     dict[str, MCA_OperatorGraphCompiler.OperatorMetadata] = {}
            self.buffers:     dict[str, MCA_TensorBuffer]   = {}
            self.variables:   dict[str, VariableHandle]     = {}
            
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
            
        def freeze(self):
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

            self.spad_space_size_per_pp = op_meta.spad_space_size_per_pp
            l1_space = device.create_l1_mem_space(self.spad_space_size_per_pp * 2, core_group.core_ids)
            
            # L1 Memory Layout
            #     
            #            (boundary direction)                                 (boundary direction)
            #     0x000000      -->             (spad_space_size_per_pp)             -->         (2 * spad_space_size_per_pp)
            #     |-- ST Area -->|<------------- LD Area --------------|-- ST Area -->|<------------- LD Area --------------|
            #     |---------------- Ping-Pong Buffer 0 ----------------|---------------- Ping-Pong Buffer 1 ----------------|
            
            self.pp_offsets = {core_id: [l1_space.allocate(core_id, self.spad_space_size_per_pp), l1_space.allocate(core_id, self.spad_space_size_per_pp)] for core_id in core_group.core_ids}
            self.pp_flags = {core_id: 0 for core_id in core_group.core_ids}
            
            self.cached_ld_tiles: dict[int, dict[int, dict[TileSignature, Pointer]]] = {core_id: {pp_idx: {} for pp_idx in [0, 1]} for core_id in core_group.core_ids}  # {core_id: {pp_idx: {tile_signature: pointer}}}
            self.cached_st_tiles: dict[int, dict[int, dict[TileSignature, tuple[Pointer, bool]]]] = {core_id: {pp_idx: {} for pp_idx in [0, 1]} for core_id in core_group.core_ids}  # {core_id: {pp_idx: {tile_signature: (pointer, is_shared)}}}
            
            self.st_ld_boundaries: dict[int, dict[int, Pointer]] = {
                core_id: {
                    pp_idx: self.pp_offsets[core_id][pp_idx] + max(op_meta.min_st_area_per_pp, op_meta.min_shared_area_per_pp)
                    for pp_idx in [0, 1]
                } 
            for core_id in core_group.core_ids}  # {core_id: {pp_idx: boundary_pointer}}
            
            self.ld_cursors: dict[int, dict[int, Pointer]] = {
                core_id: {
                    pp_idx: self.pp_offsets[core_id][pp_idx] + self.spad_space_size_per_pp 
                    for pp_idx in [0, 1]
                } 
            for core_id in core_group.core_ids}  # {core_id: {pp_idx: current_offset}}
            
            l1_space.remove()
            
        def evict_shared_tiles(self, core_id: int, evicted_tiles: Sequence[TileSignature]):
            for tile in evicted_tiles:
                for pp_idx in [0, 1]:
                    if tile not in self.cached_st_tiles[core_id][pp_idx]:
                        continue
                    
                    _, is_shared = self.cached_st_tiles[core_id][pp_idx][tile]
                    if not is_shared:
                        raise Exception(f"Trying to evict tile {tile.signature} from core {core_id} which is not marked as shared. This should never happen since only shared tiles can be evicted.")
                    
                    del self.cached_st_tiles[core_id][pp_idx][tile]
                    break
                
        def _get_st_tiles_to_be_stored_in_fragmented_st_area(self, core_id: int, new_st_tiles: list[TileSignature]) -> dict[TileSignature, Pointer]:
            pp_idx = self.pp_flags[core_id]
            overlapped_tiles = {}
            
            occupied_spaces: list[tuple[Pointer, int]] = []  # list of (start_pointer, size) tuples for occupied spaces in the current ST area (including both shared and non-shared tiles)
            for tile, (pointer, is_shared) in self.cached_st_tiles[core_id][pp_idx].items():
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
            
        def check_space_available(self, core_id: int, new_ld_tiles: list[TileSignature], new_st_tiles: list[TileSignature], is_st_tiles_shared: bool) -> bool:
            pp_idx = self.pp_flags[core_id]
            
            overlapped_st_tiles = self._get_st_tiles_to_be_stored_in_fragmented_st_area(core_id, new_st_tiles)
            new_st_area = sum(tile.tile_size for tile in new_st_tiles if tile not in overlapped_st_tiles)
            new_ld_area = sum(tile.tile_size for tile in new_ld_tiles)
            
            if self.ld_cursors[core_id][pp_idx].addr - new_ld_area < self.st_ld_boundaries[core_id][pp_idx].addr + new_st_area:
                return False
            return True
        
        def allocate_new_tiles(self, core_id: int, new_ld_tiles: list[TileSignature], new_st_tiles: list[TileSignature], is_st_tiles_shared: bool) -> dict[TileSignature, Pointer]:
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
                    
                self.cached_st_tiles[core_id][pp_idx][tile] = (new_pointer, is_st_tiles_shared)
            
            _cached_ld_ptrs = {}
            
            for tile in new_ld_tiles:
                if tile in self.cached_ld_tiles[core_id][pp_idx]:
                    shared_ptr = self.cached_ld_tiles[core_id][pp_idx][tile]
                    if shared_ptr < self.ld_cursors[core_id][pp_idx]:  
                        raise Exception("Debug")
                    if shared_ptr < self.st_ld_boundaries[core_id][pp_idx]:  
                        # del self.cached_ld_tiles[core_id][pp_idx][tile]  # ST boundary has already passed this LD pointer; drop stale cache and reallocate below.
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
        
        def clear(self, core_id: int):
            self.pp_flags[core_id] = 1 - self.pp_flags[core_id]  # switch ping-pong buffer
            
            pp_idx = self.pp_flags[core_id]
            self.ld_cursors[core_id][pp_idx] = self.pp_offsets[core_id][pp_idx] + self.spad_space_size_per_pp  # reset LD cursor to the end of the LD area
            self.cached_ld_tiles[core_id][pp_idx] = {}  # clear cached LD tiles for the new ping-pong buffer
            self.cached_st_tiles[core_id][pp_idx] = {tile: (ptr, is_shared) for tile, (ptr, is_shared) in self.cached_st_tiles[core_id][pp_idx].items() if is_shared}
                    
        def get_cached_ld_tile_ptr(self, core_id: int, tile: TileSignature) -> Pointer:
            pp_idx = self.pp_flags[core_id]
            if tile not in self.cached_ld_tiles[core_id][pp_idx]:
                return None
            return self.cached_ld_tiles[core_id][pp_idx][tile]
        
        def get_cached_st_tile_ptr(self, core_id: int, tile: TileSignature) -> Pointer:
            pp_idx = self.pp_flags[core_id]
            if tile not in self.cached_st_tiles[core_id][pp_idx]:
                return None
            return self.cached_st_tiles[core_id][pp_idx][tile][0]
        
        def get_shared_st_tile_ptr(self, tile: TileSignature) -> Pointer:
            for _, shared_cache_pps in self.cached_st_tiles.items():
                for _, shared_cache in shared_cache_pps.items():
                    if tile not in shared_cache:
                        continue
                    
                    ptr, is_shared = shared_cache[tile]
                    if is_shared:
                        return ptr
            return None
        
        def get_shared_st_tile_evictable_condidates(self, core_id: int) -> set[TileSignature]:
            evictable_tiles = set()
            pp_idx = self.pp_flags[core_id]
            for tile, (_, is_shared) in self.cached_st_tiles[core_id][pp_idx].items():
                if is_shared:
                    evictable_tiles.add(tile)
            return evictable_tiles

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
        
    def compile_grouped_target_ops(self, env: 'MCA_OperatorGraphCompiler.Environment', op_ids: set[str]) -> dict[str, MCA_CompiledOperator]:
        op_metas = {target_op_id: env.op_meta[target_op_id] for target_op_id in op_ids}
        
        # compiled op contexts
        compiled_ops   = {op_id: MCA_CompiledOperator(env, op_meta) for op_id, op_meta in op_metas.items()}
        mem_states     = {op_id: MCA_OperatorGraphCompiler.MemoryState(op_meta, env.recipe) for op_id, op_meta in op_metas.items()} 
        
        thread_progress_vars: dict[str, dict[int, VariableHandle]] = {
            op_id: {
                core_id: env.add_variable(f"thread_progress_var_{op_id}_{core_id}") 
                for core_id in op_metas[op_id].op_sig.core_group.core_ids
            }
            for op_id in op_ids
        }
        
        current_stages = {op_id: {core_id: MCA_CompiledOperator.Stage() for core_id in op_meta.op_sig.core_group.core_ids} for op_id, op_meta in op_metas.items()}
        
        # metadata and cursors for dependency analysis and scheduling
        exe_stat_cursors = {op_id: 0 for op_id in op_ids}
        prefetch_cursors = {op_id: 0 for op_id in op_ids}
        
        thread_mappings = {op_id: MCA_OperatorGraphCompiler.Thread.from_op_sig(op_meta.op_sig) for op_id, op_meta in op_metas.items()}
        cursor_limits   = {op_id: max(t.n_uop_nodes for t in tm.values()) for op_id, tm in thread_mappings.items()}
        
        tile_dependencies: dict[str, dict[TileSignature, dict[str, dict[int, int]]]] = {
            op_id: {
                tile: {}
                for tile in op_metas[op_id].op_sig.tiles[op_metas[op_id].op_sig.output_buffer_name].values()
            }
            for op_id in op_ids
        }
        
        tile_producers: dict[str, dict[TileSignature, tuple[int, int]]] = {
            op_id: {}   # {tile_signature: (core_id, uop_node_idx)}
            for op_id in op_ids
        }
        
        def mark_compiled_op_thread_progress(op_id: str):
            exe_stat_cursor = exe_stat_cursors[op_id]
            
            for core_id, current_stage in current_stages[op_id].items():
                current_stage.execute_commands.append(MCA_CompiledOperator.Command.VAR_INIT(
                    var_name=thread_progress_vars[op_id][core_id],
                    initial_value=exe_stat_cursor + 1,
                ))
            
        for op_id in op_ids:
            for core_id, current_stage in current_stages[op_id].items():
                current_stage.preprocessing_commands.append(MCA_CompiledOperator.Command.VAR_INIT(
                    var_name=thread_progress_vars[op_id][core_id], 
                    initial_value=0,
                ))

        # STAGE 1: Analyze dependencies and determine tile-level sharing relationships (for pipelining)
        for src_op_id in op_ids:
            src_meta = op_metas[src_op_id]
            
            for src_core_id, src_thread in thread_mappings[src_op_id].items():
                for uop_node_idx, uop_node in enumerate(src_thread.uop_nodes):
                    if uop_node.is_bubble:
                        continue
                    
                    tiled_op_sig = src_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                    src_o_tile = tiled_op_sig.o_tile
                    
                    tile_producers[src_op_id][src_o_tile] = (src_core_id, uop_node_idx)

            for dst_op_id in src_meta.o_tile_sharers:
                dst_meta = op_metas[dst_op_id]
                dst_thread_mapping = thread_mappings[dst_op_id]
                
                for dst_core_id, thread in dst_thread_mapping.items():
                    for uop_node_idx, uop_node in enumerate(thread.uop_nodes):
                        if uop_node.is_bubble:
                            continue
                        
                        tiled_op_sig = dst_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        
                        for i_tile in i_tiles:
                            if i_tile in tile_dependencies[src_op_id]:
                                if dst_op_id not in tile_dependencies[src_op_id][i_tile]:
                                    tile_dependencies[src_op_id][i_tile][dst_op_id] = {dst_core_id: uop_node_idx}
                                else:
                                    if dst_core_id not in tile_dependencies[src_op_id][i_tile][dst_op_id]:
                                        tile_dependencies[src_op_id][i_tile][dst_op_id][dst_core_id] = uop_node_idx
                                    else:
                                        tile_dependencies[src_op_id][i_tile][dst_op_id][dst_core_id] = max(tile_dependencies[src_op_id][i_tile][dst_op_id][dst_core_id], uop_node_idx)
                                    
        # STAGE 2: Create compiled ops and update memory states while iteratively resolving tile-level dependencies
        while any(exe_stat_cursors[target_op_id] < cursor_limits[target_op_id] for target_op_id in op_ids):
            # STEP 1: initialize prefetch cursors for all targets
            for op_id in op_ids:
                op_meta = op_metas[op_id]
                thread_mapping = thread_mappings[op_id]    
                prefetch_cursor = prefetch_cursors[op_id]
                
                i_tiles_required: dict[int, set[TileSignature]] = {core_id: set() for core_id in thread_mapping.keys()}
                
                # advance the prefetch cursor as long as the dependencies for the current uop nodes are resolved (i.e., all source tiles are available in the shared area)
                while prefetch_cursor < cursor_limits[op_id]:
                    is_resolved = True
                    
                    _cached_i_tiles_required = {core_id: set() for core_id in thread_mapping.keys()}
                    
                    for core_id, thread in thread_mapping.items():
                        if prefetch_cursor >= thread.n_uop_nodes:                      
                            continue
                        
                        uop_node = thread.uop_nodes[prefetch_cursor]
                        if uop_node.is_bubble:
                            continue
                        
                        tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        
                        for i_tile in i_tiles:
                            if op_meta.i_buf_src[i_tile.buf_name].is_buffer:
                                continue
                            
                            src_op_id = op_meta.i_buf_src[i_tile.buf_name].k
                            
                            if mem_states[src_op_id].get_shared_st_tile_ptr(i_tile) is None:
                                is_resolved = False
                                break
                            else:
                                _cached_i_tiles_required[core_id].add(i_tile)
                    
                    if not is_resolved:
                        break
                    else:
                        for core_id, cached_i_tiles in _cached_i_tiles_required.items():
                            i_tiles_required[core_id].update(cached_i_tiles)
                    
                    prefetch_cursor += 1
                
                # if the prefetch cursor has advanced, insert barriers to ensure correct synchronization
                #   - dst operator should not start executing the prefetched uops until all source operators have finished executing their prefetched uops (to ensure data correctness)
                src_op_ids = [src_type.k for src_type in op_meta.i_buf_src.values() if src_type.is_tile_shared]
                
                if len(src_op_ids) > 0 and prefetch_cursor > prefetch_cursors[op_id]:    
                    wait_conditions: dict[int, dict[tuple[str, int], int]] = {
                        core_id: {}
                        for core_id in thread_mapping.keys()
                    }
                    
                    for core_id in thread_mapping.keys():
                        for i_tile in i_tiles_required[core_id]:
                            src_op_id = op_meta.i_buf_src[i_tile.buf_name].k
                            src_core_id, src_uop_idx = tile_producers[src_op_id][i_tile]
                            wait_conditions[core_id][(src_op_id, src_core_id)] = max(wait_conditions[core_id].get((src_op_id, src_core_id), 0), src_uop_idx)
                        
                    for core_id, current_stage in current_stages[op_id].items():
                        if len(current_stage.mem_load_commands) > 0:
                            compiled_ops[op_id].add_stage(core_id, current_stage)
                            mem_states[op_id].clear(core_id)  # clear the L1 buffer for the new stage
                            current_stages[op_id][core_id] = MCA_CompiledOperator.Stage()
                        
                        for (src_op_id, src_core_id), src_uop_idx in wait_conditions[core_id].items():
                            current_stages[op_id][core_id].mem_load_commands.append(MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT(
                                var_names=[thread_progress_vars[src_op_id][src_core_id]],
                                condition=MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT.greater_equal(src_uop_idx + 1)
                            ))
        
                # update the prefetch cursor
                prefetch_cursors[op_id] = prefetch_cursor
                
            # STEP 2: update execution status cursors and shared area occupancy
            for op_id in op_ids:
                op_meta = op_metas[op_id]
                thread_mapping = thread_mappings[op_id]
                prefetch_cursor = prefetch_cursors[op_id]

                exe_iter_cnt = 0
                
                while exe_stat_cursors[op_id] < prefetch_cursor:
                    # 1) check if the current operator execution status cursor can be advanced based on the prefetch cursor positions of its consumer operators
                    is_executable = True
                    
                    for dst_op_id in op_meta.o_tile_sharers:
                        if prefetch_cursors[dst_op_id] >= cursor_limits[dst_op_id]:
                            continue    # the destination operator can be executed without waiting for the current operator since it has already prefetched all uops
                        if prefetch_cursors[dst_op_id] > exe_stat_cursors[dst_op_id]:
                            is_executable = False   # the destination operator has not yet executed all prefetched uops, so the current operator may not necessarily be executed for optimal pipelining
                            break
                        
                    if not is_executable:
                        break
                                        
                    # 3) check if there is sufficient space in the shared area for the output tiles of the current uop nodes of the operator
                    #   - If there is enough space, do not ping-pong the L1 buffer
                    #   - Otherwise, ping-pong the L1 buffer and create new stages
                    is_pp_required = False
                     
                    for core_id, thread in thread_mapping.items():
                        if exe_stat_cursors[op_id] >= thread.n_uop_nodes:
                            continue
                        
                        uop_node = thread.uop_nodes[exe_stat_cursors[op_id]]
                        if uop_node.is_bubble:
                            continue
                        
                        tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        o_tiles = [tiled_op_sig.o_tile,]
                        is_st_tiles_shared = len(op_meta.o_tile_sharers) > 0
                        
                        if not mem_states[op_id].check_space_available(core_id, i_tiles, o_tiles, is_st_tiles_shared):
                            is_pp_required = True
                            break
                    
                    if is_pp_required:
                        # 3-1) ping-pong the buffer and create new stages         
                        for core_id, current_stage in current_stages[op_id].items():
                            mem_states[op_id].clear(core_id)
                            compiled_ops[op_id].add_stage(core_id, current_stage)
                            current_stages[op_id][core_id] = MCA_CompiledOperator.Stage()  # reset current stage after ping-pong buffer switch
                            
                        # 3-2) if ping-pong is required, eliminate shared tiles that are no longer needed
                        if len(op_meta.o_tile_sharers) > 0:
                            for core_id in op_meta.op_sig.core_group.core_ids:
                                for tile in mem_states[op_id].get_shared_st_tile_evictable_condidates(core_id):
                                    is_tile_still_needed = False
                                    
                                    for dst_op_id, dst_dept in tile_dependencies[op_id][tile].items():
                                        for dst_core_id, dep_uop_idx in dst_dept.items():
                                            if exe_stat_cursors[dst_op_id] <= dep_uop_idx:
                                                is_tile_still_needed = True
                                                break

                                    if not is_tile_still_needed:
                                        _evicted_tile = tile
                                        
                                        mem_states[op_id].evict_shared_tiles(core_id, [_evicted_tile])
                                        
                                        for dst_op_id, dst_dept in tile_dependencies[op_id][_evicted_tile].items():
                                            for dst_core_id, dep_uop_idx in dst_dept.items():
                                                current_stages[op_id][core_id].mem_load_commands.append(MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT(
                                                    var_names=[thread_progress_vars[dst_op_id][dst_core_id]],
                                                    condition=MCA_CompiledOperator.Command.VAR_CONDITIONAL_WAIT.greater_equal(dep_uop_idx+1)
                                                ))
                            
                        break   # after ping-ponging the buffer and creating new stages, re-check the executability of the current uop nodes in the next iteration since the scheduling may change after stage creation
                    
                    # 4) create commands
                    for core_id, thread in thread_mapping.items():
                        if exe_stat_cursors[op_id] >= thread.n_uop_nodes:
                            continue
                        
                        uop_node = thread.uop_nodes[exe_stat_cursors[op_id]]
                        if uop_node.is_bubble:
                            continue
                        
                        current_stage = current_stages[op_id][core_id]
                        
                        tiled_op_sig = op_meta.op_sig.tiled_ops[uop_node.tiled_op_idx]
                        i_tiles = tiled_op_sig.i_tiles[uop_node.uop_idx]
                        o_tile = tiled_op_sig.o_tile
                        is_st_tiles_shared = len(op_meta.o_tile_sharers) > 0
                        
                        if uop_node.uop_idx == (tiled_op_sig.n_uops - 1):
                            _cached_ld_ptrs = mem_states[op_id].allocate_new_tiles(core_id, i_tiles, [o_tile], is_st_tiles_shared)
                        else:
                            _cached_ld_ptrs = mem_states[op_id].allocate_new_tiles(core_id, i_tiles, [], is_st_tiles_shared)
                            
                        if _cached_ld_ptrs is None:
                            # Out-of-Memory Situation
                            #   - This implies that there is not enough space in the shared area for the new tiles, which should never happen since we have already checked 
                            #     the space availability before calling this method
                            #   - The compiler should re-schedule the collocated operators so that the current operator can be executed without tile-level pipelining 
                            return None

                        for i_tile in i_tiles:
                            if i_tile not in _cached_ld_ptrs:
                                if op_meta.i_buf_src[i_tile.buf_name].is_buffer:
                                    current_stage.mem_load_commands.append(MCA_CompiledOperator.Command.MEM_LOAD_TILE(
                                        i_tile, mem_states[op_id].get_cached_ld_tile_ptr(core_id, i_tile),
                                    ))
                                else:
                                    src_op_id = op_meta.i_buf_src[i_tile.buf_name].k
                                    shared_ptr = mem_states[src_op_id].get_shared_st_tile_ptr(i_tile)
                                    if shared_ptr is None:
                                        raise Exception(f"Trying to load tile {i_tile.signature} for operator {op_id} in core {core_id} which is not cached in the shared area. This should never happen since the current operator should have already checked the cache before calling this method.")
                                    current_stage.mem_load_commands.append(MCA_CompiledOperator.Command.MEM_CPY_TILE(
                                        i_tile, shared_ptr, mem_states[op_id].get_cached_ld_tile_ptr(core_id, i_tile),
                                    ))
                                
                        current_stage.execute_commands.append(MCA_CompiledOperator.Command.EXE_UOP(
                            op_id, uop_node.tiled_op_idx, uop_node.uop_idx, 
                            i_tile_ptrs=[mem_states[op_id].get_cached_ld_tile_ptr(core_id, i_tile) for i_tile in i_tiles],
                            o_tile_ptr=mem_states[op_id].get_cached_st_tile_ptr(core_id, o_tile),
                            o_tile_sig=o_tile
                        ))
                        
                        if uop_node.output and op_meta.o_tile_store:
                            current_stage.mem_store_commands.append(MCA_CompiledOperator.Command.MEM_STORE_TILE(
                                o_tile, mem_states[op_id].get_cached_st_tile_ptr(core_id, o_tile)
                            ))
                    
                    mark_compiled_op_thread_progress(op_id)
                    exe_stat_cursors[op_id] += 1
                    exe_iter_cnt += 1
                        
        for op_id in op_ids:
            for core_id, current_stage in current_stages[op_id].items():
                compiled_ops[op_id].add_stage(core_id, current_stage)
                
                
        # TEST: check if all load tiled within the same stage do not overlap
        for op_id, compiled_op in compiled_ops.items():
            for core_id, stages in compiled_op._mappings.items():
                for stage in stages:
                    loaded_tiles: list[tuple[Pointer, TileSignature]] = []
                    for cmd in stage.mem_load_commands:
                        if isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_TILE):
                            for ptr in cmd.ptrs:
                                loaded_tiles.append((ptr, cmd.tile_sig))
                        elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_CPY_TILE):
                            for ptr in cmd.ptrs:
                                loaded_tiles.append((ptr, cmd.tile_sig))

                    for i in range(len(loaded_tiles)):
                        for j in range(i+1, len(loaded_tiles)):
                            i_st = loaded_tiles[i][0].addr
                            i_ed = i_st + loaded_tiles[i][1].tile_size
                            j_st = loaded_tiles[j][0].addr
                            j_ed = j_st + loaded_tiles[j][1].tile_size

                            if j_st < i_ed < j_ed or i_st < j_ed < i_ed:
                                raise Exception(f"Overlapping memory accesses detected in operator {op_id} in core {core_id} within the same stage. This should never happen since the compiler should have already ensured that there is no overlapping memory access within the same stage.")
        # END TEST
           
        # STAGE 3: Apply broadcasting optimization
        if env.recipe.broadcast_optimize:
            for op_id, compiled_op in compiled_ops.items():
                op_meta = op_metas[op_id]
                
                bcast_var_arrived_count = f"local_bcast_arrived_count_{op_id}"
                bcast_var_block_state = f"local_bcast_block_state_{op_id}"
                bcast_total_arrivals = len(op_meta.op_sig.core_group.core_ids)
                
                if bcast_var_arrived_count in env.variables.keys():
                    raise Exception(f"Variable name conflict for local synchronization variables for operator {op_id}. This should never happen since the variable names are generated based on operator IDs which are unique.")
                else:
                    bcast_var_arrived_count = env.add_variable(bcast_var_arrived_count, initial_value=0)
                    bcast_var_block_state = env.add_variable(bcast_var_block_state, initial_value=0)
            
                stage_cursor_limit = max(len(stages) for stages in compiled_op._mappings.values())
                
                for stage_cursor in range(stage_cursor_limit):
                    bcast_mem_load_cmds: dict[TileSignature, list[tuple[int, int]]] = {}
                    bcast_mem_copy_cmds: dict[TileSignature, list[tuple[int, int]]] = {}
                    
                    for core_id, stages in compiled_op._mappings.items():
                        if stage_cursor >= len(stages):
                            continue
                        
                        current_stage = stages[stage_cursor]
                        
                        for cmd_idx, cmd in enumerate(current_stage.mem_load_commands):
                            if isinstance(cmd, MCA_CompiledOperator.Command.MEM_LOAD_TILE):
                                key = cmd.tile_sig
                                if key not in bcast_mem_load_cmds:
                                    bcast_mem_load_cmds[key] = []
                                bcast_mem_load_cmds[key].append((core_id, cmd_idx))
                            elif isinstance(cmd, MCA_CompiledOperator.Command.MEM_CPY_TILE):
                                key = cmd.tile_sig
                                if key not in bcast_mem_copy_cmds:
                                    bcast_mem_copy_cmds[key] = []
                                bcast_mem_copy_cmds[key].append((core_id, cmd_idx))
                    
                    _tmp_bcast_cnt = 0
                    _tmp_bcast_request_traffic: dict[int, int] = {core_id: 0 for core_id in op_meta.op_sig.core_group.core_ids}
                    
                    for tile_sig, cmd_locs in bcast_mem_load_cmds.items():
                        if len(cmd_locs) <= 1:
                            continue
                        
                        _tmp_bcast_cnt += 1
                        
                        bcast_core_id, bcast_cmd_idx = min(cmd_locs, key=lambda x: _tmp_bcast_request_traffic[x[0]])
                        _tmp_bcast_request_traffic[bcast_core_id] += env.buffers[tile_sig.buf_name].tile_size
                        
                        for core_id, cmd_idx in cmd_locs:
                            if core_id == bcast_core_id and cmd_idx == bcast_cmd_idx:
                                continue
                            
                            consum_stage = compiled_op._mappings[core_id][stage_cursor]
                            bcast_stage = compiled_op._mappings[bcast_core_id][stage_cursor]
                            
                            consum_cmd: MCA_CompiledOperator.Command.MEM_LOAD_TILE = consum_stage.mem_load_commands[cmd_idx]
                            bcast_cmd: MCA_CompiledOperator.Command.MEM_LOAD_TILE = bcast_stage.mem_load_commands[bcast_cmd_idx]
                            
                            existing_ptr_addrs = {ptr.addr for ptr in bcast_cmd.ptrs}
                            for ptr in consum_cmd.ptrs:
                                if ptr.addr not in existing_ptr_addrs:
                                    bcast_cmd.ptrs.append(ptr)
                                    existing_ptr_addrs.add(ptr.addr)
                            
                            compiled_op._mappings[core_id][stage_cursor].mem_load_commands[cmd_idx] = MCA_CompiledOperator.Command.NOP()
                            
                    for tile_sig, cmd_locs in bcast_mem_copy_cmds.items():
                        if len(cmd_locs) <= 1:
                            continue
                        
                        _tmp_bcast_cnt += 1
                        
                        bcast_core_id, bcast_cmd_idx = min(cmd_locs, key=lambda x: _tmp_bcast_request_traffic[x[0]])
                        _tmp_bcast_request_traffic[bcast_core_id] += env.buffers[tile_sig.buf_name].tile_size
                        
                        for core_id, cmd_idx in cmd_locs:
                            if core_id == bcast_core_id and cmd_idx == bcast_cmd_idx:
                                continue
                            
                            consum_stage = compiled_op._mappings[core_id][stage_cursor]
                            bcast_stage = compiled_op._mappings[bcast_core_id][stage_cursor]
                            
                            consum_cmd: MCA_CompiledOperator.Command.MEM_CPY_TILE = consum_stage.mem_load_commands[cmd_idx]
                            bcast_cmd: MCA_CompiledOperator.Command.MEM_CPY_TILE = bcast_stage.mem_load_commands[bcast_cmd_idx]
                            
                            existing_dst_addrs = {ptr.addr for ptr in bcast_cmd.dst_ptrs}
                            for dst_ptr in consum_cmd.dst_ptrs:
                                if dst_ptr.addr not in existing_dst_addrs:
                                    bcast_cmd.dst_ptrs.append(dst_ptr)
                                    existing_dst_addrs.add(dst_ptr.addr)
                            
                            compiled_op._mappings[core_id][stage_cursor].mem_load_commands[cmd_idx] = MCA_CompiledOperator.Command.NOP()
                            
                    if _tmp_bcast_cnt > 0:
                        for core_id in op_meta.op_sig.core_group.core_ids:
                            stages = compiled_op._mappings[core_id]
                            if stage_cursor >= len(stages):
                                compiled_op._mappings[core_id].append(MCA_CompiledOperator.Stage())
                            
                            stages[stage_cursor].mem_load_commands.append(
                                MCA_CompiledOperator.Command.BARRIER(
                                    var_arrived_count=bcast_var_arrived_count,
                                    var_block_state=bcast_var_block_state,
                                    total_arrivals=bcast_total_arrivals,
                                )
                            )
        
        # STAGE 4: Insert global barriers at the beginning and the end of the execution of the grouped operators to ensure correct synchronization with operators outside of the group
        if f"global_barrier_arrived_count" not in env.variables.keys():
            global_var_arrived_count = env.add_variable(f"global_barrier_arrived_count", initial_value=0)
            global_var_block_state = env.add_variable(f"global_barrier_block_state", initial_value=0)
        else:
            global_var_arrived_count = env.variables[f"global_barrier_arrived_count"]
            global_var_block_state = env.variables[f"global_barrier_block_state"]
        global_total_arrivals = sum(op_metas[op_id].op_sig.core_group.n_cores for op_id in op_ids)
        
        for op_id, compiled_op in compiled_ops.items():
            for core_id, stages in compiled_op._mappings.items():
                if len(stages) == 0:
                    raise Exception(f"Debug: compiled operator {op_id} for core {core_id} has no stages??")
                    
                stages[0].preprocessing_commands.insert(0, MCA_CompiledOperator.Command.BARRIER(
                    var_arrived_count=global_var_arrived_count,
                    var_block_state=global_var_block_state,
                    total_arrivals=global_total_arrivals
                ))
                
                stages[-1].postprocessing_commands.append(MCA_CompiledOperator.Command.BARRIER(
                    var_arrived_count=global_var_arrived_count,
                    var_block_state=global_var_block_state,
                    total_arrivals=global_total_arrivals
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
