import abc
import enum
import math
from typing import Any, Sequence, Dict, List, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context.global_context import GlobalContextMemInfo
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.hardware import *


class TileSignature:
    def __init__(self, buf_name: str, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.coords: tuple[int, int, int, int] = (y_s, x_s, y_t, x_t)
        
    def depends_on(self, other: 'TileSignature') -> bool:
        return self.buf_name == other.buf_name and self.coords == other.coords

    @property
    def signature(self) -> str:
        return f"{self.buf_name}{self.coords}"
    

class CollectiveTileSignature(TileSignature):
    def __init__(self, buf_name: str, src_tiles: Sequence[TileSignature], memcpy_patterns: Sequence[dict[int, int]]):
        super().__init__(buf_name, 0, 0, 0, 0)
        
        self.src_tiles = list(src_tiles)
        self.memcpy_patterns = list(memcpy_patterns)
        self.coords = None  # override coords to None for collective tile signature
        
        for src_tile in self.src_tiles:
            if src_tile.buf_name != buf_name:
                raise ValueError("Source tile buffer names do not match collective buffer name.")
            
    def depends_on(self, other):
        return any(src_tile.depends_on(other) for src_tile in self.src_tiles)

    @property
    def signature(self) -> str:
        def tile_signature_with_pattern(tile: TileSignature, pattern: dict[int, int]) -> str:
            pattern_str = "{" + ",".join([f"{k}:{v}" for k, v in pattern.items()]) + "}"
            return f"{tile.signature}{pattern_str}"
        return f"{self.buf_name}[COLLECTIVE {', '.join([tile_signature_with_pattern(tile, pattern) for tile, pattern in zip(self.src_tiles, self.memcpy_patterns)])}]"
    
    
class TiledOperatorSignature:
    def __init__(self):
        self.i_tiles:   list[list[TileSignature]]   = []
        self.o_tile:    TileSignature               = None
        self.op_kwargs: list[dict[str, Any]]        = []
        
        if not (len(self.i_tiles) == len(self.op_kwargs)):
            raise ValueError("Length of input tiles and operation kwargs must match.")
        
    def add_uop(self, i_tiles: list[TileSignature], o_tile: TileSignature, op_kwargs: dict[str, Any]=None):
        self.i_tiles.append(i_tiles)
        if self.o_tile is None:
            self.o_tile = o_tile
        else:
            if self.o_tile.buf_name != o_tile.buf_name or self.o_tile.coords != o_tile.coords:
                raise ValueError("Output tile signature does not match existing output tile signature.")
        self.op_kwargs.append(op_kwargs if op_kwargs is not None else {})
        
    def reorder_uops(self, target_buf_name: str):
        def tile_key_fn(i_tiles: list[TileSignature]):
            for tile in i_tiles:
                if tile.buf_name == target_buf_name:
                    return tile.coords
            return (math.inf, math.inf, math.inf, math.inf)
        
        combined = list(zip(self.i_tiles, self.op_kwargs))
        combined.sort(key=lambda x: tile_key_fn(x[0]))
        self.i_tiles, self.op_kwargs = zip(*combined)
        self.i_tiles = list(self.i_tiles)
        self.op_kwargs = list(self.op_kwargs)
        
    @property
    def signature(self) -> str:
        i_sigs = [
            "[" + ", ".join([t.signature for t in tile_pair]) + "]"
            for tile_pair in self.i_tiles
        ]
        i_sig_str = " + ".join(i_sigs)
        o_sig_str = self.o_tile.signature
        return f"{i_sig_str} -> {o_sig_str}"
    
    @property
    def n_uops(self) -> int:
        return len(self.i_tiles)


class MCA_OperatorSignature:
    def __init__(self, op_type: str, op_template: Callable, op_ex_kernels: list[Callable], spad_ld_mem_space_size: int, spad_st_mem_space_size: int):
        self._op_type = op_type
        self.op_id = op_type    # will be initialized by MCA_OperatorGraphCompiler (initially set to op_type) 
        
        self.op_template = op_template
        self.op_ex_kernels = op_ex_kernels
        self.spad_ld_mem_space_size = spad_ld_mem_space_size
        self.spad_st_mem_space_size = spad_st_mem_space_size
        
        self.spad_ld_pp_mem_space_size = spad_ld_mem_space_size // 2
        self.spad_st_pp_mem_space_size = spad_st_mem_space_size // 2
        
        self._buffers: dict[str, MCA_TensorBuffer] = {}
        self._tiles: dict[str, dict[tuple[int, ...], TileSignature]] = {}
        self._tiled_ops: list[TiledOperatorSignature] = []
        self.global_kwargs: dict[str, Any] = {}
        
        self.buffer_names: list[str] = []
        self.input_buffer_names: list[str] = []
        self.output_buffer_names: list[str] = []
        
        self.core_group: MCA_CoreGroup = None
        self.spad_ld_pp_ptrs: dict[int, tuple[Pointer, Pointer]] = {}
        self.spad_st_pp_ptrs: dict[int, tuple[Pointer, Pointer]] = {}
        
    def add_buffer(self, buf_name: str, buffer: MCA_TensorBuffer, is_input: bool=False, is_output: bool=False):
        if (not is_input) and (not is_output):
            raise ValueError("Buffer must be marked as input or output.")
        
        self._buffers[buf_name] = buffer
        self._tiles[buf_name] = {}
        
        for y_s in range(buffer.shard_grid[0]):
            for x_s in range(buffer.shard_grid[1]):
                for y_t in range(buffer.tile_grid_per_shard[0]):
                    for x_t in range(buffer.tile_grid_per_shard[1]):
                        self._tiles[buf_name][(y_s, x_s, y_t, x_t)] = TileSignature(buf_name, y_s, x_s, y_t, x_t)
        
        self.buffer_names.append(buf_name)
        if is_input:
            self.input_buffer_names.append(buf_name)
        if is_output:
            self.output_buffer_names.append(buf_name)
            
        return self
    
    def new_tiled_op(self) -> TiledOperatorSignature:
        tiled_op = TiledOperatorSignature()
        self._tiled_ops.append(tiled_op)
        return tiled_op
        
    def update_global_kwargs(self, op_kwargs: dict[str, Any]):
        self.global_kwargs.update(op_kwargs)
        
    def initialize_core_group(self, core_group: MCA_CoreGroup, spad_mem_space: MCA_L1MemorySpace):
        self.core_group = core_group
        
        # ping-pong pointers for load
        self.spad_ld_pp_ptrs = {
            core_id: (
                spad_mem_space.allocate(core_id, self.spad_ld_pp_mem_space_size),
                spad_mem_space.allocate(core_id, self.spad_ld_pp_mem_space_size),
            )
            for core_id in core_group.core_ids
        }
        
        # ping-pong pointers for store
        self.spad_st_pp_ptrs = {
            core_id: (
                spad_mem_space.allocate(core_id, self.spad_st_pp_mem_space_size),
                spad_mem_space.allocate(core_id, self.spad_st_pp_mem_space_size),
            )
            for core_id in core_group.core_ids
        }
        
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
            if old_name in self.output_buffer_names:
                idx = self.output_buffer_names.index(old_name)
                self.output_buffer_names[idx] = new_name
            if old_name in self.input_buffer_names:
                idx = self.input_buffer_names.index(old_name)
                self.input_buffer_names[idx] = new_name
            if old_name in self.buffer_names:
                idx = self.buffer_names.index(old_name)
                self.buffer_names[idx] = new_name
                
    def reorder_tiled_ops_with_spatial_reuse_pattern(self, target_buf_name: str):
        if target_buf_name in self.input_buffer_names:
            for tiled_op in self._tiled_ops:
                tiled_op.reorder_uops(target_buf_name)
        else:
            def tile_key_fn(tiled_op: TiledOperatorSignature):
                o_tile = tiled_op.o_tile
                if o_tile.buf_name == target_buf_name:
                    return o_tile.coords
                return (math.inf, math.inf, math.inf, math.inf)
            self._tiled_ops.sort(key=tile_key_fn)
    
    @property
    def op_type(self):      return self._op_type
    @property
    def buffers(self):      return self._buffers
    @property
    def tiles(self):        return self._tiles
    @property
    def tiled_ops(self):    return self._tiled_ops
    
    
class MCA_CompiledOperatorGraph:
    class Command:
        class _Base(metaclass=abc.ABCMeta):
            @abc.abstractmethod
            def signature(self) -> str:
                raise NotImplementedError("Command signature method must be implemented by subclasses.")
            
        class NOP(_Base):
            def signature(self):
                return "NOP"
        
        class MEM_INIT(_Base):
            def __init__(self, ptr: Pointer, size: int):
                self.ptr = ptr
                self.size = size
                
            def signature(self):
                return f"MEM_INIT MEM@{self.ptr.addr} size={self.size}"
            
        class MEM_BROADCAST_TILE(_Base):
            def __init__(self, buf_name: str, tile_sig: TileSignature, ptrs: list[Pointer]):
                self.buf_name = buf_name
                self.tile_sig = tile_sig
                self.ptrs = ptrs
                
                for i in range(len(self.ptrs)):
                    if isinstance(self.ptrs[i], int):
                        self.ptrs[i] = Pointer(addr=self.ptrs[i])
                    
            def signature(self):
                ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.ptrs])
                return f"BROADCAST {self.tile_sig.signature} -> [{ptrs_str}]"
        
        class MEM_LOAD_TILE(_Base):
            def __init__(self, buf_name: str, tile_sig: TileSignature, ptrs: list[Pointer]):
                self.buf_name = buf_name
                self.tile_sig = tile_sig
                self.ptrs = ptrs
                
                for i in range(len(self.ptrs)):
                    if isinstance(self.ptrs[i], int):
                        self.ptrs[i] = Pointer(addr=self.ptrs[i])
                    
            def signature(self):
                ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.ptrs])
                return f"LOAD {self.tile_sig.signature} -> [{ptrs_str}]"
                
        class MEM_STORE_TILE(_Base):
            def __init__(self, buf_name: str, tile_sig: TileSignature, ptr: Pointer):
                self.buf_name = buf_name
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                    
            def signature(self):
                return f"STORE {self.tile_sig.signature} -> SPM@{self.ptr.addr}"
                
        class EXE_LOAD_CONTEXT(_Base):
            def __init__(self, buf_name: str, tile_sig: TileSignature, ptr: Pointer):
                self.buf_name = buf_name
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
                    
            def signature(self):
                return f"LOAD_CONTEXT SPM@{self.ptr.addr} -> {self.tile_sig.signature}"

        class EXE_STORE_CONTEXT(_Base):
            def __init__(self, buf_name: str, tile_sig: TileSignature, ptr: Pointer):
                self.buf_name = buf_name
                self.tile_sig = tile_sig
                self.ptr = ptr
                
                if isinstance(self.ptr, int):
                    self.ptr = Pointer(addr=self.ptr)
        
            def signature(self):
                return f"STORE_CONTEXT {self.tile_sig.signature} -> SPM@{self.ptr.addr}"

        class EXE_UOP(_Base):
            def __init__(self, op_id: str, tiled_op_idx: int, uop_idx: int, i_tile_ptrs: list[Pointer], o_tile_ptr: Pointer):
                self.op_id = op_id
                self.tiled_op_idx = tiled_op_idx
                self.uop_idx = uop_idx
                self.i_tile_ptrs = i_tile_ptrs
                self.o_tile_ptr = o_tile_ptr
                
            def signature(self):
                i_ptrs_str = ", ".join([f"SPM@{ptr.addr}" for ptr in self.i_tile_ptrs])
                o_ptr_str = f"SPM@{self.o_tile_ptr.addr}"
                return f"EXE_UOP {self.op_id} tiled_op_idx={self.tiled_op_idx} uop_idx={self.uop_idx} ({i_ptrs_str}) -> {o_ptr_str}"
                
        class BARRIER(_Base):
            def __init__(self, var_arrived_count: str, var_block_state: str, total_arrivals: int):
                self.var_arrived_count = var_arrived_count
                self.var_block_state = var_block_state
                self.total_arrivals = total_arrivals
                
            def signature(self):
                return f"BARRIER arrived_count={self.var_arrived_count} block_state={self.var_block_state} total_arrivals={self.total_arrivals}"
                
    class Stage:
        # class CommandClass(enum.Enum):
        #     PREPROCESSING = 1
        #     MEM_LOAD = 2
        #     EXECUTE = 3
        #     MEM_STORE = 4
        #     POSTPROCESSING = 5
        
        def __init__(self):
            # STAGE 1: Preprocessing & Memory Load
            #   - Preprocessing commands can be executed simultaneously
            #   - Memory load commands can be executed simultaneously
            #   - However, preprocessing commands should always be executed before memory load commands
            #
            # STAGE 2: Execute
            #   - All execute commands are executed sequentially
            #
            # STAGE 3: Memory Store & Postprocessing
            #   - Memory store commands can be executed simultaneously
            #   - Postprocessing commands can be executed simultaneously
            #   - However, memory store commands should always be executed before postprocessing commands
            
            # self.commands: dict[MCA_CompiledOperatorGraph.Stage.CommandClass, list[MCA_CompiledOperatorGraph.Command._Base]] = {
            #     MCA_CompiledOperatorGraph.Stage.CommandClass.PREPROCESSING: [],
            #     MCA_CompiledOperatorGraph.Stage.CommandClass.MEM_LOAD: [],
            #     MCA_CompiledOperatorGraph.Stage.CommandClass.EXECUTE: [],
            #     MCA_CompiledOperatorGraph.Stage.CommandClass.MEM_STORE: [],
            #     MCA_CompiledOperatorGraph.Stage.CommandClass.POSTPROCESSING: [],
            # }
            
            self.preprocessing_commands:    list[MCA_CompiledOperatorGraph.Command._Base] = []
            self.mem_load_commands:         list[MCA_CompiledOperatorGraph.Command._Base] = []
            self.execute_commands:          list[MCA_CompiledOperatorGraph.Command._Base] = []
            self.mem_store_commands:        list[MCA_CompiledOperatorGraph.Command._Base] = []
            self.postprocessing_commands:   list[MCA_CompiledOperatorGraph.Command._Base] = []
            
        # def add_cmd(self, cmd: 'MCA_CompiledOperatorGraph.Command._Base', cmd_class: 'MCA_CompiledOperatorGraph.Stage.CommandClass'):
        #     self.commands[cmd_class].append(cmd)
        #     return self
            
        # @property
        # def preprocessing_commands(self) -> 'list[MCA_CompiledOperatorGraph.Command._Base]':
        #     return self.commands[MCA_CompiledOperatorGraph.Stage.CommandClass.PREPROCESSING]
        
        # @property
        # def mem_load_commands(self) -> 'list[MCA_CompiledOperatorGraph.Command._Base]':
        #     return self.commands[MCA_CompiledOperatorGraph.Stage.CommandClass.MEM_LOAD]
        
        # @property
        # def execute_commands(self) -> 'list[MCA_CompiledOperatorGraph.Command._Base]':
        #     return self.commands[MCA_CompiledOperatorGraph.Stage.CommandClass.EXECUTE]
        
        # @property
        # def mem_store_commands(self) -> 'list[MCA_CompiledOperatorGraph.Command._Base]':
        #     return self.commands[MCA_CompiledOperatorGraph.Stage.CommandClass.MEM_STORE]
        
        # @property
        # def postprocessing_commands(self) -> 'list[MCA_CompiledOperatorGraph.Command._Base]':
        #     return self.commands[MCA_CompiledOperatorGraph.Stage.CommandClass.POSTPROCESSING]
        
        @property
        def is_bubble(self) -> bool:
            return all(len(cmds) == 0 for cmds in self.commands.values())
            
    def __init__(self, env: 'MCA_OperatorGraphCompiler.Environment', op_sig: 'MCA_OperatorSignature'):
        self._env = env
        self._op_template = op_sig.op_template
        self._op_ex_kernels = op_sig.op_ex_kernels
        self._mappings: dict[int, list[MCA_CompiledOperatorGraph.Stage]] = {}  # {core_id: [stage1, stage2, ...]}
    
    def add_stage(self, core_id: int, stage: 'MCA_CompiledOperatorGraph.Stage'):
        if core_id not in self._mappings:
            self._mappings[core_id] = []
        self._mappings[core_id].append(stage)
        
    def dispatch(self, device: MCA_DeviceBase, slot_id: int="MAIN"):
        for core_id in self.mappings.keys():
            core = device.get_npu_core(core_id)
            n_stages = len(self.mappings[core_id])
            
            for cursor in range(n_stages + 2):
                stage1_cursor = cursor
                stage2_cursor = cursor - 1
                stage3_cursor = cursor - 2
                
                kernel: KernelPrototype = self._op_template(core, self._env, self, stage1_cursor, stage2_cursor, stage3_cursor, self._op_ex_kernels)
                kernel.dispatch(slot_id)
        
    @property
    def mappings(self):
        return self._mappings
    
    def summary(self) -> dict:
        summary = {}
        for core_id, stages in self._mappings.items():
            summary[core_id] = []
            for stage in stages:
                stage_summary = {
                    "preprocessing": [cmd.signature() for cmd in stage.preprocessing_commands],
                    "mem_load": [cmd.signature() for cmd in stage.mem_load_commands],
                    "execute": [cmd.signature() for cmd in stage.execute_commands],
                    "mem_store": [cmd.signature() for cmd in stage.mem_store_commands],
                    "postprocessing": [cmd.signature() for cmd in stage.postprocessing_commands],
                }
                summary[core_id].append(stage_summary)
        return summary
    
    
class MCA_OperatorGraphCompiler:
    ALL="ALL"
    
    class OperatorRecipe:
        def __init__(
            self,
            spatial_reuse_target_buf_idx: int = 0,
            use_broadcast_optimize: bool = False,
        ):
            self.spatial_reuse_target_buf_idx = spatial_reuse_target_buf_idx
            self.use_broadcast_optimize = use_broadcast_optimize
    
    class GlobalRecipe:
        def __init__(
            self, 
            global_core_group: MCA_CoreGroup, 
            core_group_shape: int | tuple[int, int], 
            spad_mem_space: MCA_L1MemorySpace, 
            op_recipes: 'dict[str, MCA_OperatorGraphCompiler.OperatorRecipe]'=None,
            # spatial_reuse_target: dict[str, int]=None,
            # use_broadcast_optimize: bool=False,
        ):
            self.global_core_group = global_core_group
            self.core_group_shape = core_group_shape
            self.spad_mem_space = spad_mem_space
            
            self.core_groups = global_core_group.split(shape=core_group_shape)
            # self.spatial_reuse_target = spatial_reuse_target if spatial_reuse_target is not None else {}
            # self.use_broadcast_optimize = use_broadcast_optimize
            self.op_recipes = op_recipes if op_recipes is not None else {}
            
        def get_operator_recipe(self, op_sig: MCA_OperatorSignature) -> 'MCA_OperatorGraphCompiler.OperatorRecipe':
            if op_sig.op_id in self.op_recipes:
                return self.op_recipes[op_sig.op_id]
            return self.op_recipes.get(op_sig.op_type, MCA_OperatorGraphCompiler.OperatorRecipe())
    
    class Environment:
        def __init__(self):
            self.op_sigs:     dict[str, MCA_OperatorSignature]  = {}
            self.buffers:     dict[str, MCA_TensorBuffer]   = {}
            self.variables:   dict[str, VariableHandle]     = {}
            
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
            
            self.op_sigs[op_sig.op_id] = op_sig
            self.buffers.update(op_sig.buffers)
        
        def add_variable(self, var_name: str, var_handle: VariableHandle):
            if var_name in self.variables:
                raise ValueError(f"Variable with name {var_name} already exists.")
            self.variables[var_name] = var_handle
            
    class _StageState:
        def __init__(self, ld_pp_space_size: int, st_pp_space_size: int, ld_pp_ptrs: list[Pointer], st_pp_ptrs: list[Pointer]):
            self.ld_pp_space_size = ld_pp_space_size
            self.st_pp_space_size = st_pp_space_size
            
            self.ld_pp_ptrs   = ld_pp_ptrs
            self.st_pp_ptrs   = st_pp_ptrs
            
            self.cached_ld_ptrs = {}  # {tile_signature: Pointer}
            self.pp_flag        = 0   # ping-pong flag -> change this flag if there aren't enough memory space
            self.ld_offset      = 0   # current load offset within the ping-pong buffer
            self.st_offset      = 0   # current store offset within the ping-pong buffer
        
        def is_ld_available(self, tile_size: int) -> bool:
            return (self.ld_offset + tile_size) <= self.ld_pp_space_size
        
        def is_st_available(self, tile_size: int) -> bool:
            return (self.st_offset + tile_size) <= self.st_pp_space_size
        
        def is_ld_tile_cached(self, tile_sig: TileSignature) -> bool:
            return tile_sig.signature in self.cached_ld_ptrs
            
        def new_ld_ptr(self, tile_sig: TileSignature, tile_size: int) -> Pointer:
            if self.is_ld_tile_cached(tile_sig):
                raise ValueError(f"Tile {tile_sig.signature} already cached in load pointers. If you want to get the cached pointer, use get_cached_ld_ptr() method.")
            ptr_addr = self.ld_pp_ptrs[self.pp_flag].addr + self.ld_offset
            self.cached_ld_ptrs[tile_sig.signature] = Pointer(addr=ptr_addr)
            self.ld_offset += tile_size
            return self.cached_ld_ptrs[tile_sig.signature]
            
        def new_st_ptr(self, tile_size: int) -> Pointer:
            ptr_addr = self.st_pp_ptrs[self.pp_flag].addr + self.st_offset
            self.st_offset += tile_size
            return Pointer(addr=ptr_addr)
        
        def get_cached_ld_ptr(self, tile_sig: TileSignature) -> Pointer:
            if not self.is_ld_tile_cached(tile_sig):
                raise ValueError(f"Tile {tile_sig.signature} not cached in load pointers.")
            return self.cached_ld_ptrs[tile_sig.signature]
        
        def get_cached_st_ptr(self) -> Pointer:
            return Pointer(addr=self.st_pp_ptrs[self.pp_flag].addr + self.st_offset)
            
        def clear(self):
            self.cached_ld_ptrs = {}
            self.pp_flag        = 1 - self.pp_flag
            self.ld_offset      = 0
            self.st_offset      = 0
            
    def __init__(self):
        self._op_sigs: dict[str, MCA_OperatorSignature] = {}

    def new_op(self) -> MCA_OperatorSignature:
        op_id = "op_" + str(len(self._op_sigs))
        op_sig = MCA_OperatorSignature(op_type=op_id)
        self._op_sigs[op_id] = op_sig
        return op_sig
    
    def add_op(self, op_sig: MCA_OperatorSignature) -> str:
        suffix = 1
        while f"{op_sig.op_id}_{suffix}" in self._op_sigs:
            suffix += 1
        op_sig.op_id = f"{op_sig.op_id}_{suffix}"
        self._op_sigs[op_sig.op_id] = op_sig
        return op_sig.op_id
    
    @staticmethod    
    def _distribute_resources(total: int, weights: dict[str, float]) -> dict[str, int]:
        n = len(weights)
        
        if total < n:
            raise ValueError("Insufficient total resources to allocate at least 1 unit per entity.")
        
        allocations = {k: 1 for k in weights.keys()}
        remaining_resources = total - n
        
        sum_weights = sum(weights.values())

        additional_allocations = {}
        for k, w in weights.items():
            share = (w / sum_weights) * remaining_resources
            additional_allocations[k] = share
        
        for k in weights.keys():
            int_share = int(additional_allocations[k])
            allocations[k] += int_share
        
        current_sum = sum(allocations.values())
        leftover = total - current_sum
        
        if leftover > 0:
            remainders = [(k, additional_allocations[k] - int(additional_allocations[k])) for k in weights.keys()]
            remainders.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(int(leftover)):
                idx = remainders[i][0]
                allocations[idx] += 1
                
        return allocations
    
    def _compile_initialize(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', target_ops: list[str]) -> 'MCA_OperatorGraphCompiler.Environment':
        # Initialize operator recipes with core groups
        _n_cores_weight_per_core: dict[str, int] = {}
        _total_n_core_groups_available: int = len(global_recipe.core_groups)
        
        for op_id in target_ops:
            op_sig = self._op_sigs[op_id]
            _n_cores_weight_per_core[op_id] = sum(op_sig.buffers[buf_name].n_tiles for buf_name in op_sig.output_buffer_names)
            
        _n_core_groups_per_core = self._distribute_resources(_total_n_core_groups_available, _n_cores_weight_per_core)
        _core_group_alloc_offset = 0
        
        for op_id in target_ops:
            op_sig = self._op_sigs[op_id]
            n_core_groups = _n_core_groups_per_core[op_id]
            core_group = MCA_CoreGroup.merge_core_groups(global_recipe.core_groups[_core_group_alloc_offset:_core_group_alloc_offset + n_core_groups])
            _core_group_alloc_offset += n_core_groups
            
            op_sig.initialize_core_group(core_group, global_recipe.spad_mem_space)
            logger.debug(f"Initialized core group for operator {op_id}: {core_group}")
        
        # Initialize environment
        env = MCA_OperatorGraphCompiler.Environment()
        
        for op_id in target_ops:
            op_sig = self._op_sigs[op_id]
            env.add_op_sig(op_sig)
            
            target_buf_name = op_sig.buffer_names[global_recipe.get_operator_recipe(op_sig).spatial_reuse_target_buf_idx]
            op_sig.reorder_tiled_ops_with_spatial_reuse_pattern(target_buf_name)
            
        return env
            
    def _compile_create_mapping(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', target_ops: list[str]) -> dict[str, dict[int, list[int]]]:
        clustered_tiled_ops: dict[str, dict[int, list[int]]] = {}  # {op_id: {core_id: [tiled_op_idx, ...]}}  -> single cluster: [tiled_op_idx, ...]
        
        for op_id in target_ops:
            op_sig = self._op_sigs[op_id]
            target_buf_name = op_sig.buffer_names[global_recipe.get_operator_recipe(op_sig).spatial_reuse_target_buf_idx]
            
            core_ids = list(op_sig.core_group.core_ids)
            n_cores = len(core_ids)
            
            clustered_tiled_ops[op_id] = {core_id: [] for core_id in core_ids}
            
            tiled_op_indices = list(range(len(op_sig.tiled_ops)))
            
            while len(tiled_op_indices) > 0:
                # Select prime tiled op
                prime_core_id = core_ids[0]
                prime_tiled_op_idx = tiled_op_indices.pop(0)
                prime_target_buf_coords = []
                prime_tiled_op = op_sig.tiled_ops[prime_tiled_op_idx]
                
                for uop_idx in range(prime_tiled_op.n_uops):
                    for prime_i_tile in prime_tiled_op.i_tiles[uop_idx]:
                        if prime_i_tile.buf_name == target_buf_name:
                            prime_target_buf_coords.append(prime_i_tile.coords)
                            
                clustered_tiled_ops[op_id][prime_core_id].append(prime_tiled_op_idx)
                
                # calculate matching counts for other tiled ops
                tiled_ops_matching_cnt = []
                                
                for tiled_op_idx in tiled_op_indices:
                    match_cnt = 0
                    tiled_op = op_sig.tiled_ops[tiled_op_idx]
                    
                    for uop_idx in range(tiled_op.n_uops):
                        for i_tile in tiled_op.i_tiles[uop_idx]:
                            if i_tile.buf_name == target_buf_name:
                                if i_tile.coords in prime_target_buf_coords:
                                    match_cnt += 1
                                
                    tiled_ops_matching_cnt.append(match_cnt)
                    
                # select top (n_cores - 1) matching tiled ops
                n_selected_tiled_ops = min(n_cores - 1, len(tiled_op_indices))
                sorted_tiled_op_indices = sorted(range(len(tiled_op_indices)), key=lambda idx: tiled_ops_matching_cnt[idx], reverse=True)
                sorted_tiled_op_indices = sorted_tiled_op_indices[:n_selected_tiled_ops]
                sorted_tiled_op_indices.sort()
                
                for i, selected_tiled_op_idx in reversed(list(enumerate(sorted_tiled_op_indices))):
                    selected_core_id = core_ids[i + 1]
                    tiled_op_idx = tiled_op_indices[selected_tiled_op_idx]
                    clustered_tiled_ops[op_id][selected_core_id].append(tiled_op_idx)
                    tiled_op_indices.pop(selected_tiled_op_idx)
        
        return clustered_tiled_ops
    
    def _compile_generate_compiled_ops(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', env: 'MCA_OperatorGraphCompiler.Environment', clustered_tiled_ops: dict[str, dict[int, list[int]]]) -> dict[str, MCA_CompiledOperatorGraph]:
        compiled_ops: dict[str, MCA_CompiledOperatorGraph] = {op_id: MCA_CompiledOperatorGraph(env, self._op_sigs[op_id]) for op_id in self._op_sigs.keys()}
        
        for op_id, mappings in clustered_tiled_ops.items():
            compiled_op = compiled_ops[op_id]
            op_sig = self._op_sigs[op_id]
            
            for core_id, tiled_op_indices in mappings.items():
                ld_pp_ptrs = op_sig.spad_ld_pp_ptrs[core_id]
                st_pp_ptrs = op_sig.spad_st_pp_ptrs[core_id]
                
                current_stage = MCA_CompiledOperatorGraph.Stage()  # current stage being constructed (create new one if pp buffer is switched)
                current_stage_state = MCA_OperatorGraphCompiler._StageState(
                    ld_pp_space_size=op_sig.spad_ld_pp_mem_space_size,
                    st_pp_space_size=op_sig.spad_st_pp_mem_space_size,
                    ld_pp_ptrs=ld_pp_ptrs,
                    st_pp_ptrs=st_pp_ptrs,
                )
                
                for tiled_op_idx in tiled_op_indices:
                    tiled_op = op_sig.tiled_ops[tiled_op_idx]
                    
                    # Estimate memory usage for output tile
                    o_tile = tiled_op.o_tile
                    o_mem_usage = op_sig.buffers[o_tile.buf_name].tile_size
                    
                    if not current_stage_state.is_st_available(o_mem_usage):
                        compiled_op.add_stage(core_id, current_stage)
                        current_stage = MCA_CompiledOperatorGraph.Stage()
                        current_stage_state.clear()
                    
                    for uop_idx in range(tiled_op.n_uops):
                        # Estimate memory usage for input tiles (for each uop)
                        i_tiles = tiled_op.i_tiles[uop_idx]
                        i_mem_usage = sum([op_sig.buffers[tile.buf_name].tile_size for tile in i_tiles if tile.signature not in current_stage_state.cached_ld_ptrs.keys()])
                        
                        # Check if adding this uop exceeds memory limits
                        if not current_stage_state.is_ld_available(i_mem_usage):
                            # store context to the memory (if context switching occurs within the same tiled op)
                            if global_recipe.get_operator_recipe(op_sig).use_broadcast_optimize and uop_idx != 0:
                                st_ptr = current_stage_state.new_st_ptr(op_sig.buffers[o_tile.buf_name].tile_size)
                                current_stage.execute_commands.append(
                                    MCA_CompiledOperatorGraph.Command.EXE_STORE_CONTEXT(
                                        buf_name=o_tile.buf_name,
                                        tile_sig=o_tile,
                                        ptr=st_ptr
                                    )
                                )
                                current_stage.mem_store_commands.append(
                                    MCA_CompiledOperatorGraph.Command.MEM_STORE_TILE(
                                        buf_name=o_tile.buf_name,
                                        tile_sig=o_tile,
                                        ptr=st_ptr
                                    )
                                )
                            
                            # Finalize current stage and start a new stage with switched ping-pong buffers
                            compiled_op.add_stage(core_id, current_stage)
                            if global_recipe.get_operator_recipe(op_sig).use_broadcast_optimize:
                                # TODO: Remove additional bubbles in the future
                                # NOTE: Broadcast optimization requires stage-wise synchronization between cores. It implies that all cores must finish 
                                # their current stages before proceeding to the next stage where broadcast loading occurs. Therefore, the compiler inserts
                                # context switching commands to ensure that all cores can safely transition to the next stage without data hazards. However,
                                # context load/store commands cannot be placed back-to-back in a single stage due to the architecture's execution model.
                                # As a result, two additional bubble stages are added to separate these commands, ensuring correct execution order and data 
                                # integrity. This is a temporary solution, and future optimizations may eliminate bubbles by reordering stages that are not
                                # dependent on each other. (e.g., stages that operate on different buffers).
                                compiled_op.add_stage(core_id, MCA_CompiledOperatorGraph.Stage())
                                compiled_op.add_stage(core_id, MCA_CompiledOperatorGraph.Stage())
                            current_stage = MCA_CompiledOperatorGraph.Stage()
                            current_stage_state.clear()
                            
                            # load context to the memory (if context switching occurs within the same tiled op)
                            if global_recipe.get_operator_recipe(op_sig).use_broadcast_optimize and uop_idx != 0:
                                ld_ptr = current_stage_state.new_ld_ptr(o_tile, op_sig.buffers[o_tile.buf_name].tile_size)
                                current_stage.mem_load_commands.append(
                                    MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE(
                                        buf_name=o_tile.buf_name,
                                        tile_sig=o_tile,
                                        ptrs=[ld_ptr,]
                                    ),  # stage 1: load output tile
                                )
                                current_stage.execute_commands.append(
                                    MCA_CompiledOperatorGraph.Command.EXE_LOAD_CONTEXT(
                                        buf_name=o_tile.buf_name,
                                        tile_sig=o_tile,
                                        ptr=ld_ptr
                                    ),  # stage 2: load context
                                )
                        
                        # Generate commands for loading input tiles
                        i_tile_ptrs = []
                        for tile in i_tiles:
                            if current_stage_state.is_ld_tile_cached(tile):
                                tile_ptr_addr = current_stage_state.get_cached_ld_ptr(tile)
                            else:
                                tile_ptr_addr = current_stage_state.new_ld_ptr(tile, op_sig.buffers[tile.buf_name].tile_size)
                            
                                current_stage.mem_load_commands.append(
                                    MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE(
                                        buf_name=tile.buf_name, 
                                        tile_sig=tile, 
                                        ptrs=[tile_ptr_addr,],
                                    )
                                )
                            
                            i_tile_ptrs.append(tile_ptr_addr)
                            
                        # Generate command for executing uop
                        o_tile_ptr_addr = current_stage_state.get_cached_st_ptr()
                        current_stage.execute_commands.append(
                            MCA_CompiledOperatorGraph.Command.EXE_UOP(
                                op_id=op_id,
                                tiled_op_idx=tiled_op_idx,
                                uop_idx=uop_idx,
                                i_tile_ptrs=i_tile_ptrs,
                                o_tile_ptr=o_tile_ptr_addr
                            )
                        )
                        
                    # Generate command for storing output tile
                    current_stage.mem_store_commands.append(
                        MCA_CompiledOperatorGraph.Command.MEM_STORE_TILE(
                            buf_name=o_tile.buf_name, 
                            tile_sig=o_tile, 
                            ptr=current_stage_state.new_st_ptr(op_sig.buffers[o_tile.buf_name].tile_size).addr
                        )
                    )
                
                # Finalize the last stage for the core
                compiled_op.add_stage(core_id, current_stage)
                
        return compiled_ops
    
    def _compile_pipeline_optimize(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', env: 'MCA_OperatorGraphCompiler.Environment', compiled_ops: 'dict[str, MCA_CompiledOperatorGraph]') -> dict[str, MCA_CompiledOperatorGraph]:
        n_ops = len(compiled_ops)
        
        if n_ops < 2:
            return compiled_ops
        
        for dst_op_idx in range(1, n_ops):
            dst_op_id = list(compiled_ops.keys())[dst_op_idx]
            dst_compiled_op = compiled_ops[dst_op_id]
            dst_op_sig = self._op_sigs[dst_op_id]
            
            for src_op_idx in range(dst_op_idx):
                src_op_id = list(compiled_ops.keys())[src_op_idx]
                src_compiled_op = compiled_ops[src_op_id]
                src_op_sig = self._op_sigs[src_op_id]
                
                # check if there are common buffers between src and dst operators
                common_buf_names = set(src_op_sig.output_buffer_names).intersection(set(dst_op_sig.input_buffer_names))
                
                if len(common_buf_names) == 0:
                    continue
                
                var_arrived_count_name = f"pipe_b_{src_op_id}::{dst_op_id}_arrived_count"
                var_block_state_name   = f"pipe_b_{src_op_id}::{dst_op_id}_block_state"
                
                env.add_variable(var_arrived_count_name, VariableHandle(var_arrived_count_name))
                env.add_variable(var_block_state_name, VariableHandle(var_block_state_name))
                
                src_core_ids = set(src_compiled_op.mappings.keys())
                dst_core_ids = set(dst_compiled_op.mappings.keys())
                
                n_src_stages = max(len(src_compiled_op.mappings.get(core_id, [])) for core_id in src_core_ids)
                n_dst_stages = max(len(dst_compiled_op.mappings.get(core_id, [])) for core_id in dst_core_ids)
                
                dst_dept_tiles_per_stage = [set() for _ in range(n_dst_stages)]
                dst_stage_cursor = 0
                
                src_dept_tiles = set()
                
                for dst_stage_idx in range(n_dst_stages):
                    for dst_core_id in dst_core_ids:
                        if dst_stage_idx >= len(dst_compiled_op.mappings.get(dst_core_id, [])):
                            continue
                        
                        dst_stage = dst_compiled_op.mappings[dst_core_id][dst_stage_idx]
                        
                        for cmd in dst_stage.mem_load_commands:
                            if isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE) and cmd.buf_name in common_buf_names:
                                dst_dept_tiles_per_stage[dst_stage_idx].add(cmd.tile_sig.signature)
                        
                for src_stage_cursor in range(n_src_stages):
                    for src_core_id in src_core_ids:
                        if src_stage_cursor >= len(src_compiled_op.mappings.get(src_core_id, [])):
                            continue
                        
                        src_stage = src_compiled_op.mappings[src_core_id][src_stage_cursor]
                        
                        for cmd in src_stage.mem_store_commands:
                            if isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_STORE_TILE) and cmd.buf_name in common_buf_names:
                                src_dept_tiles.add(cmd.tile_sig.signature)
                    
                    barrier_inserted = False
                    
                    # check dependency with dst stages
                    while dst_stage_cursor < n_dst_stages:
                        dst_dept_tiles = dst_dept_tiles_per_stage[dst_stage_cursor]
                        is_dept_resolved = len(src_dept_tiles.intersection(dst_dept_tiles)) == len(dst_dept_tiles)
                        
                        if is_dept_resolved:
                            if not barrier_inserted:
                                total_arrivals = 0
                                
                                for core_id in src_core_ids:
                                    if src_stage_cursor < len(src_compiled_op.mappings.get(core_id, [])):
                                        total_arrivals += 1
                                        
                                for core_id in dst_core_ids:
                                    if dst_stage_cursor < len(dst_compiled_op.mappings.get(core_id, [])):
                                        total_arrivals += 1
                                
                                for core_id in src_core_ids:    
                                    src_compiled_op.mappings[core_id][src_stage_cursor].postprocessing_commands.append(
                                        MCA_CompiledOperatorGraph.Command.BARRIER(
                                            var_arrived_count=var_arrived_count_name,
                                            var_block_state=var_block_state_name,
                                            total_arrivals=total_arrivals,
                                        )
                                    )
                               
                                for core_id in dst_core_ids:
                                    dst_compiled_op.mappings[core_id][dst_stage_cursor].preprocessing_commands.append(
                                        MCA_CompiledOperatorGraph.Command.BARRIER(
                                            var_arrived_count=var_arrived_count_name,
                                            var_block_state=var_block_state_name,
                                            total_arrivals=total_arrivals,
                                        )
                                    )    
                                    
                                barrier_inserted = True
                            
                            dst_stage_cursor += 1
                        else:
                            break
                        
        return compiled_ops
    
    def _compile_broadcast_optimize(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', env: 'MCA_OperatorGraphCompiler.Environment', compiled_ops: 'dict[str, MCA_CompiledOperatorGraph]') -> dict[str, MCA_CompiledOperatorGraph]:
        for op_id, compiled_op in compiled_ops.items():
            op_sig = self._op_sigs[op_id]
            
            if not global_recipe.get_operator_recipe(op_sig).use_broadcast_optimize:
                continue
            
            n_stages = max(len(stages) for stages in compiled_op.mappings.values())
            
            # add stage-wise barriers
            var_arrived_count_name = f"bcast_b_{op_id}_arrived_count"
            var_block_state_name   = f"bcast_b_{op_id}_block_state"
            
            env.add_variable(var_arrived_count_name, VariableHandle(var_arrived_count_name))
            env.add_variable(var_block_state_name, VariableHandle(var_block_state_name))
            
            for stage_idx in range(n_stages):    
                total_arrivals = 0
                for core_id in compiled_op.mappings.keys():
                    if stage_idx < len(compiled_op.mappings[core_id]):
                        total_arrivals += 1
                        
                for core_id in compiled_op.mappings.keys():
                    if stage_idx < len(compiled_op.mappings[core_id]):
                        compiled_op.mappings[core_id][stage_idx].mem_load_commands.append(
                            MCA_CompiledOperatorGraph.Command.BARRIER(
                                var_arrived_count=var_arrived_count_name,
                                var_block_state=var_block_state_name,
                                total_arrivals=total_arrivals,
                            )
                        )
                        
            # replace load commands with broadcast load commands
            for stage_idx in range(n_stages):
                bcast_targets: dict[str, list[tuple[int, int, list[Pointer]]]] = {}  # {tile_sig: [(core_id, cmd_idx, [pointer, ...]), ...]}
                
                for core_id, stages in compiled_op.mappings.items():
                    if stage_idx < len(stages):
                        stage = stages[stage_idx]
                        
                        for cmd_idx, cmd in enumerate(stage.mem_load_commands):
                            if isinstance(cmd, MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE):
                                tile_sig = cmd.tile_sig
                                if tile_sig.signature not in bcast_targets:
                                    bcast_targets[tile_sig.signature] = []
                                bcast_targets[tile_sig.signature].append((core_id, cmd_idx, cmd.ptrs))
                
                bcast_burdens: dict[int, int]       = {core_id: 0 for core_id in compiled_op.mappings.keys()}   # {core_id: n_broadcasts}
                
                for tile_sig, targets in bcast_targets.items():
                    if len(targets) <= 1:
                        continue
                    
                    # select broadcast source core (core with least broadcast burden)
                    src_core_id = min([dst_core_id for dst_core_id, _, _ in targets], key=lambda cid: bcast_burdens[cid])
                    src_cmd_idx = None
                    
                    for dst_core_id, cmd_idx, _ in targets:
                        if dst_core_id == src_core_id:
                            src_cmd_idx = cmd_idx
                            break
                        
                    if src_cmd_idx is None:
                        raise ValueError("Broadcast source core not found.")

                    for dst_core_id, cmd_idx, ptrs in targets:
                        if dst_core_id == src_core_id:
                            continue
                        
                        stage = compiled_op.mappings[src_core_id][stage_idx]    
                        
                        cmd: MCA_CompiledOperatorGraph.Command.MEM_LOAD_TILE = stage.mem_load_commands[src_cmd_idx]
                        cmd.ptrs.extend(ptrs)  # add additional pointers for broadcast
                        bcast_burdens[src_core_id] += len(ptrs)
                    
                    for dst_core_id, cmd_idx, ptrs in targets:
                        stage = compiled_op.mappings[dst_core_id][stage_idx]
                        
                        if dst_core_id != src_core_id:
                            stage.mem_load_commands[cmd_idx] = MCA_CompiledOperatorGraph.Command.NOP()  # replace with NOP command
                        
        return compiled_ops
        
    def compile(self, global_recipe: 'MCA_OperatorGraphCompiler.GlobalRecipe', target_ops: list[str] | str="ALL") -> dict[str, MCA_CompiledOperatorGraph]:    
        if target_ops == MCA_OperatorGraphCompiler.ALL:
            target_ops = list(self._op_sigs.keys())
            
        for op_id in target_ops:
            if op_id not in self._op_sigs:
                raise ValueError(f"Operator ID {op_id} not found in the compiler.")
        
        # STEP 1: Initialize compilation environment
        env = self._compile_initialize(global_recipe, target_ops)
            
        # STEP 2: Create core to tiled operator mapping
        clustered_tiled_ops = self._compile_create_mapping(global_recipe, target_ops)
                    
        # STEP 3: Generate compiled operators
        compiled_ops = self._compile_generate_compiled_ops(global_recipe, env, clustered_tiled_ops)
        
        # STEP 4: Optional optimizations
        compiled_ops = self._compile_broadcast_optimize(global_recipe, env, compiled_ops)  # apply broadcast optimization (with respect to the operator recipes in the global recipe)
        compiled_ops = self._compile_pipeline_optimize(global_recipe, env, compiled_ops)   # apply pipeline optimization across multiple operators
        
        return compiled_ops
    

def mca_operator_method(func: Callable):
    def _mca_mapper_method_wrapper(*args, **kwargs) -> MCA_OperatorSignature:
        mapper = func(*args, **kwargs)
        if not isinstance(mapper, MCA_OperatorSignature):
            raise TypeError("The decorated function must return an instance of OperatorSignature.")
        return mapper
    return _mca_mapper_method_wrapper