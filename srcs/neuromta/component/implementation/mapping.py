import abc
import tqdm
from typing import Any, Sequence, Dict, List, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.tensor_buffer import *


__all__ = [
    "BCAST_BARRIER_ARRIVED_CNT",
    "BCAST_BARRIER_BLOCK_STATE",
    
    "TileSignature",
    "TiledOperatorSignature",
    "TiledOperatorMapping",
    "CompiledCommand",
    "CompiledStage",
    "CompiledOperator",
    "CompiledMapping",
    
    "MCA_OperatorMapper",
]


BCAST_BARRIER_ARRIVED_CNT = "BCAST_BARRIER_ARRIVED_CNT"
BCAST_BARRIER_BLOCK_STATE = "BCAST_BARRIER_BLOCK_STATE"
PIPE_BARRIER_ARRIVED_CNT  = "PIPE_BARRIER_ARRIVED_CNT"
PIPE_BARRIER_BLOCK_STATE  = "PIPE_BARRIER_BLOCK_STATE"


class TileSignature:
    def __init__(self, buf_name: str, buf: MCA_TensorBuffer, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.buf = buf
        self.coords: tuple[int, int, int, int] = (y_s, x_s, y_t, x_t)
        
        self.spm_ptr: Pointer | None = None  # to be assigned during compilation process
        
    def override_spm_ptr(self, spm_ptr: Pointer):
        new_sig = TileSignature(self.buf_name, self.buf, *self.coords)
        new_sig.spm_ptr = spm_ptr
        return new_sig
        
    def __eq__(self, value):
        if isinstance(value, TileSignature):
            return (self.buf_name == value.buf_name and self.coords == value.coords)
        return False
    
    @property
    def mem_info(self) -> GlobalContextMemInfo:
        shard_ptr = self.buf.get_shard_ptr(self.coords[0], self.coords[1])
        mem_info  = self.buf.mem_space.device.global_context.get_mem_info_by_address(addr=shard_ptr.addr)
        return mem_info
    
    @property
    def signature(self) -> str:
        return f"{self.buf_name}{self.coords}"
    

class CollectiveTileSignature(TileSignature):
    def __init__(self, buf_name: str, buf: MCA_TensorBuffer, src_tiles: Sequence[TileSignature], memcpy_patterns: Sequence[dict[int, int]]):
        super().__init__(buf_name, buf, 0, 0, 0, 0)
        
        self.src_tiles = list(src_tiles)
        self.memcpy_patterns = list(memcpy_patterns)
        self.coords = None  # override coords to None for collective tile signature
        
        for src_tile in self.src_tiles:
            if src_tile.buf_name != buf_name:
                raise ValueError("Source tile buffer names do not match collective buffer name.")
        
        self.spm_ptr: Pointer | None = None  # to be assigned during compilation process
        
    def override_spm_ptr(self, spm_ptr: Pointer):
        new_sig = CollectiveTileSignature(self.buf_name, self.buf, self.src_tiles, self.memcpy_patterns)
        new_sig.spm_ptr = spm_ptr
        return new_sig
    
    def __eq__(self, value):
        return False
    
    @property
    def mem_info(self) -> GlobalContextMemInfo:
        raise Exception("CollectiveTileSignature does not have a single memory info.")
    
    @property
    def signature(self) -> str:
        def tile_signature_with_pattern(tile: TileSignature, pattern: dict[int, int]) -> str:
            pattern_str = "{" + ",".join([f"{k}:{v}" for k, v in pattern.items()]) + "}"
            return f"{tile.signature}{pattern_str}"
        return f"{self.buf_name}[COLLECTIVE {', '.join([tile_signature_with_pattern(tile, pattern) for tile, pattern in zip(self.src_tiles, self.memcpy_patterns)])}]"
        
        
        
class TiledOperatorSignature:
    def __init__(self, i_tiles: Sequence[Sequence[TileSignature]], o_tile: TileSignature, op_kwargs: dict[str, Any]=None):
        self.i_tiles     = i_tiles
        self.o_tile      = o_tile
        self.op_kwargs   = op_kwargs if op_kwargs is not None else {}
        
    def copy(self) -> 'TiledOperatorSignature':
        i_tiles = [
            [tile.override_spm_ptr(None) for tile in tile_pair]
            for tile_pair in self.i_tiles
        ]
        o_tile = self.o_tile.override_spm_ptr(None)
        
        return TiledOperatorSignature(
            i_tiles=i_tiles, 
            o_tile=o_tile, 
            op_kwargs=self.op_kwargs.copy(), 
        )
        
    @property
    def signature(self) -> str:
        i_sigs = [
            "[" + ", ".join([t.signature for t in tile_pair]) + "]"
            for tile_pair in self.i_tiles
        ]
        i_sig_str = " + ".join(i_sigs)
        o_sig_str = self.o_tile.signature
        return f"{i_sig_str} -> {o_sig_str}"
        
        
class TiledOperatorMapping(Dict[int, List[TiledOperatorSignature]]):
    def __init__(self, core_group: MCA_CoreGroup):
        self.core_group = core_group
        
    @staticmethod
    def output_stationary(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
        # contiguous for output tile assignment
        # reuse optimization for input and weight tile assignment
        
        # STEP 1: detemine cores
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
        
        # STEP 2: collect unique output tiles
        otile_coords_arr: list[tuple[int, ...]] = []
        for op in tiled_ops:
            if op.o_tile.coords not in otile_coords_arr:
                otile_coords_arr.append(op.o_tile.coords)        
        otile_coords_arr.sort()  # sort to ensure deterministic mapping
        
        # STEP 3: assign each output tile to a core in a contiguous manner
        n_otile_per_core = math.ceil(len(otile_coords_arr) / len(core_ids))
        
        for cursor in range(len(otile_coords_arr)):
            target_core_id = core_ids[cursor // n_otile_per_core]
            otile_coords = otile_coords_arr[cursor]
            
            # assign all operators with the same output tile to the target core
            for op in tiled_ops:
                if op.o_tile.coords == otile_coords:
                    mapper[target_core_id].append(op)
        
        return mapper
    
    @staticmethod
    def round_robin(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
        
        for op_idx, op in enumerate(tiled_ops):
            target_core_id = core_ids[op_idx % len(core_ids)]
            mapper[target_core_id].append(op)
        
        return mapper
    
    @staticmethod
    def contiguous(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
        
        n_cores      = len(core_ids)
        ops_per_core = len(tiled_ops) // n_cores
        remainder    = len(tiled_ops) % n_cores
        
        cursor = 0
        for core_idx, core_id in enumerate(core_ids):
            n_ops = ops_per_core + (1 if core_idx < remainder else 0)
            for _ in range(n_ops):
                mapper[core_id].append(tiled_ops[cursor])
                cursor += 1
        
        return mapper
    
    def summary(self) -> dict[int, list[str]]:
        return {core_id: [op.signature() for op in ops] for core_id, ops in self.items()}
            
            
class CompiledCommand:
    class _Base(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def signature(self) -> str:
            pass
    
    class NOP(_Base):
        def signature(self) -> str:
            return "NOP"
    
    class VAR_BARRIER(_Base):
        def __init__(self, var_arrived_count: VariableHandle, var_block_state: VariableHandle, total_arrivals: int):
            super().__init__()
            
            self._var_arrived_count = var_arrived_count
            self._var_block_state = var_block_state
            self._total_arrivals = total_arrivals
            
        @property
        def var_arrived_count(self) -> VariableHandle:
            return self._var_arrived_count
        
        @property
        def var_block_state(self) -> VariableHandle:
            return self._var_block_state
        
        @property
        def total_arrivals(self) -> int:
            return self._total_arrivals
        
        def signature(self) -> str:
            return f"VAR_BARRIER arrived_count=@{self._var_arrived_count.handle_name} block_state=@{self._var_block_state.handle_name} total_arrivals={self._total_arrivals}"
        
    class MEM_INIT(_Base):
        def __init__(self, ptr: Pointer, size: int):
            self._ptr = ptr
            self._size = size
            
        @property
        def ptr(self) -> Pointer:
            return self._ptr
        
        @property
        def size(self) -> int:
            return self._size
        
        def signature(self) -> str:
            return f"MEM_INIT MEM@{self._ptr.addr} size={self._size}"
    
    class TILE_LOAD(_Base):
        def __init__(self, tile_sig: TileSignature):
            self._tile_sig = tile_sig
            
            if self._tile_sig.spm_ptr is None:
                raise RuntimeError("Tile SPM pointer is not assigned.")
            
            self._broadcast_dst_ptrs: list[Pointer] = []
            
        def add_broadcast_dst_ptr(self, dst_ptr: Pointer):
            self._broadcast_dst_ptrs.append(dst_ptr)
            
        @property
        def tile_sig(self) -> TileSignature:
            return self._tile_sig
        
        @property
        def broadcast_dst_ptrs(self) -> list[Pointer]:
            return self._broadcast_dst_ptrs
        
        def signature(self) -> str:
            broadcast_info = ""
            if len(self._broadcast_dst_ptrs) > 0:
                broadcast_info = f" [BROADCAST {', '.join([f'@{ptr.addr}' for ptr in self._broadcast_dst_ptrs])}]"
            return f"LOAD {self._tile_sig.signature} -> SPM@{self._tile_sig.spm_ptr.addr} {broadcast_info}"
        
    class COLLECTIVE_TILE_LOAD(_Base):
        def __init__(self, collective_tile_sig: CollectiveTileSignature):
            super().__init__()
            self._collective_tile_sig = collective_tile_sig
            
            if self._collective_tile_sig.spm_ptr is None:
                raise RuntimeError("Collective Tile SPM pointer is not assigned.")
            
        @property
        def collective_tile_sig(self) -> CollectiveTileSignature:
            return self._collective_tile_sig
        
        def signature(self):
            return f"LOAD {self._collective_tile_sig.signature} -> SPM@{self._collective_tile_sig.spm_ptr.addr}"
            
    class TILE_STORE(_Base):
        def __init__(self, tile_sig: TileSignature):
            self._tile_sig = tile_sig
            
            if self._tile_sig.spm_ptr is None:
                raise RuntimeError("Tile SPM pointer is not assigned.")
            
        @property
        def tile_sig(self) -> TileSignature:
            return self._tile_sig
        
        def signature(self) -> str:
            return f"STORE SPM@{self._tile_sig.spm_ptr.addr} -> {self._tile_sig.signature}"
            
    class TILED_OP(_Base):
        def __init__(self, op_sig: TiledOperatorSignature, inner_op_idx: int):
            self._op_sig = op_sig
            self._inner_op_idx = inner_op_idx
            
            self._pipelined_cmds: list[CompiledCommand.TILED_OP] = []
            
        @property
        def op_sig(self) -> TiledOperatorSignature:
            return self._op_sig
        
        @property
        def inner_op_idx(self) -> int:
            return self._inner_op_idx
        
        def add_pipelined_cmd(self, cmd: 'CompiledCommand.TILED_OP'):
            self._pipelined_cmds.append(cmd)
        
        def signature(self) -> str:
            tin_signature = lambda tin: tin.signature + ("" if tin.spm_ptr is None else f'@{tin.spm_ptr.addr}')
            o_tile_signature = self.op_sig.o_tile.signature + ("" if self.op_sig.o_tile.spm_ptr is None else f'@{self.op_sig.o_tile.spm_ptr.addr}')
            if self._inner_op_idx < len(self.op_sig.i_tiles) - 1:
                o_tile_signature = o_tile_signature + " (partial)"
            return f"OP {[tin_signature(tin) for tin in self.op_sig.i_tiles[self._inner_op_idx]]} -> {o_tile_signature} [inner_op_idx={self._inner_op_idx}]"
        
class CompiledStage:
    def __init__(self):
        self.preprocessings:  list[CompiledCommand._Base]   = []  # GROUP 0          : Preprocessing (e.g., barriers) -> collection of parallel commands
        self.dma_stores:      list[CompiledCommand._Base]   = []  # GROUP 1 THREAD 0 : Store output tiles from SPAD to memory
        self.dma_loads:       list[CompiledCommand._Base]   = []  # GROUP 1 THREAD 1 : Load input tiles from memory to SPAD 
        self.compute_ops:     list[CompiledCommand._Base]   = []  # GROUP 1 THREAD 2 : Compute tiled operations in SPAD
        self.postprocessings: list[CompiledCommand._Base]   = []  # GROUP 2          : Postprocessing (e.g., barriers) -> collection of parallel commands
        
    def summary(self) -> dict[str, list[str]]:
        return {
            "preprocessings":  [cmd.signature() for cmd in self.preprocessings],
            "dma_stores":      [cmd.signature() for cmd in self.dma_stores],
            "dma_loads":       [cmd.signature() for cmd in self.dma_loads],
            "compute_ops":     [cmd.signature() for cmd in self.compute_ops],
            "postprocessings": [cmd.signature() for cmd in self.postprocessings],
        }
        
class CompiledOperator:
    def __init__(self, stages: Sequence[CompiledStage], var_globals: dict[str, VariableHandle]=None):
        self.stages = list(stages)
        self._var_globals = var_globals if var_globals is not None else {}
        
    @staticmethod
    def from_tiled_ops(
        spad_ld_pp_ptrs: tuple[Pointer, Pointer], 
        spad_st_pp_ptrs: tuple[Pointer, Pointer], 
        spad_ld_pp_size: int, 
        spad_st_pp_size: int, 
        tiled_ops: Sequence[TiledOperatorSignature],
        var_globals: dict[str, VariableHandle]=None,
    ) -> 'CompiledOperator':
        stages = [CompiledStage() for _ in range(3)]
        
        dma_load_stage   = stages[0]
        compute_stage    = stages[1]
        dma_store_stage  = stages[2]
        
        spad_pp_idx = 0
        spad_pp_usage = [0, 0]
        spad_pp_ops = 0
        cached_i_tiles: dict[str, TileSignature] = {}
        
        for cursor in range(len(tiled_ops)):
            # make a copy to avoid modifying the original
            op = tiled_ops[cursor].copy()  
            
            # # NEW!: reorder input tiles for better reuse
            # def i_tile_reorder_key(i_tile: Sequence[TileSignature]):
            #     total_coords = []
            #     for tin in i_tile:
            #         if isinstance(tin, CollectiveTileSignature):
            #             total_coords.extend((0, 0, 0, 0))
            #         elif isinstance(tin, TileSignature):
            #             total_coords.extend(tin.coords)
            #         else:
            #             raise RuntimeError("Unsupported tile signature type.")
            #     return tuple(total_coords)
            # op.i_tiles = tuple(sorted(list(op.i_tiles), key=i_tile_reorder_key))
            
            for inner_op_idx in range(len(op.i_tiles)):
                total_ld_spad_usage = sum([tile.buf.tile_size for tile in op.i_tiles[inner_op_idx]])
                total_st_spad_usage = op.o_tile.buf.tile_size
                
                if (spad_ld_pp_size < spad_pp_usage[0] + total_ld_spad_usage) or (spad_st_pp_size < spad_pp_usage[1] + total_st_spad_usage):
                    # switch ping-pong buffer
                    spad_pp_idx = 1 - spad_pp_idx
                    spad_pp_usage = [0, 0]
                    spad_pp_ops   = 0
                    cached_i_tiles.clear()
                    
                    # create new DMA load and compute pipeline stages
                    dma_load_stage  = compute_stage
                    compute_stage   = dma_store_stage
                    dma_store_stage = CompiledStage()
                    
                    stages.append(dma_store_stage)

                for tin in op.i_tiles[inner_op_idx]:
                    if isinstance(tin, CollectiveTileSignature):
                        tile_buf = tin.buf
                        tile_size = tile_buf.tile_size
                        
                        tin.spm_ptr = spad_ld_pp_ptrs[spad_pp_idx] + spad_pp_usage[0]
                        
                        dma_load_stage.dma_loads.append(
                            CompiledCommand.COLLECTIVE_TILE_LOAD(
                                collective_tile_sig=tin,
                            )
                        )
                        
                        spad_pp_usage[0] += tile_size
                        if spad_pp_usage[0] > spad_ld_pp_size:
                            logger.warning(f"SPAD LOAD ping-pong buffer size exceeded: current usage={spad_pp_usage[0]}, size={spad_ld_pp_size}")
                            logger.warning(f"Consider reducing the tile size or increasing the SPAD STORE ping-pong buffer size.")
                            raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
                            
                    elif isinstance(tin, TileSignature):
                        if tin.signature in cached_i_tiles:
                            tin.spm_ptr = cached_i_tiles[tin.signature].spm_ptr  # reuse cached tile
                        else:
                            tile_buf = tin.buf
                            tile_size = tile_buf.tile_size
                            
                            tin.spm_ptr = spad_ld_pp_ptrs[spad_pp_idx] + spad_pp_usage[0]
                            cached_i_tiles[tin.signature] = tin
                        
                            dma_load_stage.dma_loads.append(
                                CompiledCommand.TILE_LOAD(
                                    tile_sig=tin,
                                )
                            )
                        
                            spad_pp_usage[0] += tile_size
                            if spad_pp_usage[0] > spad_ld_pp_size:
                                logger.warning(f"SPAD LOAD ping-pong buffer size exceeded: current usage={spad_pp_usage[0]}, size={spad_ld_pp_size}")
                                logger.warning(f"Consider reducing the tile size or increasing the SPAD STORE ping-pong buffer size.")
                                raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
                    else:
                        raise RuntimeError("Unsupported tile signature type.")
            
                compute_stage.compute_ops.append(
                    CompiledCommand.TILED_OP(
                        op_sig=op,
                        inner_op_idx=inner_op_idx
                    )
                )
                
                if inner_op_idx == len(op.i_tiles) - 1:
                    tout = op.o_tile
                    tile_buf = tout.buf
                    tile_size = tile_buf.tile_size
                    
                    tout.spm_ptr = spad_st_pp_ptrs[spad_pp_idx] + spad_pp_usage[1]
                    
                    dma_store_stage.dma_stores.append(
                        CompiledCommand.TILE_STORE(
                            tile_sig=tout
                        )
                    )
                    
                    spad_pp_usage[1] += tile_size
                    if spad_pp_usage[1] > spad_st_pp_size:
                        logger.warning(f"SPAD STORE ping-pong buffer size exceeded: current usage={spad_pp_usage[1]}, size={spad_st_pp_size}")
                        logger.warning(f"Consider reducing the tile size or increasing the SPAD STORE ping-pong buffer size.")
                        raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
                
                spad_pp_ops += 1
                    
        return CompiledOperator(stages=stages, var_globals=var_globals)
    
    @property
    def var_globals(self) -> dict[str, VariableHandle]:
        return self._var_globals
    
    def summary(self) -> list[dict[str, list[str]]]:
        return [stage.summary() for stage in self.stages]

class CompiledMapping:
    def __init__(self, mapping: TiledOperatorMapping, operators: dict[int, CompiledOperator], var_globals: dict[int, dict[str, VariableHandle]]):
        self.mapping   = mapping
        self.operators = operators
        self.var_globals = var_globals
        
    @staticmethod
    def from_tiled_op_mapping(spad_ld_pp_ptrs: dict[int, tuple[Pointer, Pointer]], spad_st_pp_ptrs: dict[int, tuple[Pointer, Pointer]], spad_ld_pp_size: int, spad_st_pp_size: int, mapping: TiledOperatorMapping) -> 'CompiledMapping':
        core_group = mapping.core_group
        operators: dict[int, CompiledOperator] = {}
        var_globals: dict[int, dict[str, VariableHandle]] = {}
        
        for core_id in core_group:
            if core_id not in spad_ld_pp_ptrs or core_id not in spad_st_pp_ptrs:
                raise RuntimeError(f"SPAD ping-pong pointers not provided for core {core_id}.")
            
            var_globals[core_id] = {}
            var_globals[core_id][BCAST_BARRIER_ARRIVED_CNT] = VariableHandle(handle_name=f"{BCAST_BARRIER_ARRIVED_CNT}_{core_id}", initial_value=0)
            var_globals[core_id][BCAST_BARRIER_BLOCK_STATE] = VariableHandle(handle_name=f"{BCAST_BARRIER_BLOCK_STATE}_{core_id}", initial_value=0)
            var_globals[core_id][PIPE_BARRIER_ARRIVED_CNT]  = VariableHandle(handle_name=f"{PIPE_BARRIER_ARRIVED_CNT}_{core_id}",  initial_value=0)
            var_globals[core_id][PIPE_BARRIER_BLOCK_STATE]  = VariableHandle(handle_name=f"{PIPE_BARRIER_BLOCK_STATE}_{core_id}",  initial_value=0)
            
            p = CompiledOperator.from_tiled_ops(
                spad_ld_pp_ptrs=spad_ld_pp_ptrs[core_id],
                spad_st_pp_ptrs=spad_st_pp_ptrs[core_id],
                spad_ld_pp_size=spad_ld_pp_size,
                spad_st_pp_size=spad_st_pp_size,
                tiled_ops=mapping[core_id],
                var_globals=var_globals[core_id],
            )
            operators[core_id] = p
            
        return CompiledMapping(mapping=mapping, operators=operators, var_globals=var_globals)
    
    def apply_broadcast_optimization(self, buf_targets: list[str]=None, n_max_bcast_bursts: int=4):
        n_stages = max([len(op.stages) for op in self.operators.values()])
        core_ids = list(self.operators.keys())
        
        barrier_target_stages: list[int] = []
        
        for stage_idx in range(n_stages):
            current_bcast_targets: dict[tuple[str, tuple[int, ...]], list[tuple[int, CompiledOperator, CompiledCommand.TILE_LOAD, int, TileSignature]]] = {}
            cached_bcast_targets: list[list[tuple[int, CompiledOperator, CompiledCommand.TILE_LOAD, int, TileSignature]]] = []
            
            for core_id in core_ids:
                op = self.operators[core_id]
                
                if stage_idx >= len(op.stages):
                    continue
                
                load_cmds = op.stages[stage_idx].dma_loads
                
                for cmd_idx, cmd in enumerate(load_cmds):
                    if not isinstance(cmd, CompiledCommand.TILE_LOAD):
                        continue
                    
                    if buf_targets is not None and cmd.tile_sig.buf_name not in buf_targets:
                        continue
                    
                    key = (cmd.tile_sig.buf_name, cmd.tile_sig.coords)
                    
                    if key not in current_bcast_targets:
                        current_bcast_targets[key] = [(core_id, op, cmd, cmd_idx, cmd.tile_sig)]
                    elif len(current_bcast_targets[key]) >= n_max_bcast_bursts:
                        cached_bcast_targets.append(current_bcast_targets[key])
                        current_bcast_targets[key] = [(core_id, op, cmd, cmd_idx, cmd.tile_sig)]
                    else:
                        current_bcast_targets[key].append((core_id, op, cmd, cmd_idx, cmd.tile_sig))
            
            load_balance_cnt: dict[int, int] = {core_id: 0 for core_id in core_ids}
            
            for target_list in current_bcast_targets.values():
                cached_bcast_targets.append(target_list)
                    
            for target_list in cached_bcast_targets:
                if len(target_list) > 1:
                    src_core_id, src_op, src_cmd, src_cmd_idx, src_tile_sig = min(target_list, key=lambda item: load_balance_cnt[item[0]])
                    load_balance_cnt[src_core_id] += len(target_list) * src_tile_sig.buf.tile_size
                    
                    for dst_core_id, dst_op, dst_cmd, dst_cmd_idx, dst_tile_sig in target_list:
                        if src_core_id == dst_core_id:
                            continue
                        
                        src_cmd.add_broadcast_dst_ptr(dst_ptr=dst_cmd.tile_sig.spm_ptr)
                        dst_op.stages[stage_idx].dma_loads[dst_cmd_idx] = CompiledCommand.NOP()
                    
                    if stage_idx not in barrier_target_stages:
                        barrier_target_stages.append(stage_idx)
        
        for stage_idx in barrier_target_stages:
            target_core_ids = core_ids
            master_core_id = target_core_ids[0]
            
            thread_count = sum([1 for core_id in target_core_ids if stage_idx < len(self.operators[core_id].stages)])
            arrived_count = self.operators[master_core_id].var_globals[BCAST_BARRIER_ARRIVED_CNT]
            barrier_state = self.operators[master_core_id].var_globals[BCAST_BARRIER_BLOCK_STATE]
            
            for target_core_id in target_core_ids:
                target_op = self.operators[target_core_id]
                if stage_idx >= len(target_op.stages):
                    continue
                target_op.stages[stage_idx].postprocessings.append(CompiledCommand.VAR_BARRIER(var_arrived_count=arrived_count, var_block_state=barrier_state, total_arrivals=thread_count))
                                
        return self
    
    def apply_pipeline_optimization(self, dst_mapping: 'CompiledMapping', src_buf_name: str, dst_buf_name: str):
        dst_n_stages = max([len(op.stages) for op in dst_mapping.operators.values()])
        src_n_stages = max([len(op.stages) for op in self.operators.values()])
        
        dst_core_ids = list(dst_mapping.operators.keys())
        src_core_ids = list(self.operators.keys())
        
        src_coords_history: set[tuple[int, ...]] = set()
        src_stage_idx = -1
        
        dst_src_barrier: list[tuple[int, int]] = []
        
        for dst_stage_idx in range(dst_n_stages):
            dst_required_coords: set[tuple[int, ...]] = set()
            
            for dst_core_id, dst_op in dst_mapping.operators.items():
                if dst_stage_idx >= len(dst_op.stages):
                    continue
                
                load_cmds = dst_op.stages[dst_stage_idx].dma_loads
                
                for cmd in load_cmds:
                    if not isinstance(cmd, (CompiledCommand.TILE_LOAD, CompiledCommand.COLLECTIVE_TILE_LOAD)):
                        continue
                    
                    if cmd.tile_sig.buf_name != dst_buf_name:
                        continue
                    
                    if isinstance(cmd, CompiledCommand.TILE_LOAD):
                        dst_required_coords.add(cmd.tile_sig.coords)
                    elif isinstance(cmd, CompiledCommand.COLLECTIVE_TILE_LOAD):
                        for src_tile in cmd.collective_tile_sig.src_tiles:
                            dst_required_coords.add(src_tile.coords)
                    
            if dst_required_coords.issubset(src_coords_history):
                continue  # all required tiles are already available from previous stages
            
            while src_stage_idx < src_n_stages:
                src_stage_idx += 1
                
                for src_core_id, src_op in self.operators.items():
                    if src_stage_idx >= len(src_op.stages):
                        continue
                    
                    store_cmds = src_op.stages[src_stage_idx].dma_stores
                    
                    for cmd in store_cmds:
                        if not isinstance(cmd, CompiledCommand.TILE_STORE):
                            continue
                        
                        if cmd.tile_sig.buf_name != src_buf_name:
                            continue
                        
                        src_coords_history.add(cmd.tile_sig.coords)
                    
                if dst_required_coords.issubset(src_coords_history):
                    break  # all required tiles are now available
            
            dst_src_barrier.append((dst_stage_idx, src_stage_idx))
        
        for dst_stage_idx, src_stage_idx in dst_src_barrier:   
            master_core_id = dst_core_ids[0]
                
            # thread_count = len(dst_core_ids) + len(src_core_ids)
            thread_count = sum([1 for core_id in dst_core_ids if dst_stage_idx < len(dst_mapping.operators[core_id].stages)]) + sum([1 for core_id in src_core_ids if src_stage_idx < len(self.operators[core_id].stages)])
            arrived_count = dst_mapping.operators[master_core_id].var_globals[PIPE_BARRIER_ARRIVED_CNT]
            barrier_state = dst_mapping.operators[master_core_id].var_globals[PIPE_BARRIER_BLOCK_STATE]
            
            for target_core_id in dst_core_ids:
                target_op = dst_mapping.operators[target_core_id]
                target_op.stages[dst_stage_idx].preprocessings.append(CompiledCommand.VAR_BARRIER(var_arrived_count=arrived_count, var_block_state=barrier_state, total_arrivals=thread_count))
                    
            for target_core_id in src_core_ids:
                target_op = self.operators[target_core_id]
                target_op.stages[src_stage_idx].postprocessings.append(CompiledCommand.VAR_BARRIER(var_arrived_count=arrived_count, var_block_state=barrier_state, total_arrivals=thread_count))
        
        return self
    
    def summary(self) -> list[dict[str, list[str]]]:
        return {
            i: p.summary() 
            for i, p in self.operators.items()
        }


class MCA_OperatorMapper:
    OUTPUT_STATIONARY = "output_stationary"
    ROUND_ROBIN       = "round_robin"
    CONTIGUOUS        = "contiguous"
    
    def __init__(
        self,
        
        core_group: MCA_CoreGroup,
        spad_ld_mem_space: MCA_L1MemorySpace,
        spad_st_mem_space: MCA_L1MemorySpace,
        
        tiled_ops: Sequence[TiledOperatorSignature],
    ):
        self._core_group = core_group
        self._tiled_ops = tiled_ops
        
        self._spad_ld_mem_space = spad_ld_mem_space
        self._spad_ld_pp_size = self._spad_ld_mem_space.size_per_owner // 2  # ping-pong buffer size per core for load
        self._spad_ld_pp_ptrs = {
            core_id: (
                self._spad_ld_mem_space.allocate(core_id, self._spad_ld_pp_size),
                self._spad_ld_mem_space.allocate(core_id, self._spad_ld_pp_size),
            )
            for core_id in self._core_group
        }
        
        self._spad_st_mem_space = spad_st_mem_space
        self._spad_st_pp_size = self._spad_st_mem_space.size_per_owner // 2  # ping-pong buffer size per core for store
        self._spad_st_pp_ptrs = {
            core_id: (
                self._spad_st_mem_space.allocate(core_id, self._spad_st_pp_size),
                self._spad_st_mem_space.allocate(core_id, self._spad_st_pp_size),
            )
            for core_id in self._core_group
        }
        
    @staticmethod
    def LINEAR(core_group: MCA_CoreGroup, spad_ld_mem_space: MCA_L1MemorySpace, spad_st_mem_space: MCA_L1MemorySpace, ifm: MCA_TensorBuffer, wgt: MCA_TensorBuffer, bias: MCA_TensorBuffer, ofm: MCA_TensorBuffer) -> 'MCA_OperatorMapper':
        ifm_shape = ifm.shape
        wgt_shape = wgt.shape
        bias_shape = bias.shape
        ofm_shape = ofm.shape
        
        if ifm_shape[0] != ofm_shape[0]:
            raise Exception(f"IFM and OFM batch size mismatch: {ifm_shape[0]} != {ofm_shape[0]}")
        if wgt_shape[0] != ofm_shape[1] or wgt_shape[0] != bias_shape[1]:
            raise Exception(f"WGT and OFM channel size mismatch: {wgt_shape[0]} != {ofm_shape[1]} != {bias_shape[1]}")
        if wgt_shape[1] != ifm_shape[1]:
            raise Exception(f"WGT and IFM feature size mismatch: {wgt_shape[1]} != {ifm_shape[1]}")
        
        ifm_shard_grid = ifm.shard_grid
        wgt_shard_grid = wgt.shard_grid
        bias_shard_grid = bias.shard_grid
        ofm_shard_grid = ofm.shard_grid
        
        if ifm_shard_grid[0] != ofm_shard_grid[0]:
            raise Exception(f"IFM and OFM shard grid batch size mismatch: {ifm_shard_grid[0]} != {ofm_shard_grid[0]}")
        if wgt_shard_grid[0] != ofm_shard_grid[1] or wgt_shard_grid[0] != bias_shard_grid[1]:
            raise Exception(f"WGT and OFM shard grid channel size mismatch: {wgt_shard_grid[0]} != {ofm_shard_grid[1]} != {bias_shard_grid[1]}")
        if wgt_shard_grid[1] != ifm_shard_grid[1]:
            raise Exception(f"WGT and IFM shard grid feature size mismatch: {wgt_shard_grid[1]} != {ifm_shard_grid[1]}")
        
        ifm_tile_shape = ifm.tile_shape
        wgt_tile_shape = wgt.tile_shape
        bias_tile_shape = bias.tile_shape
        ofm_tile_shape = ofm.tile_shape
        
        if ifm_tile_shape[0] != ofm_tile_shape[0]:
            raise Exception(f"IFM and OFM tile shape batch size mismatch: {ifm_tile_shape[0]} != {ofm_tile_shape[0]}")
        if wgt_tile_shape[0] != ofm_tile_shape[1] or wgt_tile_shape[0] != bias_tile_shape[1]:
            raise Exception(f"WGT and OFM tile shape channel size mismatch: {wgt_tile_shape[0]} != {ofm_tile_shape[1]} != {bias_tile_shape[1]}")
        if wgt_tile_shape[1] != ifm_tile_shape[1]:
            raise Exception(f"WGT and IFM tile shape feature size mismatch: {wgt_tile_shape[1]} != {ifm_tile_shape[1]}")
        
        ifm_tiles: dict[tuple[int, ...], TileSignature] = {
            (m_s, k_s, m_t, k_t): TileSignature("ifm", ifm, m_s, k_s, m_t, k_t)
            for m_s in range(ifm.shard_grid[0])
            for k_s in range(ifm.shard_grid[1])
            for m_t in range(ifm.tile_grid_per_shard[0])
            for k_t in range(ifm.tile_grid_per_shard[1])
        }
        
        wgt_tiles: dict[tuple[int, ...], TileSignature] = {
            (n_s, k_s, n_t, k_t): TileSignature("wgt", wgt, n_s, k_s, n_t, k_t)
            for n_s in range(wgt.shard_grid[0])
            for k_s in range(wgt.shard_grid[1])
            for n_t in range(wgt.tile_grid_per_shard[0])
            for k_t in range(wgt.tile_grid_per_shard[1])
        }
        
        bias_tiles: dict[tuple[int, ...], TileSignature] = {
            (0, n_s, 0, n_t): TileSignature("bias", bias, 0, n_s, 0, n_t)
            for n_s in range(bias.shard_grid[1])
            for n_t in range(bias.tile_grid_per_shard[1])
        }
        
        ofm_tiles: dict[tuple[int, ...], TileSignature] = {
            (m_s, n_s, m_t, n_t): TileSignature("ofm", ofm, m_s, n_s, m_t, n_t)
            for m_s in range(ofm.shard_grid[0])
            for n_s in range(ofm.shard_grid[1])
            for m_t in range(ofm.tile_grid_per_shard[0])
            for n_t in range(ofm.tile_grid_per_shard[1])
        }
        
        tiled_ops = [
            TiledOperatorSignature(
                i_tiles=[
                    (ifm_tiles[(m_s, k_s, m_t, k_t)], wgt_tiles[(n_s, k_s, n_t, k_t)], bias_tiles[(0, n_s, 0, n_t)]) 
                    for k_s in range(ifm.shard_grid[1]) 
                    for k_t in range(ifm.tile_grid_per_shard[1])
                ],
                o_tile=ofm_tiles[(m_s, n_s, m_t, n_t)]
            )
            for m_s in range(ofm.shard_grid[0])
            for n_s in range(ofm.shard_grid[1])
            for m_t in range(ofm.tile_grid_per_shard[0])
            for n_t in range(ofm.tile_grid_per_shard[1])
        ]
        
        return MCA_OperatorMapper(
            core_group=core_group,
            spad_ld_mem_space=spad_ld_mem_space,
            spad_st_mem_space=spad_st_mem_space,
            tiled_ops=tiled_ops,
        )
        
    @staticmethod
    def UNARY_INPLACE(core_group: MCA_CoreGroup, spad_ld_mem_space: MCA_L1MemorySpace, spad_st_mem_space: MCA_L1MemorySpace, ifm: MCA_TensorBuffer) -> 'MCA_OperatorMapper':
        ifm_tiles: dict[tuple[int, ...], TileSignature] = {
            (m_s, k_s, m_t, k_t): TileSignature("ifm", ifm, m_s, k_s, m_t, k_t)
            for m_s in range(ifm.shard_grid[0])
            for k_s in range(ifm.shard_grid[1])
            for m_t in range(ifm.tile_grid_per_shard[0])
            for k_t in range(ifm.tile_grid_per_shard[1])
        }
        
        tiled_ops = [
            TiledOperatorSignature(
                i_tiles=[(ifm_tiles[(m_s, k_s, m_t, k_t)],) ],
                o_tile=ifm_tiles[(m_s, k_s, m_t, k_t)]
            )
            for m_s in range(ifm.shard_grid[0])
            for k_s in range(ifm.shard_grid[1])
            for m_t in range(ifm.tile_grid_per_shard[0])
            for k_t in range(ifm.tile_grid_per_shard[1])
        ]
        
        return MCA_OperatorMapper(
            core_group=core_group,
            spad_ld_mem_space=spad_ld_mem_space,
            spad_st_mem_space=spad_st_mem_space,
            tiled_ops=tiled_ops,
        )
        
    @staticmethod
    def CONV2D(
        core_group: MCA_CoreGroup, spad_ld_mem_space: MCA_L1MemorySpace, spad_st_mem_space: MCA_L1MemorySpace, 
        ifm: MCA_TensorBuffer, ofm: MCA_TensorBuffer, 
        stride: Sequence[int], padding: Sequence[int], dilation: Sequence[int], groups: int=1,
        
        # additional tensors and parameters for convolution and pooling
        wgt: MCA_TensorBuffer=None, bias: MCA_TensorBuffer=None,
        window: Sequence[int]=None,
        
        # optional argument to indicate whether to use collective tile load or not
        use_collective_tile_load: bool=False,
    ) -> 'MCA_OperatorMapper':
        is_conv2d = True
        if wgt is None or bias is None:
            if window is None:
                raise Exception("Either WGT and BIAS or WINDOW must be provided for CONV2D operator mapping.")
            is_conv2d = False
        
        if isinstance(stride, int):     stride   = (stride, stride)
        if isinstance(padding, int):    padding  = (padding, padding)
        if isinstance(dilation, int):   dilation = (dilation, dilation)
        
        if len(stride) != 2:
            raise Exception(f"Stride must be an integer or a tuple of two integers, but got: {stride}")
        if len(padding) != 2:
            raise Exception(f"Padding must be an integer or a tuple of two integers, but got: {padding}")
        if len(dilation) != 2:
            raise Exception(f"Dilation must be an integer or a tuple of two integers, but got: {dilation}")
        
        N, H, W, C = ifm.shape
        if is_conv2d:
            FH, FW, K, GC = wgt.shape
        else:
            FH, FW = window
            K = C 
            
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        if is_conv2d:
            if C != ifm.shape[3]:
                raise Exception(f"Input channel mismatch between IFM and WGT: {ifm.shape[3]} != {C}")
            if K != bias.shape[1] != ofm.shape[3]:
                raise Exception(f"Output channel mismatch between WGT, BIAS and OFM: {K} != {bias.shape[1]} != {ofm.shape[3]}")
            if N != ofm.shape[0]:
                raise Exception(f"Batch size mismatch between IFM and OFM: {N} != {ofm.shape[0]}")
            if OH != ofm.shape[1]:
                raise Exception(f"Output height mismatch between computed and OFM: {OH} != {ofm.shape[1]}")
            if OW != ofm.shape[2]:
                raise Exception(f"Output width mismatch between computed and OFM: {OW} != {ofm.shape[2]}")
            if GC * groups != C:
                raise Exception(f"Input channel mismatch between IFM and WGT groups: {GC * groups} != {C}")
        else:
            if C != ifm.shape[3]:
                raise Exception(f"Input channel mismatch between IFM and WGT: {ifm.shape[3]} != {C}")
            if N != ofm.shape[0]:
                raise Exception(f"Batch size mismatch between IFM and OFM: {N} != {ofm.shape[0]}")
            if OH != ofm.shape[1]:
                raise Exception(f"Output height mismatch between computed and OFM: {OH} != {ofm.shape[1]}")
            if OW != ofm.shape[2]:
                raise Exception(f"Output width mismatch between computed and OFM: {OW} != {ofm.shape[2]}")
        
        if is_conv2d:
            ifm_tile_shape = ifm.tile_shape
            wgt_tile_shape = wgt.tile_shape
            bias_tile_shape = bias.tile_shape
            ofm_tile_shape = ofm.tile_shape
            
            if ifm_tile_shape[0] != ofm_tile_shape[0]:
                raise Exception(f"IFM and OFM tile shape batch size mismatch: {ifm_tile_shape[0]} != {ofm_tile_shape[0]}")
            if wgt_tile_shape[0] != ofm_tile_shape[1] or wgt_tile_shape[0] != bias_tile_shape[1]:
                raise Exception(f"WGT and OFM tile shape channel size mismatch: {wgt_tile_shape[0]} != {ofm_tile_shape[1]} != {bias_tile_shape[1]}")
            if wgt_tile_shape[1] != ifm_tile_shape[1]:
                raise Exception(f"WGT and IFM tile shape feature size mismatch: {wgt_tile_shape[1]} != {ifm_tile_shape[1]}")
        
            ifm_y_outer_shards, ifm_y_inner_shards, ifm_x_shards = (ifm.n_outer_shards, *ifm.shard_grid)    # NH,  NHW,  C
            wgt_y_outer_shards, wgt_y_inner_shards, wgt_x_shards = (wgt.n_outer_shards, *wgt.shard_grid)    # FHW, FHWK, C
            bias_x_shards = bias.shard_grid[-1]                                                             # K
            ofm_y_outer_shards, ofm_y_inner_shards, ofm_x_shards = (ofm.n_outer_shards, *ofm.shard_grid)    # NOH, NOHW, K
            
            if ifm_x_shards // groups != wgt_x_shards:
                raise Exception(f"IFM and WGT shard grid feature size mismatch in input channel C dimension: {ifm_x_shards} != {wgt_x_shards}")
            if ofm_x_shards != bias_x_shards or ofm_x_shards != (wgt_y_inner_shards // wgt_y_outer_shards):
                raise Exception(f"OFM, BIAS and WGT shard grid channel size mismatch in output channel K dimension: {ofm_x_shards} != {bias_x_shards} != {wgt_y_inner_shards // wgt_y_outer_shards}")
            
        else:
            ifm_tile_shape = ifm.tile_shape
            ofm_tile_shape = ofm.tile_shape
            
            if ifm_tile_shape[0] != ofm_tile_shape[0]:
                raise Exception(f"IFM and OFM tile shape batch size mismatch: {ifm_tile_shape[0]} != {ofm_tile_shape[0]}")
            if ofm_tile_shape[1] != ofm_tile_shape[1]:
                raise Exception(f"OFM tile shape channel size mismatch: {ofm_tile_shape[1]} != {ofm_tile_shape[1]}")

            ifm_y_outer_shards, ifm_y_inner_shards, ifm_x_shards = (ifm.n_outer_shards, *ifm.shard_grid)    # NH,  NHW,  C
            ofm_y_outer_shards, ofm_y_inner_shards, ofm_x_shards = (ofm.n_outer_shards, *ofm.shard_grid)    # NOH, NOHW, C
            
            if ifm_x_shards != ofm_x_shards:
                raise Exception(f"IFM and OFM shard grid feature size mismatch in input channel C dimension: {ifm_x_shards} != {ofm_x_shards}")
            
        if is_conv2d:
            wgt_tiles: dict[tuple[int, ...], TileSignature] = {
                (y_s, x_s, y_t, x_t): TileSignature("wgt", wgt, y_s, x_s, y_t, x_t)
                for y_s in range(wgt.shard_grid[0])
                for x_s in range(wgt.shard_grid[1])
                for y_t in range(wgt.tile_grid_per_shard[0])
                for x_t in range(wgt.tile_grid_per_shard[1])
            }
            
            bias_tiles: dict[tuple[int, ...], TileSignature] = {
                (0, x_s, 0, x_t): TileSignature("bias", bias, 0, x_s, 0, x_t)
                for x_s in range(bias.shard_grid[1])
                for x_t in range(bias.tile_grid_per_shard[1])
            }
            
        ifm_tiles: dict[tuple[int, ...], TileSignature] = {
            (y_s, x_s, y_t, x_t): TileSignature("ifm", ifm, y_s, x_s, y_t, x_t)
            for y_s in range(ifm.shard_grid[0])
            for x_s in range(ifm.shard_grid[1])
            for y_t in range(ifm.tile_grid_per_shard[0])
            for x_t in range(ifm.tile_grid_per_shard[1])
        }
        
        ofm_tiles: dict[tuple[int, ...], TileSignature] = {
            (y_s, x_s, y_t, x_t): TileSignature("ofm", ofm, y_s, x_s, y_t, x_t)
            for y_s in range(ofm.shard_grid[0])
            for x_s in range(ofm.shard_grid[1])
            for y_t in range(ofm.tile_grid_per_shard[0])
            for x_t in range(ofm.tile_grid_per_shard[1])
        }
        
        tiled_ops: list[TiledOperatorSignature] = []
        
        OW_N_TILES = ofm.tile_grid[0] // (N * OH)
        K_N_TILES  = ofm.tile_grid[1]
        C_N_TILES  = ifm.tile_grid[1]
        W_N_TILES  = ifm.tile_grid[0] // (N * H)
        
        OW_N_TILES_PER_SHARD = ofm.tile_grid_per_shard[0]
        W_N_TILES_PER_SHARD  = ifm.tile_grid_per_shard[0]
        
        K_N_TILES_PER_GROUP = K_N_TILES // groups
        C_N_TILES_PER_GROUP = C_N_TILES // groups
        
        with tqdm.tqdm(total=N * OH * OW_N_TILES * groups * K_N_TILES_PER_GROUP, desc="generating tiled op for conv2d mapping", leave=False, disable=(not logger.is_current_debug_log_level())) as pbar:
            for n_it in range(N):  # NO TILING over batch dimension
                for oh_it in range(OH):  # NO TILING over output height dimension
                    for ow_tile_it in range(OW_N_TILES):  # TILING over output width (32 for Tenstorrent) & not aligned with input width dimension W
                        for group_it in range(groups):  # TILING over groups
                            for k_tile_per_group_it in range(K_N_TILES_PER_GROUP):  # TILING over output channel (32 for Tenstorrent)
                                k_tile_it = group_it * K_N_TILES_PER_GROUP + k_tile_per_group_it
                                
                                tiled_op = TiledOperatorSignature(i_tiles=[], o_tile=None)
                                
                                # GET OFM tile signature
                                ofm_y_tile_idx = n_it * OH * OW_N_TILES + oh_it * OW_N_TILES + ow_tile_it
                                ofm_x_tile_idx = k_tile_it
                                ofm_tile_idx = ofm.get_shard_grid_from_tile_grid_idx(ofm_y_tile_idx, ofm_x_tile_idx)
                                
                                tiled_op.o_tile = ofm_tiles[ofm_tile_idx]
                                tiled_op.i_tiles = []
                                tiled_op.op_kwargs = {"ifm_load_kwargs": []}

                                for fh_it in range(FH):  # NO TILING over filter height dimension
                                    for fw_it in range(FW):  # NO TILING over filter width dimension
                                        for c_tile_per_group_it in range(C_N_TILES_PER_GROUP):  # TILING over input channel (32 for Tenstorrent)
                                            c_tile_it = group_it * C_N_TILES_PER_GROUP + c_tile_per_group_it
                                            
                                            if not is_conv2d and c_tile_it != 0:
                                                continue  # for pooling, only one input channel tile
                                            
                                            # IFM width and height dimension calculation
                                            h_it = oh_it * stride[0] - padding[0] + fh_it * dilation[0]   # output height to input height
                                            if h_it < 0 or h_it >= H:
                                                continue  # skip invalid output height indices due to padding
                                            
                                            ow_shard_it         = ow_tile_it // OW_N_TILES_PER_SHARD    # OW shard iterator: which OW shard the current OW tile belongs to
                                            ow_rem_tile_it      = ow_tile_it % OW_N_TILES_PER_SHARD     # OW tile iterator within the current OW shard
                                            ow_actual_tile_size = min(ofm.tile_shape[0], OW - (ow_shard_it * ofm.shard_shape[0] + ow_rem_tile_it * ofm.tile_shape[0]))              # actual OW tile size (may be smaller than ofm.tile_shape[0] at the boundary)
                                            ow_stick_idx_vec    = ow_shard_it * ofm.shard_shape[0] + ow_rem_tile_it * ofm.tile_shape[0] + torch.arange(0, ow_actual_tile_size, 1)   # stick indices in OW dimension (with respect to the actual OFM tile size)
                                            w_stick_idx_vec     = ow_stick_idx_vec * stride[1] - padding[1] + fw_it * dilation[1]    # stick indices in W dimension (converted from OW indices)
                                            w_stick_valid_mask  = (w_stick_idx_vec >= 0) & (w_stick_idx_vec < W)                     # valid mask for W dimension sticks
                                            
                                            if is_conv2d:
                                                # GET WGT tile signature
                                                wgt_y_tile_idx = fh_it * FW * K_N_TILES + fw_it * K_N_TILES + k_tile_it
                                                wgt_x_tile_idx = c_tile_per_group_it
                                                wgt_tile_idx = wgt.get_shard_grid_from_tile_grid_idx(wgt_y_tile_idx, wgt_x_tile_idx)
                                                wgt_tile = wgt_tiles[wgt_tile_idx]
                                                
                                                # GET PSUM tile signature (BIAS or partially merged OFM)
                                                psum_tile_idx = bias.get_shard_grid_from_tile_grid_idx(0, k_tile_it)
                                                psum_tile = bias_tiles[psum_tile_idx]

                                            # GET IFM tile signature with halo memcpy patterns
                                            #   - Note that IFM tile indices depend on OFM tile indices due to stride, padding, dilation
                                            #   - IFM cannot directly obtained by the tile granularity, instead need to be calculated from each
                                            #     coordinate of the required IFM sticks (or channel-wise row vectors)
                                            #   - Halo regions are handled by masking invalid coordinates and additional metadata indicating
                                            #     the memory copy patterns before getting into the lowered Conv2d computation
                                            ifm_tile_idx_with_memcpy_pattern: dict[tuple[int, ...], dict[int, int]] = {}
                                            ifm_tile_idx_offset = n_it * H * W_N_TILES + h_it * W_N_TILES
                                            
                                            for w_stick_it, (w_stick_idx, w_stick_valid) in enumerate(zip(w_stick_idx_vec, w_stick_valid_mask)):
                                                if not w_stick_valid:
                                                    continue
                                                
                                                w_stick_shard_it            = w_stick_idx // ifm.shard_shape[0]             # W stick shard iterator: which W shard the current W stick belongs to
                                                w_stick_intra_shard_idx     = (w_stick_idx % ifm.shard_shape[0])            # W stick tile iterator within the current W shard
                                                w_stick_intra_shard_tile_it = w_stick_intra_shard_idx // ifm.tile_shape[0]  # W stick intra-shard tile iterator
                                                w_stick_intra_tile_offset   = w_stick_intra_shard_idx % ifm.tile_shape[0]   # W stick offset within the intra-shard tile
                                                
                                                ifm_y_tile_idx = int(ifm_tile_idx_offset + W_N_TILES_PER_SHARD * w_stick_shard_it + w_stick_intra_shard_tile_it)
                                                ifm_x_tile_idx = c_tile_it
                                                ifm_tile_idx = ifm.get_shard_grid_from_tile_grid_idx(ifm_y_tile_idx, ifm_x_tile_idx) 
                                                    
                                                ifm_stick_src_offset = w_stick_intra_tile_offset
                                                ifm_stick_dst_offset = w_stick_it
                                                
                                                if ifm_tile_idx not in ifm_tile_idx_with_memcpy_pattern:
                                                    ifm_tile_idx_with_memcpy_pattern[ifm_tile_idx] = {}
                                            
                                                ifm_tile_idx_with_memcpy_pattern[ifm_tile_idx][ifm_stick_dst_offset] = ifm_stick_src_offset
                                                
                                            if len(ifm_tile_idx_with_memcpy_pattern) == 0:
                                                continue
                                            
                                            uop_i_tiles = []
                                            uop_kwargs = {}
                                            
                                            if use_collective_tile_load:
                                                # DMA engine handles the collective tile load and memcpy patterns
                                                uop_kwargs["use_collective_tile_load"] = True
                                                ifm_tile_sig = CollectiveTileSignature(
                                                    buf_name="ifm",
                                                    buf=ifm,
                                                    src_tiles=[ifm_tiles[idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()],
                                                    memcpy_patterns=list(ifm_tile_idx_with_memcpy_pattern.values()),
                                                )
                                                uop_i_tiles.append(ifm_tile_sig)
                                            else:
                                                # Compute core uses all the individual IFM tiles and handles the memcpy patterns internally
                                                uop_kwargs["use_collective_tile_load"] = False
                                                uop_kwargs["ifm_tile_count"] = len(ifm_tile_idx_with_memcpy_pattern)
                                                uop_kwargs["memcpy_pattern"] = list(ifm_tile_idx_with_memcpy_pattern.values())
                                                uop_i_tiles.extend([ifm_tiles[idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()])
                                                    
                                            if is_conv2d:
                                                uop_i_tiles.append(wgt_tile)
                                                uop_i_tiles.append(psum_tile)
                                                
                                            tiled_op.i_tiles.append(tuple(uop_i_tiles))
                                            tiled_op.op_kwargs["ifm_load_kwargs"].append(uop_kwargs)
                        
                                # logger.debug(f"Generated CONV2D tiled_op for OFM tile idx {tiled_op.o_tile.coords} with {len(tiled_op.i_tiles)} uops")
                                tiled_ops.append(tiled_op)
                                pbar.update(1)
        
        logger.debug(f"mapper generated {len(tiled_ops)} in total")
                               
        return MCA_OperatorMapper(
            core_group=core_group,
            spad_ld_mem_space=spad_ld_mem_space,
            spad_st_mem_space=spad_st_mem_space,
            tiled_ops=tiled_ops,
        )
        
    def FLATTEN(
        core_group: MCA_CoreGroup, 
        spad_ld_mem_space: MCA_L1MemorySpace, 
        spad_st_mem_space: MCA_L1MemorySpace, 
        ifm: MCA_TensorBuffer, 
        ofm: MCA_TensorBuffer,
    ) -> 'MCA_OperatorMapper':
        if len(ofm.shape) != 2:
            raise Exception(f"OFM tensor must be 2D for TENSOR_FLATTEN operator mapping, but got shape: {ofm.shape}")
        if ifm.dtype != ofm.dtype:
            raise Exception(f"IFM and OFM dtype mismatch: {ifm.dtype} != {ofm.dtype}")
        if ifm.numel != ofm.numel:
            raise Exception(f"IFM and OFM number of elements mismatch: {ifm.numel} != {ofm.numel}")
        if ifm.tile_shape[-1] != ofm.tile_shape[-1]:
            raise Exception(f"IFM and OFM does not have the same row-wise tile size: {ifm.tile_shape[-1]} != {ofm.tile_shape[-1]}")
        
        ifm_tiles: dict[tuple[int, ...], TileSignature] = {
            (y_s, x_s, y_t, x_t): TileSignature("ifm", ifm, y_s, x_s, y_t, x_t)
            for y_s in range(ifm.shard_grid[0])
            for x_s in range(ifm.shard_grid[1])
            for y_t in range(ifm.tile_grid_per_shard[0])
            for x_t in range(ifm.tile_grid_per_shard[1])
        }
        
        ofm_tiles: dict[tuple[int, ...], TileSignature] = {
            (y_s, x_s, y_t, x_t): TileSignature("ofm", ofm, y_s, x_s, y_t, x_t)
            for y_s in range(ofm.shard_grid[0])
            for x_s in range(ofm.shard_grid[1])
            for y_t in range(ofm.tile_grid_per_shard[0])
            for x_t in range(ofm.tile_grid_per_shard[1])
        }
        
        IFM_W = ifm.shape[-1]
        
        OFM_H_TILES = ofm.tile_grid[0]
        OFM_W_TILES = ofm.tile_grid[1]
        
        OFM_H_TILES_PER_SHARD = ofm.tile_grid_per_shard[0]
        OFM_W_TILES_PER_SHARD = ofm.tile_grid_per_shard[1]
        
        IFM_H_STRIDE = functools.reduce(lambda x, y: x * y, ifm.shape[1:-1], 1)  # stride between two adjacent IFM H sticks in terms of number of elements
        
        tiled_ops = []
        
        for ofm_h_tile_it in range(OFM_H_TILES):
            for ofm_w_tile_it in range(OFM_W_TILES):
                ofm_tile_idx = ofm.get_shard_grid_from_tile_grid_idx(ofm_h_tile_it, ofm_w_tile_it)
                
                ofm_h_shard_it         = ofm_h_tile_it // OFM_H_TILES_PER_SHARD
                ofm_h_rem_tile_it      = ofm_h_tile_it % OFM_H_TILES_PER_SHARD 
                ofm_h_actual_tile_size = min(ofm.tile_shape[0], ofm.shard_shape[0] - ofm_h_rem_tile_it * ofm.tile_shape[0])

                ofm_w_shard_it         = ofm_w_tile_it // OFM_W_TILES_PER_SHARD
                ofm_w_rem_tile_it      = ofm_w_tile_it % OFM_W_TILES_PER_SHARD
                ofm_w_actual_offset    = ofm_w_shard_it * ofm.shard_shape[1] + ofm_w_rem_tile_it * ofm.tile_shape[1]

                ifm_h_actual_offset = (ofm_h_shard_it * ofm.shard_shape[0] * IFM_H_STRIDE) + ((ofm_w_actual_offset % (IFM_H_STRIDE * IFM_W)) // IFM_W)
                ifm_w_actual_offset = ofm_w_actual_offset % IFM_W

                ifm_tile_idx_with_memcpy_pattern: dict[tuple[int, ...], dict[int, int]] = {}
                
                for ofm_h_stick_it in range(ofm_h_actual_tile_size):
                    ifm_h_stick_idx = ifm_h_actual_offset + ofm_h_stick_it * IFM_H_STRIDE
                    
                    ifm_h_shard_it = ifm_h_stick_idx // ifm.shard_shape[0]
                    ifm_h_intra_shard_idx = ifm_h_stick_idx % ifm.shard_shape[0]
                    ifm_h_intra_shard_tile_it = ifm_h_intra_shard_idx // ifm.tile_shape[0]
                    ifm_h_intra_tile_offset = ifm_h_intra_shard_idx % ifm.tile_shape[0]
                    
                    ifm_w_shard_it = ifm_w_actual_offset // ifm.shard_shape[1]
                    ifm_w_intra_shard_idx = ifm_w_actual_offset % ifm.shard_shape[1]
                    ifm_w_intra_shard_tile_it = ifm_w_intra_shard_idx // ifm.tile_shape[1]
                    
                    ifm_tile_idx = (ifm_h_shard_it, ifm_w_shard_it, ifm_h_intra_shard_tile_it, ifm_w_intra_shard_tile_it)
                    
                    if ifm_tile_idx not in ifm_tile_idx_with_memcpy_pattern:
                        ifm_tile_idx_with_memcpy_pattern[ifm_tile_idx] = {}
                        
                    ifm_tile_idx_with_memcpy_pattern[ifm_tile_idx][ofm_h_stick_it] = ifm_h_intra_tile_offset
                    
                tiled_op = TiledOperatorSignature(
                    i_tiles=[],
                    o_tile=ofm_tiles[ofm_tile_idx],
                )
                
                ifm_tile = CollectiveTileSignature(
                    buf_name="ifm",
                    buf=ifm,
                    src_tiles=[ifm_tiles[idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()],
                    memcpy_patterns=list(ifm_tile_idx_with_memcpy_pattern.values()),
                )
                
                tiled_op.i_tiles.append((ifm_tile,))
                tiled_ops.append(tiled_op)
        
        return MCA_OperatorMapper(
            core_group=core_group,
            spad_ld_mem_space=spad_ld_mem_space,
            spad_st_mem_space=spad_st_mem_space,
            tiled_ops=tiled_ops,
        )
        
                                
    def compile(self, mapping_strategy: Callable | str = OUTPUT_STATIONARY) -> CompiledMapping:
        if isinstance(mapping_strategy, str):
            mapping_strategy = getattr(TiledOperatorMapping, mapping_strategy, None)
            if mapping_strategy is None:
                raise ValueError(f"Invalid mapping strategy: {mapping_strategy}")
        
        return CompiledMapping.from_tiled_op_mapping(
            spad_ld_pp_ptrs=self._spad_ld_pp_ptrs,
            spad_st_pp_ptrs=self._spad_st_pp_ptrs,
            spad_ld_pp_size=self._spad_ld_pp_size,
            spad_st_pp_size=self._spad_st_pp_size,
            mapping=mapping_strategy(self._core_group, self._tiled_ops)
        )
        