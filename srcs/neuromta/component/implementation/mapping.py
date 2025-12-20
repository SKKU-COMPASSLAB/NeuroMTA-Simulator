import abc
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
        
        
class TiledOperatorSignature:
    def __init__(self, i_tiles: Sequence[Sequence[TileSignature]], o_tile: TileSignature, op_kwargs: dict[str, Any]=None):
        self.i_tiles    = i_tiles
        self.o_tile     = o_tile
        self.op_kwargs  = op_kwargs if op_kwargs is not None else {}
        
    def copy(self) -> 'TiledOperatorSignature':
        i_tiles = [
            [tile.override_spm_ptr(None) for tile in tile_pair]
            for tile_pair in self.i_tiles
        ]
        o_tile = self.o_tile.override_spm_ptr(None)
        return TiledOperatorSignature(i_tiles=i_tiles, o_tile=o_tile)
        
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
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
        
        for op_idx, op in enumerate(tiled_ops):
            flag = False
            
            # output stationary
            ofm_mem_info = op.o_tile.mem_info
            if ofm_mem_info.mem_type == GlobalContextMemType.L1 and ofm_mem_info.owner_core_ids[0] in core_ids:
                target_core_id = ofm_mem_info.owner_core_ids[0]
                flag = True
            
            # input stationary (prioritized)
            if not flag:
                for i in range(len(op.i_tiles[0])):
                    if op.i_tiles[0][i].mem_info.mem_type == GlobalContextMemType.L1:
                        i_mem_infos     = [tile_pair[i].mem_info for tile_pair in op.i_tiles]
                        i_core_ids      = list(set([mem_info.owner_core_ids[0] for mem_info in i_mem_infos]))
                        target_core_id  = i_core_ids[op_idx % len(i_core_ids)]
                        
                        if target_core_id in core_ids:
                            flag = True
                            break
            
            # round-robin assignment if none of the tiles are in L1
            if not flag:
                target_core_id = core_ids[op_idx % len(core_ids)]  
                
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
                    if not isinstance(cmd, CompiledCommand.TILE_LOAD):
                        continue
                    
                    if cmd.tile_sig.buf_name != dst_buf_name:
                        continue
                    
                    dst_required_coords.add(cmd.tile_sig.coords)
                    
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
        