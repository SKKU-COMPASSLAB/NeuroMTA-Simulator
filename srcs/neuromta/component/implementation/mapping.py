from typing import Any, Sequence, Dict, List

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation.tensor_buffer import *


__all__ = [
    "TileSignature",
    "TiledOperatorSignature",
    "TiledOperatorMapping",
    "CompiledCommand",
    "CompiledStage",
    "CompiledOperator",
    "CompiledMapping",
    
    "MCA_OperatorMapper",
]


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
    def from_tiled_ops(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
        
        for op_idx, op in enumerate(tiled_ops):
            flag = False
            
            # output stationary
            ofm_mem_info = op.o_tile.mem_info
            if ofm_mem_info.mem_type == GlobalContextMemType.L1:
                target_core_id = ofm_mem_info.owner_core_ids[0]
                flag = True
            
            # input stationary (prioritized)
            if not flag:
                for i in range(len(op.i_tiles[0])):
                    if op.i_tiles[0][i].mem_info.mem_type == GlobalContextMemType.L1:
                        i_mem_infos     = [tile_pair[i].mem_info for tile_pair in op.i_tiles]
                        i_core_ids      = list(set([mem_info.owner_core_ids[0] for mem_info in i_mem_infos]))
                        target_core_id  = i_core_ids[op_idx % len(i_core_ids)]
                        flag = True
                        break
            
            # round-robin assignment if none of the tiles are in L1
            if not flag:
                target_core_id = core_ids[op_idx % len(core_ids)]  
                
            mapper[target_core_id].append(op)
        
        return mapper
    
    def summary(self) -> dict[int, list[str]]:
        return {core_id: [op.signature() for op in ops] for core_id, ops in self.items()}
            
            
class CompiledCommand:
    class OP_LOCK_INCR:
        def __init__(self, increment: int):
            self._increment  = increment
        
        @property
        def increment(self) -> int:
            return self._increment
        
        def signature(self) -> str:
            return f"OP_LOCK_INCR {self._increment}"
    
    class TILE_LOAD:
        def __init__(self, tile_sig: TileSignature):
            self._tile_sig = tile_sig
            
            if self._tile_sig.spm_ptr is None:
                raise RuntimeError("Tile SPM pointer is not assigned.")
            
            self._broadcast_dst_ptrs: list[Pointer]         = []
            self._broadcast_locks:    list[VariableHandle]  = []
            
        def add_broadcast_dst_ptr(self, dst_ptr: Pointer, dst_lock: VariableHandle):
            self._broadcast_dst_ptrs.append(dst_ptr)
            self._broadcast_locks.append(dst_lock)
            
        @property
        def tile_sig(self) -> TileSignature:
            return self._tile_sig
        
        @property
        def broadcast_dst_ptrs(self) -> list[Pointer]:
            return self._broadcast_dst_ptrs
        
        @property
        def broadcast_locks(self) -> list[VariableHandle]:
            return self._broadcast_locks
        
        def signature(self) -> str:
            broadcast_info = ""
            if len(self._broadcast_dst_ptrs) > 0:
                broadcast_info = f" [BROADCAST {', '.join([f'@{ptr.addr}' for ptr in self._broadcast_dst_ptrs])}]"
            return f"LOAD {self._tile_sig.signature} -> SPM@{self._tile_sig.spm_ptr.addr} {broadcast_info}"
            
    class TILE_STORE:
        def __init__(self, tile_sig: TileSignature):
            self._tile_sig = tile_sig
            
            if self._tile_sig.spm_ptr is None:
                raise RuntimeError("Tile SPM pointer is not assigned.")
            
        @property
        def tile_sig(self) -> TileSignature:
            return self._tile_sig
        
        def signature(self) -> str:
            return f"STORE SPM@{self._tile_sig.spm_ptr.addr} -> {self._tile_sig.signature}"
            
    class TILED_OP:
        def __init__(self, op_sig: TiledOperatorSignature, inner_op_idx: int):
            self._op_sig = op_sig
            self._inner_op_idx = inner_op_idx
            
        @property
        def op_sig(self) -> TiledOperatorSignature:
            return self._op_sig
        
        @property
        def inner_op_idx(self) -> int:
            return self._inner_op_idx
        
        def signature(self) -> str:
            tin_signature = lambda tin: tin.signature if tin.spm_ptr is None else f'@{tin.spm_ptr.addr}'
            o_tile_signature = self.op_sig.o_tile.signature if self.op_sig.o_tile.spm_ptr is None else f'@{self.op_sig.o_tile.spm_ptr.addr}'
            return f"OP {[tin_signature(tin) for tile in self.op_sig.i_tiles for tin in tile]} -> {o_tile_signature} [inner_op_idx={self._inner_op_idx}]"
        
class CompiledStage:
    def __init__(self):
        self.dma_stores:  list[CompiledCommand.TILE_STORE]                               = []  # THREAD 0 STEP 1: Store output tiles from SPAD to memory
        self.dma_loads:   list[CompiledCommand.TILE_LOAD | CompiledCommand.OP_LOCK_INCR] = []  # THREAD 0 STEP 2: Load input tiles from memory to SPAD 
        self.compute_ops: list[CompiledCommand.TILED_OP]                                 = []  # THREAD 1:        Compute tiled operations in SPAD
        
    def summary(self) -> dict[str, list[str]]:
        return {
            "dma_stores":  [cmd.signature() for cmd in self.dma_stores],
            "dma_loads":   [cmd.signature() for cmd in self.dma_loads],
            "compute_ops": [cmd.signature() for cmd in self.compute_ops],
        }
        
class CompiledOperator:
    def __init__(self, stages: Sequence[CompiledStage] = []):
        self.stages = list(stages)
        self._stage_lock: VariableHandle = VariableHandle(0)
        
    @staticmethod
    def from_tiled_ops(spad_pp_ptrs: tuple[Pointer, Pointer], spad_pp_size: int, inner_tiled_op_per_pp: int, tiled_ops: Sequence[TiledOperatorSignature]) -> 'CompiledOperator':
        stages = [CompiledStage() for _ in range(3)]
        
        dma_load_stage   = stages[0]
        compute_stage    = stages[1]
        dma_store_stage  = stages[2]
        
        spad_pp_idx   = [0, 1, 0]  # ping-pong buffer index per stage
        spad_pp_usage = [0, 0]  # usage per ping-pong buffer
        spad_pp_ops   = 0  # number of inner tiled ops in the current ping-pong buffer
        cached_i_tiles: dict[str, TileSignature] = {}
        
        for cursor in range(len(tiled_ops)):
            op = tiled_ops[cursor].copy()  # make a copy to avoid modifying the original
            
            for inner_op_idx in range(len(op.i_tiles)):
                if spad_pp_ops >= inner_tiled_op_per_pp:
                    # switch ping-pong buffer
                    for i in reversed(range(3)):
                        spad_pp_idx[i] = 1 - spad_pp_idx[i]
                    spad_pp_usage = [0, 0]
                    spad_pp_ops   = 0
                    cached_i_tiles.clear()
                    
                    # create new pipeline stages
                    dma_load_stage = compute_stage
                    compute_stage  = dma_store_stage
                    dma_store_stage = CompiledStage()
                    
                    stages.append(dma_store_stage)

                for tin in op.i_tiles[inner_op_idx]:
                    if tin.signature in cached_i_tiles:
                        tin.spm_ptr = cached_i_tiles[tin.signature].spm_ptr  # reuse cached tile
                    else:
                        tile_buf = tin.buf
                        tile_size = tile_buf.tile_size
                        
                        tin.spm_ptr = spad_pp_ptrs[spad_pp_idx[0]] + spad_pp_usage[spad_pp_idx[0]]
                        cached_i_tiles[tin.signature] = tin
                    
                        dma_load_stage.dma_loads.append(
                            CompiledCommand.TILE_LOAD(
                                tile_sig=tin,
                            )
                        )
                    
                        spad_pp_usage[spad_pp_idx[0]] += tile_size
                        if spad_pp_usage[spad_pp_idx[0]] > spad_pp_size:
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
                    
                    tout.spm_ptr = spad_pp_ptrs[spad_pp_idx[2]] + spad_pp_usage[spad_pp_idx[2]]
                    
                    dma_store_stage.dma_stores.append(
                        CompiledCommand.TILE_STORE(
                            tile_sig=tout
                        )
                    )
                    
                    spad_pp_usage[spad_pp_idx[2]] += tile_size
                    if spad_pp_usage[spad_pp_idx[2]] > spad_pp_size:
                        raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
                
                spad_pp_ops += 1
                    
        return CompiledOperator(stages=stages)
    
    @property
    def stage_lock(self) -> VariableHandle:
        return self._stage_lock
    
    def summary(self) -> list[dict[str, list[str]]]:
        return [stage.summary() for stage in self.stages]

class CompiledMapping:
    def __init__(self, mapping: TiledOperatorMapping, operators: dict[int, CompiledOperator]):
        self.mapping   = mapping
        self.operators = operators
        
    @staticmethod
    def from_tiled_op_mapping(spad_pp_ptrs: dict[int, tuple[Pointer, Pointer]], spad_pp_size: int, inner_tiled_op_per_pp: int, mapping: TiledOperatorMapping) -> 'CompiledMapping':
        core_group = mapping.core_group
        operators: dict[int, CompiledOperator] = {}
        
        for core_id in core_group:
            if core_id not in spad_pp_ptrs:
                raise RuntimeError(f"SPAD ping-pong pointers not provided for core {core_id}.")
            
            p = CompiledOperator.from_tiled_ops(
                spad_pp_ptrs=spad_pp_ptrs[core_id],
                spad_pp_size=spad_pp_size,
                inner_tiled_op_per_pp=inner_tiled_op_per_pp,
                tiled_ops=mapping[core_id]
            )
            operators[core_id] = p
            
        return CompiledMapping(mapping=mapping, operators=operators)
    
    def apply_broadcast_optimization(self, buf_targets: list[str]=None):
        rr_cnt = 0  # round-robin counter
        
        for src_core_id, src_operator in self.operators.items():
            for src_stage_idx, src_stage in enumerate(src_operator.stages):
                for s_i in range(len(src_stage.dma_loads)):
                    src_load_cmd = src_stage.dma_loads[s_i]
                    if not isinstance(src_load_cmd, CompiledCommand.TILE_LOAD):
                        continue
                    
                    src_tile_sig = src_load_cmd.tile_sig
                    if (buf_targets is not None) and (src_tile_sig.buf_name not in buf_targets):
                        continue
                    
                    target_cmd_found: list[tuple[CompiledStage, int, VariableHandle]] = [(src_stage, s_i, src_operator.stage_lock)]
                    
                    for dst_core_id, dst_operator in self.operators.items():
                        if src_core_id == dst_core_id:
                            continue
                        if src_stage_idx >= len(dst_operator.stages):
                            continue
                        
                        dst_stage = dst_operator.stages[src_stage_idx]
                    
                        for d_i in range(len(dst_stage.dma_loads)):
                            dst_load_cmd = dst_stage.dma_loads[d_i]
                            if not isinstance(dst_load_cmd, CompiledCommand.TILE_LOAD):
                                continue

                            dst_tile_sig = dst_load_cmd.tile_sig
                            
                            if src_tile_sig == dst_tile_sig:
                                target_cmd_found.append((dst_stage, d_i, dst_operator.stage_lock))
                                
                    if len(target_cmd_found) > 1:
                        ss_idx = rr_cnt % len(target_cmd_found)
                        ss_stage, ss_i, ss_lock = target_cmd_found[ss_idx]
                        
                        for tt_idx, (tt_stage, tt_i, tt_lock) in enumerate(target_cmd_found):
                            if tt_idx == ss_idx:
                                continue
                            
                            ss_stage.dma_loads[ss_i].add_broadcast_dst_ptr(
                                dst_ptr=tt_stage.dma_loads[tt_i].tile_sig.spm_ptr,
                                dst_lock=tt_lock
                            )
                            tt_stage.dma_loads[tt_i] = CompiledCommand.OP_LOCK_INCR(1)
                            
                        rr_cnt += 1
        
        return self
    
    def summary(self) -> list[dict[str, list[str]]]:
        return {
            i: p.summary() 
            for i, p in self.operators.items()
        }


class MCA_OperatorMapper:
    def __init__(
        self,
        
        core_group: MCA_CoreGroup,
        spad_mem_space: MCA_L1MemorySpace,
        
        n_inner_per_outer: int,
        input_tile_size: int,
        output_tile_size: int,
        
        tiled_ops: Sequence[TiledOperatorSignature],
    ):
        self._core_group = core_group
        self._tiled_ops = tiled_ops
        
        self._spad_mem_space = spad_mem_space
        self._spad_pp_size = spad_mem_space.size_per_owner // 2  # ping-pong buffer size per core
        self._spad_pp_ptrs: dict[int, tuple[Pointer, Pointer]] = {
            core_id: (
                self._spad_mem_space.allocate(core_id, self._spad_pp_size),
                self._spad_mem_space.allocate(core_id, self._spad_pp_size),
            )
            for core_id in self._core_group
        }
        
        _n_outer_tiled_op_per_pp = self._spad_pp_size // (input_tile_size * n_inner_per_outer + output_tile_size)
        _n_remaining_inner_tiled_op_per_pp = (self._spad_pp_size % (input_tile_size * n_inner_per_outer + output_tile_size) - output_tile_size) // input_tile_size
        self._inner_tiled_op_per_pp = _n_outer_tiled_op_per_pp * n_inner_per_outer + _n_remaining_inner_tiled_op_per_pp
        
    @staticmethod
    def LINEAR(core_group: MCA_CoreGroup, spad_mem_space: MCA_L1MemorySpace, ifm_b: MCA_TensorBuffer, wgt_b: MCA_TensorBuffer, bias_b: MCA_TensorBuffer, ofm_b: MCA_TensorBuffer) -> 'MCA_OperatorMapper':
        ifm_shape = ifm_b.shape
        wgt_shape = wgt_b.shape
        bias_shape = bias_b.shape
        ofm_shape = ofm_b.shape
        
        if ifm_shape[0] != ofm_shape[0]:
            raise Exception(f"IFM and OFM batch size mismatch: {ifm_shape[0]} != {ofm_shape[0]}")
        if wgt_shape[0] != ofm_shape[1] or wgt_shape[0] != bias_shape[1]:
            raise Exception(f"WGT and OFM channel size mismatch: {wgt_shape[0]} != {ofm_shape[1]} != {bias_shape[1]}")
        if wgt_shape[1] != ifm_shape[1]:
            raise Exception(f"WGT and IFM feature size mismatch: {wgt_shape[1]} != {ifm_shape[1]}")
        
        ifm_shard_grid = ifm_b.shard_grid
        wgt_shard_grid = wgt_b.shard_grid
        bias_shard_grid = bias_b.shard_grid
        ofm_shard_grid = ofm_b.shard_grid
        
        if ifm_shard_grid[0] != ofm_shard_grid[0]:
            raise Exception(f"IFM and OFM shard grid batch size mismatch: {ifm_shard_grid[0]} != {ofm_shard_grid[0]}")
        if wgt_shard_grid[0] != ofm_shard_grid[1] or wgt_shard_grid[0] != bias_shard_grid[1]:
            raise Exception(f"WGT and OFM shard grid channel size mismatch: {wgt_shard_grid[0]} != {ofm_shard_grid[1]} != {bias_shard_grid[1]}")
        if wgt_shard_grid[1] != ifm_shard_grid[1]:
            raise Exception(f"WGT and IFM shard grid feature size mismatch: {wgt_shard_grid[1]} != {ifm_shard_grid[1]}")
        
        ifm_tile_shape = ifm_b.tile_shape
        wgt_tile_shape = wgt_b.tile_shape
        bias_tile_shape = bias_b.tile_shape
        ofm_tile_shape = ofm_b.tile_shape
        
        if ifm_tile_shape[0] != ofm_tile_shape[0]:
            raise Exception(f"IFM and OFM tile shape batch size mismatch: {ifm_tile_shape[0]} != {ofm_tile_shape[0]}")
        if wgt_tile_shape[0] != ofm_tile_shape[1] or wgt_tile_shape[0] != bias_tile_shape[1]:
            raise Exception(f"WGT and OFM tile shape channel size mismatch: {wgt_tile_shape[0]} != {ofm_tile_shape[1]} != {bias_tile_shape[1]}")
        if wgt_tile_shape[1] != ifm_tile_shape[1]:
            raise Exception(f"WGT and IFM tile shape feature size mismatch: {wgt_tile_shape[1]} != {ifm_tile_shape[1]}")
        
        ifm_tiles: dict[tuple[int, ...], TileSignature] = {
            (m_s, k_s, m_t, k_t): TileSignature("ifm", ifm_b, m_s, k_s, m_t, k_t)
            for m_s in range(ifm_b.shard_grid[0])
            for k_s in range(ifm_b.shard_grid[1])
            for m_t in range(ifm_b.tile_grid_per_shard[0])
            for k_t in range(ifm_b.tile_grid_per_shard[1])
        }
        
        wgt_tiles: dict[tuple[int, ...], TileSignature] = {
            (n_s, k_s, n_t, k_t): TileSignature("wgt", wgt_b, n_s, k_s, n_t, k_t)
            for n_s in range(wgt_b.shard_grid[0])
            for k_s in range(wgt_b.shard_grid[1])
            for n_t in range(wgt_b.tile_grid_per_shard[0])
            for k_t in range(wgt_b.tile_grid_per_shard[1])
        }
        
        bias_tiles: dict[tuple[int, ...], TileSignature] = {
            (0, n_s, 0, n_t): TileSignature("bias", bias_b, 0, n_s, 0, n_t)
            for n_s in range(bias_b.shard_grid[1])
            for n_t in range(bias_b.tile_grid_per_shard[1])
        }
        
        ofm_tiles: dict[tuple[int, ...], TileSignature] = {
            (m_s, n_s, m_t, n_t): TileSignature("ofm", ofm_b, m_s, n_s, m_t, n_t)
            for m_s in range(ofm_b.shard_grid[0])
            for n_s in range(ofm_b.shard_grid[1])
            for m_t in range(ofm_b.tile_grid_per_shard[0])
            for n_t in range(ofm_b.tile_grid_per_shard[1])
        }
        
        tiled_ops = [
            TiledOperatorSignature(
                i_tiles=[
                    (ifm_tiles[(m_s, k_s, m_t, k_t)], wgt_tiles[(n_s, k_s, n_t, k_t)], bias_tiles[(0, n_s, 0, n_t)]) 
                    for k_s in range(ifm_b.shard_grid[1]) 
                    for k_t in range(ifm_b.tile_grid_per_shard[1])
                ],
                o_tile=ofm_tiles[(m_s, n_s, m_t, n_t)]
            )
            for m_s in range(ofm_b.shard_grid[0])
            for n_s in range(ofm_b.shard_grid[1])
            for m_t in range(ofm_b.tile_grid_per_shard[0])
            for n_t in range(ofm_b.tile_grid_per_shard[1])
        ]
        
        return MCA_OperatorMapper(
            core_group=core_group,
            spad_mem_space=spad_mem_space,
            n_inner_per_outer=ifm_b.shard_grid[1] * ifm_b.tile_grid_per_shard[1],
            input_tile_size=ifm_b.tile_size + wgt_b.tile_size + bias_b.tile_size,
            output_tile_size=ofm_b.tile_size,
            tiled_ops=tiled_ops,
        )
        
    def compile(self) -> CompiledMapping:
        return CompiledMapping.from_tiled_op_mapping(
            spad_pp_ptrs=self._spad_pp_ptrs,
            spad_pp_size=self._spad_pp_size,
            inner_tiled_op_per_pp=self._inner_tiled_op_per_pp,
            mapping=TiledOperatorMapping.from_tiled_ops(self._core_group, self._tiled_ops)
        )
        