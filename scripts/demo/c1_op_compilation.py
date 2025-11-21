import os
import json
import torch
from typing import Any, Sequence, Dict, List

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


class TileSignature:
    def __init__(self, buf_name: str, buf: MCA_TensorBuffer, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.buf = buf
        self.y_s = y_s
        self.x_s = x_s
        self.y_t = y_t
        self.x_t = x_t
        
        self.spm_ptr: Pointer | None = None  # to be assigned during pipeline generation
        
    def override_spm_ptr(self, spm_ptr: Pointer):
        new_sig = TileSignature(
            buf_name=self.buf_name,
            buf=self.buf,
            y_s=self.y_s,
            x_s=self.x_s,
            y_t=self.y_t,
            x_t=self.x_t,
        )
        new_sig.spm_ptr = spm_ptr
        return new_sig
        
    def __eq__(self, value):
        if isinstance(value, TileSignature):
            return (self.buf_name == value.buf_name and
                    self.y_s == value.y_s and
                    self.x_s == value.x_s and
                    self.y_t == value.y_t and
                    self.x_t == value.x_t)
        
        return False
    
    @property
    def mem_info(self) -> GlobalContextMemInfo:
        shard_ptr = self.buf.get_shard_ptr(self.y_s, self.x_s)
        mem_info  = self.buf.mem_space.device.global_context.get_mem_info_by_address(addr=shard_ptr.addr)
        return mem_info
    
    @property
    def signature(self) -> str:
        return f"{self.buf_name}({self.y_s},{self.x_s},{self.y_t},{self.x_t})"
        
        
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
    def os_mapping(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
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
        self.dma_stores:  list[CompiledCommand.TILE_STORE]                           = []  # THREAD 0 STEP 1: Store output tiles from SPAD to memory
        self.dma_loads:   list[CompiledCommand.TILE_LOAD | CompiledCommand.OP_LOCK_INCR] = []  # THREAD 0 STEP 2: Load input tiles from memory to SPAD 
        self.compute_ops: list[CompiledCommand.TILED_OP]                             = []  # THREAD 1:        Compute tiled operations in SPAD
        
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
    def from_mapping(spad_pp_ptrs: dict[int, tuple[Pointer, Pointer]], spad_pp_size: int, inner_tiled_op_per_pp: int, mapping: TiledOperatorMapping) -> 'CompiledMapping':
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
    
    def apply_broadcast_optimization(self, buf_name: str=None):
        rr_cnt = 0  # round-robin counter
        
        for src_core_id, src_operator in self.operators.items():
            for src_stage_idx, src_stage in enumerate(src_operator.stages):
                for s_i in range(len(src_stage.dma_loads)):
                    src_load_cmd = src_stage.dma_loads[s_i]
                    if not isinstance(src_load_cmd, CompiledCommand.TILE_LOAD):
                        continue
                    
                    src_tile_sig = src_load_cmd.tile_sig
                    if (buf_name is not None) and (src_tile_sig.buf_name != buf_name):
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
        
        
def linear_os_compute_cmd(core: NPUCore, compute_cmd: CompiledCommand.TILED_OP, debug_output: bool=False):
    op_sig = compute_cmd.op_sig
    inner_op_idx = compute_cmd.inner_op_idx
    
    ifm_sig = op_sig.i_tiles[inner_op_idx][0]
    wgt_sig = op_sig.i_tiles[inner_op_idx][1]
    bias_sig = op_sig.i_tiles[inner_op_idx][2]
    ofm_sig = op_sig.o_tile
    
    ifm  = DataContainer(shape=ifm_sig.buf.tile_shape, dtype=ifm_sig.buf.dtype)
    wgt  = DataContainer(shape=wgt_sig.buf.tile_shape, dtype=wgt_sig.buf.dtype)
    bias = DataContainer(shape=bias_sig.buf.tile_shape, dtype=bias_sig.buf.dtype)
    ofm  = DataContainer(shape=ofm_sig.buf.tile_shape, dtype=ofm_sig.buf.dtype)
    
    preload_psum = (inner_op_idx == 0)
    flush_ofm    = (inner_op_idx == len(op_sig.i_tiles) - 1)
    
    if inner_op_idx == 0:
        core.mxu_reconfigure(dtype=ifm_sig.buf.dtype, acc_dtype=ofm_sig.buf.dtype)
    
    core.local_mem_page_read(ifm_sig.spm_ptr, ifm_sig.buf.tile_size, ifm)
    core.local_mem_page_read(wgt_sig.spm_ptr, wgt_sig.buf.tile_size, wgt)
    if preload_psum:
        core.local_mem_page_read(bias_sig.spm_ptr, bias_sig.buf.tile_size, bias)
    
    if debug_output:
        core.debug_core_with_ambiguous_func(
            lambda wgt: logger.info(f"CORE: {core.core_id} WGT tile ({wgt_sig.y_s}, {wgt_sig.x_s}, {wgt_sig.y_t}, {wgt_sig.x_t}):\n{wgt.data.flatten().view(wgt.dtype).reshape(wgt.shape)}"),
            wgt
        )
        core.debug_core_with_ambiguous_func(
            lambda ifm: logger.info(f"CORE: {core.core_id} IFM tile ({ifm_sig.y_s}, {ifm_sig.x_s}, {ifm_sig.y_t}, {ifm_sig.x_t}):\n{ifm.data.flatten().view(ifm.dtype).reshape(ifm.shape)}"),
            ifm
        )
        if preload_psum:
            core.debug_core_with_ambiguous_func(
                lambda bias: logger.info(f"CORE: {core.core_id} BIAS tile ({bias_sig.y_s}, {bias_sig.x_s}, {bias_sig.y_t}, {bias_sig.x_t}):\n{bias.data.flatten().view(bias.dtype).reshape(bias.shape)}"),
                bias
            )

    core.mxu_tiled_gemm(
        ifm, wgt, bias, ofm,
        preload_psum=preload_psum,
        flush_ofm=flush_ofm,
        wgt_transposed=True,
        psum_vectored=True,
    )
    
    if debug_output:
        core.debug_core_with_ambiguous_func(
            lambda ofm: logger.info(f"CORE: {core.core_id} OFM tile ({ofm_sig.y_s}, {ofm_sig.x_s}, {ofm_sig.y_t}, {ofm_sig.x_t}):\n{ofm.data.flatten().view(ofm.dtype).reshape(ofm.shape)}"),
            ofm
        )
    
    if flush_ofm:
        core.local_mem_page_write(ofm_sig.spm_ptr, ofm_sig.buf.tile_size, ofm)


@jit_prototype
def main(core: NPUCore, operator: CompiledOperator, debug_output: bool=False):
    for stage in operator.stages:
        with new_parallel_thread("DMA"):
            # DMA STORE
            for store_cmd in stage.dma_stores:
                tile_sig = store_cmd.tile_sig
                dst_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_write_args(tile_sig.y_s, tile_sig.x_s, tile_sig.y_t, tile_sig.x_t)
                if debug_output:
                    core.debug_core_with_ambiguous_func(
                        logger.info,
                        f"CORE: {core.core_id} STORE SPM@{tile_sig.spm_ptr.addr} -> {tile_sig.signature} <{src_size} {src_row_size} {src_row_stride} {dst_row_stride}>"
                    )
                core.local_mem_copy(dst_ptr, tile_sig.spm_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
            
            core.async_rpc_wait_all()
            
            # DMA LOAD
            for load_cmd in stage.dma_loads:
                if isinstance(load_cmd, CompiledCommand.OP_LOCK_INCR):
                    core.var_atomic_increase(operator.stage_lock, load_cmd._increment)  # BROADCAST: increase the stage lock and wait for the response from the source core
                    continue
                
                tile_sig = load_cmd.tile_sig
                src_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_read_args(tile_sig.y_s, tile_sig.x_s, tile_sig.y_t, tile_sig.x_t)
                if debug_output:
                    core.debug_core_with_ambiguous_func(
                        logger.info,
                        f"CORE: {core.core_id} LOAD {tile_sig.signature} -> SPM@{tile_sig.spm_ptr.addr} <{src_size} {src_row_size} {src_row_stride} {dst_row_stride}>"
                    )
                
                if len(load_cmd.broadcast_dst_ptrs) > 0:  # BROADCAST: broadcast optimization
                    core.local_mem_broadcast(load_cmd.broadcast_dst_ptrs + [tile_sig.spm_ptr,], src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
                else:
                    core.local_mem_copy(tile_sig.spm_ptr, src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
                
            core.async_rpc_wait_all()
            
            for load_cmd in stage.dma_loads:
                if isinstance(load_cmd, CompiledCommand.TILE_LOAD):
                    for lock in load_cmd.broadcast_locks:  # BROADCAST: notify dst cores that the broadcast load is complete
                        core.var_atomic_increase(lock, -1)
                        
            core.var_atomic_wait(operator.stage_lock, 0)  # BORADCAST: wait for all broadcast loads to complete
                
        with new_parallel_thread("COMPUTE"):
            # COMPUTE
            for compute_cmd in stage.compute_ops:
                linear_os_compute_cmd(core, compute_cmd, debug_output=debug_output)
                
        core.parallel_merge()
    

if __name__ == "__main__":
    import time
    
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group((0, 0), (4, 4))
    
    M, N, K = 512, 512, 512
    Ms, Ns, Ks = 4, 4, 4
    Mt, Nt, Kt = 32, 32, 32
    dtype = torch.int8
    acc_dtype = torch.int32
    
    ifm  = torch.arange(M * K, dtype=dtype).reshape(M, K)
    wgt  = torch.arange(N * K, dtype=dtype).reshape(N, K)
    bias = torch.arange(N, dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    ifm_tile_size  = Mt * Kt * dtype.itemsize
    wgt_tile_size  = Nt * Kt * dtype.itemsize
    bias_tile_size = Nt * acc_dtype.itemsize
    ofm_tile_size  = Mt * Nt * acc_dtype.itemsize
    
    ifm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_grid)
    param_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    ofm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_grid)
    space_pp_space  = device.create_l1_mem_space(parse_mem_cap_str("128KB"), core_group=core_grid)
    
    ifm_b  = MCA_TensorBuffer(mem_space=ifm_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks), blocked_mapping=True).tiling(tile_shape=(Mt, Kt)).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks)                      ).tiling(tile_shape=(Nt, Kt)).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns)                      ).tiling(tile_shape=(1,  Nt)).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=ofm_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=True).tiling(tile_shape=(Mt, Nt)).allocate()
    
    spad_pp_size = parse_mem_cap_str("64KB")
    spad_pp_ptrs = {
        core_id: (
            space_pp_space.allocate(core_id=core_id, size=spad_pp_size),
            space_pp_space.allocate(core_id=core_id, size=spad_pp_size),
        )
        for core_id in core_grid.core_ids
    }
    
    _n_inner_per_outer = ifm_b.tile_grid[1]  # number of total Kt tiles
    _n_outer_tiled_op_per_pp = spad_pp_size // ((ifm_tile_size + wgt_tile_size + bias_tile_size) * _n_inner_per_outer + ofm_tile_size)
    _n_remaining_inner_tiled_op_per_pp = (spad_pp_size % ((ifm_tile_size + wgt_tile_size + bias_tile_size) * _n_inner_per_outer + ofm_tile_size) - ofm_tile_size) // (ifm_tile_size + wgt_tile_size + bias_tile_size)
    inner_tiled_op_per_pp = _n_outer_tiled_op_per_pp * _n_inner_per_outer + _n_remaining_inner_tiled_op_per_pp
    print(f"Inner tiled operations per ping-pong buffer: {inner_tiled_op_per_pp}")
    
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
    
    tiled_ops_mapping = TiledOperatorMapping.os_mapping(
        core_group=core_grid,
        tiled_ops=tiled_ops
    )
    
    compiled_mapping = CompiledMapping.from_mapping(
        spad_pp_ptrs=spad_pp_ptrs,
        spad_pp_size=spad_pp_size,
        inner_tiled_op_per_pp=inner_tiled_op_per_pp,
        mapping=tiled_ops_mapping
    )
    
    compiled_mapping.apply_broadcast_optimization()
    
    for core_id, operator in compiled_mapping.operators.items():
        core = device.get_npu_core(core_id=core_id)
        kernel = main(core, operator)
        kernel.dispatch("MAIN")
        
    with MonitoringWindow() as monitor:
        for core_id in core_grid.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
        
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(compiled_mapping.summary(), f, indent=4)
        logger.info(f"Pipelined mapping summary saved to '{tmp_ouput_path}'.")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * M * N * K
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")