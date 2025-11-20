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
    
    @staticmethod
    def is_mapping(core_group: MCA_CoreGroup, tiled_ops: Sequence[TiledOperatorSignature]) -> 'TiledOperatorMapping':
        core_ids = core_group.core_ids
        mapper = TiledOperatorMapping(core_group=core_group)
        
        for core_id in core_ids:
            mapper[core_id] = []
            
        for op_idx, op in enumerate(tiled_ops):
            flag = False
            
            # input stationary (prioritized)
            for i in range(len(op.i_tiles[0])):
                if op.i_tiles[0][i].mem_info.mem_type == GlobalContextMemType.L1:
                    i_mem_infos     = [tile_pair[i].mem_info for tile_pair in op.i_tiles]
                    i_core_ids      = list(set([mem_info.owner_core_ids[0] for mem_info in i_mem_infos]))
                    target_core_id  = i_core_ids[op_idx % len(i_core_ids)]
                    flag = True
                    break
            
            # output stationary
            if not flag:
                ofm_mem_info = op.o_tile.mem_info
                if ofm_mem_info.mem_type == GlobalContextMemType.L1:
                    target_core_id = ofm_mem_info.owner_core_ids[0]
                    flag = True
            
            # round-robin assignment if none of the tiles are in L1
            if not flag:
                target_core_id = core_ids[op_idx % len(core_ids)]  

            mapper[target_core_id].append(op)
            
        return mapper
    
    def summary(self) -> dict[int, list[str]]:
        return {core_id: [op.signature() for op in ops] for core_id, ops in self.items()}
            
            
class CompiledPipelineCommand:
    class TILE_LOAD:
        def __init__(self, tile_sig: TileSignature):
            self._tile_sig = tile_sig
            
            if self._tile_sig.spm_ptr is None:
                raise RuntimeError("Tile SPM pointer is not assigned.")
            
        @property
        def tile_sig(self) -> TileSignature:
            return self._tile_sig
        
        def signature(self) -> str:
            return f"LOAD {self._tile_sig.signature} -> SPM@{self._tile_sig.spm_ptr.addr}"
            
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
        
class CompiledPipelineStage:
    def __init__(self):
        self.dma_stores:  list[CompiledPipelineCommand.TILE_STORE] = []  # THREAD 0 STEP 1: Store output tiles from SPAD to memory
        self.dma_loads:   list[CompiledPipelineCommand.TILE_LOAD]  = []  # THREAD 0 STEP 2: Load input tiles from memory to SPAD 
        self.compute_ops: list[CompiledPipelineCommand.TILED_OP]   = []  # THREAD 1:        Compute tiled operations in SPAD
        
    def summary(self) -> dict[str, list[str]]:
        return {
            "dma_stores":  [cmd.signature() for cmd in self.dma_stores],
            "dma_loads":   [cmd.signature() for cmd in self.dma_loads],
            "compute_ops": [cmd.signature() for cmd in self.compute_ops],
        }
        
class CompiledPipeline:
    def __init__(self, stages: Sequence[CompiledPipelineStage] = []):
        self.stages = list(stages)
        
    @staticmethod
    def from_tiled_ops(spad_pp_ptrs: tuple[Pointer, Pointer], spad_pp_size: int, inner_tiled_op_per_pp: int, tiled_ops: Sequence[TiledOperatorSignature]) -> 'CompiledPipeline':
        stages = [CompiledPipelineStage() for _ in range(3)]
        
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
                        spad_pp_idx[i]    = 1 - spad_pp_idx[i]
                    spad_pp_usage = [0, 0]
                    spad_pp_ops   = 0
                    cached_i_tiles.clear()
                    
                    # create new pipeline stages
                    dma_load_stage = compute_stage
                    compute_stage  = dma_store_stage
                    dma_store_stage = CompiledPipelineStage()
                    
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
                            CompiledPipelineCommand.TILE_LOAD(
                                tile_sig=tin,
                            )
                        )
                    
                        spad_pp_usage[spad_pp_idx[0]] += tile_size
                        if spad_pp_usage[spad_pp_idx[0]] > spad_pp_size:
                            raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
            
                compute_stage.compute_ops.append(
                    CompiledPipelineCommand.TILED_OP(
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
                        CompiledPipelineCommand.TILE_STORE(
                            tile_sig=tout
                        )
                    )
                    
                    spad_pp_usage[spad_pp_idx[2]] += tile_size
                    if spad_pp_usage[spad_pp_idx[2]] > spad_pp_size:
                        raise RuntimeError("Not enough SPAD space for ping-pong buffering.")
                
                spad_pp_ops += 1
                    
        return CompiledPipeline(stages=stages)
    
    def summary(self) -> list[dict[str, list[str]]]:
        return [stage.summary() for stage in self.stages]

class CompiledPipelineMapping:
    def __init__(self, mapping: TiledOperatorMapping, pipelines: dict[int, CompiledPipeline]):
        self.mapping   = mapping
        self.pipelines = pipelines
        
    @staticmethod
    def from_mapping(spad_pp_ptrs: dict[int, tuple[Pointer, Pointer]], spad_pp_size: int, inner_tiled_op_per_pp: int, mapping: TiledOperatorMapping) -> 'CompiledPipelineMapping':
        core_group = mapping.core_group
        pipelines: dict[int, CompiledPipeline] = {}
        
        for core_id in core_group:
            if core_id not in spad_pp_ptrs:
                raise RuntimeError(f"SPAD ping-pong pointers not provided for core {core_id}.")
            
            p = CompiledPipeline.from_tiled_ops(
                spad_pp_ptrs=spad_pp_ptrs[core_id],
                spad_pp_size=spad_pp_size,
                inner_tiled_op_per_pp=inner_tiled_op_per_pp,
                tiled_ops=mapping[core_id]
            )
            pipelines[core_id] = p
            
        return CompiledPipelineMapping(mapping=mapping, pipelines=pipelines)
    
    def summary(self) -> list[dict[str, list[str]]]:
        return {
            i: p.summary() 
            for i, p in self.pipelines.items()
        }


@jit_prototype
def main(core: NPUCore, pipeline: CompiledPipeline):
    core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
    
    for stage in pipeline.stages:
        with new_parallel_thread("DMA"):
            # DMA STORE
            for store_cmd in stage.dma_stores:
                tile_sig = store_cmd.tile_sig
                dst_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_write_args(tile_sig.y_s, tile_sig.x_s, tile_sig.y_t, tile_sig.x_t)
                core.debug_core_with_ambiguous_func(
                    logger.info,
                    f"CORE: {core.core_id} STORE SPM@{tile_sig.spm_ptr.addr} -> {tile_sig.signature} <{src_size} {src_row_size} {src_row_stride} {dst_row_stride}>"
                )
                core.local_mem_copy(dst_ptr, tile_sig.spm_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
            
            core.async_rpc_wait_all()
            
            # DMA LOAD
            for load_cmd in stage.dma_loads:
                tile_sig = load_cmd.tile_sig
                src_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = tile_sig.buf.get_tile_ptr_read_args(tile_sig.y_s, tile_sig.x_s, tile_sig.y_t, tile_sig.x_t)
                core.debug_core_with_ambiguous_func(
                    logger.info,
                    f"CORE: {core.core_id} LOAD {tile_sig.signature} -> SPM@{tile_sig.spm_ptr.addr} <{src_size} {src_row_size} {src_row_stride} {dst_row_stride}>"
                )
                core.local_mem_copy(tile_sig.spm_ptr, src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride, nowait=True)
                
            core.async_rpc_wait_all()
                
        with new_parallel_thread("COMPUTE"):
            for compute_cmd in stage.compute_ops:
                op_sig = compute_cmd.op_sig
                inner_op_idx = compute_cmd.inner_op_idx
                
                ifm_sig = op_sig.i_tiles[inner_op_idx][0]
                wgt_sig = op_sig.i_tiles[inner_op_idx][1]
                ofm_sig = op_sig.o_tile
                
                ifm  = DataContainer(shape=ifm_sig.buf.tile_shape, dtype=ifm_sig.buf.dtype)
                wgt  = DataContainer(shape=wgt_sig.buf.tile_shape, dtype=wgt_sig.buf.dtype)
                psum = DataContainer(shape=ofm_sig.buf.tile_shape, dtype=ofm_sig.buf.dtype)
                ofm  = DataContainer(shape=ofm_sig.buf.tile_shape, dtype=ofm_sig.buf.dtype)
                
                core.local_mem_page_read(ifm_sig.spm_ptr, ifm_sig.buf.tile_size, ifm)
                core.debug_core_with_ambiguous_func(
                    lambda ifm: logger.info(f"CORE: {core.core_id} IFM tile ({ifm_sig.y_s}, {ifm_sig.x_s}, {ifm_sig.y_t}, {ifm_sig.x_t}):\n{ifm.data.flatten().view(ifm.dtype).reshape(ifm.shape)}"),
                    ifm
                )
                core.local_mem_page_read(wgt_sig.spm_ptr, wgt_sig.buf.tile_size, wgt)
                core.debug_core_with_ambiguous_func(
                    lambda wgt: logger.info(f"CORE: {core.core_id} WGT tile ({wgt_sig.y_s}, {wgt_sig.x_s}, {wgt_sig.y_t}, {wgt_sig.x_t}):\n{wgt.data.flatten().view(wgt.dtype).reshape(wgt.shape)}"),
                    wgt
                )
                
                flush_ofm = (inner_op_idx == len(op_sig.i_tiles) - 1)
                
                core.mxu_tiled_gemm(
                    ifm, wgt, psum, ofm,
                    flush_ofm=flush_ofm,
                    wgt_transposed=True,
                )
                
                core.debug_core_with_ambiguous_func(
                    lambda ofm: logger.info(f"CORE: {core.core_id} OFM tile ({ofm_sig.y_s}, {ofm_sig.x_s}, {ofm_sig.y_t}, {ofm_sig.x_t}):\n{ofm.data.flatten().view(ofm.dtype).reshape(ofm.shape)}"),
                    ofm
                )
                
                if flush_ofm:
                    core.local_mem_page_write(ofm_sig.spm_ptr, ofm_sig.buf.tile_size, ofm)
                
        core.parallel_merge()
    

if __name__ == "__main__":
    import time
    
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group((0, 0), (2, 2))
    
    M, N, K = 60, 60, 60
    Ms, Ns, Ks = 2, 2, 2
    Mt, Nt, Kt = 32, 32, 32
    dtype = torch.int32
    
    ifm = torch.arange(M * K, dtype=dtype).reshape(M, K)
    wgt = torch.arange(N * K, dtype=dtype).reshape(N, K)
    ofm = torch.zeros((M, N), dtype=dtype)
    
    ifm_size = ifm.numel() * ifm.dtype.itemsize
    wgt_size = wgt.numel() * wgt.dtype.itemsize
    ofm_size = ofm.numel() * ofm.dtype.itemsize
    
    ifm_tile_size = Mt * Kt * dtype.itemsize
    wgt_tile_size = Nt * Kt * dtype.itemsize
    ofm_tile_size = Mt * Nt * dtype.itemsize
    
    l1_spad_size_per_bank = parse_mem_cap_str("128KB")
    l1_io_space_size_per_bank = parse_mem_cap_str("512KB")
    main_mem_size_per_channel = ifm_size + wgt_size + ofm_size + 1024  # extra buffer
    
    l1_spad_space   = device.create_l1_mem_space(parse_mem_cap_str("128KB"), core_group=core_grid)
    l1_io_bank0     = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_grid)
    l1_io_bank1     = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_grid)
    main_mem_space  = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b = MCA_TensorBuffer(mem_space=l1_io_bank0,    shape=ifm.shape, dtype=ifm.dtype, shard_grid=(Ms, Ks), blocked_mapping=True).tiling(tile_shape=(Mt, Kt)).allocate().update(ifm)
    wgt_b = MCA_TensorBuffer(mem_space=main_mem_space, shape=wgt.shape, dtype=wgt.dtype, shard_grid=(Ns, Ks)                      ).tiling(tile_shape=(Nt, Kt)).allocate().update(wgt)
    ofm_b = MCA_TensorBuffer(mem_space=l1_io_bank1,    shape=ofm.shape, dtype=ofm.dtype, shard_grid=(Ms, Ns), blocked_mapping=True).tiling(tile_shape=(Mt, Nt)).allocate()
    
    spad_pp_size = parse_mem_cap_str("64KB")
    spad_pp_ptrs = {
        core_id: (
            l1_spad_space.allocate(core_id=core_id, size=spad_pp_size),
            l1_spad_space.allocate(core_id=core_id, size=spad_pp_size),
        )
        for core_id in core_grid.core_ids
    }
    
    _n_inner_per_outer = ifm_b.tile_grid[1]  # number of total Kt tiles
    _n_outer_tiled_op_per_pp = spad_pp_size // ((ifm_tile_size + wgt_tile_size) * _n_inner_per_outer + ofm_tile_size)
    _n_remaining_inner_tiled_op_per_pp = (spad_pp_size % ((ifm_tile_size + wgt_tile_size) * _n_inner_per_outer + ofm_tile_size) - ofm_tile_size) // (ifm_tile_size + wgt_tile_size)
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
                (ifm_tiles[(m_s, k_s, m_t, k_t)], wgt_tiles[(n_s, k_s, n_t, k_t)]) 
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
    
    pipeline_mapping = CompiledPipelineMapping.from_mapping(
        spad_pp_ptrs=spad_pp_ptrs,
        spad_pp_size=spad_pp_size,
        inner_tiled_op_per_pp=inner_tiled_op_per_pp,
        mapping=tiled_ops_mapping
    )
    
    for core_id, pipeline in pipeline_mapping.pipelines.items():
        core = device.get_npu_core(core_id=core_id)
        kernel = main(core, pipeline)
        kernel.dispatch("MAIN")
        
    st = time.time()
    device.run_kernels()
    ed = time.time()
        
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(pipeline_mapping.summary(), f, indent=4)
        logger.info(f"Pipelined mapping summary saved to '{tmp_ouput_path}'.")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    simulated = ofm_b.restore()
    reference = torch.matmul(ifm, wgt.t())
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")