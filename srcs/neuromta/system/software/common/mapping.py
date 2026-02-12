import torch
import functools
import tqdm
from typing import Sequence

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.implementation import *
from neuromta.component.implementation.mapping import *


__all__ = [
    "MCA_MAPPER_LINEAR",
    "MCA_MAPPER_UNARY_INPLACE",
    "MCA_MAPPER_CONV2D",
    "MCA_MAPPER_FLATTEN",
]

def MCA_MAPPER_LINEAR(op_sig: MCA_OperatorSignature) -> MCA_OperatorSignature:
    ifm = op_sig.buffers["ifm"]
    wgt = op_sig.buffers["wgt"]
    bias = op_sig.buffers["bias"]
    ofm = op_sig.buffers["ofm"]
    
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
    
    for m_s in range(ofm.shard_grid[0]):
        for n_s in range(ofm.shard_grid[1]):
            for m_t in range(ofm.tile_grid_per_shard[0]):
                for n_t in range(ofm.tile_grid_per_shard[1]):
                    tiled_op = op_sig.new_tiled_op()
                    
                    for k_s in range(ifm.shard_grid[1]):
                        for k_t in range(ifm.tile_grid_per_shard[1]):
                            tiled_op.add_uop(
                                i_tiles=[
                                    op_sig.tiles["ifm"][(m_s, k_s, m_t, k_t)],
                                    op_sig.tiles["wgt"][(n_s, k_s, n_t, k_t)],
                                    op_sig.tiles["bias"][(0, n_s, 0, n_t)],
                                ],
                                o_tile=op_sig.tiles["ofm"][(m_s, n_s, m_t, n_t)],
                            )
    
    return op_sig
    
def MCA_MAPPER_UNARY_INPLACE(op_sig: MCA_OperatorSignature) -> MCA_OperatorSignature:
    ifm = op_sig.buffers["ifm"]
    
    for m_s in range(ifm.shard_grid[0]):
        for k_s in range(ifm.shard_grid[1]):
            for m_t in range(ifm.tile_grid_per_shard[0]):
                for k_t in range(ifm.tile_grid_per_shard[1]):
                    tiled_op = op_sig.new_tiled_op()
                    tiled_op.add_uop(
                        i_tiles=[op_sig.tiles["ifm"][(m_s, k_s, m_t, k_t)]],
                        o_tile=op_sig.tiles["ifm"][(m_s, k_s, m_t, k_t)],
                    )
    
    return op_sig
    
def MCA_MAPPER_CONV2D(
    op_sig: MCA_OperatorSignature,
    is_conv2d: bool=True,
    use_collective_tile_load: bool=False,
) -> MCA_OperatorSignature:
    ifm = op_sig.buffers["ifm"]
    if is_conv2d:
        wgt = op_sig.buffers["wgt"]
        bias = op_sig.buffers["bias"]
    ofm = op_sig.buffers["ofm"]
    
    stride = op_sig.global_kwargs.get("stride", (1, 1))
    padding = op_sig.global_kwargs.get("padding", (0, 0))
    dilation = op_sig.global_kwargs.get("dilation", (1, 1))
    groups = op_sig.global_kwargs.get("groups", 1)
    window = op_sig.global_kwargs.get("window", None)
    
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
    
    # if is_conv2d:
    #     print(ifm.shape, wgt.shape)
    
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
        
    # if is_conv2d:
    #     wgt_tiles: dict[tuple[int, ...], TileSignature] = {
    #         (y_s, x_s, y_t, x_t): TileSignature("wgt", wgt, y_s, x_s, y_t, x_t)
    #         for y_s in range(wgt.shard_grid[0])
    #         for x_s in range(wgt.shard_grid[1])
    #         for y_t in range(wgt.tile_grid_per_shard[0])
    #         for x_t in range(wgt.tile_grid_per_shard[1])
    #     }
        
    #     bias_tiles: dict[tuple[int, ...], TileSignature] = {
    #         (0, x_s, 0, x_t): TileSignature("bias", bias, 0, x_s, 0, x_t)
    #         for x_s in range(bias.shard_grid[1])
    #         for x_t in range(bias.tile_grid_per_shard[1])
    #     }
        
    # ifm_tiles: dict[tuple[int, ...], TileSignature] = {
    #     (y_s, x_s, y_t, x_t): TileSignature("ifm", ifm, y_s, x_s, y_t, x_t)
    #     for y_s in range(ifm.shard_grid[0])
    #     for x_s in range(ifm.shard_grid[1])
    #     for y_t in range(ifm.tile_grid_per_shard[0])
    #     for x_t in range(ifm.tile_grid_per_shard[1])
    # }
    
    # ofm_tiles: dict[tuple[int, ...], TileSignature] = {
    #     (y_s, x_s, y_t, x_t): TileSignature("ofm", ofm, y_s, x_s, y_t, x_t)
    #     for y_s in range(ofm.shard_grid[0])
    #     for x_s in range(ofm.shard_grid[1])
    #     for y_t in range(ofm.tile_grid_per_shard[0])
    #     for x_t in range(ofm.tile_grid_per_shard[1])
    # }
    
    # tiled_ops: list[TiledOperatorSignature] = []
    
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
                            
                            # tiled_op = TiledOperatorSignature(i_tiles=[], o_tile=None)
                            tiled_op = op_sig.new_tiled_op()
                            
                            # GET OFM tile signature
                            ofm_y_tile_idx = n_it * OH * OW_N_TILES + oh_it * OW_N_TILES + ow_tile_it
                            ofm_x_tile_idx = k_tile_it
                            ofm_tile_idx = ofm.get_shard_grid_from_tile_grid_idx(ofm_y_tile_idx, ofm_x_tile_idx)
                            ofm_tile = op_sig.tiles["ofm"][ofm_tile_idx]
                            # tiled_op.o_tile = ofm_tiles[ofm_tile_idx]
                            # tiled_op.i_tiles = []
                            # tiled_op.op_kwargs = {"ifm_load_kwargs": []}

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
                                            # wgt_tile = wgt_tiles[wgt_tile_idx]
                                            wgt_tile = op_sig.tiles["wgt"][wgt_tile_idx]
                                            
                                            # GET PSUM tile signature (BIAS or partially merged OFM)
                                            psum_tile_idx = bias.get_shard_grid_from_tile_grid_idx(0, k_tile_it)
                                            # psum_tile = bias_tiles[psum_tile_idx]
                                            psum_tile = op_sig.tiles["bias"][psum_tile_idx]

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
                                            ifm_x_tile_idx = c_tile_it if is_conv2d else k_tile_it
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
                                                src_tiles=[op_sig.tiles["ifm"][idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()],
                                                memcpy_patterns=list(ifm_tile_idx_with_memcpy_pattern.values()),
                                            )
                                            uop_i_tiles.append(ifm_tile_sig)
                                        else:
                                            # Compute core uses all the individual IFM tiles and handles the memcpy patterns internally
                                            uop_kwargs["use_collective_tile_load"] = False
                                            uop_kwargs["ifm_tile_count"] = len(ifm_tile_idx_with_memcpy_pattern)
                                            uop_kwargs["memcpy_pattern"] = list(ifm_tile_idx_with_memcpy_pattern.values())
                                            uop_i_tiles.extend([op_sig.tiles["ifm"][idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()])
                                                
                                        if is_conv2d:
                                            uop_i_tiles.append(wgt_tile)
                                            uop_i_tiles.append(psum_tile)
                                        
                                        tiled_op.add_uop(
                                            i_tiles=uop_i_tiles,
                                            o_tile=ofm_tile,
                                            op_kwargs=uop_kwargs,
                                        )    
                                        
                                        # tiled_op.i_tiles.append(tuple(uop_i_tiles))
                                        # tiled_op.op_kwargs["ifm_load_kwargs"].append(uop_kwargs)
                    
                            # logger.debug(f"Generated CONV2D tiled_op for OFM tile idx {tiled_op.o_tile.coords} with {len(tiled_op.i_tiles)} uops")
                            # tiled_ops.append(tiled_op)
                            pbar.update(1)
    
    logger.debug(f"mapper generated {len(op_sig.tiled_ops)} in total")
                            
    return op_sig

def MCA_MAPPER_FLATTEN(
    op_sig: MCA_OperatorSignature,
) -> MCA_OperatorSignature:
    ifm = op_sig.buffers["ifm"]
    ofm = op_sig.buffers["ofm"]
    
    if len(ofm.shape) != 2:
        raise Exception(f"OFM tensor must be 2D for TENSOR_FLATTEN operator mapping, but got shape: {ofm.shape}")
    if ifm.dtype != ofm.dtype:
        raise Exception(f"IFM and OFM dtype mismatch: {ifm.dtype} != {ofm.dtype}")
    if ifm.numel != ofm.numel:
        raise Exception(f"IFM and OFM number of elements mismatch: {ifm.numel} != {ofm.numel}")
    if ifm.tile_shape[-1] != ofm.tile_shape[-1]:
        raise Exception(f"IFM and OFM does not have the same row-wise tile size: {ifm.tile_shape[-1]} != {ofm.tile_shape[-1]}")
    
    # ifm_tiles: dict[tuple[int, ...], TileSignature] = {
    #     (y_s, x_s, y_t, x_t): TileSignature("ifm", ifm, y_s, x_s, y_t, x_t)
    #     for y_s in range(ifm.shard_grid[0])
    #     for x_s in range(ifm.shard_grid[1])
    #     for y_t in range(ifm.tile_grid_per_shard[0])
    #     for x_t in range(ifm.tile_grid_per_shard[1])
    # }
    
    # ofm_tiles: dict[tuple[int, ...], TileSignature] = {
    #     (y_s, x_s, y_t, x_t): TileSignature("ofm", ofm, y_s, x_s, y_t, x_t)
    #     for y_s in range(ofm.shard_grid[0])
    #     for x_s in range(ofm.shard_grid[1])
    #     for y_t in range(ofm.tile_grid_per_shard[0])
    #     for x_t in range(ofm.tile_grid_per_shard[1])
    # }
    
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
                
            # tiled_op = TiledOperatorSignature(
            #     i_tiles=[],
            #     o_tile=op_sig.tiles["ofm"][ofm_tile_idx],
            # )
            tiled_op = op_sig.new_tiled_op()
            
            ifm_tile = CollectiveTileSignature(
                buf_name="ifm",
                buf=ifm,
                src_tiles=[op_sig.tiles["ifm"][idx] for idx in ifm_tile_idx_with_memcpy_pattern.keys()],
                memcpy_patterns=list(ifm_tile_idx_with_memcpy_pattern.values()),
            )
            
            tiled_op.add_uop(
                i_tiles=[ifm_tile],
                o_tile=op_sig.tiles["ofm"][ofm_tile_idx],
            )
            
            # tiled_op.i_tiles.append((ifm_tile,))
            # tiled_ops.append(tiled_op)
    
    return op_sig