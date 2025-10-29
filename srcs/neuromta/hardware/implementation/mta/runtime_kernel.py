import math
import torch

from neuromta.framework import *
from neuromta.hardware.core import *
from neuromta.hardware.context import *
from neuromta.hardware.implementation.common import *


__all__ = [
    "MTA_RT_KERNEL_TILED_LINEAR_BURST_K",
    "MTA_RT_KERNEL_TILED_CONV2D_BURST_FHW_C",
    "MTA_RT_KERNEL_TILED_MAXPOOL2D_BURST_FHW_C",
    "MTA_RT_KERNEL_TILED_RELU",
]


@MCA_RT_KERNEL
def MTA_RT_KERNEL_TILED_LINEAR_BURST_K(
    core: NPUCore,
    
    buf_ifm:  BufferPointer, 
    buf_wgt:  BufferPointer, 
    buf_bias: BufferPointer | None,
    buf_ofm:  BufferPointer,
    
    m_tile_num: int,
    n_tile_num: int,
    k_tile_num: int,

    m_it: int,
    n_it: int,

    dtype: torch.dtype, acc_dtype: torch.dtype,
    accumulate_psum: bool=False
):  
    containers = [DataContainer() for _ in range(4)]
    
    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
    
    bias_page_idx = n_it
    ofm_page_idx = (m_it * n_tile_num) + n_it

    if buf_bias is not None:
        core.mem_read_with_container(buf_bias[bias_page_idx], containers[2])

    for k_it in range(k_tile_num):
        ifm_page_idx = (m_it * k_tile_num) + k_it
        wgt_page_idx = (n_it * k_tile_num) + k_it
        
        core.mem_container_init(containers[0], shape=core.mxu_context.ifm_tile_shape, dtype=dtype)
        core.mem_read_with_container(buf_ifm[ifm_page_idx], containers[0], offset=0)
        core.mem_read_with_container(buf_wgt[wgt_page_idx], containers[1])

        preload_psum = True if k_it == 0 else False
        flush_ofm    = True if ((k_it == (k_tile_num - 1)) and not accumulate_psum) else False
        
        core.mxu_tiled_gemm(
            *containers,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
            wgt_transposed=True,    # TODO: assume that the weight matrix is transposed (for contiguous memory access)
            psum_vectored=True,     # TODO: assume taht the psum is a bias vector, not the partial sum matrix
        )
    
    if accumulate_psum:
        core.mem_container_init(containers[0], shape=core.mxu_context.ofm_tile_shape, dtype=acc_dtype)
        core.mem_read_with_container(buf_ofm[ofm_page_idx], containers[0])
        core.mxu_tiled_elemwise(
            op=MXUElementwiseOp.ADD,
            src=containers[0],
            dst=containers[3],
            flush_ofm=True,
        )
    
    core.mem_write_with_container(buf_ofm[ofm_page_idx], containers[3])
    
    
@MCA_RT_KERNEL
def MTA_RT_KERNEL_TILED_CONV2D_BURST_FHW_C(
    core: NPUCore,
    
    buf_ifm:  BufferPointer,            # memory layout: ROW_MAJOR_REFERENCE / shape=(N*H*W, C)
    buf_wgt:  BufferPointer,            # memory layout: ROW_MAJOR_REFERENCE / shape=(FH*FW*K, C)
    buf_bias: BufferPointer | None,     # memory layout: ROW_MAJOR_REFERENCE / shape=(1, K)
    buf_ofm:  BufferPointer,            # memory layout: ROW_MAJOR_REFERENCE / shape=(N*OH*OW, K)
    
    ifm_shape: tuple[int, int, int, int],   # (N, H, W, C)
    wgt_shape: tuple[int, int, int, int],   # (FH, FW, K, C)
    pad_shape: tuple[int, int],             # (PH, PW)
    stride_shape: tuple[int, int],          # (SH, SW)
    dilation_shape: tuple[int, int],        # (DH, DW)
    
    ow_tile_num: int,
    w_tile_num: int,
    c_tile_num: int,
    k_tile_num: int,
    
    ow_tile_size: int,
    w_tile_size: int,
    c_tile_size: int,
    
    n_it: int,
    oh_it: int,
    ow_tile_it: int,
    k_tile_it: int,

    dtype: torch.dtype,
    acc_dtype: torch.dtype,
    accumulate_psum: bool=False,
):
    _, H, W, _ = ifm_shape
    FH, FW, _, _ = wgt_shape
    PH, PW = pad_shape
    SH, SW = stride_shape
    DH, DW = dilation_shape
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1
    
    ofm_page_idx  = (n_it * OH * ow_tile_num * k_tile_num) + (oh_it * ow_tile_num * k_tile_num) + (ow_tile_it * k_tile_num) + k_tile_it
    bias_page_idx = k_tile_it
    
    fh_min = max(0,  math.ceil((PH - oh_it) / DH))
    fh_max = min(FH, math.ceil((H + PH - oh_it) / DH))
        
    containers = [DataContainer() for _ in range(4)]
        
    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
    
    if buf_bias is not None:
        core.mem_read_with_container(buf_bias[bias_page_idx], containers[2])
    
    for fh_it in range(fh_min, fh_max, 1):
        for fw_it in range(FW):
            h_it = (SH * oh_it) + (DH * fh_it) - PH
            
            for c_tile_it in range(c_tile_num):
                wgt_page_idx  = (fh_it * FW * k_tile_num * c_tile_num) + (fw_it * k_tile_num * c_tile_num) + (k_tile_it * c_tile_num) + c_tile_it
            
                core.mem_read_with_container(buf_wgt[wgt_page_idx], containers[1])
                
                core.mem_container_init(containers[0], shape=(ow_tile_size, c_tile_size), dtype=dtype)                    
                copy_layout_pattern: dict[int, list[tuple[int, int]]] = {}
                
                for ow_intra_tile_it in range(ow_tile_size):
                    ow_it = ow_tile_it * ow_tile_size + ow_intra_tile_it
                    w_it = (SW * ow_it) + (DW * fw_it) - PW
                    w_tile_it = w_it // w_tile_size
                    
                    ifm_page_idx = (n_it * H * w_tile_num * c_tile_num) + (h_it * w_tile_num * c_tile_num) + (w_tile_it * c_tile_num) + c_tile_it
                        
                    if 0 <= h_it < H and 0 <= w_it < W:
                        if ifm_page_idx not in copy_layout_pattern.keys():
                            copy_layout_pattern[ifm_page_idx] = []
                        
                        dst_seg_id = ow_intra_tile_it
                        src_seg_id = (w_it % w_tile_size)
                        
                        copy_layout_pattern[ifm_page_idx].append((dst_seg_id, src_seg_id, 0, 0, c_tile_size * dtype.itemsize))
                        
                for ifm_page_idx, pattern in copy_layout_pattern.items():
                    core.mem_read_with_container(
                        buf_ifm[ifm_page_idx], 
                        containers[0], 
                        copy_layout_width=c_tile_size * dtype.itemsize,
                        copy_layout_pattern=pattern,
                    )
                
                preload_psum = (fh_it == fh_min and fw_it == 0 and c_tile_it == 0)
                flush_ofm    = (fh_it == (fh_max - 1) and fw_it == (FW - 1) and c_tile_it == (c_tile_num - 1)) and not accumulate_psum
                
                core.mxu_tiled_gemm(
                    *containers,
                    preload_psum=preload_psum,
                    flush_ofm=flush_ofm,
                    wgt_transposed=True,    # TODO: assume that the weight matrix is transposed (for contiguous memory access)
                    psum_vectored=True,     # TODO: assume taht the psum is a bias vector, not the partial sum matrix
                )
                
    if accumulate_psum:
        core.mem_container_init(containers[0], shape=core.mxu_context.ifm_tile_shape, dtype=dtype)
        core.mem_read_with_container(buf_ofm[ofm_page_idx], containers[0])
        core.mxu_tiled_elemwise(
            op=MXUElementwiseOp.ADD,
            src=containers[0],
            dst=containers[3],
            flush_ofm=True,
        )
    
    core.mem_write_with_container(buf_ofm[ofm_page_idx], containers[3])
    
    
@MCA_RT_KERNEL
def MTA_RT_KERNEL_TILED_MAXPOOL2D_BURST_FHW_C(
    core: NPUCore,
    
    buf_ifm:  BufferPointer,            # memory layout: ROW_MAJOR_REFERENCE / shape=(N*H*W, C)
    buf_ofm:  BufferPointer,            # memory layout: ROW_MAJOR_REFERENCE / shape=(N*OH*OW, K)
    
    ifm_shape: tuple[int, int, int, int],   # (N, H, W, C)
    kernel_shape: tuple[int, int],          # (FH, FW)
    pad_shape: tuple[int, int],             # (PH, PW)
    stride_shape: tuple[int, int],          # (SH, SW)
    dilation_shape: tuple[int, int],        # (DH, DW)
    
    ow_tile_num: int,
    w_tile_num: int,
    c_tile_num: int,
    
    ow_tile_size: int,
    w_tile_size: int,
    c_tile_size: int,
    
    n_it: int,
    oh_it: int,
    ow_tile_it: int,
    c_tile_it: int,

    dtype: torch.dtype,
    acc_dtype: torch.dtype,
    accumulate_psum: bool=False,
):
    _, H, W, _ = ifm_shape
    FH, FW = kernel_shape
    PH, PW = pad_shape
    SH, SW = stride_shape
    DH, DW = dilation_shape
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1

    ofm_page_idx = (n_it * OH * ow_tile_num * c_tile_num) + (oh_it * ow_tile_num * c_tile_num) + (ow_tile_it * c_tile_num) + c_tile_it

    fh_min = max(0,  math.ceil((PH - oh_it) / DH))
    fh_max = min(FH, math.ceil((H + PH - oh_it) / DH))
        
    containers = [DataContainer() for _ in range(3)]
        
    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
    
    core.mem_container_init(containers[1], shape=(ow_tile_size, c_tile_size), dtype=dtype)
    
    for fh_it in range(fh_min, fh_max, 1):
        for fw_it in range(FW):
            h_it = (SH * oh_it) + (DH * fh_it) - PH

            core.mem_container_init(containers[0], shape=(ow_tile_size, c_tile_size), dtype=dtype)                    
            copy_layout_pattern: dict[int, list[tuple[int, int]]] = {}
            
            for ow_intra_tile_it in range(ow_tile_size):
                ow_it = ow_tile_it * ow_tile_size + ow_intra_tile_it
                w_it = (SW * ow_it) + (DW * fw_it) - PW
                w_tile_it = w_it // w_tile_size
                
                ifm_page_idx = (n_it * H * w_tile_num * c_tile_num) + (h_it * w_tile_num * c_tile_num) + (w_tile_it * c_tile_num) + c_tile_it
                    
                if 0 <= h_it < H and 0 <= w_it < W:
                    if ifm_page_idx not in copy_layout_pattern.keys():
                        copy_layout_pattern[ifm_page_idx] = []
                    
                    dst_seg_id = ow_intra_tile_it
                    src_seg_id = (w_it % w_tile_size)
                    
                    copy_layout_pattern[ifm_page_idx].append((dst_seg_id, src_seg_id, 0, 0, c_tile_size * dtype.itemsize))
                    
            for ifm_page_idx, pattern in copy_layout_pattern.items():
                core.mem_read_with_container(
                    buf_ifm[ifm_page_idx], 
                    containers[0], 
                    copy_layout_width=c_tile_size * dtype.itemsize,
                    copy_layout_pattern=pattern,
                )
            
            preload_psum = (fh_it == fh_min and fw_it == 0)
            flush_ofm    = (fh_it == (fh_max - 1) and fw_it == (FW - 1)) and not accumulate_psum
            
            core.mxu_tiled_maxpool(
                *containers,
                preload_psum=preload_psum,
                flush_ofm=flush_ofm,
            )
            
    if accumulate_psum:
        core.mem_container_init(containers[0], shape=core.mxu_context.ifm_tile_shape, dtype=dtype)
        core.mem_read_with_container(buf_ofm[ofm_page_idx], containers[0])
        core.mxu_tiled_elemwise(
            op=MXUElementwiseOp.CMP_MAX,
            src=containers[0],
            dst=containers[1],
            flush_ofm=True,
        )
    
    core.mem_write_with_container(buf_ofm[ofm_page_idx], containers[2])
                    
    
@MCA_RT_KERNEL
def MTA_RT_KERNEL_TILED_RELU(
    core: NPUCore,
    
    buf_src:  BufferPointer,
    buf_dst:  BufferPointer,
    
    dtype: torch.dtype,
    vlen: int=None,
):  
    if buf_src.page_size != buf_dst.page_size:
        raise Exception(f"Source and destination buffer must have the same size.")
    if buf_src.page_size % (vlen * dtype.itemsize) != 0:
        raise Exception(f"Buffer size {buf_src.page_size} is not divisible by vector register length {(vlen * dtype.itemsize)}.")
    if vlen is None:
        vlen = core.vpu_context.vlen_max
    
    n_reg_per_page = math.ceil(buf_src.page_size / (vlen * dtype.itemsize))
    n_vreg_num     = core.vpu_context.get_vreg_num_with_config(vlen=vlen, vdtype=dtype)
    n_bursts       = math.ceil(n_reg_per_page / n_vreg_num)
    
    container = DataContainer()
    
    core.vpu_reconfigure(vlen=vlen, vdtype=dtype)
    
    for page_idx in range(buf_src.n_pages):
        core.mem_read_with_container(buf_src[page_idx], container)
        
        for step in range(n_bursts):
            st = step * n_vreg_num
            ed = min(st + n_vreg_num, n_reg_per_page)
            burst_len = ed - st
            offset = st * vlen * dtype.itemsize

            core.vpu_load_reg(container, 0, burst_len=burst_len, offset=offset)
            core.vpu_execute(VPUOperator.RELU, vreg_a=0, inplace=True, burst_len=burst_len)
            core.vpu_store_reg(container, 0, burst_len=burst_len, offset=offset)
            
        core.mem_write_with_container(buf_dst[page_idx], container)
        

@MCA_RT_KERNEL
def MTA_RT_KERNEL_TILED_FLATTEN_4D(
    core: NPUCore,
    
    buf_src:  BufferPointer,
    buf_dst:  BufferPointer,
    
    dtype: torch.dtype,
    
    src_shape: tuple[int, int, int, int],  # (N, H, W, C)
    dst_shape: tuple[int, int],            # (COL, ROW)
    page_shape: tuple[int, int],           # (COL, ROW)
    
    target_dst_page_indice: list[int]
):
    SN, SH, SW, SC = src_shape
    DC, DR = dst_shape
    PC, PR = page_shape
    
    if SC % PR != 0:
        raise Exception(f"Flatten operation requires that the source C dimension ({SC}) is divisible by the page ROW size ({PR}).")
    if DR % PR != 0:
        raise Exception(f"Flatten operation requires that the destination ROW dimension ({DR}) is divisible by the page ROW size ({PR}).")
    
    n_sw_tiles = math.ceil(SW / PC)
    n_sc_tiles = math.ceil(SC / PR)
    
    sw_pad = (n_sw_tiles * PC) - SW
    
    n_dc_tiles = math.ceil(DC / PC)
    n_dr_tiles = math.ceil(DR / PR)
    
    dc_pad = (n_dc_tiles * PC) - DC
    
    container = DataContainer()
    
    for dst_page_idx in target_dst_page_indice:
        dc_tile_it = (dst_page_idx // n_dr_tiles)
        dr_tile_it = (dst_page_idx % n_dr_tiles)
        
        core.mem_container_init(container, shape=(PC, PR), dtype=dtype)
        
        src_page_idx_offset = (dc_tile_it * PC * n_sw_tiles * n_sc_tiles) + dr_tile_it
        