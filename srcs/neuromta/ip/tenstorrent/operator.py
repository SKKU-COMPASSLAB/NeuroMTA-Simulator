import torch
import math

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *
from neuromta.ip.tenstorrent.rt_kernel import *


__all__ = [
    "TT_RT_DMA_LOAD",
    "TT_RT_DMA_STORE",
    "TT_RT_LINEAR",
    "TT_RT_RELU",
    "TT_RT_CONV2D",
]
    
    
@MCA_RT_OPERATOR
def TT_RT_DMA_LOAD(device: TenstorrentDevice, core_grid: MTA_CoreGrid, main_buf: MCA_TensorBuffer, l1_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    l1_buf = MCA_TensorBuffer(shape=main_buf.tensor_shape, dtype=main_buf.tensor_dtype, layout=l1_layout, device=device, core_ids=core_grid.core_ids)
    
    for core_id in l1_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)

        with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
            page_indice = l1_buf.get_page_idx_by_owner(core.core_id)
            buf_src = main_buf.get_reference_by_page_idx(*page_indice)
            buf_dst = l1_buf.get_reference_by_page_idx(*page_indice)

            core.mem_buffer_copy(buf_dst, buf_src, n_pages=len(page_indice))

    return l1_buf
      
@MCA_RT_OPERATOR
def TT_RT_DMA_STORE(device: TenstorrentDevice, core_grid: MTA_CoreGrid, l1_buf: MCA_TensorBuffer, main_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    main_buf = MCA_TensorBuffer(shape=l1_buf.tensor_shape, dtype=l1_buf.tensor_dtype, layout=main_layout, device=device, core_ids=core_grid.core_ids)

    for core_id in l1_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)
        
        with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
            page_indice = l1_buf.get_page_idx_by_owner(core_id)
            buf_src = l1_buf.get_reference_by_page_idx(*page_indice)
            buf_dst = main_buf.get_reference_by_page_idx(*page_indice)

            core.mem_buffer_copy(buf_dst, buf_src, n_pages=len(page_indice))

    return main_buf

@MCA_RT_OPERATOR
def TT_RT_LINEAR(
    device: TenstorrentDevice, 
    core_grid: MTA_CoreGrid,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    
    dtype:      torch.dtype = torch.float32,
    acc_dtype:  torch.dtype = torch.float32,
) -> MCA_TensorBuffer:
    
    M, K = buf_ifm.tensor_shape
    N, KW = buf_wgt.tensor_shape
    NB = buf_bias.tensor_shape[0] if buf_bias is not None else N
    
    assert K == KW, f"[ERROR] The second dimension of input tensor (K={K}) must match the second dimension of weight tensor (KW={KW})."
    assert N == NB, f"[ERROR] The first dimension of weight tensor (N={N}) must match the first dimension of bias tensor (NB={NB})."
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Bias buffer must be allocated in L1 memory.")
    
    m_tile_num = buf_ifm.y_n_pages
    n_tile_num = buf_wgt.y_n_pages
    k_tile_num = buf_ifm.x_n_pages
    
    buf_ofm = MCA_TensorBuffer(shape=(M, N), dtype=acc_dtype, layout=buf_ifm.layout, device=device, core_ids=core_grid.core_ids)
    
    for m_it in range(m_tile_num):
        for n_it in range(n_tile_num):
            core_id = core_grid[m_it % core_grid.shape[0], n_it % core_grid.shape[1]]
            core = device.get_npu_core(core_id=core_id)

            TT_RT_KERNEL_TILED_LINEAR_BURST_K(
                core,

                buf_ifm  = buf_ifm.reference, 
                buf_wgt  = buf_wgt.reference, 
                buf_bias = None if buf_bias is None else buf_bias.reference,
                buf_ofm  = buf_ofm.reference,
                
                m_tile_num = m_tile_num,
                n_tile_num = n_tile_num,
                k_tile_num = k_tile_num,
                
                m_it = m_it,
                n_it = n_it,
                
                dtype = dtype, 
                acc_dtype = acc_dtype,
            )

    return buf_ofm

@MCA_RT_OPERATOR
def TT_RT_CONV2D(
    device: TenstorrentDevice, 
    core_grid: MTA_CoreGrid,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    
    stride:    tuple[int, int] = (1, 1),
    padding:   tuple[int, int] = (0, 0),
    dilation:  tuple[int, int] = (1, 1),
    
    dtype:      torch.dtype = torch.float32,
    acc_dtype:  torch.dtype = torch.float32,
    cb_n_pages: int         = 8, 
) -> MCA_TensorBuffer:
    N, H, W, C   = buf_ifm.tensor_shape
    FH, FW, K, C = buf_wgt.tensor_shape
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Bias buffer must be allocated in L1 memory.")
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1
    
    buf_ofm = MCA_TensorBuffer(shape=(N, OH, OW, K), dtype=acc_dtype, layout=buf_ifm.layout, device=device, core_ids=core_grid.core_ids)

    ow_tile_size = buf_ofm.layout.y_page_size
    w_tile_size  = buf_ifm.layout.y_page_size
    c_tile_size  = buf_ifm.layout.x_page_size
    k_tile_size  = buf_ofm.layout.x_page_size

    buf_ofm_n_pages = buf_ofm.n_pages
    n_ofm_pages_per_core = math.ceil(buf_ofm_n_pages / len(core_grid.core_ids))

    w_tile_num  = math.ceil(W  / w_tile_size)
    ow_tile_num = math.ceil(OW / ow_tile_size)
    c_tile_num  = math.ceil(C  / c_tile_size)
    k_tile_num  = math.ceil(K  / k_tile_size)
    
    ofm_tile_cnt = 0
    
    for n_it in range(N):  # batch dimension
        for oh_it in range(OH):  # output height dimension
            for ow_tile_it in range(ow_tile_num):  # output width tile dimension
                for k_tile_it in range(k_tile_num):  # kernel tile dimension
                    core_idx = ofm_tile_cnt // n_ofm_pages_per_core
                    core_id  = core_grid.core_ids[core_idx]
                    
                    TT_RT_KERNEL_TILED_CONV2D_BURST_FHW_C(
                        core = device.get_core_from_id(core_id=core_id),
            
                        buf_ifm = buf_ifm.reference,
                        buf_wgt = buf_wgt.reference,
                        buf_bias = buf_bias.reference if buf_bias is not None else None,
                        buf_ofm = buf_ofm.reference,
                        
                        ifm_shape = buf_ifm.buffer_shape,  # NOTE: use buffer_shape instead of tensor_shape to consider padding applied to the rows and columns MCA_TensorBuffer
                        wgt_shape = buf_wgt.buffer_shape,  # NOTE: use buffer_shape instead of tensor_shape to consider padding applied to the rows and columns MCA_TensorBuffer
                        pad_shape = (PH, PW),
                        stride_shape = (SH, SW),
                        dilation_shape = (DH, DW),
                        
                        ow_tile_num = ow_tile_num,
                        w_tile_num  = w_tile_num,
                        c_tile_num  = c_tile_num,
                        k_tile_num  = k_tile_num,
                        
                        ow_tile_size = ow_tile_size,
                        w_tile_size  = w_tile_size,
                        c_tile_size  = c_tile_size,
                        
                        n_it = n_it,
                        oh_it = oh_it,
                        ow_tile_it = ow_tile_it,
                        k_tile_it = k_tile_it,

                        dtype = dtype,
                        acc_dtype = acc_dtype,
                    )
                    
                    ofm_tile_cnt += 1
                
    return buf_ofm

@MCA_RT_OPERATOR
def TT_RT_RELU(
    device: TenstorrentDevice, 
    core_grid: MTA_CoreGrid,
    
    buf_src:  MCA_TensorBuffer,
    dtype:    torch.dtype = None, 
    inplace:  bool = False,
) -> MCA_TensorBuffer:
    
    if buf_src.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Source buffer must be allocated in L1 memory.")
    
    if dtype is None:
        dtype = buf_src.tensor_dtype
    
    if not inplace:
        buf_dst = MCA_TensorBuffer(shape=buf_src.tensor_shape, dtype=dtype, layout=buf_src.layout, device=device, core_ids=core_grid.core_ids)
    else:
        if buf_src.tensor_dtype != dtype:
            raise Exception(f"[ERROR] In in-place operation, source and destination buffer must have the same data type.")
        buf_dst = buf_src
    
    for core_id in core_grid.core_ids:
        page_indice = buf_dst.get_page_idx_by_owner(core_id)
        core = device.get_npu_core(core_id=core_id)
        
        TT_RT_KERNEL_TILED_RELU(
            core,
            
            buf_src = buf_src.get_reference_by_page_idx(*page_indice),
            buf_dst = buf_dst.get_reference_by_page_idx(*page_indice),

            dtype = dtype,
            vlen = core.vpu_context.vlen_max   # TODO: hard-coded vector length (should be configurable)
        )
    
    return buf_dst
