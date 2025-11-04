import torch
import math

from neuromta.framework import *
from neuromta.component.implementation.common import *
from neuromta.component.implementation.mta.runtime_kernel import *


__all__ = [
    "MTA_RT_LINEAR",
    "MTA_RT_RELU",
    "MTA_RT_CONV2D",
    "MTA_RT_MAXPOOL2D",
]


@MCA_RT_OPERATOR
def MTA_RT_LINEAR(
    device: MTA_DeviceBase, 
    core_grid: MTA_CoreGrid,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    buf_ofm:  MCA_TensorBuffer,
    
    accumulate_psum: bool = False
) -> MCA_TensorBuffer:
    
    M, K = buf_ifm.tensor_shape
    N, KW = buf_wgt.tensor_shape
    NB = buf_bias.tensor_shape[0] if buf_bias is not None else N
    
    dtype = buf_ifm.tensor_dtype
    acc_dtype = buf_ofm.tensor_dtype
    
    assert K == KW, f"[ERROR] The second dimension of input tensor (K={K}) must match the second dimension of weight tensor (KW={KW})."
    assert N == NB, f"[ERROR] The first dimension of weight tensor (N={N}) must match the first dimension of bias tensor (NB={NB})."
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Bias buffer must be allocated in L1 memory.")
    
    if buf_wgt.tensor_dtype != dtype:
        raise Exception(f"The data type of weight buffer {buf_wgt.tensor_dtype} does not match the expected data type {dtype}.")
    if buf_bias is not None and buf_bias.tensor_dtype != acc_dtype:
        raise Exception(f"The data type of bias buffer {buf_bias.tensor_dtype} does not match the expected data type {acc_dtype}.")
    if buf_ofm.tensor_shape != (M, N):
        raise Exception(f"The shape of output feature map buffer {buf_ofm.tensor_shape} does not match the expected shape {(M, N)}.")
    if buf_ofm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Output feature map buffer must be allocated in L1 memory.")
    
    m_tile_num = buf_ifm.y_n_pages
    n_tile_num = buf_wgt.y_n_pages
    k_tile_num = buf_ifm.x_n_pages
    
    for m_it in range(m_tile_num):
        for n_it in range(n_tile_num):
            core_id = core_grid[m_it % core_grid.shape[0], n_it % core_grid.shape[1]]
            core = device.get_npu_core(core_id=core_id)

            MTA_RT_KERNEL_TILED_LINEAR_BURST_K(
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
                accumulate_psum = accumulate_psum,
            )

    # return buf_ofm

@MCA_RT_OPERATOR
def MTA_RT_CONV2D(
    device: MTA_DeviceBase, 
    core_grid: MTA_CoreGrid,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    buf_ofm:  MCA_TensorBuffer,
    
    stride:    tuple[int, int] = (1, 1),
    padding:   tuple[int, int] = (0, 0),
    dilation:  tuple[int, int] = (1, 1),
    
    accumulate_psum: bool = False,
) -> MCA_TensorBuffer:
    N, H, W, C   = buf_ifm.tensor_shape
    FH, FW, K, C = buf_wgt.tensor_shape
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1
    
    dtype = buf_ifm.tensor_dtype
    acc_dtype = buf_ofm.tensor_dtype
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Bias buffer must be allocated in L1 memory.")
    
    if buf_wgt.tensor_dtype != dtype:
        raise Exception(f"The data type of weight buffer {buf_wgt.tensor_dtype} does not match the expected data type {dtype}.")
    if buf_bias is not None and buf_bias.tensor_dtype != acc_dtype:
        raise Exception(f"The data type of bias buffer {buf_bias.tensor_dtype} does not match the expected data type {acc_dtype}.")
    if buf_ofm.tensor_shape != (N, OH, OW, K):
        raise Exception(f"The shape of output feature map buffer {buf_ofm.tensor_shape} does not match the expected shape {(N, OH, OW, K)}.")
    if buf_ofm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Output feature map buffer must be allocated in L1 memory.")

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
        for k_tile_it in range(k_tile_num):  # kernel tile dimension
            for ow_tile_it in range(ow_tile_num):  # output width tile dimension
                for oh_it in range(OH):  # output height dimension
                    core_idx = ofm_tile_cnt // n_ofm_pages_per_core
                    core_id  = core_grid.core_ids[core_idx]
                    
                    MTA_RT_KERNEL_TILED_CONV2D_BURST_FHW_C(
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
                        accumulate_psum = accumulate_psum,
                    )
                    
                    ofm_tile_cnt += 1

@MCA_RT_OPERATOR
def MTA_RT_MAXPOOL2D(
    device: MTA_DeviceBase, 
    core_grid: MTA_CoreGrid,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_ofm:  MCA_TensorBuffer,
    
    kernel:    tuple[int, int] = (2, 2),
    stride:    tuple[int, int] = (1, 1),
    padding:   tuple[int, int] = (0, 0),
    dilation:  tuple[int, int] = (1, 1),
    
    accumulate_psum: bool = False
) -> MCA_TensorBuffer:
    N, H, W, C   = buf_ifm.tensor_shape
    FH, FW = kernel
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1
    
    dtype = buf_ifm.tensor_dtype
    acc_dtype = buf_ofm.tensor_dtype
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Input feature map buffer must be allocated in L1 memory.")
    if buf_ofm.tensor_shape != (N, OH, OW, C):
        raise Exception(f"The shape of output feature map buffer {buf_ofm.tensor_shape} does not match the expected shape {(N, OH, OW, C)}.")
    if buf_ofm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Output feature map buffer must be allocated in L1 memory.")

    ow_tile_size = buf_ofm.layout.y_page_size
    w_tile_size  = buf_ifm.layout.y_page_size
    c_tile_size  = buf_ifm.layout.x_page_size

    buf_ofm_n_pages = buf_ofm.n_pages
    n_ofm_pages_per_core = math.ceil(buf_ofm_n_pages / len(core_grid.core_ids))

    w_tile_num  = math.ceil(W  / w_tile_size)
    ow_tile_num = math.ceil(OW / ow_tile_size)
    c_tile_num  = math.ceil(C  / c_tile_size)
    
    ofm_tile_cnt = 0
    
    for n_it in range(N):  # batch dimension
        for oh_it in range(OH):  # output height dimension
            for ow_tile_it in range(ow_tile_num):  # output width tile dimension
                for c_tile_it in range(c_tile_num):
                    core_idx = ofm_tile_cnt // n_ofm_pages_per_core
                    core_id  = core_grid.core_ids[core_idx]
                    
                    MTA_RT_KERNEL_TILED_MAXPOOL2D_BURST_FHW_C(
                        core = device.get_core_from_id(core_id=core_id),
            
                        buf_ifm = buf_ifm.reference,
                        buf_ofm = buf_ofm.reference,
                        
                        ifm_shape = buf_ifm.buffer_shape,  # NOTE: use buffer_shape instead of tensor_shape to consider padding applied to the rows and columns MCA_TensorBuffer
                        kernel_shape = (FH, FW),
                        pad_shape = (PH, PW),
                        stride_shape = (SH, SW),
                        dilation_shape = (DH, DW),
                        
                        ow_tile_num = ow_tile_num,
                        w_tile_num  = w_tile_num,
                        c_tile_num  = c_tile_num,
                        
                        ow_tile_size = ow_tile_size,
                        w_tile_size  = w_tile_size,
                        c_tile_size  = c_tile_size,
                        
                        n_it = n_it,
                        oh_it = oh_it,
                        ow_tile_it = ow_tile_it,
                        c_tile_it = c_tile_it,

                        dtype = dtype,
                        acc_dtype = acc_dtype,
                        accumulate_psum = accumulate_psum,
                    )
                    
                    ofm_tile_cnt += 1

@MCA_RT_OPERATOR
def MTA_RT_RELU(
    device: MTA_DeviceBase, 
    core_grid: MTA_CoreGrid,
    
    buf_src:  MCA_TensorBuffer,
    buf_dst:  MCA_TensorBuffer | None = None,
    
    inplace:  bool = False,
) -> MCA_TensorBuffer:
    
    if buf_src.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Source buffer must be allocated in L1 memory.")
    if buf_dst is not None and buf_dst.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"Destination buffer must be allocated in L1 memory.")
    
    dtype = buf_src.tensor_dtype
    
    if inplace:
        if buf_src.tensor_dtype != dtype:
            raise Exception(f"In in-place operation, source and destination buffer must have the same data type.")
        buf_dst = buf_src
    
    for core_id in core_grid.core_ids:
        page_indice = buf_dst.get_page_idx_by_owner(core_id)
        core = device.get_npu_core(core_id=core_id)
        
        if len(page_indice) == 0:
            continue
        
        MTA_RT_KERNEL_TILED_RELU(
            core,
            
            buf_src = buf_src.get_reference_by_page_idx(*page_indice),
            buf_dst = buf_dst.get_reference_by_page_idx(*page_indice),

            dtype = dtype,
            vlen = buf_src.reference.page_size // dtype.itemsize,
        )       
       