import math
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *
from neuromta.ip.tenstorrent.software.rt_kernel import *


__all__ = [
    "TT_RT_DMA_LOAD",
    "TT_RT_DMA_STORE",
    "TT_RT_LINEAR",
]
    
    
@MCA_RT_OPERATOR
def TT_RT_DMA_LOAD(device: TenstorrentDevice, core_grid: MTA_CoreGrid, main_buf: MCA_TensorBuffer, l1_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    l1_buf = MCA_TensorBuffer(shape=main_buf.tensor_shape, dtype=main_buf.tensor_dtype, layout=l1_layout, device=device, core_ids=core_grid.core_ids)
    core_id_to_page_idx_map = l1_buf.get_core_id_to_page_coord_mapping()
    
    for core_id in l1_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)

        page_coords = core_id_to_page_idx_map[core_id]
        buf_src = main_buf.get_multiple_pages_reference(*page_coords)
        buf_dst = l1_buf.get_multiple_pages_reference(*page_coords)
        
        TT_RT_KERNEL_MEM_BUFFER_COPY(core, src=buf_src, dst=buf_dst, n_pages=len(page_coords))
    
    return l1_buf
    
        
@MCA_RT_OPERATOR
def TT_RT_DMA_STORE(device: TenstorrentDevice, core_grid: MTA_CoreGrid, l1_buf: MCA_TensorBuffer, main_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    main_buf = MCA_TensorBuffer(shape=l1_buf.tensor_shape, dtype=l1_buf.tensor_dtype, layout=main_layout, device=device, core_ids=core_grid.core_ids)
    core_id_to_page_idx_map = l1_buf.get_core_id_to_page_coord_mapping()
    
    for core_id in l1_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)

        page_coords = core_id_to_page_idx_map[core_id]
        buf_src = l1_buf.get_multiple_pages_reference(*page_coords)
        buf_dst = main_buf.get_multiple_pages_reference(*page_coords)

        TT_RT_KERNEL_MEM_BUFFER_COPY(core, src=buf_src, dst=buf_dst, n_pages=len(page_coords))
        
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
    cb_n_pages: int         = 8, 
) -> MCA_TensorBuffer:
    
    M, K = buf_ifm.tensor_shape
    N, KW = buf_wgt.tensor_shape
    _, NB = buf_bias.tensor_shape if buf_bias is not None else (1, N)
    
    assert K == KW, f"[ERROR] The second dimension of input tensor (K={K}) must match the second dimension of weight tensor (KW={KW})."
    assert N == NB, f"[ERROR] The first dimension of weight tensor (N={N}) must match the first dimension of bias tensor (NB={NB})."
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Bias buffer must be allocated in L1 memory.")
    
    buf_ifm = buf_ifm
    buf_wgt = buf_wgt
    buf_bias = buf_bias
    
    dtype = dtype
    acc_dtype = acc_dtype
    
    cb_n_pages = cb_n_pages
    
    k_tile_num = buf_ifm.x_shard_grid * buf_ifm.x_page_num_per_shard
    load_burst_len = cb_n_pages // 2
    
    ofm_layout = MCA_TensorMemoryLayout(
        mem_type=MCA_TensorMemoryType.L1,
        grid_shape=core_grid.shape,
        page_shape=(32, 32),
    )
    
    buf_ofm = MCA_TensorBuffer(shape=(M, N), dtype=acc_dtype, layout=ofm_layout, device=device, core_ids=core_grid.core_ids)
    
    cb_ifm_ptrs:  dict[int, BufferPointer] = {core_id: BufferPointer() for core_id in core_grid.core_ids}
    cb_wgt_ptrs:  dict[int, BufferPointer] = {core_id: BufferPointer() for core_id in core_grid.core_ids}
    cb_bias_ptrs: dict[int, BufferPointer] = {core_id: BufferPointer() for core_id in core_grid.core_ids}
    cb_ofm_ptrs:  dict[int, BufferPointer] = {core_id: BufferPointer() for core_id in core_grid.core_ids}

    for core_id in core_grid.core_ids:
        core = device.get_npu_core(core_id=core_id)
        
        TT_RT_KERNEL_LOCAL_CB_ALLOCATE(core, cb_ifm_ptrs[core_id], buf_ifm.reference.resolve(is_read=True).page_size, cb_n_pages)
        TT_RT_KERNEL_LOCAL_CB_ALLOCATE(core, cb_wgt_ptrs[core_id], buf_wgt.reference.resolve(is_read=True).page_size, cb_n_pages)
        TT_RT_KERNEL_LOCAL_CB_ALLOCATE(core, cb_ofm_ptrs[core_id], buf_ofm.reference.resolve(is_read=True).page_size, cb_n_pages)
        if buf_bias is not None:
            TT_RT_KERNEL_LOCAL_CB_ALLOCATE(core, cb_bias_ptrs[core_id], buf_bias.reference.resolve(is_read=True).page_size, cb_n_pages)

    for y_pi in range(buf_ofm.y_page_num_per_shard):
        for x_pi in range(buf_ofm.x_page_num_per_shard):
            for y_si in range(buf_ofm.y_shard_grid):
                for x_si in range(buf_ofm.x_shard_grid):
                    core_id = core_grid[y_si, x_si]
                    core = device.get_npu_core(core_id=core_id)

                    m_pi = y_si * buf_ofm.y_page_num_per_shard + y_pi
                    n_pi = x_si * buf_ofm.x_page_num_per_shard + x_pi
                    
                    TT_RT_KERNEL_TILED_LINEAR_BURST_K(
                        core,
                        
                        buf_ifm  = buf_ifm.get_row_contiguous_reference(m_pi), 
                        buf_wgt  = buf_wgt.get_row_contiguous_reference(n_pi) , 
                        buf_bias = None if buf_bias is None else buf_bias.reference[n_pi],
                        buf_ofm  = buf_ofm.get_page_reference((y_si, x_si), (y_pi, x_pi)),
                            
                        cb_ifm   = cb_ifm_ptrs[core_id], 
                        cb_wgt   = cb_wgt_ptrs[core_id], 
                        cb_bias  = cb_bias_ptrs[core_id], 
                        cb_ofm   = cb_ofm_ptrs[core_id],
                        
                        k_tile_num = k_tile_num, 
                        load_burst_len = load_burst_len,
                        dtype = dtype, 
                        acc_dtype = acc_dtype,
                    )

    for core_id in core_grid.core_ids:
        core = device.get_npu_core(core_id=core_id)
        
        TT_RT_KERNEL_LOCAL_CB_DEALLOCATE(core, cb_ifm_ptrs[core_id])
        TT_RT_KERNEL_LOCAL_CB_DEALLOCATE(core, cb_wgt_ptrs[core_id])
        TT_RT_KERNEL_LOCAL_CB_DEALLOCATE(core, cb_ofm_ptrs[core_id])
        if buf_bias is not None:
            TT_RT_KERNEL_LOCAL_CB_DEALLOCATE(core, cb_bias_ptrs[core_id])
            
    return buf_ofm
