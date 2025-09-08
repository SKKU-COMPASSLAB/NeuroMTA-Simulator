import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig, TenstorrentDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".logs", FILENAME)


@core_kernel_method
def read_kernel(
    core: NPUCore,

    bf_ifm_ptr:  Reference,
    bf_wgt_ptr:  Reference,
    bf_psum_ptr: Reference,
    
    cb_ifm_ptr:  Reference,
    cb_wgt_ptr:  Reference,
    cb_psum_ptr: Reference,
    
    n_seq_pages: int,   # number of sequential pages (especially for IFM and WGT)
):
    # load psum to the l1 circular buffer
    core.cb_reserve_back(cb_psum_ptr, 1)
    core.async_noc_buffer_read(cb_psum_ptr[0], bf_psum_ptr[0])
    core.cb_push_back(cb_psum_ptr, 1)
    
    # load ifm and wgt to the l1 circular buffer
    for i in range(n_seq_pages):
        core.cb_reserve_back(cb_ifm_ptr, 1)
        core.cb_reserve_back(cb_wgt_ptr, 1)
        
        core.async_noc_buffer_read(cb_ifm_ptr[0], bf_ifm_ptr[i])
        core.async_noc_buffer_read(cb_wgt_ptr[0], bf_wgt_ptr[i])
        
        core.cb_push_back(cb_ifm_ptr, 1)
        core.cb_push_back(cb_wgt_ptr, 1)
        
@core_kernel_method
def compute_kernel(
    core: NPUCore,    
    
    cb_ifm_ptr:  Reference,
    cb_wgt_ptr:  Reference,
    cb_psum_ptr: Reference,
    cb_ofm_ptr:  Reference,
    
    k_tile_num: int
):  
    core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
    
    core.cb_wait_front(cb_psum_ptr, 1)
    core.cb_reserve_back(cb_ofm_ptr, 1)
    
    for k_it in range(k_tile_num):
        preload_psum = True if (k_it == 0) else False
        flush_ofm    = True if (k_it == k_tile_num - 1) else False

        core.cb_wait_front(cb_ifm_ptr, 1)
        core.cb_wait_front(cb_wgt_ptr, 1)
        
        core.mxu_tiled_gemm(
            ifm_ptr=cb_ifm_ptr[0],
            wgt_ptr=cb_wgt_ptr[0],
            psum_ptr=cb_psum_ptr[0],
            ofm_ptr=cb_ofm_ptr[0],
            preload_wgt=False,
            preload_psum=preload_psum,
            flush_ofm=flush_ofm,
        )
        
        core.cb_pop_front(cb_ifm_ptr, 1)
        core.cb_pop_front(cb_wgt_ptr, 1)
    
    core.cb_pop_front(cb_psum_ptr, 1)
    core.cb_push_back(cb_ofm_ptr, 1)
    
@core_kernel_method
def write_kernel(
    core: NPUCore,
    
    bf_ofm_ptr: Reference,
    cb_ofm_ptr: Reference,
):
    core.cb_wait_front(cb_ofm_ptr, 1)
    core.async_noc_buffer_write(bf_ofm_ptr[0], cb_ofm_ptr[0])
    core.cb_pop_front(cb_ofm_ptr, 1)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = TenstorrentConfig.BLACKHOLE()

    device = TenstorrentDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 512
    N = 512
    K = 128
    dtype = torch.int32
    acc_dtype = torch.int32
    
    m_tile = 32
    n_tile = 32
    k_tile = 32
    
    m_tile_num = M // m_tile
    n_tile_num = N // n_tile
    k_tile_num = K // k_tile
    
    ifm_tile_shape = (m_tile, k_tile)
    wgt_tile_shape = (k_tile, n_tile)
    ofm_tile_shape = (m_tile, n_tile)
    
    ifm_tile_num = m_tile_num * k_tile_num
    wgt_tile_num = k_tile_num * n_tile_num
    ofm_tile_num = m_tile_num * n_tile_num
    
    ifm_tile_size = m_tile * k_tile * dtype.itemsize
    wgt_tile_size = k_tile * n_tile * dtype.itemsize
    ofm_tile_size = m_tile * n_tile * acc_dtype.itemsize

    n_cores = min(m_tile_num * n_tile_num, len(device.npu_cores))
    npu_core_group = device.npu_core_ids[:n_cores]
    dma_core_group = device.dma_core_ids
    cb_n_pages = 8
    
    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum: torch.Tensor = torch.arange(0, M * N, dtype=acc_dtype).reshape(M, N)
    ofm:  torch.Tensor = torch.zeros((M, N), dtype=acc_dtype)

    tiled_ifm  = ifm.reshape(m_tile_num, m_tile, k_tile_num, k_tile).permute(0, 2, 1, 3)
    tiled_wgt  = wgt.reshape(k_tile_num, k_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)
    tiled_psum = psum.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(0, 2, 1, 3)
    tiled_ofm  = ofm.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(0, 2, 1, 3)

    ifm_size  = ifm.numel()  * ifm.element_size()
    wgt_size  = wgt.numel()  * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size  = ofm.numel()  * ofm.element_size()
    
    bf_ifm_ptr:  Reference = device.create_sharded_main_buffer(page_size=ifm_tile_size, n_pages=ifm_tile_num)
    bf_wgt_ptr:  Reference = device.create_sharded_main_buffer(page_size=wgt_tile_size, n_pages=wgt_tile_num)
    bf_psum_ptr: Reference = device.create_sharded_main_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num)
    bf_ofm_ptr:  Reference = device.create_sharded_main_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num)

    cb_ifm_ptrs:  list[Reference] = device.create_local_l1_circular_buffer(page_size=ifm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_group)
    cb_wgt_ptrs:  list[Reference] = device.create_local_l1_circular_buffer(page_size=wgt_tile_size, n_pages=cb_n_pages, core_ids=npu_core_group)
    cb_psum_ptrs: list[Reference] = device.create_local_l1_circular_buffer(page_size=ofm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_group)
    cb_ofm_ptrs:  list[Reference] = device.create_local_l1_circular_buffer(page_size=ofm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_group)

    device.set_ptr_content(bf_ifm_ptr, tiled_ifm)
    device.set_ptr_content(bf_wgt_ptr, tiled_wgt)
    device.set_ptr_content(bf_psum_ptr, tiled_psum)
    device.set_ptr_content(bf_ofm_ptr, tiled_ofm)
        
    for m_it in range(m_tile_num):
        for n_it in range(n_tile_num):
            core_idx = (m_it * n_tile_num + n_it) % n_cores

            core_id = npu_core_group[core_idx]
            core = device.get_core_from_id(core_id=core_id)
            
            core_bf_ifm_ptr  = bf_ifm_ptr[m_it * k_tile_num:(m_it + 1) * k_tile_num]
            core_bf_wgt_ptr  = bf_wgt_ptr[n_it * k_tile_num:(n_it + 1) * k_tile_num]
            core_bf_psum_ptr = bf_psum_ptr[m_it * n_tile_num + n_it]
            core_bf_ofm_ptr  = bf_ofm_ptr[m_it * n_tile_num + n_it]

            core_cb_ifm_ptr  = cb_ifm_ptrs[core_idx]  if n_cores > 1 else cb_ifm_ptrs
            core_cb_wgt_ptr  = cb_wgt_ptrs[core_idx]  if n_cores > 1 else cb_wgt_ptrs
            core_cb_psum_ptr = cb_psum_ptrs[core_idx] if n_cores > 1 else cb_psum_ptrs
            core_cb_ofm_ptr  = cb_ofm_ptrs[core_idx]  if n_cores > 1 else cb_ofm_ptrs

            kernel1 = read_kernel(core, core_bf_ifm_ptr, core_bf_wgt_ptr, core_bf_psum_ptr, core_cb_ifm_ptr, core_cb_wgt_ptr, core_cb_psum_ptr, n_seq_pages=k_tile_num)
            kernel2 = compute_kernel(core, core_cb_ifm_ptr, core_cb_wgt_ptr, core_cb_psum_ptr, core_cb_ofm_ptr, k_tile_num=k_tile_num)
            kernel3 = write_kernel(core, core_bf_ofm_ptr, core_cb_ofm_ptr)
            
            core.dispatch_main_kernel("read",  kernel=kernel1)
            core.dispatch_main_kernel("compute", kernel=kernel2)
            core.dispatch_main_kernel("write", kernel=kernel3)

    st = time.time()
    device.run_kernels(verbose=True, max_steps=-1, save_trace=True, save_trace_dir=TRACE_DIR)
    ed = time.time()
    
    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm, wgt) + psum
    simulated = device.get_ptr_content(bf_ofm_ptr, shape=(m_tile_num, n_tile_num, m_tile, n_tile), dtype=acc_dtype).permute(0, 2, 1, 3).reshape(M, N)

    # print(f"\n=== REFERENCE ===\n{reference}")
    # print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")