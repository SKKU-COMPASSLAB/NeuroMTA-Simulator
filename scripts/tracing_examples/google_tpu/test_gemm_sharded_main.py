import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import GoogleTPUConfig, GoogleTPUDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".traces", FILENAME)


@core_kernel_method
def compute_kernel(
    core: NPUCore,
    
    bf_ifm_ptr: Reference,
    bf_wgt_ptr: Reference,
    bf_psum_ptr: Reference,
    bf_ofm_ptr: Reference,
    
    spm_ifm_ptr: Reference,
    spm_wgt_ptr: Reference,
    spm_psum_ptr: Reference,
    spm_ofm_ptr: Reference,

    m_tile_num: int,
    n_tile_num: int,
    k_tile_num: int,
    
    n_threads: int=1
):  
    load_sem = Pointer()
    mxu_sem = Pointer()
    store_sem = Pointer()
    thread_sem = [Pointer() for _ in range(n_threads)]
    
    core.var_allocate(load_sem, initial_value=0)
    core.var_allocate(mxu_sem, initial_value=0)
    core.var_allocate(store_sem, initial_value=0)
    
    for thread_id in range(n_threads):
        core.var_allocate(thread_sem[thread_id], initial_value=0)
    
    core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
    
    for n_it in range(n_tile_num):
        thread_id = n_it % n_threads
        
        with new_parallel_thread():
            core.var_compare_and_swap(thread_sem[thread_id], 0, 1)
            
            for k_it in range(k_tile_num):
                core.var_compare_and_swap(load_sem, 0, 1)
                core.async_buffer_read(spm_wgt_ptr[thread_id], bf_wgt_ptr[n_it * k_tile_num + k_it])
                core.async_rpc_wait_all()
                core.var_compare_and_swap(load_sem, 1, 0)
                
                for m_it in range(m_tile_num):
                    if k_it == 0:
                        psum_tile_bf_ptr = bf_psum_ptr[n_it * m_tile_num + m_it]
                    else:
                        psum_tile_bf_ptr = bf_ofm_ptr[n_it * m_tile_num + m_it]

                    core.var_compare_and_swap(load_sem, 0, 1)
                    core.async_buffer_read(spm_ifm_ptr[thread_id], bf_ifm_ptr[k_it * m_tile_num + m_it])
                    core.async_buffer_read(spm_psum_ptr[thread_id], psum_tile_bf_ptr)
                    core.async_rpc_wait_all()
                    core.var_compare_and_swap(load_sem, 1, 0)

                    core.var_compare_and_swap(mxu_sem, 0, 1)
                    core.mxu_tiled_gemm(
                        spm_ifm_ptr[thread_id], spm_wgt_ptr[thread_id], spm_psum_ptr[thread_id], spm_ofm_ptr[thread_id], 
                        preload_wgt=True,   # cannot reuse preloaded weights due to interference between threads
                        preload_psum=False,
                        flush_ofm=True,
                    )
                    core.var_compare_and_swap(mxu_sem, 1, 0)

                    core.var_compare_and_swap(store_sem, 0, 1)
                    core.async_buffer_write(bf_ofm_ptr[n_it * m_tile_num + m_it], spm_ofm_ptr[thread_id])
                    core.async_rpc_wait_all()
                    core.var_compare_and_swap(store_sem, 1, 0)
                    
            core.var_compare_and_swap(thread_sem[thread_id], 1, 0)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = GoogleTPUConfig.V4()

    device = GoogleTPUDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    tracer_hub = TracerHub()
    for core_id, core in device.cores.items():
        tracer = Tracer()
        tracer.register_core(core)
        tracer_hub.register_tracer(f"{type(core).__name__}_{core.core_id}", tracer)
    
    M = 512
    N = 512
    K = 128
    dtype = torch.int32
    acc_dtype = torch.int32
    
    m_tile = 128
    n_tile = 128
    k_tile = 128
    
    n_thread = 2  # assumes double buffering
    
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

    npu_core_id = device.npu_core_ids[0]
    
    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum: torch.Tensor = torch.arange(0, M * N, dtype=acc_dtype).reshape(M, N)
    ofm:  torch.Tensor = torch.zeros((M, N), dtype=acc_dtype)

    tiled_ifm  = ifm.reshape(m_tile_num, m_tile, k_tile_num, k_tile).permute(2, 0, 1, 3)
    tiled_wgt  = wgt.reshape(k_tile_num, k_tile, n_tile_num, n_tile).permute(0, 2, 1, 3)
    tiled_psum = psum.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)
    tiled_ofm  = ofm.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)

    ifm_size  = ifm.numel()  * ifm.element_size()
    wgt_size  = wgt.numel()  * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size  = ofm.numel()  * ofm.element_size()
    
    bf_ifm_ptr:  Reference = device.create_sharded_main_buffer(page_size=ifm_tile_size, n_pages=ifm_tile_num)
    bf_wgt_ptr:  Reference = device.create_sharded_main_buffer(page_size=wgt_tile_size, n_pages=wgt_tile_num)
    bf_psum_ptr: Reference = device.create_sharded_main_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num)
    bf_ofm_ptr:  Reference = device.create_sharded_main_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num)

    spm_ifm_ptr:  Reference = device.create_local_l1_buffer(page_size=ifm_tile_size, n_pages=n_thread, core_ids=npu_core_id)
    spm_wgt_ptr:  Reference = device.create_local_l1_buffer(page_size=wgt_tile_size, n_pages=n_thread, core_ids=npu_core_id)
    spm_psum_ptr: Reference = device.create_local_l1_buffer(page_size=ofm_tile_size, n_pages=n_thread, core_ids=npu_core_id)
    spm_ofm_ptr:  Reference = device.create_local_l1_buffer(page_size=ofm_tile_size, n_pages=n_thread, core_ids=npu_core_id)

    device.set_ptr_content(bf_ifm_ptr, tiled_ifm)
    device.set_ptr_content(bf_wgt_ptr, tiled_wgt)
    device.set_ptr_content(bf_psum_ptr, tiled_psum)
    device.set_ptr_content(bf_ofm_ptr, tiled_ofm)
    
    core = device.get_npu_core(core_id=npu_core_id)
    kernel = compute_kernel(
        core, 
        bf_ifm_ptr, bf_wgt_ptr, bf_psum_ptr, bf_ofm_ptr, 
        spm_ifm_ptr, spm_wgt_ptr, spm_psum_ptr, spm_ofm_ptr, 
        m_tile_num, n_tile_num, k_tile_num,
        n_threads=n_thread,
    )
    core.dispatch_main_kernel("compute", kernel=kernel)
    
    st = time.time()
    device.run_kernels(verbose=True)
    ed = time.time()
    
    tracer_hub.save_traces(TRACE_DIR)
    
    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm, wgt) + psum
    simulated = device.get_ptr_content(bf_ofm_ptr, shape=(n_tile_num, m_tile_num, m_tile, n_tile), dtype=acc_dtype).permute(1, 2, 0, 3).reshape(M, N)

    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\n=== RESULT COMPARISON ===\n{(reference == simulated).to(torch.int8)}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")