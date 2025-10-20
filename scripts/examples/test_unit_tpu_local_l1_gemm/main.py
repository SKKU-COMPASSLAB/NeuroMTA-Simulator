import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import GoogleTPUConfig, GoogleTPUDevice


TRACE_DIR = os.path.join(os.path.dirname(__file__), ".traces")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")


@jit_prototype
def dma_kernel(
    core: NPUCore,
    
    bf_ifm_ptr: BufferPointer,
    bf_wgt_ptr: BufferPointer,
    bf_psum_ptr: BufferPointer,
    bf_ofm_ptr: BufferPointer,
    
    cb_ifm_ptr: BufferPointer,
    cb_wgt_ptr: BufferPointer,
    cb_psum_ptr: BufferPointer,
    cb_ofm_ptr: BufferPointer,
    
    m_tile_num: int,
    n_tile_num: int,
    k_tile_num: int,
    
    load_burst_len: int,
):
    for n_it in range(n_tile_num):
        for k_it in range(k_tile_num):
            for m_it_st in range(0, m_tile_num, load_burst_len):
                m_it_ed = min(m_it_st + load_burst_len, m_tile_num)
                m_it_n_pages = m_it_ed - m_it_st
                
                if k_it == 0:
                    psum_tile_bf_ptr = bf_psum_ptr[n_it * m_tile_num + m_it_st : n_it * m_tile_num + m_it_ed]
                else:
                    psum_tile_bf_ptr = bf_ofm_ptr[n_it * m_tile_num + m_it_st : n_it * m_tile_num + m_it_ed]
                
                if m_it_st == 0:
                    with new_parallel_thread():
                        core.cb_reserve_back(cb_wgt_ptr, n_pages=1)
                        core.mem_buffer_copy(cb_wgt_ptr[0], bf_wgt_ptr[n_it * k_tile_num + k_it], 1)
                        core.cb_push_back(cb_wgt_ptr, n_pages=1)

                with new_parallel_thread():
                    core.cb_reserve_back(cb_ifm_ptr, n_pages=m_it_n_pages)
                    core.mem_buffer_copy(cb_ifm_ptr[:m_it_n_pages], bf_ifm_ptr[k_it * m_tile_num + m_it_st : k_it * m_tile_num + m_it_ed], n_pages=m_it_n_pages)
                    core.cb_push_back(cb_ifm_ptr, n_pages=m_it_n_pages)
                
                with new_parallel_thread():
                    core.cb_reserve_back(cb_psum_ptr, n_pages=m_it_n_pages)
                    core.mem_buffer_copy(cb_psum_ptr[:m_it_n_pages], psum_tile_bf_ptr[:m_it_n_pages], n_pages=m_it_n_pages)
                    core.cb_push_back(cb_psum_ptr, n_pages=m_it_n_pages)
                    
                core.parallel_merge()
                
                core.cb_wait_front(cb_ofm_ptr, n_pages=m_it_n_pages)
                core.mem_buffer_copy(bf_ofm_ptr[n_it * m_tile_num + m_it_st:n_it * m_tile_num + m_it_ed], cb_ofm_ptr[:m_it_n_pages], n_pages=m_it_n_pages)
                core.cb_pop_front(cb_ofm_ptr, n_pages=m_it_n_pages)
                
                core.parallel_merge()

                
@jit_prototype
def compute_kernel(
    core: NPUCore,    
    
    cb_ifm_ptr:  BufferPointer,
    cb_wgt_ptr:  BufferPointer,
    cb_psum_ptr: BufferPointer,
    cb_ofm_ptr:  BufferPointer,
    
    m_tile_num: int,
    n_tile_num: int,
    k_tile_num: int,
    
    dtype: torch.dtype,
    acc_dtype: torch.dtype,
):  
    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
    
    containers = [DataContainer() for _ in range(4)]
    
    for n_it in range(n_tile_num):
        for k_it in range(k_tile_num):            
            for m_it in range(m_tile_num):
                if m_it == 0:
                    core.cb_wait_front(cb_wgt_ptr, n_pages=1)
                core.cb_wait_front(cb_ifm_ptr, n_pages=1)
                core.cb_wait_front(cb_psum_ptr, n_pages=1)
                core.cb_reserve_back(cb_ofm_ptr, n_pages=1)
                
                if m_it == 0:
                    core.mem_read_with_container(cb_wgt_ptr[0], containers[1])
                core.mem_read_with_container(cb_ifm_ptr[0], containers[0])
                core.mem_read_with_container(cb_psum_ptr[0], containers[2])

                core.mxu_tiled_gemm(
                    *containers, 
                    preload_wgt=(m_it == 0),
                    preload_psum=False,
                    flush_ofm=True,
                )
                
                core.mem_write_with_container(cb_ofm_ptr[0], containers[3])
                
                if m_it == (m_tile_num - 1):
                    core.cb_pop_front(cb_wgt_ptr, n_pages=1)
                core.cb_pop_front(cb_ifm_ptr, n_pages=1)
                core.cb_pop_front(cb_psum_ptr, n_pages=1)
                core.cb_push_back(cb_ofm_ptr, n_pages=1)


if __name__ == "__main__":
    logger.set_print_options(LogLevel.DEBUG)
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = GoogleTPUConfig.V4()

    device = GoogleTPUDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 128
    N = 256
    K = 128
    dtype = torch.int8
    acc_dtype = torch.int32
    
    m_tile = 128
    n_tile = 128
    k_tile = 128
    
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
    
    cb_n_pages = 8
    load_burst_len = cb_n_pages // 2  # burst length should be the double of the number of circular buffer pages for double buffering
    
    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum: torch.Tensor = torch.arange(0, M * N, dtype=acc_dtype).reshape(M, N)
    ofm:  torch.Tensor = torch.zeros((M, N), dtype=acc_dtype)

    tiled_ifm  = ifm.reshape(m_tile_num, m_tile, k_tile_num, k_tile).permute(2, 0, 1, 3)
    tiled_wgt  = wgt.reshape(k_tile_num, k_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)
    tiled_psum = psum.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)
    tiled_ofm  = ofm.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(2, 0, 1, 3)

    ifm_size  = ifm.numel()  * ifm.element_size()
    wgt_size  = wgt.numel()  * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size  = ofm.numel()  * ofm.element_size()
    
    bf_ifm_ptr:  BufferPointer = device.create_local_l1_buffer(page_size=ifm_tile_size, n_pages=ifm_tile_num, core_ids=npu_core_id)
    bf_wgt_ptr:  BufferPointer = device.create_local_l1_buffer(page_size=wgt_tile_size, n_pages=wgt_tile_num, core_ids=npu_core_id)
    bf_psum_ptr: BufferPointer = device.create_local_l1_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num, core_ids=npu_core_id)
    bf_ofm_ptr:  BufferPointer = device.create_local_l1_buffer(page_size=ofm_tile_size, n_pages=ofm_tile_num, core_ids=npu_core_id)

    cb_ifm_ptr:  BufferPointer = device.create_local_l1_circular_buffer(page_size=ifm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_id)
    cb_wgt_ptr:  BufferPointer = device.create_local_l1_circular_buffer(page_size=wgt_tile_size, n_pages=cb_n_pages, core_ids=npu_core_id)
    cb_psum_ptr: BufferPointer = device.create_local_l1_circular_buffer(page_size=ofm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_id)
    cb_ofm_ptr:  BufferPointer = device.create_local_l1_circular_buffer(page_size=ofm_tile_size, n_pages=cb_n_pages, core_ids=npu_core_id)

    device.set_ptr_content(bf_ifm_ptr, tiled_ifm)
    device.set_ptr_content(bf_wgt_ptr, tiled_wgt)
    device.set_ptr_content(bf_psum_ptr, tiled_psum)
    device.set_ptr_content(bf_ofm_ptr, tiled_ofm)
    
    core = device.get_npu_core(core_id=npu_core_id)
    kernel1 = dma_kernel(core, bf_ifm_ptr, bf_wgt_ptr, bf_psum_ptr, bf_ofm_ptr, cb_ifm_ptr, cb_wgt_ptr, cb_psum_ptr, cb_ofm_ptr, m_tile_num, n_tile_num, k_tile_num, load_burst_len=load_burst_len)
    kernel2 = compute_kernel(core, cb_ifm_ptr, cb_wgt_ptr, cb_psum_ptr, cb_ofm_ptr, m_tile_num, n_tile_num, k_tile_num, dtype=dtype, acc_dtype=acc_dtype)
    core.dispatch_main_kernel("dma", kernel=kernel1)
    core.dispatch_main_kernel("compute", kernel=kernel2)
    
    tracer_hub = TracerHub()
    for core_id, core in device.cores.items():
        tracer = Tracer()
        tracer.register_core(core)
        tracer_hub.register_tracer(f"{type(core).__name__}_{core.core_id}", tracer)
        
    profiler_hub = ProfilerHub()
    for core_id, core in device.cores.items():
        if isinstance(core, NPUCore) and (not core.is_idle):
            profiler = CommandUtilizationProfiler(core)
            profiler_hub.register_profiler(f"{type(core).__name__}_{core.core_id}", profiler)

    # with MonitoringWindow() as monitor:
    #     for core_id, core in device.cores.items():
    #         if isinstance(core, NPUCore) and (not core.is_idle):
    #             pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
    #             pbar.bind_core(core)
        
    st = time.time()
    device.run_kernels()
    ed = time.time()
    
    tracer_hub.save_traces(TRACE_DIR)
    profiler_hub.save_profiles(PROFILE_DIR)

    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm.to(dtype=acc_dtype), wgt.to(dtype=acc_dtype)) + psum
    simulated = device.get_ptr_content(bf_ofm_ptr, shape=(n_tile_num, m_tile_num, m_tile, n_tile), dtype=acc_dtype).permute(1, 2, 0, 3).reshape(M, N)
    
    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")