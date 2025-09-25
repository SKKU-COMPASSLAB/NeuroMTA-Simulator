import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.hardware.analyzer.icnt_core_analyzer import IcntCoreAnalyzer
from neuromta.hardware.analyzer.main_mem_core_analyzer import MainMemCoreAnalyzer
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig, TenstorrentDevice


TRACE_DIR = os.path.join(os.path.dirname(__file__), ".traces")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), ".analysis")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


class RT_TT_SHARDED_L1_GEMM(MCA_RuntimeKernel):
    def __init__(
        self,
    
        bf_ifm_ptr:  BufferPointer,
        bf_wgt_ptr:  BufferPointer,
        bf_psum_ptr: BufferPointer,
        bf_ofm_ptr:  BufferPointer,

        cb_ifm_ptr:  BufferPointer,
        cb_wgt_ptr:  BufferPointer,
        cb_psum_ptr: BufferPointer,
        cb_ofm_ptr:  BufferPointer,
        
        k_tile_num: int,
        load_burst_len: int, 
        
        dtype: torch.dtype,
        acc_dtype: torch.dtype,
    ):
        super().__init__()
        
        self.bf_ifm_ptr  = bf_ifm_ptr
        self.bf_wgt_ptr  = bf_wgt_ptr
        self.bf_psum_ptr = bf_psum_ptr
        self.bf_ofm_ptr  = bf_ofm_ptr
        
        self.cb_ifm_ptr  = cb_ifm_ptr
        self.cb_wgt_ptr  = cb_wgt_ptr
        self.cb_psum_ptr = cb_psum_ptr
        self.cb_ofm_ptr  = cb_ofm_ptr
        
        self.k_tile_num = k_tile_num
        self.load_burst_len = load_burst_len
        
        self.dtype = dtype
        self.acc_dtype = acc_dtype
    
    def RD_KERNEL(self, core: NPUCore):
        for i in range(0, self.k_tile_num, self.load_burst_len):
            ed = min(i + self.load_burst_len, self.k_tile_num)
            n_pages = ed - i
            
            if i == 0:
                core.cb_reserve_back(self.cb_psum_ptr, 1)
            core.cb_reserve_back(self.cb_ifm_ptr, n_pages)
            core.cb_reserve_back(self.cb_wgt_ptr, n_pages)
            
            if i == 0:
                with new_parallel_thread("LD_PSUM"):
                    core.mem_buffer_copy(self.cb_psum_ptr[0], self.bf_psum_ptr[0], 1)
            
            with new_parallel_thread("LD_IFM"):
                core.mem_buffer_copy(self.cb_ifm_ptr[0:n_pages], self.bf_ifm_ptr[i:ed], n_pages)
            
            with new_parallel_thread("LD_WGT"):
                core.mem_buffer_copy(self.cb_wgt_ptr[0:n_pages], self.bf_wgt_ptr[i:ed], n_pages)
                
            core.parallel_merge()
            
            if i == 0:
                core.cb_push_back(self.cb_psum_ptr, 1)
            core.cb_push_back(self.cb_ifm_ptr, n_pages)
            core.cb_push_back(self.cb_wgt_ptr, n_pages)
    
    def EX_KERNEL(self, core: NPUCore):  
        core.mxu_reconfigure(dtype=self.dtype, acc_dtype=self.acc_dtype)
        
        containers = [DataContainer() for _ in range(4)]
        
        core.cb_wait_front(self.cb_psum_ptr, 1)
        core.cb_reserve_back(self.cb_ofm_ptr, 1)
        
        core.mem_read_with_container(self.cb_psum_ptr[0], containers[2])
        
        for k_it in range(self.k_tile_num):
            preload_psum = True if (k_it == 0) else False
            flush_ofm    = True if (k_it == (self.k_tile_num - 1)) else False

            core.cb_wait_front(self.cb_ifm_ptr, 1)
            core.cb_wait_front(self.cb_wgt_ptr, 1)
            
            core.mem_read_with_container(self.cb_ifm_ptr[0], containers[0])
            core.mem_read_with_container(self.cb_wgt_ptr[0], containers[1])
            
            core.mxu_tiled_gemm(
                *containers,
                preload_wgt=False,
                preload_psum=preload_psum,
                flush_ofm=flush_ofm,
                wgt_transposed=True,
                psum_vectored=True,
            )
            
            core.cb_pop_front(self.cb_ifm_ptr, 1)
            core.cb_pop_front(self.cb_wgt_ptr, 1)
            
        core.mem_write_with_container(self.cb_ofm_ptr[0], containers[3])
        
        core.cb_pop_front(self.cb_psum_ptr, 1)
        core.cb_push_back(self.cb_ofm_ptr, 1)
    
    def WR_KERNEL(self, core: NPUCore):
        core.cb_wait_front(self.cb_ofm_ptr, 1)
        core.mem_buffer_copy(self.bf_ofm_ptr[0], self.cb_ofm_ptr[0], 1)
        core.cb_pop_front(self.cb_ofm_ptr, 1)


if __name__ == "__main__":
    logger.set_print_options(LogLevel.DEBUG)
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = TenstorrentConfig.BLACKHOLE()

    device = TenstorrentDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 512
    N = 512
    K = 512
    dtype = torch.int8
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
    
    ifm_tile_num  = m_tile_num * k_tile_num
    wgt_tile_num  = k_tile_num * n_tile_num
    psum_tile_num = n_tile_num
    ofm_tile_num  = m_tile_num * n_tile_num
    
    ifm_tile_size  = m_tile * k_tile * dtype.itemsize
    wgt_tile_size  = k_tile * n_tile * dtype.itemsize
    psum_tile_size = n_tile * acc_dtype.itemsize
    ofm_tile_size  = m_tile * n_tile * acc_dtype.itemsize

    core_grid_shape = (4, 4)
    core_ids = device.get_npu_core_grid(offset=(0, 0), shape=core_grid_shape).core_ids
    n_cores = len(core_ids)
    
    cb_n_pages = 4
    load_burst_len = cb_n_pages // 2  # should be less than cb_n_pages to avoid deadlock
    
    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum: torch.Tensor = torch.arange(0, N, dtype=acc_dtype).flatten()
    ofm:  torch.Tensor = torch.zeros((M, N), dtype=acc_dtype)

    tiled_ifm  = ifm.reshape(m_tile_num, m_tile, k_tile_num, k_tile).permute(0, 2, 1, 3)
    tiled_wgt  = wgt.T.reshape(n_tile_num, n_tile, k_tile_num, k_tile).permute(0, 2, 1, 3)
    tiled_psum = psum.reshape(n_tile_num, n_tile)
    tiled_ofm  = ofm.reshape(m_tile_num, m_tile, n_tile_num, n_tile).permute(0, 2, 1, 3)

    ifm_size  = ifm.numel()  * ifm.element_size()
    wgt_size  = wgt.numel()  * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size  = ofm.numel()  * ofm.element_size()
    
    bf_ifm_ptr:  BufferPointer = device.create_sharded_l1_buffer(page_size=ifm_tile_size,  n_pages=ifm_tile_num,  core_ids=core_ids)
    bf_wgt_ptr:  BufferPointer = device.create_sharded_l1_buffer(page_size=wgt_tile_size,  n_pages=wgt_tile_num,  core_ids=core_ids)
    bf_psum_ptr: BufferPointer = device.create_sharded_l1_buffer(page_size=psum_tile_size, n_pages=psum_tile_num, core_ids=core_ids)
    bf_ofm_ptr:  BufferPointer = device.create_sharded_l1_buffer(page_size=ofm_tile_size,  n_pages=ofm_tile_num,  core_ids=core_ids)

    cb_ifm_ptrs:  list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=ifm_tile_size,  n_pages=cb_n_pages, core_ids=core_ids)
    cb_wgt_ptrs:  list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=wgt_tile_size,  n_pages=cb_n_pages, core_ids=core_ids)
    cb_psum_ptrs: list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=psum_tile_size, n_pages=cb_n_pages, core_ids=core_ids)
    cb_ofm_ptrs:  list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=ofm_tile_size,  n_pages=cb_n_pages, core_ids=core_ids)

    device.set_ptr_content(bf_ifm_ptr, tiled_ifm)
    device.set_ptr_content(bf_wgt_ptr, tiled_wgt)
    device.set_ptr_content(bf_psum_ptr, tiled_psum)
    device.set_ptr_content(bf_ofm_ptr, tiled_ofm)
        
    for m_it in range(m_tile_num):
        for n_it in range(n_tile_num):
            core_idx = (m_it * n_tile_num + n_it) % n_cores

            core_id = core_ids[core_idx]
            core = device.get_core_from_id(core_id=core_id)
            
            core_bf_ifm_ptr  = bf_ifm_ptr[m_it * k_tile_num:(m_it + 1) * k_tile_num]
            core_bf_wgt_ptr  = bf_wgt_ptr[n_it * k_tile_num:(n_it + 1) * k_tile_num]
            core_bf_psum_ptr = bf_psum_ptr[n_it]
            core_bf_ofm_ptr  = bf_ofm_ptr[m_it * n_tile_num + n_it]

            core_cb_ifm_ptr  = cb_ifm_ptrs[core_idx]  if n_cores > 1 else cb_ifm_ptrs
            core_cb_wgt_ptr  = cb_wgt_ptrs[core_idx]  if n_cores > 1 else cb_wgt_ptrs
            core_cb_psum_ptr = cb_psum_ptrs[core_idx] if n_cores > 1 else cb_psum_ptrs
            core_cb_ofm_ptr  = cb_ofm_ptrs[core_idx]  if n_cores > 1 else cb_ofm_ptrs

            rt_kernel = RT_TT_SHARDED_L1_GEMM(
                bf_ifm_ptr=core_bf_ifm_ptr,
                bf_wgt_ptr=core_bf_wgt_ptr,
                bf_psum_ptr=core_bf_psum_ptr,
                bf_ofm_ptr=core_bf_ofm_ptr,
                
                cb_ifm_ptr=core_cb_ifm_ptr,
                cb_wgt_ptr=core_cb_wgt_ptr,
                cb_psum_ptr=core_cb_psum_ptr,
                cb_ofm_ptr=core_cb_ofm_ptr,
                
                k_tile_num=k_tile_num,
                load_burst_len=load_burst_len,
                
                dtype=dtype,
                acc_dtype=acc_dtype,
            )
            
            rt_kernel.dispatch(core=core)

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
            
    icnt_core_tracer = IcntCoreAnalyzer(device.icnt_core)

    with MonitoringWindow() as monitor:
        for core_id, core in device.cores.items():
            if isinstance(core, NPUCore) and (not core.is_idle):
                pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=40)
                pbar.bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    tracer_hub.save_traces(TRACE_DIR)
    profiler_hub.save_profiles(PROFILE_DIR)
    icnt_core_tracer.save_traces(os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv"))
    icnt_core_tracer.save_bandwidth_analysis(os.path.join(ANALYSIS_DIR, "icnt_core_bandwidth_analysis.csv"), bin_size=1)

    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm.to(dtype=acc_dtype), wgt.to(dtype=acc_dtype)) + psum
    simulated = device.get_ptr_content(bf_ofm_ptr, shape=(m_tile_num, n_tile_num, m_tile, n_tile), dtype=acc_dtype).permute(0, 2, 1, 3).reshape(M, N)

    # print(f"\n=== REFERENCE ===\n{reference}")
    # print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")