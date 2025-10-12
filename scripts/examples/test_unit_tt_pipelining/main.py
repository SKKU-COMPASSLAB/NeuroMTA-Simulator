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



@jit_prototype
def read_kernel(core: NPUCore, pipe_in_ptr: BufferPointer, pipe_out_ptr: BufferPointer | None, core_rd_ptr: BufferPointer, n_pages: int):
    for page_idx in range(n_pages):
        if pipe_in_ptr.is_circular:
            core.cb_wait_front(pipe_in_ptr, 1)
            pipe_in_page_idx = 0
        else:
            pipe_in_page_idx = page_idx
            
        if pipe_out_ptr is not None:
            with new_parallel_thread():
                core.cb_reserve_back(pipe_out_ptr, 1)
                core.mem_buffer_copy(pipe_out_ptr[0], pipe_in_ptr[pipe_in_page_idx], n_pages=1)
                core.cb_push_back(pipe_out_ptr, 1)
        
        if core_rd_ptr.is_circular:
            with new_parallel_thread():
                core.cb_reserve_back(core_rd_ptr, 1)
                core.mem_buffer_copy(core_rd_ptr[0], pipe_in_ptr[pipe_in_page_idx], n_pages=1)
                core.cb_push_back(core_rd_ptr, 1)
                
        core.parallel_merge()
        
        if pipe_in_ptr.is_circular:
            core.cb_pop_front(pipe_in_ptr, 1)
            
@jit_prototype
def compute_kernel(core: NPUCore, core_rd_ptr: BufferPointer, core_wr_ptr: BufferPointer, n_pages: int):
    for page_idx in range(n_pages):
        core.cb_wait_front(core_rd_ptr, 1)
        core.cb_reserve_back(core_wr_ptr, 1)
        
        core.mem_buffer_copy(core_wr_ptr[0], core_rd_ptr[0], n_pages=1)
        
        core.cb_push_back(core_wr_ptr, 1)
        core.cb_pop_front(core_rd_ptr, 1)



if __name__ == "__main__":
    logger.set_print_options(LogLevel.DEBUG)
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = TenstorrentConfig.BLACKHOLE()

    device = TenstorrentDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    page_numel = 8
    dtype = torch.int32
    page_size = page_numel * dtype.itemsize
    n_pages = 4
    
    tensor = torch.arange(n_pages * page_numel, dtype=dtype).reshape(n_pages, page_numel)
    
    core_ids = device.get_npu_core_grid(offset=(0, 0), shape=(1, 4)).core_ids
    n_cores = len(core_ids)
    
    bf_main:        BufferPointer       = device.create_sharded_main_buffer(page_size=page_size, n_pages=n_pages, channel_id=list(range(device.mem_context.main_config.ch_num)))
    cb_pipe:        list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=page_size, n_pages=2, core_ids=core_ids)
    cb_core_rd_ptr: list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=page_size, n_pages=2, core_ids=core_ids)
    cb_core_wr_ptr: list[BufferPointer] = device.create_local_l1_circular_buffer(page_size=page_size, n_pages=n_pages, core_ids=core_ids)

    device.set_ptr_content(bf_main, tensor)
    
    for pipe_stage, core_id in enumerate(core_ids):
        core = device.cores[core_id]
        
        pipe_in_ptr = cb_pipe[pipe_stage - 1] if pipe_stage > 0 else bf_main
        pipe_out_ptr = cb_pipe[pipe_stage] if pipe_stage < n_cores - 1 else None
        
        kernel1 = read_kernel(core, pipe_in_ptr, pipe_out_ptr, cb_core_rd_ptr[pipe_stage], n_pages)
        kernel2 = compute_kernel(core, cb_core_rd_ptr[pipe_stage], cb_core_wr_ptr[pipe_stage], n_pages)
        
        core.dispatch_main_kernel("read", kernel1)
        core.dispatch_main_kernel("compute", kernel2)
        
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
    main_mem_core_tracer = MainMemCoreAnalyzer(device.main_mem_core)

    with MonitoringWindow() as monitor:
        for core_id, core in device.cores.items():
            if isinstance(core, NPUCore) and (not core.is_idle):
                pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
                pbar.bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    tracer_hub.save_traces(TRACE_DIR)
    profiler_hub.save_profiles(PROFILE_DIR)
    icnt_core_tracer.save_traces(os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv"))
    icnt_core_tracer.save_bandwidth_analysis(os.path.join(ANALYSIS_DIR, "icnt_core_bandwidth_analysis.csv"), bin_size=1)
    main_mem_core_tracer.save_traces(os.path.join(ANALYSIS_DIR, "main_mem_core_trace.csv"))
    main_mem_core_tracer.save_bandwidth_analysis(os.path.join(ANALYSIS_DIR, "main_mem_core_bandwidth_analysis.csv"), bin_size=1)

    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")

    for idx, core_id in enumerate(core_ids):
        print(f"\ncore {core_id} output:")
        print(device.get_ptr_content(cb_core_wr_ptr[idx], shape=(n_pages, page_numel), dtype=torch.int32))