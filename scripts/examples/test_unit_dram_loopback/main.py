import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.mta.tenstorrent import TenstorrentConfig, TenstorrentDevice

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.mta.tenstorrent import TenstorrentConfig, TenstorrentDevice

TRACE_DIR = os.path.join(os.path.dirname(__file__), ".traces")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), ".analysis")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


@jit_prototype
def kernel_main(core: NPUCore, dst_ref: BufferPointer, src_ref: BufferPointer, n_pages: int):
    core.mem_buffer_copy(dst_ref=src_ref, src_ref=dst_ref, n_pages=n_pages)


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = TenstorrentConfig.BLACKHOLE()

    device = TenstorrentDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    page_size = 32 * 32 * 4
    n_pages = 4
    dtype = torch.int32
    
    npu_core = device.npu_cores[0]
    
    src_ref = device.create_sharded_main_buffer(page_size, n_pages)
    dst_ref = device.create_sharded_main_buffer(page_size, n_pages)

    for i in range(n_pages):
        content = torch.zeros(page_size // dtype.itemsize, dtype=dtype).fill_(i+1)
        device.set_ptr_content(src_ref[i], content)
    
    kernel = kernel_main(npu_core, src_ref, dst_ref, n_pages=n_pages)
    npu_core.dispatch_main_kernel("main", kernel=kernel)
    
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
    icnt_core_tracer.save_traces(os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv"))
    icnt_core_tracer.save_bandwidth_analysis(os.path.join(ANALYSIS_DIR, "icnt_core_bandwidth_analysis.csv"), bin_size=1)
    main_mem_core_tracer.save_traces(os.path.join(ANALYSIS_DIR, "main_mem_core_trace.csv"))
    main_mem_core_tracer.save_bandwidth_analysis(os.path.join(ANALYSIS_DIR, "main_mem_core_bandwidth_analysis.csv"), bin_size=1)
    
    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    print(f"\n=== INPUT  BUFFER ===\n{device.get_ptr_content(src_ref[0], shape=(-1,), dtype=dtype)}")
    print(f"\n=== OUTPUT BUFFER ===\n{device.get_ptr_content(dst_ref[0], shape=(-1,), dtype=dtype)}")
