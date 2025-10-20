import os
import sys
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent import *

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

visualizer_enabled = False
try:
    from scripts.examples.utils.visualize import visualize_bandwidth_utilization_graph
    visualizer_enabled = True
except ImportError as e:
    logger.warning(f"failed to import visualize module: {e}")


TRACE_DIR = os.path.join(os.path.dirname(__file__), ".traces")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), ".analysis")
ICNT_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv")
ICNT_CORE_BW_ANALYSIS_FNAME = os.path.join(ANALYSIS_DIR, "icnt_core_bandwidth_analysis.csv")
MAIN_MEM_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "main_mem_core_trace.csv")
MAIN_MEM_CORE_BW_ANALYSIS_FNAME = os.path.join(ANALYSIS_DIR, "main_mem_core_bandwidth_analysis.csv")
IMG_SAVE_FNAME = os.path.join(ANALYSIS_DIR, "bandwidth_utilization.png")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


if __name__ == "__main__":
    logger.set_print_options(LogLevel.DEBUG)
    torch.set_printoptions(linewidth=1024, sci_mode=False)
    
    config = TenstorrentConfig.BLACKHOLE()

    device = TenstorrentDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 512
    N = 512
    K = 512
    dtype = torch.int8
    acc_dtype = torch.int32

    core_grid = device.get_npu_core_grid(offset=(0, 0), shape=(4, 4))

    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N).T  # (N, K)
    bias: torch.Tensor = torch.arange(0, N, dtype=acc_dtype).flatten()
    
    main_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=(32, 32))
    l1_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))
    core_ids = core_grid.core_ids

    main_buf_ifm  = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=main_layout, device=device).allocate(initial=ifm)
    main_buf_wgt  = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=main_layout, device=device).allocate(initial=wgt)
    main_buf_psum = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=main_layout.overrides(page_shape=(1, 32)), device=device).allocate(initial=bias)
    main_buf_ofm  = MCA_TensorBuffer(shape=(M, N),    dtype=acc_dtype, layout=main_layout, device=device).allocate()
    
    l1_buf_ifm    = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=l1_layout, device=device, core_ids=core_ids).allocate()
    l1_buf_wgt    = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=l1_layout, device=device, core_ids=core_ids).allocate()
    l1_buf_psum   = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=l1_layout.overrides(page_shape=(1, 32)), device=device, core_ids=core_ids).allocate()
    l1_buf_ofm    = MCA_TensorBuffer(shape=(M, N),    dtype=acc_dtype, layout=l1_layout, device=device, core_ids=core_ids).allocate()
    
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_ifm, dst_buf=l1_buf_ifm)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_wgt, dst_buf=l1_buf_wgt)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_psum, dst_buf=l1_buf_psum)
    
    MCA_RT_GLOBAL_SYNC(device, core_grid.core_ids)

    TT_RT_LINEAR(
        device=device, core_grid=core_grid,
        buf_ifm=l1_buf_ifm, buf_wgt=l1_buf_wgt, buf_bias=l1_buf_psum, buf_ofm=l1_buf_ofm,
        dtype=dtype, acc_dtype=acc_dtype,
    )
    
    MCA_RT_GLOBAL_SYNC(device, core_grid.core_ids)
    
    MCA_RT_DMA_STORE(device, src_buf=l1_buf_ofm, dst_buf=main_buf_ofm)
    
    
    tracer_hub = TracerHub()
    profiler_hub = ProfilerHub()
    
    for core_id, core in device.cores.items():
        tracer = Tracer()
        tracer.register_core(core)
        tracer_hub.register_tracer(f"{type(core).__name__}_{core.core_id}", tracer)
        
    for core_id in core_grid.core_ids:
        core = device.get_npu_core(core_id=core_id)
        profiler = CommandUtilizationProfiler(core)
        profiler_hub.register_profiler(f"{type(core).__name__}_{core.core_id}", profiler)
            
    
    with MonitoringWindow() as monitor:
        for core_id in core_grid.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            pbar.bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
        
    tracer_hub.save_traces(TRACE_DIR)
    profiler_hub.save_profiles(PROFILE_DIR)
    


    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm.to(dtype=acc_dtype), wgt.T.to(dtype=acc_dtype)) + bias
    simulated = main_buf_ofm.restore()

    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")
    
    for core_id, core in device.cores.items():
        if isinstance(core, NPUCore) and (not core.is_idle):
            print(f"\n=== NPUCore {core_id} RUNNING CONTEXT (EXCEPTION CAUSED BY PRETERMINATION)")
            for slot_id, kernel in core._dispatched_main_kernels.items():
                print(f"Slot {slot_id}: {kernel.callstack}")
                for cmd in kernel.recursive_current_commands(core):
                    print(f"  - {cmd}")
            for slot_id, kernel in core._dispatched_rpc_kernels.items():
                print(f"Slot {slot_id}: {kernel.callstack}")
                for cmd in kernel.recursive_current_commands(core):
                    print(f"  - {cmd}")