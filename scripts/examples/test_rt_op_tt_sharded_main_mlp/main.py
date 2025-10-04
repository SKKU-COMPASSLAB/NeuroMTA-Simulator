import os
import sys
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.hardware.analyzer.icnt_core_analyzer import IcntCoreAnalyzer
from neuromta.hardware.analyzer.main_mem_core_analyzer import MainMemCoreAnalyzer
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
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 512
    N = 512
    K = 512
    dtype = torch.int8
    acc_dtype = torch.int32

    core_grid = device.get_npu_core_grid(offset=(0, 0), shape=(4, 4))

    ifm:  torch.Tensor = torch.randint(-32, 32, (M * K,)).to(dtype=dtype).reshape(M, K)
    wgt:  torch.Tensor = torch.randint(-32, 32, (K * N,)).to(dtype=dtype).reshape(K, N).T  # (N, K)
    bias: torch.Tensor = torch.randint(-32, 32, (N,)).to(dtype=acc_dtype).flatten()

    # ifm = ifm - 0.5
    # wgt = wgt - 0.5
    # bias = bias - 0.5

    main_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=(32, 32))
    l1_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))
    core_ids = core_grid.core_ids

    main_buf_ifm  = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=main_layout, device=device)
    main_buf_wgt  = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=main_layout, device=device)
    main_buf_psum = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=main_layout.overrides(page_shape=(1, 32)), device=device)

    main_buf_ifm.update(ifm)
    main_buf_wgt.update(wgt)
    main_buf_psum.update(bias)
    
    l1_buf_ifm  = TT_RT_DMA_LOAD(device, core_grid, main_buf=main_buf_ifm,  l1_layout=l1_layout)
    l1_buf_wgt  = TT_RT_DMA_LOAD(device, core_grid, main_buf=main_buf_wgt,  l1_layout=l1_layout)
    l1_buf_psum = TT_RT_DMA_LOAD(device, core_grid, main_buf=main_buf_psum, l1_layout=l1_layout.overrides(page_shape=(1, 32)))
    
    MCA_RT_GLOBAL_SYNC(device, core_grid.core_ids)

    l1_buf_ofm = TT_RT_LINEAR(
        device=device, core_grid=core_grid,
        buf_ifm=l1_buf_ifm, buf_wgt=l1_buf_wgt, buf_bias=l1_buf_psum,
        dtype=dtype, acc_dtype=acc_dtype,
    )
    
    MCA_RT_GLOBAL_SYNC(device, core_grid.core_ids)
    
    l1_buf_ofm = TT_RT_RELU(device, core_grid, l1_buf_ofm, inplace=True)  # TODO: is this layer fusion??
    
    MCA_RT_GLOBAL_SYNC(device, core_grid.core_ids)
    
    main_buf_ofm = TT_RT_DMA_STORE(device, core_grid, l1_buf=l1_buf_ofm, main_layout=main_layout)
    
    
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
            
    icnt_core_tracer = IcntCoreAnalyzer(device.icnt_core)
    main_mem_core_tracer = MainMemCoreAnalyzer(device.main_mem_core)
    
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
    icnt_core_tracer.save_traces(ICNT_CORE_TRACE_FNAME)
    icnt_core_tracer.save_bandwidth_analysis(ICNT_CORE_BW_ANALYSIS_FNAME, bin_size=1)
    main_mem_core_tracer.save_traces(MAIN_MEM_CORE_TRACE_FNAME)
    main_mem_core_tracer.save_bandwidth_analysis(MAIN_MEM_CORE_BW_ANALYSIS_FNAME, bin_size=1)
    
    try:
        if visualizer_enabled:
            visualize_bandwidth_utilization_graph(
                PROFILE_DIR,
                ICNT_CORE_TRACE_FNAME,
                MAIN_MEM_CORE_TRACE_FNAME,
                IMG_SAVE_FNAME
            )
    except Exception as e:
        logger.warning(f"failed to visualize the bandwidth utilization graph: {e}")


    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    linear_ref = torch.matmul(ifm.to(dtype=acc_dtype), wgt.T.to(dtype=acc_dtype)) + bias
    simulated = main_buf_ofm.restore()
    
    n_neg_vals_linear_ref = (linear_ref < 0).sum().item()
    n_neg_vals_simulated = (simulated < 0).sum().item()
    
    linear_ref = torch.nn.functional.relu(linear_ref)
    
    print(f"\nnumber of negative values in linear output: {n_neg_vals_linear_ref}")
    print(f"number of negative values in simulated output: {n_neg_vals_simulated}")
    print(f"simulation terminated with valid result: {n_neg_vals_simulated == 0}")
    print(f"simulation terminated with valid result: {torch.allclose(linear_ref, simulated, atol=1e-2)}")