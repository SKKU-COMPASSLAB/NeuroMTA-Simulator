import os
import sys
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.hardware.analyzer.icnt_core_analyzer import IcntCoreAnalyzer
from neuromta.hardware.analyzer.main_mem_core_analyzer import MainMemCoreAnalyzer
from neuromta.ip.google_tpu import *

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
    
    config = GoogleTPUConfig.V4()

    device = GoogleTPUDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 4
    N = 384
    K = 508
    dtype = torch.int8
    acc_dtype = torch.int32

    ifm:  torch.Tensor = torch.arange(0, M * K, dtype=dtype).reshape(M, K).T
    wgt:  torch.Tensor = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    bias: torch.Tensor = torch.arange(0, N, dtype=acc_dtype).reshape(1, N).T
    
    main_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=(128, 128))
    l1_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(128, 128))
    core_id = device.npu_core_ids[0]

    main_buf_ifm  = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=main_layout, device=device)
    main_buf_wgt  = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=main_layout, device=device)
    main_buf_bias = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=main_layout.overrides(page_shape=(128, 1)), device=device)
    main_buf_ofm  = MCA_TensorBuffer(shape=(N, M), dtype=acc_dtype, layout=main_layout, device=device)
    
    l1_buf_ifm    = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=l1_layout, device=device, core_ids=[core_id])
    l1_buf_wgt    = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=l1_layout, device=device, core_ids=[core_id])
    l1_buf_bias   = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=l1_layout.overrides(page_shape=(128, 1)), device=device, core_ids=[core_id])
    l1_buf_ofm    = MCA_TensorBuffer(shape=(N, M), dtype=acc_dtype, layout=l1_layout, device=device, core_ids=[core_id])

    main_buf_ifm.update(ifm)
    main_buf_wgt.update(wgt)
    main_buf_bias.update(bias)
    
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_ifm, dst_buf=l1_buf_ifm)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_wgt, dst_buf=l1_buf_wgt)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_bias, dst_buf=l1_buf_bias)
    
    TPU_RT_LINEAR(
        device=device, core_id=core_id,
        buf_ifm=l1_buf_ifm, buf_wgt=l1_buf_wgt, buf_bias=l1_buf_bias, buf_ofm=l1_buf_ofm,
        dtype=dtype, acc_dtype=acc_dtype,
    )
    
    MCA_RT_DMA_STORE(device, src_buf=l1_buf_ofm, dst_buf=main_buf_ofm)
    
    tracer_hub = TracerHub()
    profiler_hub = ProfilerHub()
    
    for core in device.cores.values():
        tracer = Tracer()
        tracer.register_core(core)
        tracer_hub.register_tracer(f"{type(core).__name__}_{core.core_id}", tracer)
        
    core = device.get_npu_core(core_id=core_id)
    profiler = CommandUtilizationProfiler(core)
    profiler_hub.register_profiler(f"{type(core).__name__}_{core.core_id}", profiler)
            
    main_mem_core_tracer = MainMemCoreAnalyzer(device.main_mem_core)
    
    
    with MonitoringWindow() as monitor:
        core = device.get_npu_core(core_id=core_id)
        pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
        pbar.bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
        
        
    tracer_hub.save_traces(TRACE_DIR)
    profiler_hub.save_profiles(PROFILE_DIR)
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
    
    reference = torch.matmul(ifm.to(dtype=acc_dtype).T, wgt.to(dtype=acc_dtype)) + bias.T
    simulated = main_buf_ofm.restore().T

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