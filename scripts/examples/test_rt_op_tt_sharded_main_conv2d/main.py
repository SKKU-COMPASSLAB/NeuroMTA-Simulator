import os
import sys
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent import *
from neuromta.hardware.companions.booksim import BookSim2
from neuromta.hardware.companions.dramsim import DRAMSim3

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
    
    N, H, W, C = 1, 224, 224, 32
    K = 128
    FH, FW = 3, 3
    SH, SW = 1, 1
    PH, PW = 1, 1
    DH, DW = 1, 1
    
    dtype = torch.float32
    acc_dtype = torch.float32

    core_grid = device.get_npu_core_grid(offset=(0, 0), shape=(6, 6))

    ifm:  torch.Tensor = torch.randint(0, 16, (N * H * W * C,)).to(dtype=dtype).reshape(N, H, W, C)
    wgt:  torch.Tensor = torch.randint(0, 16, (FH * FW * K * C,)).to(dtype=dtype).reshape(FH, FW, K, C)
    bias: torch.Tensor = torch.randint(0, 16, (K,)).to(dtype=acc_dtype).flatten()
    
    l1_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))
    main_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=(32, 32))
    core_ids = core_grid.core_ids

    main_buf_ifm  = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=main_layout, device=device, core_ids=core_ids).allocate(initial=ifm)
    main_buf_wgt  = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=main_layout, device=device, core_ids=core_ids).allocate(initial=wgt)
    main_buf_bias = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=main_layout.overrides(page_shape=(1, 32)), device=device, core_ids=core_ids).allocate(initial=bias)
    main_buf_ofm  = MCA_TensorBuffer(shape=(N, H, W, K), dtype=acc_dtype, layout=main_layout, device=device, core_ids=core_ids).allocate()

    l1_buf_ifm  = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=l1_layout, device=device, core_ids=core_ids).allocate()
    l1_buf_wgt  = MCA_TensorBuffer(shape=wgt.shape,  dtype=wgt.dtype,  layout=l1_layout, device=device, core_ids=core_ids).allocate()
    l1_buf_bias = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=l1_layout.overrides(page_shape=(1, 32)), device=device, core_ids=core_ids).allocate()
    l1_buf_ofm  = MCA_TensorBuffer(shape=(N, H, W, K), dtype=acc_dtype, layout=l1_layout, device=device, core_ids=core_ids).allocate()
    
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_ifm, dst_buf=l1_buf_ifm)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_wgt, dst_buf=l1_buf_wgt)
    MCA_RT_DMA_LOAD(device, src_buf=main_buf_bias, dst_buf=l1_buf_bias)

    MCA_RT_GLOBAL_SYNC(device, core_ids=core_ids)
    
    TT_RT_CONV2D(
        device = device,
        core_grid = core_grid,

        buf_ifm  = l1_buf_ifm,
        buf_wgt  = l1_buf_wgt,
        buf_bias = l1_buf_bias,
        buf_ofm  = l1_buf_ofm,

        stride   = (SH, SW),
        padding  = (PH, PW),
        dilation = (DH, DW),
        
        dtype = dtype,
        acc_dtype = acc_dtype,
    )
    
    MCA_RT_GLOBAL_SYNC(device, core_ids=core_ids)
    
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
        
    booksim_module: BookSim2 = device.companion_core.get_companion_module(device.cmap_context.config.booksim_module_id)
    dramsim_module: DRAMSim3 = device.companion_core.get_companion_module(device.cmap_context.config.dramsim_module_id)
    booksim_module.enable_bandwidth_profiling(resolution=10)
    dramsim_module.enable_bandwidth_profiling(resolution=10)

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
    booksim_module.save_bandwidth_profiles_as_file(os.path.join(ANALYSIS_DIR, "booksim2"))
    dramsim_module.save_bandwidth_profiles_as_file(os.path.join(ANALYSIS_DIR, "dramsim3"))

    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.conv2d(
        input = ifm.permute(0, 3, 1, 2).to(dtype=acc_dtype),
        weight = wgt.permute(2, 3, 0, 1).to(dtype=acc_dtype),
        bias = bias,
        stride = (SH, SW),
        padding = (PH, PW),
        dilation = (DH, DW),
    ).permute(0, 2, 3, 1)

    simulated = main_buf_ofm.restore()

    print(f"\n=== REFERENCE ===\n{reference.reshape(-1, K)}")
    print(f"\n=== SIMULATED ===\n{simulated.reshape(-1, K)}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")
    
    for core_id, core in device.cores.items():
        if isinstance(core, NPUCore) and (not core.is_idle):
            print(f"\n=== NPUCore {core_id} RUNNING CONTEXT (EXCEPTION CAUSED BY PRETERMINATION)")
            for slot_id, kernel in core._dispatched_main_kernels.items():
                print(f"Slot {slot_id}: {kernel.callstack}")
                for cmd in kernel.recursive_current_commands(core):
                    print(f"  - {cmd}")