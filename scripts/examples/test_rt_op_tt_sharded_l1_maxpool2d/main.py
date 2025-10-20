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
    
    N, H, W, C = 1, 6, 6, 64
    FH, FW = 2, 2
    SH, SW = 2, 2
    PH, PW = 0, 0
    DH, DW = 1, 1
    
    OH = (H + 2 * PH - DH * (FH-1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (FW-1) - 1) // SW + 1
    
    dtype = torch.int32
    acc_dtype = torch.int32

    core_grid = device.get_npu_core_grid(offset=(0, 0), shape=(8, 8))

    ifm:  torch.Tensor = torch.randint(0, 16, (N * H * W * C,)).to(dtype=dtype).reshape(N, H, W, C)
    
    layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))
    core_ids = core_grid.core_ids

    buf_ifm = MCA_TensorBuffer(shape=ifm.shape,  dtype=ifm.dtype,  layout=layout, device=device, core_ids=core_ids)
    buf_ofm = MCA_TensorBuffer(shape=(N, OH, OW, C), dtype=dtype, layout=layout, device=device, core_ids=core_ids)
    
    buf_ifm.update(ifm)

    TT_RT_MAXPOOL2D(
        device = device,
        core_grid = core_grid,
        
        buf_ifm = buf_ifm,
        buf_ofm = buf_ofm,
        
        kernel = (FH, FW),
        stride = (SH, SW),
        padding = (PH, PW),
        dilation = (DH, DW),
        
        dtype = dtype,
        acc_dtype = acc_dtype,
    )
    
    
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
    
    reference = torch.max_pool2d(
        input = ifm.permute(0, 3, 1, 2).to(dtype=acc_dtype),
        kernel_size = (FH, FW),
        stride = (SH, SW),
        padding = (PH, PW),
        dilation = (DH, DW),
    ).permute(0, 2, 3, 1)
    
    simulated = buf_ofm.restore()

    print(f"\n=== REFERENCE ===\n{reference.reshape(-1, C)}")
    print(f"\n=== SIMULATED ===\n{simulated.reshape(-1, C)}")
    print(f"\nnumber of mismatched elements: {torch.sum(reference != simulated)} / {torch.numel(reference)}")
    print(f"simulation terminated with valid result: {torch.allclose(reference, simulated)}")
    
    for core_id, core in device.cores.items():
        if isinstance(core, NPUCore) and (not core.is_idle):
            print(f"\n=== NPUCore {core_id} RUNNING CONTEXT (EXCEPTION CAUSED BY PRETERMINATION)")
            for slot_id, kernel in core._dispatched_main_kernels.items():
                print(f"Slot {slot_id}: {kernel.callstack}")
                for cmd in kernel.recursive_current_commands(core):
                    print(f"  - {cmd}")
