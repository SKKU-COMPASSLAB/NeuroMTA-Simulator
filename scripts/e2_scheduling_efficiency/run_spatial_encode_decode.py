import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.nn.qwen2_5_omni import *
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler

if __package__:
    from .common import KernelProfile, WorkloadProfile, save_profiles, vision_encode, text_decode_request, prepare_kv_cache
else:
    from common import KernelProfile, WorkloadProfile, save_profiles, vision_encode, text_decode_request, prepare_kv_cache


ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, ".cache")
LOG_DIR = os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(__file__))[0])
KERNEL_PROFILE_DIR = os.path.join(LOG_DIR, "kernel_profile")
KERNEL_PROFILE_CSV_FMT = os.path.join(KERNEL_PROFILE_DIR, "kernel_profile_{workload_id}.csv")
WORKLOAD_PROFILE_DIR = os.path.join(LOG_DIR, "workload_profile")
WORKLOAD_PROFILE_CSV_FMT = os.path.join(WORKLOAD_PROFILE_DIR, "workload_profile_{workload_id}.csv")

os.makedirs(KERNEL_PROFILE_DIR, exist_ok=True)
os.makedirs(WORKLOAD_PROFILE_DIR, exist_ok=True)


def run(
    n_video_requests: int = 10,
    video_interval:   int = 500_000_000,    # interval between video requests = 0.5s (slo for video request = 0.5s)
    text_arrival:     int = 500_000_000,    # arrival time of the text request = 0.5s
    prompt_length:    int = 1024,
    decode_tokens:    int = 512,
    ttft:             int = 1_000_000_000,  # time to first token = 1s
    tpot:             int = 50_000_000,     # time per output token = 50ms
    enable_debug_log: bool = True,
    mesh_accel_config_name: str = "large"
):
    if enable_debug_log:
        logger.set_print_options(log_level=LogLevel.DEBUG)
    else:
        logger.set_print_options(log_level=LogLevel.INFO)

    if mesh_accel_config_name == "large":
        config = MeshAcceleratorConfig.large()
    elif mesh_accel_config_name == "medium":
        config = MeshAcceleratorConfig.medium()
    elif mesh_accel_config_name == "small":
        config = MeshAcceleratorConfig.small()
    else:
        raise ValueError("Invalid mesh_accel_config_name")

    device = MeshAccelerator(**config).initialize()

    with MeshDeviceRuntimeContext(
        device=device, 
        default_tile_shape=(32, 32), 
        default_dtype=torch.bfloat16, 
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=enable_debug_log)
    ) as context:
        model = Qwen2_5_Omni()
        kv_cache = mesh_kv_cache(batch_size=1, max_seq_len=8192, num_layers=36, num_kv_heads=2, head_dim=128)
        prepare_kv_cache(kv_cache, prompt_length)
        
        for i in range(n_video_requests):
            arrival_cycle = i * video_interval
            vision_encode(SpatialCompiler, context, model, kv_cache, arrival_cycle=arrival_cycle)
        text_decode_request(SpatialCompiler, context, model, kv_cache, decode_tokens=decode_tokens, arrival_cycle=text_arrival)

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("The Qwen2.5-Omni workload did not complete.")

        workload_profile = WorkloadProfile()
        for workload in runtime.workloads:
            kernel_profile = KernelProfile()
            
            if workload.workload_id.startswith("encode."):
                slo = video_interval
            elif workload.workload_id.startswith("text.prefill."):
                slo = ttft
            elif workload.workload_id.startswith("text.decode."):
                slo = tpot * decode_tokens

            for kernel in workload.kernel_log:
                kernel_profile.add(kernel.compiled_kernel.kernel_desc, kernel.completion_cycle - kernel.start_cycle, int(kernel.placement.core_mesh.size))
            workload_profile.add(workload.workload_id, workload.arrival_cycle, workload.start_cycle, workload.completion_cycle, slo)
            
            save_profiles(kernel_profile, KERNEL_PROFILE_CSV_FMT.format(workload_id=workload.workload_id))
        save_profiles(workload_profile, WORKLOAD_PROFILE_CSV_FMT.format(workload_id=workload.workload_id))

        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}; simulation time: {simulation_time:.6f} s")
        logger.info(f"Kernel profiles: {KERNEL_PROFILE_DIR}")
        logger.info(f"Workload profiles: {WORKLOAD_PROFILE_DIR}")
        

if __name__ == "__main__":
    run(n_video_requests=5, prompt_length=128, decode_tokens=10)
