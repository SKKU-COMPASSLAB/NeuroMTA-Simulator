import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from neuromta.system.hardware.mesh_accelerator import MESH_TILE_COORDS, MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.api import MeshDeviceRuntimeContext, mesh_conv2d, mesh_linear, mesh_parameter, mesh_relu, mesh_sum, mesh_tensor
from neuromta.system.software.implementation.sequential import *
from neuromta.system.software.implementation.virtual import *
from neuromta.system.software.implementation.preemptive import *
from neuromta.system.software.implementation.spatial import *
from neuromta.system.software.utils.scheduler import MeshSchedulingDomain


def linear_kernel_workload(M, N, K, dtype=torch.bfloat16):
    x = mesh_tensor(M, K, dtype=dtype).as_intermediate()
    w = mesh_parameter(N, K, dtype=dtype).as_weight()
    b = mesh_parameter(N, dtype=dtype).as_weight()
    return mesh_linear(x, w, b)


def conv2d_kernel_workload(N, C, H, W, K, R, S, stride, padding, dilation, dtype=torch.bfloat16):
    x = mesh_tensor(N, H, W, C, tile_shape=(1, 1, 32, 32), dtype=dtype).as_intermediate()
    w = mesh_parameter(R, S, K, C, tile_shape=(1, 1, 32, 32), dtype=dtype).as_weight()
    b = mesh_parameter(K, dtype=dtype).as_weight()
    return mesh_conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation)


def elementwise_kernel_workload(M, N, dtype=torch.bfloat16):
    x = mesh_tensor(M, N, dtype=dtype).as_intermediate()
    return mesh_relu(x)


def reduction_kernel_workload(M, N, dtype=torch.bfloat16):
    x = mesh_tensor(M, N, dtype=dtype).as_intermediate()
    return mesh_sum(x, dim=-1)


WORKLOADS: dict[str, Callable] = {
    "linear_large": lambda: linear_kernel_workload(32, 1024, 1024),
    "linear_medium": lambda: linear_kernel_workload(32, 512, 512),
    "linear_small": lambda: linear_kernel_workload(32, 256, 256),
    "conv2d_large": lambda: conv2d_kernel_workload(1, 64, 32, 32, 128, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_medium": lambda: conv2d_kernel_workload(1, 32, 16, 16, 64, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_small": lambda: conv2d_kernel_workload(1, 16, 8, 8, 32, 3, 3, stride=1, padding=1, dilation=1),
    "elementwise_large": lambda: elementwise_kernel_workload(1024, 1024),
    "elementwise_medium": lambda: elementwise_kernel_workload(512, 512),
    "elementwise_small": lambda: elementwise_kernel_workload(256, 256),
    "reduction_large": lambda: reduction_kernel_workload(1024, 1024),
    "reduction_medium": lambda: reduction_kernel_workload(512, 512),
    "reduction_small": lambda: reduction_kernel_workload(256, 256),
}


@dataclass(frozen=True)
class KernelExecutionProfile:
    workload: str
    start_cycle: int
    completion_cycle: int
    core_count: int

    @property
    def execution_cycles(self) -> int:
        return self.completion_cycle - self.start_cycle
    

def create_device() -> MeshAccelerator:
    config = MeshAcceleratorConfig.medium(
        icnt_mesh_shape=(4, 10),
        ccg_tile_coords=MESH_TILE_COORDS(shape=(4, 8), offset=(0, 1)),
        dma_tile_coords=MESH_TILE_COORDS(shape=(4, 10), offset=(0, 0), stride=(1, 9)),
    )
    device = MeshAccelerator(**config).initialize()
    return device


def execute_workloads_with_sequential(device: MeshAccelerator, workload_id_1: str, workload_id_2: str) -> tuple[tuple[KernelExecutionProfile, ...], int, float]:
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SequentialRuntime(device),
    ) as context:
        with context.new_compiler_context(compiler=SequentialCompiler()):
            WORKLOADS[workload_id_1]()
        with context.new_compiler_context(compiler=SequentialCompiler()):
            WORKLOADS[workload_id_2]()

        runtime = context.runtime
        
        wall_start = time.perf_counter()
        jobs = runtime.run()
        wall_time = time.perf_counter() - wall_start
        
        states = runtime.workloads
        if len(jobs) != 2 or any(len(state.kernel_log) != 1 for state in states):
            raise RuntimeError("Each profiling workload must execute exactly one kernel.")
        
        profiles = tuple(KernelExecutionProfile(name, int(state.kernel_log[0].start_cycle), int(state.kernel_log[0].completion_cycle), int(state.kernel_log[0].placement.core_mesh.size)) for name, state in zip([workload_id_1, workload_id_2], states))
        elapsed_cycles = max(profile.completion_cycle for profile in profiles) - min(profile.start_cycle for profile in profiles)
        
        return profiles, elapsed_cycles, wall_time


def execute_workloads_with_virtual(device: MeshAccelerator, workload_id_1: str, workload_id_2: str) -> tuple[tuple[KernelExecutionProfile, ...], int, float]:
    ccg_mesh_shape = device.get_ccg_tile_mesh().shape
    ccg_mesh_1 = device.get_ccg_tile_mesh()[:, :ccg_mesh_shape[1] // 2].copy()
    ccg_mesh_2 = device.get_ccg_tile_mesh()[:, ccg_mesh_shape[1] // 2:].copy()
    
    dma_mesh_shape = device.get_dma_tile_mesh().shape
    dma_mesh_1 = device.get_dma_tile_mesh()[:, :dma_mesh_shape[1] // 2].copy()
    dma_mesh_2 = device.get_dma_tile_mesh()[:, dma_mesh_shape[1] // 2:].copy()
    
    domain1 = MeshSchedulingDomain("partition.1", ccg_mesh_1, dma_mesh_1.flatten().tolist())
    domain2 = MeshSchedulingDomain("partition.2", ccg_mesh_2, dma_mesh_2.flatten().tolist())
    
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=VirtualRuntime(device, instances=[domain1, domain2], require_full_coverage=False, exclusive_dma=False),
    ) as context:
        with context.new_compiler_context(compiler=VirtualCompiler(manual_ccg_tile_mesh=ccg_mesh_1)):
            WORKLOADS[workload_id_1]()
        with context.new_compiler_context(compiler=VirtualCompiler(manual_ccg_tile_mesh=ccg_mesh_2)):
            WORKLOADS[workload_id_2]()

        runtime = context.runtime
        
        wall_start = time.perf_counter()
        jobs = runtime.run()
        wall_time = time.perf_counter() - wall_start
        
        states = runtime.workloads
        if len(jobs) != 2 or any(len(state.kernel_log) != 1 for state in states):
            raise RuntimeError("Each profiling workload must execute exactly one kernel.")
        
        profiles = tuple(KernelExecutionProfile(name, int(state.kernel_log[0].start_cycle), int(state.kernel_log[0].completion_cycle), int(state.kernel_log[0].placement.core_mesh.size)) for name, state in zip([workload_id_1, workload_id_2], states))
        elapsed_cycles = max(profile.completion_cycle for profile in profiles) - min(profile.start_cycle for profile in profiles)
        
        return profiles, elapsed_cycles, wall_time
    

def execute_workloads_with_preemptive(device: MeshAccelerator, workload_id_1: str, workload_id_2: str) -> tuple[tuple[KernelExecutionProfile, ...], int, float]:
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=PreemptiveRuntime(device),
    ) as context:
        with context.new_compiler_context(compiler=PreemptiveCompiler()):
            WORKLOADS[workload_id_1]()
        with context.new_compiler_context(compiler=PreemptiveCompiler()):
            WORKLOADS[workload_id_2]()

        runtime = context.runtime
        
        wall_start = time.perf_counter()
        jobs = runtime.run()
        wall_time = time.perf_counter() - wall_start
        
        states = runtime.workloads
        if len(jobs) != 2 or any(len(state.kernel_log) != 1 for state in states):
            raise RuntimeError("Each profiling workload must execute exactly one kernel.")
        
        profiles = tuple(KernelExecutionProfile(name, int(state.kernel_log[0].start_cycle), int(state.kernel_log[0].completion_cycle), int(state.kernel_log[0].placement.core_mesh.size)) for name, state in zip([workload_id_1, workload_id_2], states))
        elapsed_cycles = max(profile.completion_cycle for profile in profiles) - min(profile.start_cycle for profile in profiles)
        
        return profiles, elapsed_cycles, wall_time
    

def execute_workloads_with_spatial(device: MeshAccelerator, workload_id_1: str, workload_id_2: str, decision_hook=None) -> tuple[tuple[KernelExecutionProfile, ...], int, float]:
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SpatialRuntime(device, scheduler=SpatialScheduler(decision_hook=decision_hook)),
    ) as context:
        with context.new_compiler_context(compiler=SpatialCompiler()):
            WORKLOADS[workload_id_1]()
        with context.new_compiler_context(compiler=SpatialCompiler()):
            WORKLOADS[workload_id_2]()

        runtime = context.runtime
        
        wall_start = time.perf_counter()
        jobs = runtime.run()
        wall_time = time.perf_counter() - wall_start
        
        states = runtime.workloads
        if len(jobs) != 2 or any(len(state.kernel_log) != 1 for state in states):
            raise RuntimeError("Each profiling workload must execute exactly one kernel.")
        
        profiles = tuple(KernelExecutionProfile(name, int(state.kernel_log[0].start_cycle), int(state.kernel_log[0].completion_cycle), int(state.kernel_log[0].placement.core_mesh.size)) for name, state in zip([workload_id_1, workload_id_2], states))
        elapsed_cycles = max(profile.completion_cycle for profile in profiles) - min(profile.start_cycle for profile in profiles)
        
        return profiles, elapsed_cycles, wall_time
