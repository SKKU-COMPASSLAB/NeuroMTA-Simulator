import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from neuromta.system.hardware.mesh_accelerator import MESH_TILE_COORDS, MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.api import MeshDeviceRuntimeContext, mesh_conv2d, mesh_linear, mesh_parameter, mesh_relu, mesh_sum, mesh_tensor
from neuromta.system.software.implementation.virtual import VirtualRuntime, VirtualCompiler
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
    "conv2d_large": lambda: conv2d_kernel_workload(32, 64, 32, 32, 128, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_medium": lambda: conv2d_kernel_workload(32, 32, 16, 16, 64, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_small": lambda: conv2d_kernel_workload(32, 16, 8, 8, 32, 3, 3, stride=1, padding=1, dilation=1),
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

    @property
    def execution_cycles(self) -> int:
        return self.completion_cycle - self.start_cycle


@dataclass(frozen=True)
class MeshPartitions:
    full_ccg_tile_mesh: np.ndarray
    lhs_ccg_tile_mesh: np.ndarray
    rhs_ccg_tile_mesh: np.ndarray
    full_dma_ids: tuple[int, ...]


def create_device() -> tuple[MeshAccelerator, MeshPartitions]:
    config = MeshAcceleratorConfig.medium(
        icnt_mesh_shape=(4, 10),
        ccg_tile_coords=MESH_TILE_COORDS(shape=(4, 8), offset=(0, 1)),
        dma_tile_coords=MESH_TILE_COORDS(shape=(4, 10), offset=(0, 0), stride=(1, 9)),
    )
    device = MeshAccelerator(**config).initialize()
    full_ccg_tile_mesh = device.get_ccg_tile_mesh().copy()
    lhs_ccg_tile_mesh = full_ccg_tile_mesh[:, :full_ccg_tile_mesh.shape[1] // 2].copy()
    rhs_ccg_tile_mesh = full_ccg_tile_mesh[:, full_ccg_tile_mesh.shape[1] // 2:].copy()
    full_dma_ids = tuple(int(dma_id) for dma_id in device.global_context.config.dma_tile_ids)
    return device, MeshPartitions(full_ccg_tile_mesh, lhs_ccg_tile_mesh, rhs_ccg_tile_mesh, full_dma_ids)


def execute_workloads(device: MeshAccelerator, requests: tuple[tuple[str, np.ndarray, tuple[int, ...]], ...]) -> tuple[tuple[KernelExecutionProfile, ...], int, float]:
    domains = tuple(MeshSchedulingDomain(f"partition.{index}", ccg_tile_mesh, dma_ids) for index, (_, ccg_tile_mesh, dma_ids) in enumerate(requests))
    
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=VirtualRuntime(device, instances=domains, require_full_coverage=False, exclusive_dma=False),
    ) as context:
        for index, ((workload_name, ccg_tile_mesh, _), domain) in enumerate(zip(requests, domains)):
            with context.new_compiler_context(
                compiler=VirtualCompiler(manual_ccg_tile_mesh=ccg_tile_mesh),
            ):
                WORKLOADS[workload_name]()

        runtime = context.runtime
        
        wall_start = time.perf_counter()
        jobs = runtime.run()
        wall_time = time.perf_counter() - wall_start
        states = runtime.workloads
        if len(jobs) != len(requests) or any(len(state.kernel_log) != 1 for state in states):
            raise RuntimeError("Each profiling workload must execute exactly one kernel.")
        profiles = tuple(KernelExecutionProfile(name, int(state.kernel_log[0].start_cycle), int(state.kernel_log[0].completion_cycle)) for (name, _, _), state in zip(requests, states))
        elapsed_cycles = max(profile.completion_cycle for profile in profiles) - min(profile.start_cycle for profile in profiles)
        return profiles, elapsed_cycles, wall_time
