import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "srcs"))

from neuromta.system.hardware.mesh_accelerator import MESH_TILE_COORDS, MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.api import MeshDeviceRuntimeContext, mesh_conv2d, mesh_linear, mesh_parameter, mesh_relu, mesh_sdpa, mesh_sum, mesh_tensor, set_attention_type
from neuromta.system.software.implementation.virtual import VirtualCompiler, VirtualRuntime
from neuromta.system.software.utils.scheduler import MeshSchedulingDomain


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULT_PATH = CACHE_DIR / "kernel_latency.csv"
MESH_SHAPES = tuple((rows, columns) for rows in (1, 2, 4, 8) for columns in (1, 4, 8))


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


def sdpa_kernel_workload(q_heads, kv_heads, q_length, kv_length, head_dim, dtype=torch.bfloat16):
    # A causal GQA step: Q is produced locally, while K/V reside in the KV cache.
    tile_shape = (1, 1, 32, 32)
    q = mesh_tensor(1, q_heads, q_length, head_dim, tile_shape=tile_shape, dtype=dtype).as_intermediate()
    k = mesh_tensor(1, kv_heads, kv_length, head_dim, tile_shape=tile_shape, dtype=dtype).to_device_memory()
    v = mesh_tensor(1, kv_heads, kv_length, head_dim, tile_shape=tile_shape, dtype=dtype).to_device_memory()
    return mesh_sdpa(q, k, v, is_causal=True)


WORKLOADS: dict[str, Callable] = {
    "linear_large": lambda: linear_kernel_workload(32, 2048, 2048),
    "linear_medium": lambda: linear_kernel_workload(32, 1024, 1024),
    "linear_small": lambda: linear_kernel_workload(32, 256, 256),
    "conv2d_large": lambda: conv2d_kernel_workload(32, 64, 32, 32, 128, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_medium": lambda: conv2d_kernel_workload(32, 32, 16, 16, 64, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_small": lambda: conv2d_kernel_workload(32, 16, 8, 8, 32, 3, 3, stride=1, padding=1, dilation=1),
    "elementwise_large": lambda: elementwise_kernel_workload(2048, 2048),
    "elementwise_medium": lambda: elementwise_kernel_workload(512, 512),
    "elementwise_small": lambda: elementwise_kernel_workload(256, 256),
    "reduction_large": lambda: reduction_kernel_workload(2048, 2048),
    "reduction_medium": lambda: reduction_kernel_workload(512, 512),
    "reduction_small": lambda: reduction_kernel_workload(256, 256),
    "sdpa_small": lambda: sdpa_kernel_workload(4, 4, 32, 128, 64),
    "sdpa_medium": lambda: sdpa_kernel_workload(8, 2, 64, 512, 128),
    "sdpa_large": lambda: sdpa_kernel_workload(16, 2, 128, 2048, 128),
}


def profile(configuration: tuple[str, tuple[int, int]]) -> tuple[str, int, int, int]:
    workload_name, shape = configuration
    rows, columns = shape
    config = MeshAcceleratorConfig.medium(
        icnt_mesh_shape=(8, 10),
        ccg_tile_coords=MESH_TILE_COORDS((8, 8), offset=(0, 1)),
        dma_tile_coords=MESH_TILE_COORDS((8, 10), offset=(0, 0), stride=(1, 9)),
    )
    device = MeshAccelerator(**config).initialize()
    core_mesh = device.get_ccg_tile_mesh()[:rows, :columns].copy()
    dma_ids = tuple(device.get_dma_tile_mesh().reshape(-1).tolist())
    domain = MeshSchedulingDomain("profile", core_mesh, dma_ids)

    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=VirtualRuntime(device, instances=[domain], require_full_coverage=False),
    ) as context:
        with context.new_compiler_context(compiler=VirtualCompiler(manual_ccg_tile_mesh=core_mesh)):
            WORKLOADS[workload_name]()
        context.runtime.run()
        
        states = context.runtime.workloads
        start_cycle = states[0].kernel_log[0].start_cycle
        completion_cycle = states[0].kernel_log[-1].completion_cycle
        cycles = int(completion_cycle - start_cycle)
        if cycles <= 0:
            raise RuntimeError(f"Invalid kernel latency for {workload_name} on {rows}x{columns}: {cycles}")
        return workload_name, rows, columns, cycles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Profile kernel latency across core mesh shapes.")
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS), default=tuple(WORKLOADS))
    parser.add_argument("--mesh-shapes", nargs="+", choices=tuple(f"{r}x{c}" for r, c in MESH_SHAPES), default=tuple(f"{r}x{c}" for r, c in MESH_SHAPES))
    parser.add_argument("--max-parallel", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--use-fused-attention", action="store_true", help="Use fused attention for profiling (default: False)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    if args.use_fused_attention:
        set_attention_type("fused")
    else:
        set_attention_type("sequential")
    shapes = tuple(tuple(map(int, shape.split("x"))) for shape in args.mesh_shapes)
    configurations = tuple(dict.fromkeys(itertools.product(args.workloads, shapes)))
    started = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(profile, configuration): configuration for configuration in configurations}
        for future in as_completed(futures):
            configuration = futures[future]
            try:
                results[configuration] = future.result()
            except Exception as error:
                raise RuntimeError(f"Profiling failed for {configuration[0]} on {configuration[1]}") from error

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("workload", "mesh_rows", "mesh_columns", "latency_cycles"))
        writer.writerows(results[configuration] for configuration in configurations)
    temporary_path.replace(RESULT_PATH)
    print(f"Completed {len(configurations)} configurations in {time.perf_counter() - started:.3f} s: {RESULT_PATH}")


if __name__ == "__main__":
    main()
