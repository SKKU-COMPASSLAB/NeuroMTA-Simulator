import csv
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from neuromta.framework.logger import LogLevel, logger
from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.api import MeshDeviceRuntimeContext, MeshMemoryType, MeshTensorDescriptor, MeshTensorType, MeshWorkloadSchedulingHint, mesh_copy, mesh_linear, mesh_relu, mesh_tensor
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor
from neuromta.system.software.utils.runtime import MeshDeviceRuntime, MeshDeviceRuntimeWorkloadState
from neuromta.system.software.utils.scheduler import MeshFRFCFSScheduler, MeshSchedulingDomain

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
BENCHMARK_IDS = ("compute.large", "memory.small")
PERIOD_CYCLES = 500_000
SLO_CYCLES = 400_000
DEFAULT_DURATION_CYCLES = 5_000_000
DEFAULT_WARMUP_CYCLES = 1_000_000
DEFAULT_CCG_TOPS = 0.05


def _parameter(context: MeshDeviceRuntimeContext, namespace: str, name: str, *shape: int) -> MeshTensorDescriptor:
    tensor = MeshTensorDescriptor(shape=shape, tile_shape=context.default_tile_shape[-min(len(context.default_tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
    tensor.parameter_id = f"e3_2.{namespace}/{name}"
    tensor.preferred_mem = MeshMemoryType.LOCAL_CACHE
    return tensor


class ComputeIntensiveBenchmark:
    input_shape = (64, 256)
    output_shape = (64, 128)
    input_preferred_mem = None

    def __init__(self, context: MeshDeviceRuntimeContext):
        self.weights = (
            _parameter(context, "compute", "linear0.weight", 256, 256),
            _parameter(context, "compute", "linear1.weight", 256, 256),
            _parameter(context, "compute", "linear2.weight", 128, 256),
        )
        self.biases = (
            _parameter(context, "compute", "linear0.bias", 1, 256),
            _parameter(context, "compute", "linear1.bias", 1, 256),
            _parameter(context, "compute", "linear2.bias", 1, 128),
        )

    def forward(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        x = mesh_relu(mesh_linear(x, self.weights[0], self.biases[0]))
        x = mesh_relu(mesh_linear(x, self.weights[1], self.biases[1]))
        return mesh_linear(x, self.weights[2], self.biases[2])


class MemoryIntensiveBenchmark:
    input_shape = (512, 512)
    output_shape = input_shape
    input_preferred_mem = MeshMemoryType.DEVICE_MEMORY
    copy_count = 4

    def __init__(self, context: MeshDeviceRuntimeContext):
        self.context = context

    def forward(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        for _ in range(self.copy_count):
            x = mesh_copy(x, preferred_mem=MeshMemoryType.DEVICE_MEMORY)
        return x


@dataclass
class KernelProfileEntry:
    kernel_type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    count: int = 0
    total_latency: int = 0
    core_count_sum: int = 0
    core_count_square_sum: int = 0

    def add(self, latency: int, core_count: int) -> None:
        self.count += 1
        self.total_latency += latency
        self.core_count_sum += core_count
        self.core_count_square_sum += core_count * core_count

    def to_csv_entry(self) -> str:
        average_latency = self.total_latency / self.count
        core_count_average = self.core_count_sum / self.count
        core_count_variance = max(0.0, self.core_count_square_sum / self.count - core_count_average * core_count_average)
        input_shapes = "|".join("x".join(str(dim) for dim in shape) for shape in self.input_shapes)
        output_shapes = "|".join("x".join(str(dim) for dim in shape) for shape in self.output_shapes)
        return f"{self.kernel_type},{input_shapes},{output_shapes},{self.count},{average_latency},{core_count_average},{math.sqrt(core_count_variance)}"


@dataclass
class KernelProfile:
    entries: dict[tuple[str, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], KernelProfileEntry] = field(default_factory=dict)

    def add(self, kernel_desc: MeshKernelDescriptor, latency: int, core_count: int) -> None:
        kernel_type = kernel_desc.kernel_type.name
        input_shapes = tuple(tensor.shape for tensor in kernel_desc.input_tensors)
        output_shapes = tuple(tensor.shape for tensor in kernel_desc.output_tensors)
        key = (kernel_type, input_shapes, output_shapes)
        if key not in self.entries:
            self.entries[key] = KernelProfileEntry(kernel_type, input_shapes, output_shapes)
        self.entries[key].add(latency, core_count)

    def to_csv(self) -> str:
        header = "kernel_type,input_shapes,output_shapes,count,average_latency,core_count_average,core_count_stddev"
        return header + "\n" + "\n".join(entry.to_csv_entry() for entry in self.entries.values())


@dataclass(frozen=True)
class WorkloadProfileEntry:
    workload_id: str
    benchmark_id: str
    arrival_cycle: int
    start_cycle: int
    completion_cycle: int
    slo: int

    def to_csv_entry(self) -> str:
        response_time = self.completion_cycle - self.arrival_cycle
        execution_time = self.completion_cycle - self.start_cycle
        slack = self.slo - response_time
        return f"{self.workload_id},{self.benchmark_id},{self.arrival_cycle},{self.start_cycle},{self.completion_cycle},{self.slo},{response_time},{execution_time},{slack},{slack >= 0}"


@dataclass
class WorkloadProfile:
    entries: list[WorkloadProfileEntry] = field(default_factory=list)

    def add(self, workload_id: str, benchmark_id: str, arrival_cycle: int, start_cycle: int, completion_cycle: int, slo: int) -> None:
        self.entries.append(WorkloadProfileEntry(workload_id, benchmark_id, arrival_cycle, start_cycle, completion_cycle, slo))

    def to_csv(self) -> str:
        header = "workload_id,benchmark_id,arrival_cycle,start_cycle,completion_cycle,slo,response_time,execution_time,slack,deadline_met"
        return header + "\n" + "\n".join(entry.to_csv_entry() for entry in self.entries)


@dataclass(frozen=True)
class WorkloadRequest:
    workload_id: str
    benchmark_id: str
    arrival_cycle: int
    slo: int = SLO_CYCLES


def create_device(ccg_tops: float) -> MeshAccelerator:
    return MeshAccelerator(**MeshAcceleratorConfig.small(ccg_tops=ccg_tops)).initialize()


def create_scheduler() -> MeshFRFCFSScheduler:
    return MeshFRFCFSScheduler(starvation_cycles=SLO_CYCLES, candidate_window=16)


def create_virtual_domains(device: MeshAccelerator) -> tuple[MeshSchedulingDomain, ...]:
    core_mesh = device.get_ccg_tile_mesh()
    dma_ids = tuple(device.global_context.config.dma_tile_ids)
    if core_mesh.shape != (4, 2) or len(dma_ids) != 4:
        raise ValueError(f"The e3.2 virtual partition requires a 4x2 CCG mesh and four DMA tiles, got {core_mesh.shape} and {len(dma_ids)} DMA tiles.")
    return (
        MeshSchedulingDomain("compute.large", core_mesh[:3, :].copy(), dma_ids[:2]),
        MeshSchedulingDomain("memory.small", core_mesh[3:, :].copy(), dma_ids[2:]),
    )


def build_requests(duration_cycles: int) -> list[WorkloadRequest]:
    return [WorkloadRequest(f"{benchmark_id}.{request_index}", benchmark_id, arrival_cycle) for request_index, arrival_cycle in enumerate(range(0, duration_cycles, PERIOD_CYCLES), start=1) for benchmark_id in BENCHMARK_IDS]


def submit_workloads(context: MeshDeviceRuntimeContext, compiler_type: type, requests: list[WorkloadRequest], domain_ids: dict[str, str] | None = None) -> None:
    benchmarks = {"compute.large": ComputeIntensiveBenchmark(context), "memory.small": MemoryIntensiveBenchmark(context)}
    for request in requests:
        benchmark = benchmarks[request.benchmark_id]
        domain_id = None if domain_ids is None else domain_ids[request.benchmark_id]
        scheduling_hint = MeshWorkloadSchedulingHint(priority=1, weight=1.0, max_wait_cycles=request.slo)
        with context.new_compiler_context(compiler_type(), arrival_cycle=request.arrival_cycle, workload_id=request.workload_id, scheduling_hint=scheduling_hint, domain_id=domain_id):
            input_tensor = mesh_tensor(*benchmark.input_shape, tile_shape=context.default_tile_shape, preferred_mem=benchmark.input_preferred_mem)
            output = benchmark.forward(input_tensor)
            if output.shape != benchmark.output_shape:
                raise RuntimeError(f"Unexpected {request.benchmark_id} output shape: {output.shape}")


def save_decision_profile(runtime: MeshDeviceRuntime, path: str) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("cycle", "scheduler", "kernel_ids", "core_meshes", "memory_bank_ids", "benefit"))
        for record in runtime.decision_log:
            kernel_ids = "|".join(record["kernels"])
            core_meshes = ";".join(f"{'x'.join(str(size) for size in mesh.shape)}:{'|'.join(str(core_id) for core_id in np.asarray(mesh).reshape(-1))}" for mesh in record["core_meshes"])
            memory_bank_ids = ";".join("|".join(str(bank_id) for bank_id in bank_ids) for bank_ids in record["memory_banks"])
            writer.writerow((record["cycle"], record["scheduler"], kernel_ids, core_meshes, memory_bank_ids, record["benefit"]))


def save_profiles(runtime: MeshDeviceRuntime, experiment_name: str, warmup_cycles: int, requests: list[WorkloadRequest], elapsed_seconds: float, completed_jobs: int, simulation_completion_cycle: int) -> None:
    output_dir = os.path.join(CACHE_DIR, experiment_name)
    kernel_profile_dir = os.path.join(output_dir, "kernel_profile")
    os.makedirs(kernel_profile_dir, exist_ok=True)
    request_by_id = {request.workload_id: request for request in requests}
    kernel_profiles = {benchmark_id: KernelProfile() for benchmark_id in BENCHMARK_IDS}
    workload_profile = WorkloadProfile()
    measured_counts = {benchmark_id: 0 for benchmark_id in BENCHMARK_IDS}
    for workload in runtime.workloads:
        request = request_by_id[workload.workload_id]
        if workload.arrival_cycle < warmup_cycles:
            continue
        if workload.state != MeshDeviceRuntimeWorkloadState.COMPLETED or workload.start_cycle is None or workload.completion_cycle is None:
            raise RuntimeError(f"Measured workload '{workload.workload_id}' did not complete.")
        measured_counts[request.benchmark_id] += 1
        workload_profile.add(workload.workload_id, request.benchmark_id, workload.arrival_cycle, workload.start_cycle, workload.completion_cycle, request.slo)
        for kernel in workload.kernel_log:
            if kernel.start_cycle is None or kernel.completion_cycle is None or kernel.placement is None:
                raise RuntimeError(f"Kernel '{kernel.compiled_kernel.kernel_desc.name}' of workload '{workload.workload_id}' has incomplete profile data.")
            kernel_profiles[request.benchmark_id].add(kernel.compiled_kernel.kernel_desc, kernel.completion_cycle - kernel.start_cycle, int(kernel.placement.core_mesh.size))
    workload_profile.entries.sort(key=lambda entry: (entry.arrival_cycle, entry.workload_id))
    for benchmark_id, profile in kernel_profiles.items():
        with open(os.path.join(kernel_profile_dir, f"kernel_profile_{benchmark_id}.csv"), "w") as file:
            file.write(profile.to_csv())
    with open(os.path.join(output_dir, "workload_profile.csv"), "w") as file:
        file.write(workload_profile.to_csv())
    save_decision_profile(runtime, os.path.join(output_dir, "scheduler_profile.csv"))
    deadline_misses = sum(entry.completion_cycle - entry.arrival_cycle > entry.slo for entry in workload_profile.entries)
    metadata = {"experiment": experiment_name, "warmup_cycles": warmup_cycles, "submitted_workloads": len(requests), "measured_workloads": sum(measured_counts.values()), "measured_by_benchmark": measured_counts, "completed_jobs": completed_jobs, "deadline_misses": deadline_misses, "simulation_completion_cycle": simulation_completion_cycle, "wall_time_seconds": elapsed_seconds}
    with open(os.path.join(output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def run_experiment(experiment_name: str, compiler_type: type, runtime_builder, duration_cycles: int = DEFAULT_DURATION_CYCLES, warmup_cycles: int = DEFAULT_WARMUP_CYCLES, ccg_tops: float = DEFAULT_CCG_TOPS, enable_debug_log: bool = False, domain_ids: dict[str, str] | None = None) -> tuple[int, int, float, str]:
    if duration_cycles <= 0:
        raise ValueError("duration_cycles must be positive.")
    if warmup_cycles < 0 or warmup_cycles >= duration_cycles:
        raise ValueError("warmup_cycles must be non-negative and smaller than duration_cycles.")
    if ccg_tops <= 0:
        raise ValueError("ccg_tops must be positive.")
    logger.set_print_options(log_level=LogLevel.DEBUG if enable_debug_log else LogLevel.INFO)
    device = create_device(ccg_tops)
    runtime = runtime_builder(device, enable_debug_log)
    requests = build_requests(duration_cycles)
    context = MeshDeviceRuntimeContext(device=device, default_tile_shape=(32, 32), default_dtype=torch.bfloat16, runtime=runtime)
    with context:
        submit_workloads(context, compiler_type, requests, domain_ids)
    start_time = time.perf_counter()
    completed_jobs = runtime.run()
    elapsed_seconds = time.perf_counter() - start_time
    if not all(workload.is_completed for workload in runtime.workloads):
        raise RuntimeError("Not all submitted workloads completed.")
    save_profiles(runtime, experiment_name, warmup_cycles, requests, elapsed_seconds, len(completed_jobs), device.timestamp)
    return len(completed_jobs), device.timestamp, elapsed_seconds, os.path.join(CACHE_DIR, experiment_name)
