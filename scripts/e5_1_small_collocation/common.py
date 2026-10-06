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
from neuromta.system.software.api import MeshDeviceRuntimeContext, MeshTensorDescriptor
from neuromta.system.software.nn.yolox_nano import YOLOXNano
from neuromta.system.software.utils.descriptor import MeshKernelDescriptor
from neuromta.system.software.utils.runtime import MeshDeviceRuntime, MeshDeviceRuntimeWorkloadState
from neuromta.system.software.utils.scheduler import MeshSchedulingDomain, MeshWorkloadSchedulingHint
from neuromta.system.software.implementation._common import MeshFRFCFSScheduler

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CAMERAS = ("front", "left", "right", "rear")
DETECTOR_SLO = 50_000_000
DEFAULT_CCG_TOPS = 0.5


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
    arrival_cycle: int
    start_cycle: int
    completion_cycle: int
    slo: int

    def to_csv_entry(self) -> str:
        response_time = self.completion_cycle - self.arrival_cycle
        execution_time = self.completion_cycle - self.start_cycle
        slack = self.slo - response_time
        return f"{self.workload_id},{self.arrival_cycle},{self.start_cycle},{self.completion_cycle},{self.slo},{response_time},{execution_time},{slack},{slack >= 0}"


@dataclass
class WorkloadProfile:
    entries: list[WorkloadProfileEntry] = field(default_factory=list)

    def add(self, workload_id: str, arrival_cycle: int, start_cycle: int, completion_cycle: int, slo: int) -> None:
        self.entries.append(WorkloadProfileEntry(workload_id, arrival_cycle, start_cycle, completion_cycle, slo))

    def to_csv(self) -> str:
        header = "workload_id,arrival_cycle,start_cycle,completion_cycle,slo,response_time,execution_time,slack,deadline_met"
        return header + "\n" + "\n".join(entry.to_csv_entry() for entry in self.entries)


@dataclass(frozen=True)
class WorkloadRequest:
    workload_id: str
    workload_class: str
    arrival_cycle: int
    slo: int
    priority: int
    weight: float
    max_wait_cycles: int


def create_device(ccg_tops: float) -> MeshAccelerator:
    config = MeshAcceleratorConfig.small(ccg_tops=ccg_tops)
    return MeshAccelerator(**config).initialize()


def create_scheduler() -> MeshFRFCFSScheduler:
    return MeshFRFCFSScheduler(starvation_cycles=10_000_000, candidate_window=16)


def create_virtual_domains(device: MeshAccelerator) -> tuple[MeshSchedulingDomain, ...]:
    core_mesh = device.get_ccg_tile_mesh()
    dma_ids = tuple(device.global_context.config.dma_tile_ids)
    if core_mesh.shape != (4, 2) or len(dma_ids) != 4:
        raise ValueError(f"The e5.1 virtual partition requires a 4x2 CCG mesh and four DMA tiles, got {core_mesh.shape} and {len(dma_ids)} DMA tiles.")
    return tuple(
        MeshSchedulingDomain(f"vision.{camera}", core_mesh[index:index + 1, :].copy(), (dma_ids[index],))
        for index, camera in enumerate(CAMERAS)
    )


def build_requests() -> list[WorkloadRequest]:
    return [WorkloadRequest(f"camera.det.{camera}.1", "camera.det", 0, DETECTOR_SLO, 2, 2.0, 15_000_000) for camera in CAMERAS]


def submit_workloads(context: MeshDeviceRuntimeContext, compiler_type: type, requests: list[WorkloadRequest], domain_ids: dict[str, str] | None = None) -> None:
    detector = YOLOXNano(image_size=416, num_classes=10, depth=0.33, width=0.25)
    for request in requests:
        domain_id = None if domain_ids is None else domain_ids[request.workload_id]
        scheduling_hint = MeshWorkloadSchedulingHint(priority=request.priority, weight=request.weight, max_wait_cycles=request.max_wait_cycles)
        with context.new_compiler_context(compiler_type(), arrival_cycle=request.arrival_cycle, workload_id=request.workload_id, scheduling_hint=scheduling_hint, domain_id=domain_id):
            input_tensor = MeshTensorDescriptor(detector.image_shape(), tile_shape=(1, 1) + tuple(context.default_tile_shape), dtype=context.default_dtype)
            outputs = detector.forward(input_tensor)
            expected_shapes = tuple((1, detector.image_size[0] // stride, detector.image_size[1] // stride, detector.num_classes + 5) for stride in (8, 16, 32))
            if tuple(output.shape for output in outputs) != expected_shapes:
                raise RuntimeError(f"Unexpected YOLOX-Nano output shapes: {tuple(output.shape for output in outputs)}")


def save_profiles(runtime: MeshDeviceRuntime, experiment_name: str, requests: list[WorkloadRequest], elapsed_seconds: float, completed_jobs: int, simulation_completion_cycle: int, ccg_tops: float) -> None:
    output_dir = os.path.join(CACHE_DIR, experiment_name)
    kernel_profile_dir = os.path.join(output_dir, "kernel_profile")
    os.makedirs(kernel_profile_dir, exist_ok=True)
    request_by_id = {request.workload_id: request for request in requests}
    kernel_profiles = {"camera.det": KernelProfile()}
    workload_profile = WorkloadProfile()
    measured_counts = {workload_class: 0 for workload_class in kernel_profiles}
    for workload in runtime.workloads:
        workload_id = workload.workload_id
        request = request_by_id[workload_id]
        if workload.state != MeshDeviceRuntimeWorkloadState.COMPLETED or workload.start_cycle is None or workload.completion_cycle is None:
            raise RuntimeError(f"Measured workload '{workload_id}' did not complete.")
        measured_counts[request.workload_class] += 1
        workload_profile.add(workload_id, workload.arrival_cycle, workload.start_cycle, workload.completion_cycle, request.slo)
        for kernel in workload.kernel_log:
            if kernel.start_cycle is None or kernel.completion_cycle is None or kernel.placement is None:
                raise RuntimeError(f"Kernel '{kernel.compiled_kernel.kernel_desc.name}' of workload '{workload_id}' has incomplete profile data.")
            kernel_profiles[request.workload_class].add(kernel.compiled_kernel.kernel_desc, kernel.completion_cycle - kernel.start_cycle, int(kernel.placement.core_mesh.size))
    workload_profile.entries.sort(key=lambda entry: (entry.arrival_cycle, entry.workload_id))
    for old_profile in ("lane.seg", "driver.monitor"):
        old_path = os.path.join(kernel_profile_dir, f"kernel_profile_{old_profile}.csv")
        if os.path.exists(old_path):
            os.remove(old_path)
    for workload_class, profile in kernel_profiles.items():
        with open(os.path.join(kernel_profile_dir, f"kernel_profile_{workload_class}.csv"), "w") as file:
            file.write(profile.to_csv())
    with open(os.path.join(output_dir, "workload_profile.csv"), "w") as file:
        file.write(workload_profile.to_csv())
    save_decision_profile(runtime, os.path.join(output_dir, "scheduler_profile.csv"))
    deadline_misses = sum(entry.completion_cycle - entry.arrival_cycle > entry.slo for entry in workload_profile.entries)
    metadata = {
        "experiment": experiment_name,
        "warmup_cycles": 0,
        "ccg_tops": ccg_tops,
        "submitted_workloads": len(requests),
        "measured_workloads": sum(measured_counts.values()),
        "measured_by_class": measured_counts,
        "completed_jobs": completed_jobs,
        "deadline_misses": deadline_misses,
        "simulation_completion_cycle": simulation_completion_cycle,
        "wall_time_seconds": elapsed_seconds,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def save_decision_profile(runtime: MeshDeviceRuntime, path: str) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("cycle", "scheduler", "kernel_ids", "core_meshes", "memory_bank_ids", "benefit"))
        for record in runtime.decision_log:
            kernel_ids = "|".join(record["kernels"])
            core_meshes = ";".join(f"{'x'.join(str(size) for size in mesh.shape)}:{'|'.join(str(core_id) for core_id in np.asarray(mesh).reshape(-1))}" for mesh in record["core_meshes"])
            memory_bank_ids = ";".join("|".join(str(bank_id) for bank_id in bank_ids) for bank_ids in record["memory_banks"])
            writer.writerow((record["cycle"], record["scheduler"], kernel_ids, core_meshes, memory_bank_ids, record["benefit"]))


def run_experiment(experiment_name: str, compiler_type: type, runtime_builder, ccg_tops: float = DEFAULT_CCG_TOPS, enable_debug_log: bool = False, domain_ids: dict[str, str] | None = None) -> tuple[int, int, float, str]:
    if ccg_tops <= 0:
        raise ValueError("ccg_tops must be positive.")
    logger.set_print_options(log_level=LogLevel.DEBUG if enable_debug_log else LogLevel.INFO)
    device = create_device(ccg_tops)
    runtime = runtime_builder(device, enable_debug_log)
    requests = build_requests()
    context = MeshDeviceRuntimeContext(device=device, default_tile_shape=(32, 32), default_dtype=torch.bfloat16, runtime=runtime)
    with context:
        submit_workloads(context, compiler_type, requests, domain_ids)
    total_kernels = sum(len(workload.compiled_workload.kernel_stats_map) for workload in runtime.workloads)
    start_time = time.perf_counter()

    def report_progress(completed: int, cycle: int) -> None:
        logger.info(f"{experiment_name:<20s}: {completed}/{total_kernels} kernels, cycle {cycle}, elapsed {time.perf_counter() - start_time:.1f} s")

    completed_jobs = runtime.run(collect_jobs=False, progress_callback=report_progress)
    elapsed_seconds = time.perf_counter() - start_time
    if not all(workload.is_completed for workload in runtime.workloads):
        raise RuntimeError("Not all submitted workloads completed.")
    save_profiles(runtime, experiment_name, requests, elapsed_seconds, completed_jobs, device.timestamp, ccg_tops)
    return completed_jobs, device.timestamp, elapsed_seconds, os.path.join(CACHE_DIR, experiment_name)
