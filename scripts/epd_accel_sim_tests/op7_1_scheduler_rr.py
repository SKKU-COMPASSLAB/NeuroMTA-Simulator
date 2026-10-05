import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.api import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def main():
    logger.set_print_options(log_level=LogLevel.DEBUG)

    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()

    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=False),
    ) as context:
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="dma_blocker"):
            x = mesh_tensor(1, 2048)
            weight = mesh_parameter(1408, 2048)
            y = mesh_linear(x, weight)
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="cache_stream"):
            x = mesh_tensor(32, 256)
            for _ in range(48):
                x = mesh_relu(x)
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=1, workload_id="dma_waiter"):
            x = mesh_tensor(1, 2048)
            weight = mesh_parameter(1024, 2048)
            y = mesh_linear(x, weight)
        for index in range(5):
            with context.new_compiler_context(SpatialCompiler(), arrival_cycle=1, workload_id=f"cache_batch_{index:02d}"):
                x = mesh_tensor(32, 2048)
                y = mesh_relu(x)

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("At least one Linear workload did not complete.")
        concurrent_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]

        logger.info(f"Simulation time: {simulation_time:.6f} s")
        logger.info(f"Scheduler: {runtime.kernel_materializer.scheduler.name}")
        for workload in runtime.workloads:
            logger.info(f"workload={workload.workload_id} arrival={workload.arrival_cycle} completion={workload.completion_cycle} latency={workload.completion_cycle - workload.arrival_cycle}")
        logger.info(f"Concurrent decisions: {len(concurrent_decisions)}")
        for decision_index, decision in enumerate(runtime.decision_log):
            logger.info(f"decision={decision_index} cycle={decision['cycle']} benefit={decision['benefit']:.3f}")
            for kernel_id, core_ids, dma_ids in zip(decision["kernels"], decision["core_meshes"], decision["memory_banks"]):
                logger.info(f"  kernel={kernel_id} cores={core_ids} dma={dma_ids}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
        logger.info(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
