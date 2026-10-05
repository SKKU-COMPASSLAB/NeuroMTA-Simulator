import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.api import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def workload1(x: MeshTensorDescriptor):
    w1 = mesh_tensor(1024, 1024).as_weight()
    b1 = mesh_tensor(1, 1024).as_weight()
    w2 = mesh_tensor(256, 1024).as_weight()
    b2 = mesh_tensor(1, 256).as_weight()
    w3 = mesh_tensor(10, 256).as_weight()
    b3 = mesh_tensor(1, 10).as_weight()

    o1 = mesh_linear(x, w1, bias=b1)
    o2 = mesh_linear(o1, w2, bias=b2)
    y = mesh_linear(o2, w3, bias=b3)

    return y

def workload2(x: MeshTensorDescriptor):
    w1 = mesh_tensor(1024, 1024).as_weight()
    b1 = mesh_tensor(1, 1024).as_weight()
    w2 = mesh_tensor(256, 1024).as_weight()
    b2 = mesh_tensor(1, 256).as_weight()
    w3 = mesh_tensor(10, 256).as_weight()
    b3 = mesh_tensor(1, 10).as_weight()

    o1 = mesh_linear(x, w1, bias=b1)
    o2 = mesh_linear(o1, w2, bias=b2)
    y = mesh_linear(o2, w3, bias=b3)

    return y

def main():
    logger.set_print_options(log_level=LogLevel.DEBUG)

    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()

    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=True),
    ) as context:
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="workload1") as compiler:
            x = mesh_tensor(32, 1024)
            y = workload1(x)
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="workload2") as compiler:
            x = mesh_tensor(32, 1024)
            y = workload2(x)

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("At least one Linear workload did not complete.")
        concurrent_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]

        logger.info(f"Simulation time: {simulation_time:.6f} s")
        logger.info(f"Concurrent decisions: {len(concurrent_decisions)}")
        for decision_index, decision in enumerate(runtime.decision_log):
            logger.info(f"decision={decision_index} cycle={decision['cycle']} benefit={decision['benefit']:.3f}")
            for kernel_id, core_ids, dma_ids in zip(decision["kernels"], decision["core_meshes"], decision["memory_banks"]):
                logger.info(f"  kernel={kernel_id} cores={core_ids} dma={dma_ids}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
        logger.info(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
