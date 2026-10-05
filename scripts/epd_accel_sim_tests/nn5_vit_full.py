import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.nn.vit import *
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def main():
    logger.set_print_options(log_level=LogLevel.INFO)

    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()

    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=True),
    ) as context:
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="ViT.forward") as compiler:
            model = ViT()
            x = MeshTensorDescriptor(shape=model.image_shape(), tile_shape=(1, 1, 32, 32), dtype=torch.bfloat16)
            y = model.forward(x)
            if y.shape != (1, 1000):
                raise RuntimeError(f"Unexpected ViT output shape: {y.shape}")

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("The ViT workload did not complete.")
        concurrent_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]
        kernel_stats = {}
        for workload in runtime.workloads:
            for kernel in workload.kernel_log:
                kernel_type = kernel.compiled_kernel.kernel_desc.kernel_type.value
                kernel_cycles = kernel.completion_cycle - kernel.dispatch_cycle
                count, cycles = kernel_stats.get(kernel_type, (0, 0))
                kernel_stats[kernel_type] = count + 1, cycles + kernel_cycles
        slowest_kernels = sorted((kernel for workload in runtime.workloads for kernel in workload.kernel_log), key=lambda kernel: kernel.completion_cycle - kernel.dispatch_cycle, reverse=True)[:5]

        logger.info(f"Model config: image={model.image_size}, patch={model.patch_size}, layers={model.num_layers}, heads={model.num_heads}, hidden={model.hidden_dim}, mlp={model.mlp_dim}, classes={model.num_classes}")
        logger.info(f"Input shape: {x.shape}")
        logger.info(f"Output shape: {y.shape}")
        logger.info(f"Simulation time: {simulation_time:.6f} s")
        logger.info(f"Simulated cycles: {device.timestamp}")
        logger.info(f"Concurrent decisions: {len(concurrent_decisions)}")
        for kernel_type, (count, cycles) in sorted(kernel_stats.items()):
            logger.info(f"kernel_type={kernel_type} count={count} cycles={cycles}")
        for kernel in slowest_kernels:
            logger.info(f"slowest_kernel={kernel.execution_id} cycles={kernel.completion_cycle - kernel.dispatch_cycle} cores={tuple(kernel.placement.core_mesh.flatten().tolist())} active_dma={kernel.placement.memory_bank_ids}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
        logger.info(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
