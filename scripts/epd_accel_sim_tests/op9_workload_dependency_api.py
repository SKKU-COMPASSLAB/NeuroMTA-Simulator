import os
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
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=False)
    ) as context:
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="parent_a"):
            mesh_relu(mesh_tensor(32, 32))
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="parent_b"):
            mesh_relu(mesh_tensor(32, 32))
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="child", dependent_workload_ids=("parent_a", "parent_b")):
            mesh_relu(mesh_tensor(32, 32))

        runtime = context.runtime
        jobs = runtime.run()
        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("At least one dependent workload did not complete.")
        parent_completion_cycle = max(runtime.get_workload("parent_a").completion_cycle, runtime.get_workload("parent_b").completion_cycle)
        child_dispatch_cycle = runtime.get_workload("child").kernel_log[0].dispatch_cycle
        if child_dispatch_cycle < parent_completion_cycle:
            raise RuntimeError("The child workload started before all parent workloads completed.")
        logger.info(f"Parent completion cycle: {parent_completion_cycle}")
        logger.info(f"Child dispatch cycle: {child_dispatch_cycle}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")


if __name__ == "__main__":
    main()
