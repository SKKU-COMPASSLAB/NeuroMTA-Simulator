import time

import numpy as np
import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.api import *
from neuromta.system.software.implementation.calm import CALMCompiler, CALMRuntime, CALMScheduler


def main():
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    
    manual_ccg_tile_mesh = device.get_ccg_tile_mesh()[:2, :].copy()
    
    with MeshDeviceRuntimeContext(
        device=device, 
        default_tile_shape=(32, 32), 
        default_dtype=torch.bfloat16, 
        runtime=CALMRuntime(device, scheduler=CALMScheduler(), enable_debug_log=False)
    ) as context:    
        with context.new_compiler_context(CALMCompiler(manual_ccg_tile_mesh=manual_ccg_tile_mesh), arrival_cycle=0, workload_id="relu"):
            mesh_relu(mesh_tensor(128, 128))
        
        runtime = context.runtime
        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if runtime.workloads[0].state != MeshDeviceRuntimeWorkloadState.COMPLETED:
            raise RuntimeError("The manually placed workload did not complete.")
        runtime_ccg_tile_mesh = runtime.decision_log[0]["core_meshes"][0]
        if not np.array_equal(runtime_ccg_tile_mesh, manual_ccg_tile_mesh):
            raise RuntimeError(f"Runtime placement {runtime_ccg_tile_mesh.tolist()} differs from manual placement {manual_ccg_tile_mesh.tolist()}.")

    print(f"Manual core grid: {manual_ccg_tile_mesh.shape}")
    print(f"Manual core IDs: \n{manual_ccg_tile_mesh}")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Completed {len(jobs)} kernel at cycle {device.timestamp}.")


if __name__ == "__main__":
    main()
