import time

import torch

from neuromta.system.hardware import *
from neuromta.system.software.implementation.sequential import SequentialCompiler, SequentialRuntime
from neuromta.system.software.api import *
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def main():
    config = MeshAcceleratorConfig.stacked_3d_dram_npu()
    device = MeshAccelerator(**config).initialize()
    
    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(128, 128),
        default_dtype=torch.bfloat16,
        runtime=SequentialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=False),
    ) as context:
        with context.new_compiler_context(SequentialCompiler(), arrival_cycle=0, workload_id="Linear.forward") as compiler:
            ifm = mesh_tensor(4096, 4096).as_intermediate()
            ofm = mesh_relu(ifm)

        runtime = context.runtime
        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

    if runtime.workloads[0].state != MeshDeviceRuntimeWorkloadState.COMPLETED:
        raise RuntimeError("The Elementwise workload did not complete.")

    decision = runtime.decision_log[0]
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Runtime core IDs: {decision['core_meshes'][0].flatten().tolist()}")
    print(f"Runtime DMA IDs: {decision['memory_banks'][0]}")
    print(f"Completed {len(jobs)} kernel at cycle {device.timestamp}.")


if __name__ == "__main__":
    main()
