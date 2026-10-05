import time

import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import MESH_KERNEL_REDUCTION
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)

    ifm = MeshTensorDescriptor(shape=(4096, 4096), tile_shape=(32, 32), dtype=torch.bfloat16).to_local_cache()
    ofm = MeshTensorDescriptor(shape=(4096, 1), tile_shape=(32, 1), dtype=torch.bfloat16).to_local_cache()

    compiler = SpatialCompiler()
    compiler.add_kernel(MESH_KERNEL_REDUCTION(ifm=ifm, ofm=ofm, ops_per_input_element=1))
    compile_start = time.perf_counter()
    workload = compiler.compile()
    compilation_time = time.perf_counter() - compile_start

    runtime = SpatialRuntime(device_desc)
    runtime.submit(workload, workload_id="reduction")
    simulation_start = time.perf_counter()
    jobs = runtime.run()
    simulation_time = time.perf_counter() - simulation_start

    if runtime.workloads[0].state != MeshDeviceRuntimeWorkloadState.COMPLETED:
        raise RuntimeError("The Reduction workload did not complete.")

    decision = runtime.decision_log[0]
    print(f"Compilation time: {compilation_time:.6f} s")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Runtime core grid: {decision['core_meshes'][0].shape}")
    print(f"Runtime core IDs: {decision['core_meshes'][0]}")
    print(f"Runtime DMA IDs: {decision['memory_banks'][0]}")
    print(f"Completed {len(jobs)} kernel at cycle {device.timestamp}.")


if __name__ == "__main__":
    main()
