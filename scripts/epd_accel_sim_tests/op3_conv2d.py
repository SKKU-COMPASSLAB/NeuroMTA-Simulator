import time

import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_CONV2D
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)

    ifm = MeshTensorDescriptor(shape=(1, 128, 128, 32), tile_shape=(32, 32), dtype=torch.bfloat16)
    wgt = MeshTensorDescriptor(shape=(3, 3, 64, 32), tile_shape=(32, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
    ofm = MeshTensorDescriptor(shape=(1, 128, 128, 64), tile_shape=(32, 32), dtype=torch.bfloat16)

    compiler = SpatialCompiler()
    compiler.add_kernel(MESH_KERNEL_CONV2D(ifm=ifm, wgt=wgt, ofm=ofm, padding=1))
    compile_start = time.perf_counter()
    workload = compiler.compile()
    compilation_time = time.perf_counter() - compile_start

    runtime = SpatialRuntime(device_desc)
    runtime.submit(workload, workload_id="conv2d")
    simulation_start = time.perf_counter()
    jobs = runtime.run()
    simulation_time = time.perf_counter() - simulation_start

    if runtime.workloads[0].state != MeshDeviceRuntimeWorkloadState.COMPLETED:
        raise RuntimeError("The Conv2D workload did not complete.")

    decision = runtime.decision_log[0]
    print(f"Compilation time: {compilation_time:.6f} s")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Runtime core grid: {decision['core_meshes'][0].shape}")
    print(f"Weight shape (HWKC): {wgt.shape}")
    print(f"Weight tile grid: {wgt.tile_grid_shape} ({wgt.get_n_tiles()} tiles)")
    print(f"Runtime core IDs: {decision['core_meshes'][0]}")
    print(f"Runtime DMA IDs: {decision['memory_banks'][0]}")
    print(f"Resident weight tensors: {sum(stats.tensor_desc.tensor_type == MeshTensorType.WEIGHT and stats.is_placed for stats in workload.tensor_stats_map.values())}")
    print(f"Completed {len(jobs)} kernel at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights(workload)} resident weight tensors.")


if __name__ == "__main__":
    main()
