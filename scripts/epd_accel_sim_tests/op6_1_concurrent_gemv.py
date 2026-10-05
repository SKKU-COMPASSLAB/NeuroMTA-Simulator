import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_LINEAR
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)

    linear_512_ifm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)
    linear_512_wgt = MeshTensorDescriptor(shape=(4096, 4096), tile_shape=(32, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
    linear_512_ofm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)
    linear_1024_ifm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)
    linear_1024_wgt = MeshTensorDescriptor(shape=(4096, 4096), tile_shape=(32, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
    linear_1024_ofm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)

    compiler_512 = SpatialCompiler()
    compiler_512.add_kernel(MESH_KERNEL_LINEAR(ifm=linear_512_ifm, wgt=linear_512_wgt, ofm=linear_512_ofm))
    compiler_1024 = SpatialCompiler()
    compiler_1024.add_kernel(MESH_KERNEL_LINEAR(ifm=linear_1024_ifm, wgt=linear_1024_wgt, ofm=linear_1024_ofm))
    compile_start = time.perf_counter()
    workload_512 = compiler_512.compile()
    workload_1024 = compiler_1024.compile()
    compilation_time = time.perf_counter() - compile_start

    runtime = SpatialRuntime(device_desc)
    runtime.submit(workload_512, arrival_cycle=0, workload_id="linear_512x512")
    runtime.submit(workload_1024, arrival_cycle=0, workload_id="linear_512x1024")
    simulation_start = time.perf_counter()
    jobs = runtime.run()
    simulation_time = time.perf_counter() - simulation_start

    if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
        raise RuntimeError("At least one Linear workload did not complete.")
    concurrent_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]

    print(f"Compilation time: {compilation_time:.6f} s")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Concurrent decisions: {len(concurrent_decisions)}")
    for decision_index, decision in enumerate(runtime.decision_log):
        print(f"decision={decision_index} cycle={decision['cycle']} benefit={decision['benefit']:.3f}")
        for kernel_id, core_ids, dma_ids in zip(decision["kernels"], decision["core_meshes"], decision["memory_banks"]):
            print(f"  kernel={kernel_id} cores={core_ids} dma={dma_ids}")
    print(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
