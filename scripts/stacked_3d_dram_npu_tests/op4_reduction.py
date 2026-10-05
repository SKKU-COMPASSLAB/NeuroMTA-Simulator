import time

import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.sequential import SequentialCompiler, SequentialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor
from neuromta.system.software.utils.kernel import MESH_KERNEL_REDUCTION
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    config = MeshAcceleratorConfig.stacked_3d_dram_npu()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)

    ifm = MeshTensorDescriptor(shape=(4096, 4096), tile_shape=(128, 128), dtype=torch.bfloat16)
    ofm = MeshTensorDescriptor(shape=(4096, 1), tile_shape=(128, 1), dtype=torch.bfloat16)

    compiler = SequentialCompiler()
    compiler.add_kernel(MESH_KERNEL_REDUCTION(ifm=ifm, ofm=ofm, ops_per_input_element=1))
    compile_start = time.perf_counter()
    workload = compiler.compile(device_desc)
    compilation_time = time.perf_counter() - compile_start

    runtime = SequentialRuntime(device_desc, workload)
    simulation_start = time.perf_counter()
    jobs = runtime.run()
    simulation_time = time.perf_counter() - simulation_start

    if runtime.workloads[0].state != MeshDeviceRuntimeWorkloadState.COMPLETED:
        raise RuntimeError("The Reduction workload did not complete.")

    compiled_kernel = workload.compiled_kernels[0]
    decision = runtime.decision_log[0]
    print(f"Compilation time: {compilation_time:.6f} s")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Compiled core grid: {compiled_kernel.core_grid_shape}")
    print(f"Runtime core IDs: {decision['core_meshes'][0]}")
    print(f"Runtime DMA IDs: {decision['memory_banks'][0]}")
    print(f"Completed {len(jobs)} kernel at cycle {device.timestamp}.")


if __name__ == "__main__":
    main()
