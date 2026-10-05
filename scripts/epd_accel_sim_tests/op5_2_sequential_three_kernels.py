import time

import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_ELEMENTWISE, MESH_KERNEL_LINEAR, MESH_KERNEL_REDUCTION
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)

    linear_ifm = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16)
    linear_wgt = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
    linear_ofm = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16)
    elementwise_ifm = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16)
    elementwise_ofm = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16)
    reduction_ifm = MeshTensorDescriptor(shape=(512, 512), tile_shape=(32, 32), dtype=torch.bfloat16)
    reduction_ofm = MeshTensorDescriptor(shape=(512, 1), tile_shape=(32, 1), dtype=torch.bfloat16)

    compiler = SpatialCompiler()
    compiler.add_kernel(MESH_KERNEL_LINEAR(ifm=linear_ifm, wgt=linear_wgt, ofm=linear_ofm))
    compiler.add_kernel(MESH_KERNEL_ELEMENTWISE(ifms=elementwise_ifm, ofm=elementwise_ofm, ops_per_element=1))
    compiler.add_kernel(MESH_KERNEL_REDUCTION(ifm=reduction_ifm, ofm=reduction_ofm, ops_per_input_element=1))
    compile_start = time.perf_counter()
    workload = compiler.compile()
    compilation_time = time.perf_counter() - compile_start

    runtime = SpatialRuntime(device_desc)
    runtime.submit(workload, arrival_cycle=0, workload_id="linear")
    simulation_start = time.perf_counter()
    jobs = runtime.run()
    simulation_time = time.perf_counter() - simulation_start

    if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
        raise RuntimeError("At least one concurrent workload did not complete.")
    dispatch_groups = {}
    for decision in runtime.decision_log:
        dispatch_groups.setdefault(decision["cycle"], []).extend(decision["kernels"])
    concurrent_dispatch_groups = {cycle: kernel_ids for cycle, kernel_ids in dispatch_groups.items() if len(kernel_ids) > 1}

    print(f"Compilation time: {compilation_time:.6f} s")
    print(f"Simulation time: {simulation_time:.6f} s")
    print(f"Runtime co-location: {bool(concurrent_dispatch_groups)}")
    print(f"Concurrent dispatch groups: {len(concurrent_dispatch_groups)}")
    for cycle, kernel_ids in sorted(concurrent_dispatch_groups.items()):
        print(f"concurrent_cycle={cycle} kernels={tuple(kernel_ids)}")
    for decision_index, decision in enumerate(runtime.decision_log):
        print(f"decision={decision_index} cycle={decision['cycle']} benefit={decision['benefit']:.3f}")
        for kernel_id, core_ids, memory_bank_ids in zip(decision["kernels"], decision["core_meshes"], decision["memory_banks"]):
            print(f"  kernel={kernel_id} cores={core_ids} memory_banks={memory_bank_ids}")
    print(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
