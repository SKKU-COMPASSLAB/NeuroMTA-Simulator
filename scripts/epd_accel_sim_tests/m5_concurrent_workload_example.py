import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.compiler import MeshDeviceCompiledWorkload
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_ELEMENTWISE, MESH_KERNEL_LINEAR
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def create_tensor(shape: tuple[int, int], tensor_type: MeshTensorType=MeshTensorType.INTERMEDIATE) -> MeshTensorDescriptor:
    tile_shape = (min(32, shape[0]), min(32, shape[1]))
    return MeshTensorDescriptor(shape=shape, tile_shape=tile_shape, dtype=torch.bfloat16, tensor_type=tensor_type)


def build_mlp(device_desc: MeshDeviceDescriptor, batch_size: int, layer_dims: tuple[int, ...]) -> MeshDeviceCompiledWorkload:
    if len(layer_dims) < 2:
        raise ValueError("An MLP requires at least two layer dimensions.")

    compiler = SpatialCompiler()
    activation = create_tensor((batch_size, layer_dims[0]))
    for layer_index, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        weight = create_tensor((output_dim, input_dim), MeshTensorType.WEIGHT)
        bias = create_tensor((1, output_dim), MeshTensorType.WEIGHT)
        linear_output = create_tensor((batch_size, output_dim))
        compiler.add_kernel(MESH_KERNEL_LINEAR(ifm=activation, wgt=weight, bias=bias, ofm=linear_output))
        if layer_index == len(layer_dims) - 2:
            activation = linear_output
            continue
        activated_output = create_tensor((batch_size, output_dim))
        compiler.add_kernel(MESH_KERNEL_ELEMENTWISE(ifms=linear_output, ofm=activated_output, ops_per_element=1))
        activation = activated_output
    return compiler.compile(device_desc)


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)
    digit_classifier = build_mlp(device_desc, batch_size=32, layer_dims=(784, 256, 128, 10))
    auxiliary_classifier = build_mlp(device_desc, batch_size=16, layer_dims=(784, 192, 64, 10))

    runtime = SpatialRuntime(device_desc)
    runtime.submit(digit_classifier, arrival_cycle=0, workload_id="digit_classifier")
    runtime.submit(auxiliary_classifier, arrival_cycle=0, workload_id="auxiliary_classifier")
    runtime.warmup(digit_classifier)
    runtime.warmup(auxiliary_classifier)
    jobs = runtime.run()

    if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
        raise RuntimeError("At least one concurrent MLP workload did not complete.")
    parallel_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]
    if not parallel_decisions:
        raise RuntimeError("The runtime did not select a concurrent kernel placement.")

    for decision_index, decision in enumerate(runtime.decision_log):
        print(f"decision={decision_index} cycle={decision['cycle']} benefit={decision['benefit']:.3f}")
        for kernel_id, core_ids, dma_ids in zip(decision["kernels"], decision["core_meshes"], decision["memory_banks"]):
            print(f"  kernel={kernel_id} cores={core_ids} dma={dma_ids}")
    for workload in runtime.workloads:
        print(f"workload={workload.workload_id} state={workload.state.value} completion_cycle={workload.completion_cycle}")
    print(f"Concurrent scheduling decisions: {len(parallel_decisions)}")
    print(f"Completed {len(jobs)} kernels from two MLP workloads at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")


if __name__ == "__main__":
    main()
