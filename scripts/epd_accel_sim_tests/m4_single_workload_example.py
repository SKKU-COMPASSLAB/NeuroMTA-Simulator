import torch

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.compiler import MeshDeviceCompiledWorkload
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_ELEMENTWISE, MESH_KERNEL_LINEAR
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


MNIST_IMAGE_SIZE = 28 * 28
MNIST_CLASS_COUNT = 10


def create_activation(shape: tuple[int, int]) -> MeshTensorDescriptor:
    tile_shape = (min(32, shape[0]), min(32, shape[1]))
    return MeshTensorDescriptor(shape=shape, tile_shape=tile_shape, dtype=torch.bfloat16)


def create_weight(shape: tuple[int, int]) -> MeshTensorDescriptor:
    tile_shape = (min(32, shape[0]), min(32, shape[1]))
    return MeshTensorDescriptor(shape=shape, tile_shape=tile_shape, dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)


def build_mnist_mlp(device_desc: MeshDeviceDescriptor, batch_size: int=32) -> MeshDeviceCompiledWorkload:
    layer_dims = (MNIST_IMAGE_SIZE, 256, 128, MNIST_CLASS_COUNT)
    compiler = SpatialCompiler()
    activation = create_activation((batch_size, layer_dims[0]))

    for layer_index, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        weight = create_weight((output_dim, input_dim))
        bias = create_weight((1, output_dim))
        linear_output = create_activation((batch_size, output_dim))
        compiler.add_kernel(MESH_KERNEL_LINEAR(ifm=activation, wgt=weight, bias=bias, ofm=linear_output))
        if layer_index == len(layer_dims) - 2:
            activation = linear_output
            continue
        relu_output = create_activation((batch_size, output_dim))
        compiler.add_kernel(MESH_KERNEL_ELEMENTWISE(ifms=linear_output, ofm=relu_output, ops_per_element=1))
        activation = relu_output

    return compiler.compile(device_desc)


def main():
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    device_desc = MeshDeviceDescriptor(device)
    workload = build_mnist_mlp(device_desc)

    runtime = SpatialRuntime(device_desc, workload)
    weight_placements = runtime.warmup(workload)
    jobs = runtime.run()
    invocation = runtime.workloads[0]

    if invocation.state != MeshDeviceRuntimeWorkloadState.COMPLETED:
        raise RuntimeError("The MNIST MLP workload did not complete.")

    print(f"MNIST MLP kernels: {len(workload.compiled_kernels)}")
    print(f"Resident weight tensors: {len(weight_placements)}")
    for decision_index, decision in enumerate(runtime.decision_log):
        kernel_id = decision["kernels"][0]
        print(f"decision={decision_index} cycle={decision['cycle']} kernel={kernel_id} cores={decision['core_meshes'][0]} dma={decision['memory_banks'][0]}")
    print(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights(workload)} resident weight tensors.")


if __name__ == "__main__":
    main()
