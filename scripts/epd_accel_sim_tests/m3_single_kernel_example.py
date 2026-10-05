import torch

from neuromta.framework.debug_utils import *
from neuromta.framework.logger import *

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorConfig
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.utils.descriptor import MeshDeviceDescriptor, MeshTensorDescriptor, MeshTensorType
from neuromta.system.software.utils.kernel import MESH_KERNEL_LINEAR
from neuromta.system.software.utils.runtime import MeshDeviceRuntimeWorkloadState


def main():
    logger.set_print_options(LogLevel.DEBUG)
    
    with print_log_execution_time(desc="execution_time"):
        config = MeshAcceleratorConfig.medium()
        device = MeshAccelerator(**config).initialize()
        device_desc = MeshDeviceDescriptor(device)

        compiler = SpatialCompiler()

        ifm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)
        wgt = MeshTensorDescriptor(shape=(4096, 4096), tile_shape=(32, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
        bias = MeshTensorDescriptor(shape=(1, 4096), tile_shape=(1, 32), dtype=torch.bfloat16, tensor_type=MeshTensorType.WEIGHT)
        ofm = MeshTensorDescriptor(shape=(32, 4096), tile_shape=(32, 32), dtype=torch.bfloat16)

        linear = MESH_KERNEL_LINEAR(ifm=ifm, wgt=wgt, bias=bias, ofm=ofm)
        compiler.add_kernel(linear)
        workload = compiler.compile(device_desc)

        runtime = SpatialRuntime(device_desc, workload)
        weight_placements = runtime.warmup(workload)
        jobs = runtime.run()
        invocation = runtime.workloads[0]
        decision = runtime.decision_log[0]

        if invocation.state != MeshDeviceRuntimeWorkloadState.COMPLETED:
            raise RuntimeError("The Linear kernel did not complete.")

    print(f"Configured CCG mesh: {device_desc.ccg_tile_mesh.shape}")
    print(f"Compiled kernel: {workload.compiled_kernels[0].kernel_id}")
    print(f"Resident weight tensors: {tuple(weight_placements)}")
    print(f"Runtime core IDs: {decision['core_meshes'][0]}")
    print(f"Runtime DMA IDs: {decision['memory_banks'][0]}")
    print(f"Completed {len(jobs)} Linear kernel at cycle {device.timestamp}.")
    print(f"Deallocated {runtime.deallocate_weights(workload)} resident weight tensors.")


if __name__ == "__main__":
    main()
