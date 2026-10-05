import numpy as np
import enum
import copy
from abc import ABC

from neuromta.system.software.utils.descriptor import *


__all__ = [
    "MeshDeviceCompiledWorkload",
    "MeshDeviceCompiler",
    "MeshDeviceCompiledAction",
    "MeshDeviceActionType",
    "MeshDeviceCompiledTensorStats",
    "MeshDeviceCompiledKernelStats",
]


class MeshDeviceActionType(enum.Enum):
    PLACE_TENSOR    = "PLACE_TENSOR"
    RUN_KERNEL      = "RUN_KERNEL"


class MeshDeviceCompiledAction:
    def __init__(
        self,
        action_type: MeshDeviceActionType,
        **action_kwargs
    ):
        self.action_type = action_type
        self._action_kwargs = action_kwargs

    @classmethod
    def place_tensor(cls, tensor_id: str):
        return cls(
            action_type=MeshDeviceActionType.PLACE_TENSOR,
            tensor_id=tensor_id
        )

    @classmethod
    def run_kernel(cls, kernel_id: str, place_tensor_ids: list[str]=None, release_tensor_ids: list[str]=None):
        if place_tensor_ids is None:
            place_tensor_ids = []
        if release_tensor_ids is None:
            release_tensor_ids = []

        return cls(
            action_type=MeshDeviceActionType.RUN_KERNEL,
            kernel_id=kernel_id,
            place_tensor_ids=place_tensor_ids,
            release_tensor_ids=release_tensor_ids,
        )

    @property
    def tensor_id(self) -> str:
        if self.action_type == MeshDeviceActionType.PLACE_TENSOR:
            return self._action_kwargs.get("tensor_id", None)
        else:
            raise ValueError(f"Action type '{self.action_type}' does not have a tensor_id.")

    @property
    def kernel_id(self) -> str:
        if self.action_type == MeshDeviceActionType.RUN_KERNEL:
            return self._action_kwargs.get("kernel_id", None)
        else:
            raise ValueError(f"Action type '{self.action_type}' does not have a kernel_id.")

    @property
    def place_tensor_ids(self) -> list[str]:
        if self.action_type == MeshDeviceActionType.RUN_KERNEL:
            return self._action_kwargs.get("place_tensor_ids", [])
        else:
            raise ValueError(f"Action type '{self.action_type}' does not have place_tensor_ids.")

    @property
    def release_tensor_ids(self) -> list[str]:
        if self.action_type == MeshDeviceActionType.RUN_KERNEL:
            return self._action_kwargs.get("release_tensor_ids", [])
        else:
            raise ValueError(f"Action type '{self.action_type}' does not have release_tensor_ids.")

    def __repr__(self):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self._action_kwargs.items())
        return f"{self.action_type.value}({kwargs_str})"

class MeshDeviceCompiledTensorStats:
    def __init__(self, tensor_desc: MeshTensorDescriptor):
        self.tensor_desc = tensor_desc
        self._mem_type = None
        self._tile_placement: dict[tuple[int, ...], MeshMemoryBankDescriptor] = {}
        self._is_placed = False

    def place(self, tile_placement: dict[tuple[int, ...], MeshMemoryBankDescriptor]):
        grid_shape = self.tensor_desc.tile_grid_shape
        if any(len(coord) != len(grid_shape) or any(index < 0 or index >= size for index, size in zip(coord, grid_shape)) for coord in tile_placement):
            raise ValueError("Invalid tile coordinates in tile_placement.")

        self._tile_placement.update(tile_placement)
        self._mem_type = next(iter(tile_placement.values())).mem_type
        self._is_placed = True

        return self

    def unplace(self):
        self._tile_placement.clear()
        self._is_placed = False
        self._mem_type = None

        return self

    @property
    def tile_placement(self) -> dict[tuple[int, ...], MeshMemoryBankDescriptor]:
        return self._tile_placement

    @property
    def mem_type(self) -> MeshMemoryType:
        return self._mem_type

    @property
    def is_placed(self) -> bool:
        return self._is_placed

    def __repr__(self):
        return f"Tensor(shape={self.tensor_desc.shape}, dtype={self.tensor_desc.dtype})"


class MeshDeviceCompiledKernelStats:
    def __init__(self, kernel_desc: MeshKernelDescriptor):
        self.kernel_desc = kernel_desc
        self._ccg_tile_mesh: np.ndarray = None

    def place(self, ccg_tile_mesh: np.ndarray):
        self._ccg_tile_mesh = ccg_tile_mesh
        return self

    def unplace(self):
        self._ccg_tile_mesh = None

    @property
    def ccg_tile_mesh(self) -> np.ndarray:
        return self._ccg_tile_mesh

    @property
    def is_placed(self) -> bool:
        return self._ccg_tile_mesh is not None

    def __repr__(self):
        _placement_details = ", is_placed=False"
        if self.is_placed:
            _placement_details = f", n_ccg_tiles={self.ccg_tile_mesh.size}, ccg_tile_mesh_shape={self.ccg_tile_mesh.shape}"
        return f"Kernel(type={self.kernel_desc.kernel_type.name}{_placement_details})"


class MeshDeviceCompiledWorkload:
    def __init__(self):
        self._tensor_stats_map: dict[str, MeshDeviceCompiledTensorStats] = {}
        self._kernel_stats_map: dict[str, MeshDeviceCompiledKernelStats] = {}

        self._warmup_actions: list[MeshDeviceCompiledAction] = []
        self._main_actions: list[MeshDeviceCompiledAction] = []

        self._tensor_obj_to_id_map: dict[int, str] = {}

    def add_tensor_stats(self, tensor_id: str, tensor_desc: MeshTensorDescriptor):
        if tensor_id in self._tensor_stats_map:
            raise ValueError(f"Tensor ID '{tensor_id}' already exists in the workload.")

        self._tensor_stats_map[tensor_id] = MeshDeviceCompiledTensorStats(tensor_desc=tensor_desc)
        self._tensor_obj_to_id_map[id(tensor_desc)] = tensor_id

    def add_kernel_stats(self, kernel_id: str, kernel_desc: MeshKernelDescriptor):
        if kernel_id in self._kernel_stats_map:
            raise ValueError(f"Kernel ID '{kernel_id}' already exists in the workload.")

        self._kernel_stats_map[kernel_id] = MeshDeviceCompiledKernelStats(kernel_desc=kernel_desc)

    def get_tensor_stats(self, tensor_id: str | MeshTensorDescriptor) -> MeshDeviceCompiledTensorStats:
        if isinstance(tensor_id, MeshTensorDescriptor):
            tensor_id = self.get_tensor_id(tensor_id)
        if tensor_id not in self._tensor_stats_map:
            raise ValueError(f"Tensor ID '{tensor_id}' does not exist in the workload.")

        return self._tensor_stats_map[tensor_id]

    def get_kernel_stats(self, kernel_id: str) -> MeshDeviceCompiledKernelStats:
        if kernel_id not in self._kernel_stats_map:
            raise ValueError(f"Kernel ID '{kernel_id}' does not exist in the workload.")

        return self._kernel_stats_map[kernel_id]

    def get_tensor_id(self, tensor_ptr: int | MeshTensorDescriptor) -> str:
        if isinstance(tensor_ptr, MeshTensorDescriptor):
            tensor_ptr = id(tensor_ptr)
        tensor_id = self._tensor_obj_to_id_map.get(tensor_ptr, None)
        if tensor_id is None:
            raise ValueError("The provided MeshTensorDescriptor does not exist in the workload.")
        return tensor_id

    def is_tensor_desc_included(self, tensor_desc: MeshTensorDescriptor) -> bool:
        return id(tensor_desc) in self._tensor_obj_to_id_map

    @property
    def tensor_stats_map(self) -> dict[str, MeshDeviceCompiledTensorStats]:
        return self._tensor_stats_map

    @property
    def kernel_stats_map(self) -> dict[str, MeshDeviceCompiledKernelStats]:
        return self._kernel_stats_map

    @property
    def warmup_actions(self) -> list[MeshDeviceCompiledAction]:
        return self._warmup_actions

    @property
    def main_actions(self) -> list[MeshDeviceCompiledAction]:
        return self._main_actions


class MeshDeviceCompiler(ABC):
    """
    Introduction
    ------------
    TBD
    """

    def __init__(self, enable_fusion: bool = True, manual_ccg_tile_mesh: np.ndarray = None):
        if not isinstance(enable_fusion, bool):
            raise TypeError("enable_fusion must be a boolean.")
        if manual_ccg_tile_mesh is not None:
            manual_ccg_tile_mesh = np.asarray(manual_ccg_tile_mesh, dtype=int)
            if manual_ccg_tile_mesh.ndim != 2 or manual_ccg_tile_mesh.size == 0:
                raise ValueError("manual_ccg_tile_mesh must be a non-empty 2D array.")
            if len(set(manual_ccg_tile_mesh.flatten().tolist())) != manual_ccg_tile_mesh.size:
                raise ValueError("manual_ccg_tile_mesh must not contain duplicate CCG IDs.")

        self.enable_fusion = enable_fusion
        self.manual_ccg_tile_mesh = None if manual_ccg_tile_mesh is None else manual_ccg_tile_mesh.copy()

        self._kernels: list[MeshKernelDescriptor] = []

    ###########################################################################
    # Public Compiler Interface
    ###########################################################################

    def add_kernel(self, kernel_desc: MeshKernelDescriptor):
        if not isinstance(kernel_desc, MeshKernelDescriptor):
            raise TypeError("kernel_desc must be an instance of MeshKernelDescriptor.")
        self._kernels.append(kernel_desc)

    def compile(self) -> MeshDeviceCompiledWorkload:
        consumers = {id(tensor): 0 for kernel in self._kernels for tensor in kernel.input_tensors + kernel.output_tensors}
        for kernel in self._kernels:
            for tensor in kernel.input_tensors:
                consumers[id(tensor)] += 1
        kernels = []
        for original in self._kernels:
            kernel_desc = copy.copy(original)
            kernel_desc.input_tensors = list(original.input_tensors)
            kernel_desc.output_tensors = list(original.output_tensors)
            kernel_desc.kernel_kwargs = dict(original.kernel_kwargs)
            if self.enable_fusion and kernel_desc.kernel_type == MeshKernelType.ELEMENTWISE and kernel_desc.enable_fusion and kernels:
                previous = kernels[-1]
                internal = previous.output_tensors[0] if len(previous.output_tensors) == 1 else None
                output = kernel_desc.output_tensors[0] if len(kernel_desc.output_tensors) == 1 else None
                supported = previous.kernel_type in (MeshKernelType.LINEAR, MeshKernelType.REDUCTION) or previous.kernel_type == MeshKernelType.CONV2D and previous.get_kwargs("operation") == "conv2d" or previous.kernel_type == MeshKernelType.ELEMENTWISE and previous.enable_fusion
                tensors = previous.input_tensors + kernel_desc.input_tensors + ([internal, output] if internal is not None and output is not None else [])
                can_fuse = supported and internal is not None and output is not None and any(tensor is internal for tensor in kernel_desc.input_tensors) and consumers[id(internal)] == 1 and not any(tensor.is_view or tensor.storage_desc.is_persistent for tensor in tensors) and not internal.fusion_barrier and not output.fusion_barrier and internal.tensor_type == MeshTensorType.INTERMEDIATE and output.tensor_type == MeshTensorType.INTERMEDIATE and internal.reserved_shape == internal.shape and output.reserved_shape == output.shape and output not in previous.input_tensors + kernel_desc.input_tensors and (internal.shape, internal.tile_shape, internal.dtype) == (output.shape, output.tile_shape, output.dtype)
                if can_fuse:
                    fused_ops = kernel_desc.get_required_kwargs("ops_per_element")
                    if previous.kernel_type == MeshKernelType.ELEMENTWISE:
                        previous.set_kwargs(ops_per_element=previous.get_required_kwargs("ops_per_element") + fused_ops)
                    else:
                        previous.set_kwargs(extra_ops_per_output_element=previous.get_kwargs("extra_ops_per_output_element", default=0.0) + fused_ops)
                    previous.input_tensors.extend(tensor for tensor in kernel_desc.input_tensors if tensor is not internal and all(tensor is not existing for existing in previous.input_tensors))
                    previous.output_tensors[0] = output
                    previous.fused_operations += (original,)
                    continue
            kernels.append(kernel_desc)

        workload = MeshDeviceCompiledWorkload()
        kernel_cnt = 0
        tensor_cnt = 0
        tensor_obj_ref_cnt = {}

        # STEP 1: Compile all kernels and tensors, and register them in the workload
        for kernel_desc in kernels:
            # 1-1: register kernel descriptor
            new_kernel_id = f"kernel.{kernel_cnt}"
            kernel_cnt += 1
            workload.add_kernel_stats(new_kernel_id, kernel_desc)
            if self.manual_ccg_tile_mesh is not None:
                workload.get_kernel_stats(new_kernel_id).place(self.manual_ccg_tile_mesh.copy())

            # 1-2: register tensor descriptors (skip duplicate descriptors)
            for tensor_desc in kernel_desc.input_tensors + kernel_desc.output_tensors:
                if tensor_desc.is_view and not workload.is_tensor_desc_included(tensor_desc.storage_desc):
                    new_tensor_id = f"tensor.{tensor_cnt}"
                    tensor_cnt += 1
                    workload.add_tensor_stats(new_tensor_id, tensor_desc.storage_desc)
                if not workload.is_tensor_desc_included(tensor_desc):
                    new_tensor_id = f"tensor.{tensor_cnt}"
                    tensor_cnt += 1
                    workload.add_tensor_stats(new_tensor_id, tensor_desc)
        for kernel_desc in kernels:
            for tensor_desc in kernel_desc.input_tensors:
                tensor_obj_ref_cnt[tensor_desc.storage_id] = tensor_obj_ref_cnt.get(tensor_desc.storage_id, 0) + 1

        # STEP 2: Generate the action sequence for the workload
        _placed_tensors = set()

        # 2-1: WARMUP >> Place all weight tensors (usually, they are placed in the device memory before the workload execution)
        for tensor_id, tensor_stat in workload.tensor_stats_map.items():
            tensor_desc = tensor_stat.tensor_desc
            if tensor_desc.tensor_type == MeshTensorType.WEIGHT:
                workload.warmup_actions.append(MeshDeviceCompiledAction.place_tensor(tensor_id=tensor_id))
                _placed_tensors.add(tensor_id)

        # 2-2: MAIN >> Run all kernels in the order they were added
        for kernel_id, kernel_stat in workload.kernel_stats_map.items():
            kernel_desc = kernel_stat.kernel_desc
            kernel_place_tensor_ids = []
            kernel_release_tensor_ids = []

            # Place intermediate tensors if they haven't been placed yet
            for tensor_desc in kernel_desc.input_tensors + kernel_desc.output_tensors:
                storage_desc = tensor_desc.storage_desc
                tensor_id = workload.get_tensor_id(storage_desc)

                if tensor_id not in _placed_tensors:
                    # workload.main_actions.append(MeshDeviceCompiledAction.place_tensor(tensor_id=tensor_id))
                    kernel_place_tensor_ids.append(tensor_id)
                    _placed_tensors.add(tensor_id)

            for tensor_desc in kernel_desc.input_tensors:
                tensor_obj_ref_cnt[tensor_desc.storage_id] -= 1
            releasable = {tensor_desc.storage_desc for tensor_desc in kernel_desc.input_tensors + kernel_desc.output_tensors if tensor_obj_ref_cnt.get(tensor_desc.storage_id, 0) == 0}
            for tensor_desc in releasable:
                tensor_id = workload.get_tensor_id(tensor_desc)
                if tensor_desc.tensor_type != MeshTensorType.WEIGHT and not tensor_desc.is_persistent and tensor_id in _placed_tensors:
                    kernel_release_tensor_ids.append(tensor_id)
                    _placed_tensors.remove(tensor_id)

            # Run the kernel
            workload.main_actions.append(MeshDeviceCompiledAction.run_kernel(
                kernel_id=kernel_id,
                place_tensor_ids=kernel_place_tensor_ids,
                release_tensor_ids=kernel_release_tensor_ids
            ))

        return workload
