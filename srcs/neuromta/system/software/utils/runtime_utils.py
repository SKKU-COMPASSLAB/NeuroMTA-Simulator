from dataclasses import dataclass, replace

import numpy as np
import numba

from neuromta.system.software.utils.kernel import MeshKernelType
from neuromta.system.software.utils.descriptor import MeshTensorDescriptor, MeshKernelDescriptor, MeshMemoryType


__all__ = [
    "_Mapping",
    "create_mapping",
    "convert_2d_grid_to_nd",
    "convert_nd_grid_to_2d",
    "create_input_output_grids",
    "create_linear_mapping_requistes",
    "create_elementwise_mapping_requistes",
    "create_conv2d_mapping_requistes",
    "create_reduction_mapping_requistes",
    "create_sdpa_mapping_requistes",
    "TileRef",
    "CoreTask",
    "TileLocation",
    "LocatedTask",
    "BufferRegion",
    "ScratchPlan",
    "ResidencyPlan",
    "OutputGroup",
    "PlannedTransfer",
    "GroupTransfers",
    "StagePlan",
    "collect_core_tasks",
    "resolve_tile_locations",
    "plan_scratch_regions",
    "plan_full_residency",
    "group_output_tasks",
    "plan_group_transfers",
    "schedule_core_stages",
    "validate_materialization_plan",
]


class _Mapping:
    def __init__(
        self,
        core_mesh: np.ndarray,          # shape: [mesh height, mesh width] -> each element refers to a core ID
        output_grid: np.ndarray,        # shape: [output tensor height, output tensor width, *output tile coords]
        input_grids: dict[np.ndarray],  # key: input tensor name, value: shape: [input tensor height, input tensor width, *input tile coords]
        ops: dict[tuple[int, ...], dict[list[tuple[int, ...]]]], # key: output tile coords, value: dict with keys 'input_tensor_id' and 'input_tile_coords

        output_mapping: np.ndarray,
    ):
        """
        Output Mapping:
            * shape: [mesh height, mesh width, n output tiles per core, *output tile coords]
        """

        self.core_mesh   = core_mesh
        self.output_grid = output_grid
        self.input_grids = input_grids
        self.ops = ops
        self.output_mapping = output_mapping
        
        self._core_id_to_coords = {core_id: tuple(coords) for coords, core_id in np.ndenumerate(core_mesh)}
    
    def get_core_coords(self, core_id: int) -> tuple[int, int]:
        return self._core_id_to_coords[core_id]
    
    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self.core_mesh.shape


def create_mapping(
    core_mesh: np.ndarray,          # shape: [mesh height, mesh width] -> each element refers to a core ID
    output_grid: np.ndarray,        # shape: [output tensor height, output tensor width, *output tile coords]
    input_grids: dict[np.ndarray],  # key: input tensor name, value: shape: [input tensor height, input tensor width, *input tile coords]
    ops: dict[tuple[int, ...], dict[str, list[tuple[int, ...]]]],  # key: output tile coords, value: dict with keys 'input_tensor_id' and 'input_tile_coords'
) -> _Mapping:
    """
    Create a mapping from output tiles to input tiles based on the provided operations and mesh shape.

    Args:
        core_mesh (np.ndarray): A two-dimensional array of core IDs.
        output_grid (np.ndarray): The output grid with shape [output tensor height, output tensor width, *output tile coords].
        input_grids (dict[np.ndarray]): A dictionary where keys are input tensor names and values are input grids with shape [input tensor height, input tensor width, *input tile coords].
        ops :
            * key: output tile coords
            * value: dict with keys 'input_tensor_id' and 'input_tile_coords'

    Returns:
        mapping information
    """

    # Mapping Creation Logic
    #   - This function creates a mapping from output/input tiles to mesh cores based on the provided operations and mesh shape.
    #   - One of the major challenges is ensuring that the mapping distributes the workload evenly across the mesh.
    #   - The mapping assumes that the output tile computations are independent with each other.
    #
    # Output Tile Mapping Algorithm
    #   - The shape of the output grid may not match the shape of the mesh.
    #   - The algorithm needs to handle the case where the output grid is larger or smaller than the mesh.
    #   - The algorithm folds the output grid to fit the mesh if the output grid is larger than the mesh.
    #   - The algorithm splits the output grid into multiple sub-grids and co-locate them on the mesh if the output grid is smaller than the mesh or underutilization occurs.
    #
    # Optimization
    #   - In order to reduce the mapping creation overhead, the algorithm should be optimized with the `numba` library.

    output_grid = convert_nd_grid_to_2d(output_grid)
    input_grids = {name: convert_nd_grid_to_2d(grid) for name, grid in input_grids.items()}

    core_mesh = np.asarray(core_mesh)
    if core_mesh.ndim != 2 or any(size <= 0 for size in core_mesh.shape):
        raise ValueError("core_mesh must be a nonempty two-dimensional array.")
    mesh_height, mesh_width = core_mesh.shape
    output_grid = np.asarray(output_grid)
    if output_grid.ndim < 3 or not np.issubdtype(output_grid.dtype, np.integer):
        raise ValueError("output_grid must contain integer tile coordinates in its trailing dimensions.")

    # Flatten the tile coordinates so they can be matched with the keys in ops.
    output_height, output_width = output_grid.shape[:2]
    output_coord_size = int(np.prod(output_grid.shape[2:]))
    output_coords = output_grid.reshape(output_height * output_width, output_coord_size)
    output_keys = [tuple(int(value) for value in coord) for coord in output_coords]
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("output_grid contains duplicate tile coordinates.")
    if set(ops) - set(output_keys):
        raise ValueError("ops contains output tile coordinates absent from output_grid.")

    # Check that every dependency in ops names a tile present in its input grid.
    input_grids = {name: np.asarray(grid) for name, grid in input_grids.items()}
    input_coords = {}
    for name, grid in input_grids.items():
        if grid.ndim < 3 or not np.issubdtype(grid.dtype, np.integer):
            raise ValueError(f"Input grid '{name}' must contain integer tile coordinates in its trailing dimensions.")
        coords = {tuple(int(value) for value in coord) for coord in grid.reshape(grid.shape[0] * grid.shape[1], int(np.prod(grid.shape[2:])))}
        if len(coords) != grid.shape[0] * grid.shape[1]:
            raise ValueError(f"Input grid '{name}' contains duplicate tile coordinates.")
        input_coords[name] = coords
    for output_key, inputs in ops.items():
        if set(inputs) - set(input_grids):
            raise ValueError(f"Operation for output tile {output_key} refers to an unknown input grid.")
        for name, coords in inputs.items():
            if any(tuple(coord) not in input_coords[name] for coord in coords):
                raise ValueError(f"Operation for output tile {output_key} refers to a tile absent from input grid '{name}'.")

    # Keep one compiled assignment function for large grids; small grids avoid JIT startup cost.
    assign = getattr(create_mapping, "_assign", None)
    if assign is None:
        @numba.njit
        def assign(height, width, mesh_h, mesh_w):
            tile_count = height * width
            core_count = mesh_h * mesh_w
            owners = np.empty(tile_count, dtype=np.int64)
            slots = np.empty(tile_count, dtype=np.int64)
            counts = np.zeros(core_count, dtype=np.int64)
            base, remainder = divmod(tile_count, core_count)
            for index in range(tile_count):
                # Fold the output row and column onto the mesh, then find room if that core is full.
                preferred = (index // width % mesh_h) * mesh_w + index % width % mesh_w
                core = preferred
                for offset in range(core_count):
                    candidate = (preferred + offset) % core_count
                    if counts[candidate] < base + (candidate < remainder):
                        core = candidate
                        break
                owners[index] = core
                slots[index] = counts[core]
                counts[core] += 1
            return owners, slots
        create_mapping._assign = assign
    tile_count = len(output_keys)
    core_count = mesh_height * mesh_width
    owners, slots = assign(output_height, output_width, mesh_height, mesh_width) if tile_count >= 1024 else assign.py_func(output_height, output_width, mesh_height, mesh_width)

    # Use -1 for unused slots when cores receive different numbers of output tiles.
    tiles_per_core = (tile_count + core_count - 1) // core_count
    output_mapping = np.full((mesh_height, mesh_width, tiles_per_core, *output_grid.shape[2:]), -1, dtype=np.int64)
    output_mapping.reshape(core_count, tiles_per_core, output_coord_size)[owners, slots] = output_coords

    return _Mapping(core_mesh, output_grid, input_grids, ops, output_mapping)

def convert_nd_grid_to_2d(grid: np.ndarray) -> np.ndarray:
    return grid.reshape(-1, grid.shape[-2], grid.shape[-1])

def convert_2d_grid_to_nd(grid: np.ndarray, nd_shape: tuple[int, ...]) -> np.ndarray:
    return grid.reshape(*nd_shape, *grid.shape[-2:])

def create_input_output_grids(kernel_desc: MeshKernelDescriptor) -> tuple[np.ndarray, dict[np.ndarray]]:
    """
    Create the output grid and input grids for a given kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid and a dictionary of input grids.
    """
    output_grid = np.moveaxis(np.indices(kernel_desc.output_tensors[0].tile_grid_shape), 0, -1)
    input_grids = {
        i: np.moveaxis(np.indices(tensor.tile_grid_shape), 0, -1)
        for i, tensor in enumerate(kernel_desc.input_tensors)
    }
    return output_grid, input_grids


def create_linear_mapping_requistes(kernel_desc: MeshKernelDescriptor):
    """
    Create the output grid, input grids, and operations for a linear kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid, a dictionary of input grids, and a dictionary of operations.
    """

    if kernel_desc.kernel_type != MeshKernelType.LINEAR:
        raise ValueError("Kernel descriptor must be of type LINEAR.")

    ifm, wgt = kernel_desc.input_tensors[:2]
    ofm = kernel_desc.output_tensors[0]
    bias = kernel_desc.input_tensors[2] if len(kernel_desc.input_tensors) > 2 else None
    transpose_ifm = kernel_desc.get_required_kwargs("transpose_ifm")
    transpose_wgt = kernel_desc.get_required_kwargs("transpose_wgt")
    output_grid, input_grids = create_input_output_grids(kernel_desc)

    def batch_coord(tensor: MeshTensorDescriptor, output_batch: tuple[int, ...], matrix_rank: int) -> tuple[int, ...]:
        batch_shape = tensor.tile_grid_shape[:-matrix_rank]
        if len(batch_shape) > len(output_batch):
            raise ValueError("Input batch rank exceeds output batch rank.")
        aligned_output = output_batch[-len(batch_shape):] if batch_shape else ()
        aligned_shape = ofm.tile_grid_shape[:-matrix_rank][-len(batch_shape):] if batch_shape else ()
        if any(size not in (1, output_size) for size, output_size in zip(batch_shape, aligned_shape)):
            raise ValueError("Input batch tile grid is incompatible with output batch tile grid.")
        return tuple(0 if size == 1 else index for size, index in zip(batch_shape, aligned_output))

    ops = {}
    for output_coord in np.ndindex(*ofm.tile_grid_shape):
        output_batch, row, column = output_coord[:-2], output_coord[-2], output_coord[-1]
        row_start, column_start = row * ofm.tile_shape[-2], column * ofm.tile_shape[-1]
        row_stop = min(row_start + ofm.tile_shape[-2], ofm.shape[-2])
        column_stop = min(column_start + ofm.tile_shape[-1], ofm.shape[-1])
        ifm_batch = batch_coord(ifm, output_batch, 2)
        wgt_batch = batch_coord(wgt, output_batch, 2)
        ifm_output_axis = -1 if transpose_ifm else -2
        wgt_output_axis = -1 if transpose_wgt else -2
        ifm_rows = range(row_start // ifm.tile_shape[ifm_output_axis], (row_stop - 1) // ifm.tile_shape[ifm_output_axis] + 1)
        wgt_columns = range(column_start // wgt.tile_shape[wgt_output_axis], (column_stop - 1) // wgt.tile_shape[wgt_output_axis] + 1)
        ifm_reductions = range(ifm.tile_grid_shape[-2 if transpose_ifm else -1])
        wgt_reductions = range(wgt.tile_grid_shape[-2 if transpose_wgt else -1])
        ifm_tiles = [ifm_batch + ((reduction, ifm_row) if transpose_ifm else (ifm_row, reduction)) for ifm_row in ifm_rows for reduction in ifm_reductions]
        wgt_tiles = [wgt_batch + ((reduction, wgt_column) if transpose_wgt else (wgt_column, reduction)) for wgt_column in wgt_columns for reduction in wgt_reductions]
        inputs = {0: ifm_tiles, 1: wgt_tiles}
        if bias is not None:
            bias_batch = batch_coord(bias, output_coord[:-1], 1)
            bias_columns = range(column_start // bias.tile_shape[-1], (column_stop - 1) // bias.tile_shape[-1] + 1)
            inputs[2] = [bias_batch + (bias_column,) for bias_column in bias_columns]
        ops[output_coord] = inputs
    return output_grid, input_grids, ops

def create_elementwise_mapping_requistes(kernel_desc: MeshKernelDescriptor):
    """
    Create the output grid, input grids, and operations for an elementwise kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid, a dictionary of input grids, and a dictionary of operations.
    """

    if kernel_desc.kernel_type not in (MeshKernelType.ELEMENTWISE, MeshKernelType.MEMCOPY):
        raise ValueError("Kernel descriptor must be of type ELEMENTWISE or MEMCOPY.")

    if len(kernel_desc.output_tensors) != 1:
        raise ValueError("Elementwise and mapped MemCopy kernels require one output tensor.")
    output_grid, input_grids = create_input_output_grids(kernel_desc)
    output_shape = kernel_desc.output_tensors[0].tile_grid_shape
    ops = {}
    for output_coord in np.ndindex(*output_shape):
        inputs = {}
        for tensor_id, tensor in enumerate(kernel_desc.input_tensors):
            input_shape = tensor.tile_grid_shape
            if len(input_shape) > len(output_shape) or any(dim not in (1, output_shape[len(output_shape) - len(input_shape) + index]) for index, dim in enumerate(input_shape)):
                raise ValueError("Elementwise input tile grid cannot broadcast to the output tile grid.")
            inputs[tensor_id] = [tuple(0 if dim == 1 else output_coord[len(output_shape) - len(input_shape) + index] for index, dim in enumerate(input_shape))]
        ops[output_coord] = inputs
    return output_grid, input_grids, ops

def create_conv2d_mapping_requistes(kernel_desc: MeshKernelDescriptor):
    """
    Create the output grid, input grids, and operations for a conv2d kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid, a dictionary of input grids, and a dictionary of operations.
    """

    if kernel_desc.kernel_type != MeshKernelType.CONV2D:
        raise ValueError("Kernel descriptor must be of type CONV2D.")

    ofm = kernel_desc.output_tensors[0]
    ifm = kernel_desc.input_tensors[0]
    is_pooling = kernel_desc.get_required_kwargs("operation") != "conv2d"
    wgt = None if is_pooling else kernel_desc.input_tensors[1]
    bias = kernel_desc.input_tensors[2] if not is_pooling and len(kernel_desc.input_tensors) > 2 else None
    output_grid, input_grids = create_input_output_grids(kernel_desc)

    filter_positions = list(np.ndindex(*kernel_desc.get_required_kwargs("kernel_size")))
    stride_height, stride_width = kernel_desc.get_required_kwargs("stride")
    padding_top, _, padding_left, _ = kernel_desc.get_required_kwargs("padding")
    dilation_height, dilation_width = kernel_desc.get_required_kwargs("dilation")
    groups = kernel_desc.get_required_kwargs("groups") if wgt is not None else 1
    input_channels_per_group = ifm.shape[3] // groups
    output_channels_per_group = ofm.shape[3] // groups
    ops = {}
    for batch, row, column, output_channel in np.ndindex(*ofm.tile_grid_shape):
        row_start, column_start = row * ofm.tile_shape[1], column * ofm.tile_shape[2]
        row_stop = min(row_start + ofm.tile_shape[1], ofm.shape[1])
        column_stop = min(column_start + ofm.tile_shape[2], ofm.shape[2])
        channel_start = output_channel * ofm.tile_shape[3]
        channel_stop = min(channel_start + ofm.tile_shape[3], ofm.shape[3])
        spatial_tiles = set()
        for output_row in range(row_start, row_stop):
            for output_column in range(column_start, column_stop):
                for kernel_row, kernel_column in filter_positions:
                    input_row = output_row * stride_height + kernel_row * dilation_height - padding_top
                    input_column = output_column * stride_width + kernel_column * dilation_width - padding_left
                    if 0 <= input_row < ifm.shape[1] and 0 <= input_column < ifm.shape[2]:
                        spatial_tiles.add((kernel_row, kernel_column, input_row // ifm.tile_shape[1], input_column // ifm.tile_shape[2]))
        ifm_tiles, wgt_tiles = set(), set()
        for group in range(channel_start // output_channels_per_group, (channel_stop - 1) // output_channels_per_group + 1):
            group_channel_start = max(channel_start, group * output_channels_per_group)
            group_channel_stop = min(channel_stop, (group + 1) * output_channels_per_group)
            input_channel_start = group * input_channels_per_group if wgt is not None else group_channel_start
            input_channel_stop = (group + 1) * input_channels_per_group if wgt is not None else group_channel_stop
            ifm_channels = range(input_channel_start // ifm.tile_shape[3], (input_channel_stop - 1) // ifm.tile_shape[3] + 1)
            wgt_output_channels = range(group_channel_start // wgt.tile_shape[2], (group_channel_stop - 1) // wgt.tile_shape[2] + 1) if wgt is not None else ()
            wgt_input_channels = range((input_channels_per_group - 1) // wgt.tile_shape[3] + 1) if wgt is not None else ()
            for kernel_row, kernel_column, input_row_tile, input_column_tile in spatial_tiles:
                for input_channel in ifm_channels:
                    ifm_tiles.add((batch, input_row_tile, input_column_tile, input_channel))
                for output_channel_tile in wgt_output_channels:
                    for input_channel_tile in wgt_input_channels:
                        wgt_tiles.add((kernel_row, kernel_column, output_channel_tile, input_channel_tile))
        inputs = {0: sorted(ifm_tiles)}
        if wgt is not None:
            inputs[1] = sorted(wgt_tiles)
        if bias is not None:
            bias_channels = range(channel_start // bias.tile_shape[-1], (channel_stop - 1) // bias.tile_shape[-1] + 1)
            inputs[2] = [(0,) * (len(bias.tile_grid_shape) - 1) + (bias_channel,) for bias_channel in bias_channels]
        ops[(batch, row, column, output_channel)] = inputs

    return output_grid, input_grids, ops

def create_reduction_mapping_requistes(kernel_desc: MeshKernelDescriptor):
    """
    Create the output grid, input grids, and operations for a reduction kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid, a dictionary of input grids, and a dictionary of operations.
    """

    if kernel_desc.kernel_type != MeshKernelType.REDUCTION or len(kernel_desc.input_tensors) != 1 or len(kernel_desc.output_tensors) != 1:
        raise ValueError("Kernel descriptor must contain one REDUCTION input and one output.")
    ifm, ofm = kernel_desc.input_tensors[0], kernel_desc.output_tensors[0]
    keepdim = ofm.shape == ifm.shape[:-1] + (1,)
    if not keepdim and ofm.shape != ifm.shape[:-1]:
        raise ValueError("Reduction output must reduce the last input dimension.")
    if ofm.tile_grid_shape[:len(ifm.tile_grid_shape) - 1] != ifm.tile_grid_shape[:-1]:
        raise ValueError("Reduction output tiles must align with the preserved input dimensions.")
    output_grid, input_grids = create_input_output_grids(kernel_desc)
    ops = {}
    for output_coord in np.ndindex(*ofm.tile_grid_shape):
        prefix = output_coord[:-1] if keepdim else output_coord
        input_tiles = [prefix + (reduction,) for reduction in range(ifm.tile_grid_shape[-1])]
        ops[output_coord] = {0: input_tiles}
    return output_grid, input_grids, ops

def create_sdpa_mapping_requistes(kernel_desc: MeshKernelDescriptor):
    """
    Create the output grid, input grids, and operations for a SDPA kernel descriptor.

    Args:
        kernel_desc (MeshKernelDescriptor): The kernel descriptor containing input and output tensor descriptors.

    Returns:
        tuple: A tuple containing the output grid, a dictionary of input grids, and a dictionary of operations.
    """

    if kernel_desc.kernel_type != MeshKernelType.SDPA or len(kernel_desc.input_tensors) not in (3, 4) or len(kernel_desc.output_tensors) != 1:
        raise ValueError("Kernel descriptor must contain Q, K, V, optional mask, and one SDPA output.")
    q, k, v = kernel_desc.input_tensors[:3]
    ofm = kernel_desc.output_tensors[0]
    mask = kernel_desc.input_tensors[3] if len(kernel_desc.input_tensors) == 4 else None
    output_grid, input_grids = create_input_output_grids(kernel_desc)
    head_group_size = q.shape[1] // k.shape[1]
    past_length = k.shape[2] - q.shape[2]
    ops = {}
    for batch, query_head, query_tile, output_dim_tile in np.ndindex(*ofm.tile_grid_shape):
        kv_head = query_head // head_group_size
        query_tiles = [(batch, query_head, query_tile, dim_tile) for dim_tile in range(q.tile_grid_shape[3])]
        query_end = min((query_tile + 1) * q.tile_shape[2], q.shape[2])
        key_end = min(k.shape[2], past_length + query_end) if kernel_desc.get_required_kwargs("is_causal") else k.shape[2]
        kv_tiles = range((key_end + k.tile_shape[2] - 1) // k.tile_shape[2])
        key_tiles = [(batch, kv_head, kv_tile, dim_tile) for kv_tile in kv_tiles for dim_tile in range(k.tile_grid_shape[3])]
        value_tiles = [(batch, kv_head, kv_tile, output_dim_tile) for kv_tile in kv_tiles]
        inputs = {0: query_tiles, 1: key_tiles, 2: value_tiles}
        if mask is not None:
            mask_tiles = []
            for kv_tile in kv_tiles:
                score_coord = (batch, query_head, query_tile, kv_tile)[-len(mask.tile_grid_shape):]
                mask_tiles.append(tuple(0 if dim == 1 else index for dim, index in zip(mask.tile_grid_shape, score_coord)))
            inputs[3] = list(dict.fromkeys(mask_tiles))
        ops[(batch, query_head, query_tile, output_dim_tile)] = inputs
    return output_grid, input_grids, ops

@dataclass(frozen=True)
class TileRef:
    tensor_id: int
    coord: tuple[int, ...]
    is_output: bool = False


@dataclass(frozen=True)
class CoreTask:
    core_id: int
    output: TileRef
    inputs: tuple[tuple[int, tuple[TileRef, ...]], ...]
    scratch_bytes: int = 0


@dataclass(frozen=True)
class TileLocation:
    ref: TileRef
    storage_key: tuple[int, tuple[int, ...]]
    mem_type: MeshMemoryType
    owner_id: int
    addr: int
    size: int


@dataclass(frozen=True)
class LocatedTask:
    core_id: int
    output: TileLocation
    inputs: tuple[tuple[int, tuple[TileLocation, ...]], ...]
    scratch_bytes: int = 0


@dataclass(frozen=True)
class BufferRegion:
    core_id: int
    addr: int
    size: int


@dataclass(frozen=True)
class ScratchPlan:
    core_id: int
    ld_regions: tuple[BufferRegion, ...]
    st_regions: tuple[BufferRegion, ...]
    combined_regions: tuple[BufferRegion, ...]
    ld_slots: tuple[tuple[BufferRegion, ...], ...]
    st_slots: tuple[tuple[BufferRegion, ...], ...]
    output_local: bool


@dataclass(frozen=True)
class ResidencyPlan:
    full_resident: bool
    buffer_locations: dict[int, dict[tuple[int, tuple[int, ...]], TileLocation]]
    preloads: dict[int, tuple[TileLocation, ...]]
    writebacks: dict[int, tuple[TileLocation, ...]]
    scratch_regions: dict[int, tuple[BufferRegion, ...]]


@dataclass(frozen=True)
class OutputGroup:
    core_id: int
    index: int
    output_group_id: int
    tasks: tuple[LocatedTask, ...]
    source_tasks: tuple[LocatedTask, ...]
    input_tiles: tuple[TileLocation, ...]
    first_chunk: bool
    last_chunk: bool


@dataclass(frozen=True)
class PlannedTransfer:
    tile: TileLocation
    buffer: BufferRegion
    consumers: tuple[TileRef, ...] = ()


@dataclass(frozen=True)
class GroupTransfers:
    loads: tuple[PlannedTransfer, ...]
    local_reads: tuple[TileLocation, ...]
    stores: tuple[PlannedTransfer, ...]
    outputs: tuple[TileLocation, ...]
    scratch_regions: tuple[BufferRegion, ...]


@dataclass(frozen=True)
class StagePlan:
    core_id: int
    kind: str
    group: OutputGroup | None
    ld_slot: int | None
    st_slot: int | None
    loads: tuple[PlannedTransfer, ...]
    local_reads: tuple[TileLocation, ...]
    stores: tuple[PlannedTransfer, ...]
    outputs: tuple[TileLocation, ...]
    scratch_regions: tuple[BufferRegion, ...]




def collect_core_tasks(mapping: _Mapping, scratch_bytes_per_output: dict[tuple[int, ...], int] | None = None) -> dict[int, list[CoreTask]]:
    scratch_bytes_per_output = {} if scratch_bytes_per_output is None else scratch_bytes_per_output
    tasks = {}
    for coords, core_id in np.ndenumerate(mapping.core_mesh):
        core_id = int(core_id)
        if core_id in tasks:
            raise ValueError(f"Core ID {core_id} occurs more than once in the mesh.")
        tasks[core_id] = []
        for tile in mapping.output_mapping[coords]:
            output_coord = tuple(int(value) for value in np.asarray(tile).reshape(-1))
            if all(value == -1 for value in output_coord):
                continue
            if any(value < 0 for value in output_coord) or output_coord not in mapping.ops:
                raise ValueError(f"Invalid or missing operation for output tile {output_coord}.")
            scratch_bytes = int(scratch_bytes_per_output.get(output_coord, 0))
            if scratch_bytes < 0:
                raise ValueError("Per-output scratch size must be nonnegative.")
            inputs = tuple((int(tensor_id), tuple(TileRef(int(tensor_id), tuple(int(value) for value in input_coord)) for input_coord in coords)) for tensor_id, coords in mapping.ops[output_coord].items())
            tasks[core_id].append(CoreTask(core_id, TileRef(0, output_coord, True), inputs, scratch_bytes))
    return tasks


def resolve_tile_locations(tasks: dict[int, list[CoreTask]], input_placements: list, output_placements: list) -> dict[int, list[LocatedTask]]:
    if len(output_placements) != 1:
        raise ValueError("The current mapping requires exactly one output tensor.")
    def locate(ref: TileRef) -> TileLocation:
        placements = output_placements if ref.is_output else input_placements
        if ref.tensor_id < 0 or ref.tensor_id >= len(placements):
            raise ValueError(f"Unknown tensor ID in tile reference {ref}.")
        placement = placements[ref.tensor_id]
        bank = placement.tile_placement.get(ref.coord)
        if bank is None:
            raise ValueError(f"No placement for tile {ref}.")
        descriptor = placement.tensor_desc
        size = descriptor.get_tile_size()
        if bank.size < size:
            raise ValueError(f"Placement is smaller than tile {ref}.")
        storage_key = (descriptor.storage_id, descriptor.map_tile_coord_to_storage(ref.coord))
        return TileLocation(ref, storage_key, bank.mem_type, int(bank.owner_id), int(bank.addr), int(size))
    return {core_id: [LocatedTask(core_id, locate(task.output), tuple((tensor_id, tuple(locate(ref) for ref in refs)) for tensor_id, refs in task.inputs), task.scratch_bytes) for task in core_tasks] for core_id, core_tasks in tasks.items()}


def _split_regions(regions: tuple[BufferRegion, ...]) -> tuple[tuple[BufferRegion, ...], tuple[BufferRegion, ...]]:
    half = sum(region.size for region in regions) // 2
    slots = [[], []]
    remaining = [half, half]
    for region in regions:
        addr, available = region.addr, region.size
        for index in range(2):
            used = min(available, remaining[index])
            if used:
                slots[index].append(BufferRegion(region.core_id, addr, used))
                addr += used
                available -= used
                remaining[index] -= used
    return tuple(slots[0]), tuple(slots[1])


def _pack_regions(sizes: list[int], regions: tuple[BufferRegion, ...]) -> tuple[BufferRegion, ...] | None:
    free = [[region.addr, region.size, region.core_id] for region in regions]
    result = [None] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda item: sizes[item], reverse=True):
        size = sizes[index]
        if size < 0:
            raise ValueError("Scratch allocation size must be nonnegative.")
        if size == 0:
            continue
        target = next((entry for entry in free if entry[1] >= size), None)
        if target is None:
            return None
        result[index] = BufferRegion(target[2], target[0], size)
        target[0] += size
        target[1] -= size
    return tuple(result)


def _without_regions(regions: tuple[BufferRegion, ...], occupied: tuple[BufferRegion, ...]) -> tuple[BufferRegion, ...]:
    free = []
    for region in regions:
        pieces = [(region.addr, region.addr + region.size)]
        for used in occupied:
            if used.core_id != region.core_id:
                continue
            pieces = [(start, min(end, used.addr)) for start, end in pieces if start < used.addr] + [(max(start, used.addr + used.size), end) for start, end in pieces if end > used.addr + used.size]
        free.extend(BufferRegion(region.core_id, start, end - start) for start, end in pieces if start < end)
    return tuple(free)


def plan_scratch_regions(core_mesh: np.ndarray, ld_buffer_ptrs: dict[int, tuple[int, int]], st_buffer_ptrs: dict[int, tuple[int, int]], output_locations: dict[int, list[LocatedTask]]) -> dict[int, ScratchPlan]:
    plans = {}
    for core_id in (int(value) for value in np.asarray(core_mesh).flat):
        if core_id in plans or core_id not in ld_buffer_ptrs or core_id not in st_buffer_ptrs:
            raise ValueError(f"Core {core_id} has duplicate or missing LD/ST reservations.")
        ld_addr, ld_size = ld_buffer_ptrs[core_id]
        st_addr, st_size = st_buffer_ptrs[core_id]
        if ld_size <= 0 or st_size <= 0 or min(ld_addr, st_addr) < 0 or (ld_addr < st_addr + st_size and st_addr < ld_addr + ld_size):
            raise ValueError(f"Invalid or overlapping LD/ST reservations for core {core_id}.")
        output_types = {task.output.mem_type for task in output_locations.get(core_id, ())}
        if len(output_types) > 1:
            raise ValueError("Mixed output memory types on one core need separate plans.")
        output_local = output_types != {MeshMemoryType.DEVICE_MEMORY}
        ld_regions = (BufferRegion(core_id, int(ld_addr), int(ld_size)),)
        st_regions = (BufferRegion(core_id, int(st_addr), int(st_size)),)
        combined = ld_regions + st_regions
        ld_slots = _split_regions(combined if output_local else ld_regions)
        st_slots = () if output_local else _split_regions(st_regions)
        plans[core_id] = ScratchPlan(core_id, ld_regions, st_regions, combined, ld_slots, st_slots, output_local)
    return plans


def plan_full_residency(located_tasks: dict[int, list[LocatedTask]], scratch_plans: dict[int, ScratchPlan], local_cache_vacancy: dict | None = None) -> ResidencyPlan:
    buffer_locations = {}
    preloads = {}
    writebacks = {}
    scratch_regions = {}
    for core_id, scratch in scratch_plans.items():
        inputs = {}
        outputs = {}
        tasks = located_tasks.get(core_id, ())
        for task in tasks:
            for _, locations in task.inputs:
                for location in locations:
                    if location.mem_type == MeshMemoryType.DEVICE_MEMORY:
                        inputs.setdefault(location.storage_key, location)
            if task.output.mem_type == MeshMemoryType.DEVICE_MEMORY:
                outputs.setdefault(task.output.storage_key, task.output)
        if set(inputs) & set(outputs):
            return ResidencyPlan(False, {}, {}, {}, {})
        locations = list(inputs.values()) + list(outputs.values())
        scratch_sizes = [task.scratch_bytes for task in tasks if task.scratch_bytes]
        packed = _pack_regions([location.size for location in locations] + scratch_sizes, scratch.combined_regions)
        if packed is None:
            return ResidencyPlan(False, {}, {}, {}, {})
        buffer_locations[core_id] = {location.storage_key: TileLocation(location.ref, location.storage_key, MeshMemoryType.LOCAL_CACHE, core_id, region.addr, location.size) for location, region in zip(locations, packed)}
        preloads[core_id] = tuple(inputs.values())
        writebacks[core_id] = tuple(outputs.values())
        scratch_regions[core_id] = packed[len(locations):]
    if set(located_tasks) != set(scratch_plans):
        raise ValueError("Located tasks and fixed buffer plans must cover the same cores.")
    return ResidencyPlan(True, buffer_locations, preloads, writebacks, scratch_regions)


def _unique_group_inputs(tasks: list[LocatedTask]) -> tuple[TileLocation, ...]:
    return tuple(dict((location.storage_key, location) for task in tasks for _, inputs in task.inputs for location in inputs if location.mem_type == MeshMemoryType.DEVICE_MEMORY).values())


def group_output_tasks(tasks: list[LocatedTask], scratch_plan: ScratchPlan, residency_plan: ResidencyPlan, split_task=None, merge_partial: bool=False) -> list[OutputGroup]:
    if residency_plan.full_resident:
        return [OutputGroup(scratch_plan.core_id, 0, 0, tuple(tasks), tuple(tasks), (), True, True)] if tasks else []
    groups = []
    current = []
    output_group_id = 0

    def make_group(fragments: list[LocatedTask], sources: list[LocatedTask], first: bool, last: bool) -> OutputGroup:
        return OutputGroup(scratch_plan.core_id, len(groups), output_group_id, tuple(fragments), tuple(sources), _unique_group_inputs(fragments), first, last)

    def fits(candidate: list[LocatedTask], index: int, owner: int, partial: bool) -> bool:
        local_partial = scratch_plan.output_local and partial
        ld_slots = _split_regions(scratch_plan.ld_regions) if local_partial else scratch_plan.ld_slots
        ld_regions = ld_slots[index % 2]
        st_regions = (_split_regions(scratch_plan.st_regions) if local_partial else scratch_plan.st_slots)[owner % 2] if not scratch_plan.output_local or local_partial else ()
        scratch_size = sum(task.scratch_bytes for task in candidate)
        ld_scratch = _pack_regions([scratch_size], ld_regions) if scratch_plan.output_local and not partial and scratch_size else ()
        if ld_scratch is None:
            return False
        free_ld = _without_regions(ld_regions, ld_scratch)
        outputs = [task.output for task in candidate if task.output.mem_type == MeshMemoryType.DEVICE_MEMORY or local_partial]
        st_sizes = [location.size for location in outputs] + ([scratch_size] if (not scratch_plan.output_local or local_partial) and scratch_size else [])
        return _pack_regions(st_sizes, st_regions) is not None and _pack_regions([location.size for location in _unique_group_inputs(candidate)], free_ld) is not None

    def finish_current():
        nonlocal current, output_group_id
        if current:
            groups.append(make_group(current, current, True, True))
            current = []
            output_group_id += 1

    for task in tasks:
        if task.core_id != scratch_plan.core_id:
            raise ValueError("Output task belongs to a different core.")
        if current and not fits(current + [task], len(groups), output_group_id, False):
            finish_current()
        if fits([task], len(groups), output_group_id, False):
            current.append(task)
            continue
        if split_task is None:
            if len(task.inputs) != 1 or not task.inputs[0][1]:
                raise ValueError(f"Output tile {task.output.ref.coord} needs kernel-specific contribution groups or exceeds an ST slot.")
            tensor_id, locations = task.inputs[0]
            units = [((tensor_id, (location,)),) for location in locations]
        else:
            units = split_task(task)
            if not units:
                raise ValueError(f"Output tile {task.output.ref.coord} has no splittable contributions.")
        chunks = []
        chunk_units = []

        def fragment_for(units):
            by_tensor = {}
            for unit in units:
                for tensor_id, locations in unit:
                    by_tensor.setdefault(tensor_id, {})
                    for location in locations:
                        by_tensor[tensor_id].setdefault(location.storage_key, location)
            inputs = tuple((tensor_id, tuple(locations.values())) for tensor_id, locations in by_tensor.items())
            return replace(task, inputs=inputs)

        for unit in units:
            candidate = fragment_for(chunk_units + [unit])
            if chunk_units and not fits([candidate], len(groups) + len(chunks), output_group_id, True):
                chunks.append(fragment_for(chunk_units))
                chunk_units = []
                candidate = fragment_for([unit])
            if not fits([candidate], len(groups) + len(chunks), output_group_id, True):
                raise ValueError(f"Output tile {task.output.ref.coord} has a contribution that cannot fit its LD/ST slots.")
            chunk_units.append(unit)
        chunks.append(fragment_for(chunk_units))
        if len(chunks) < 2:
            raise ValueError(f"Output tile {task.output.ref.coord} cannot fit one complete fixed-buffer stage.")
        for index, fragment in enumerate(chunks):
            groups.append(make_group([fragment], [task], index == 0, index == len(chunks) - 1))
        output_group_id += 1
    finish_current()
    if not merge_partial:
        return groups
    sequences = []
    index = 0
    while index < len(groups):
        first = groups[index]
        end = index + 1
        while end < len(groups) and groups[end].output_group_id == first.output_group_id:
            end += 1
        sequences.append(groups[index:end])
        index = end
    merged = []
    pending = []
    output_group_id = 0

    def flush_pending():
        nonlocal output_group_id
        if not pending:
            return
        for depth in range(len(pending[0])):
            fragments = [task for sequence in pending for task in sequence[depth].tasks]
            sources = [task for sequence in pending for task in sequence[depth].source_tasks]
            merged.append(OutputGroup(scratch_plan.core_id, len(merged), output_group_id, tuple(fragments), tuple(sources), _unique_group_inputs(fragments), depth == 0, depth == len(pending[0]) - 1))
        output_group_id += 1
        pending.clear()

    for sequence in sequences:
        if len(sequence) == 1:
            flush_pending()
            group = sequence[0]
            merged.append(replace(group, index=len(merged), output_group_id=output_group_id))
            output_group_id += 1
            continue
        candidate = pending + [sequence]
        if pending and (len(sequence) != len(pending[0]) or any(not fits([task for item in candidate for task in item[depth].tasks], len(merged) + depth, output_group_id, True) for depth in range(len(sequence)))):
            flush_pending()
            candidate = [sequence]
        pending = candidate
    flush_pending()
    return merged


def plan_group_transfers(group: OutputGroup, residency_plan: ResidencyPlan, scratch_plan: ScratchPlan) -> GroupTransfers:
    if group.core_id != scratch_plan.core_id:
        raise ValueError("Output group belongs to a different core.")
    input_uses = {}
    local_reads = {}
    for task in group.tasks:
        for _, inputs in task.inputs:
            for location in inputs:
                input_uses.setdefault(location.storage_key, []).append(location.ref)
                effective = residency_plan.buffer_locations[group.core_id].get(location.storage_key, location) if residency_plan.full_resident else location
                if effective.mem_type == MeshMemoryType.LOCAL_CACHE:
                    local_reads.setdefault(location.storage_key, replace(effective, ref=location.ref))
    if residency_plan.full_resident:
        outputs = tuple(residency_plan.buffer_locations[group.core_id].get(task.output.storage_key, task.output) for task in group.tasks)
        return GroupTransfers((), tuple(local_reads.values()), (), outputs, residency_plan.scratch_regions[group.core_id])
    partial = not (group.first_chunk and group.last_chunk)
    local_partial = scratch_plan.output_local and partial
    ld_slots = _split_regions(scratch_plan.ld_regions) if local_partial else scratch_plan.ld_slots
    ld_regions = ld_slots[group.index % 2]
    st_regions = (_split_regions(scratch_plan.st_regions) if local_partial else scratch_plan.st_slots)[group.output_group_id % 2] if not scratch_plan.output_local or local_partial else ()
    scratch_size = sum(task.scratch_bytes for task in group.tasks)
    ld_scratch = _pack_regions([scratch_size], ld_regions) if scratch_plan.output_local and not partial and scratch_size else ()
    if ld_scratch is None:
        raise ValueError("Fixed LD scratch capacity changed after grouping.")
    free_ld = _without_regions(ld_regions, ld_scratch)
    packed_ld = _pack_regions([location.size for location in group.input_tiles], free_ld)
    buffered = [task.output for task in group.tasks if task.output.mem_type == MeshMemoryType.DEVICE_MEMORY or local_partial]
    packed_st = _pack_regions([location.size for location in buffered] + ([scratch_size] if (not scratch_plan.output_local or local_partial) and scratch_size else []), st_regions)
    if packed_ld is None or packed_st is None:
        raise ValueError("Fixed buffer capacity changed after grouping.")
    loads = tuple(PlannedTransfer(location, packed_ld[index], tuple(input_uses[location.storage_key])) for index, location in enumerate(group.input_tiles))
    stores = tuple(PlannedTransfer(location, packed_st[index]) for index, location in enumerate(buffered) if group.last_chunk)
    buffered_locations = {location.storage_key: TileLocation(location.ref, location.storage_key, MeshMemoryType.LOCAL_CACHE, group.core_id, packed_st[index].addr, location.size) for index, location in enumerate(buffered)}
    outputs = tuple(buffered_locations.get(task.output.storage_key, task.output) for task in group.tasks)
    scratch_regions = ld_scratch if scratch_plan.output_local and not partial else packed_st[len(buffered):]
    return GroupTransfers(loads, tuple(local_reads.values()), stores, outputs, scratch_regions)


def schedule_core_stages(groups: dict[int, list[OutputGroup]], transfers: dict[int, list[GroupTransfers]], scratch_plans: dict[int, ScratchPlan], residency_plan: ResidencyPlan) -> dict[int, list[StagePlan]]:
    if set(groups) != set(scratch_plans) or set(transfers) != set(scratch_plans):
        raise ValueError("Group and transfer plans must cover every core.")
    stages = {}
    for core_id, scratch in scratch_plans.items():
        if len(groups[core_id]) != len(transfers[core_id]):
            raise ValueError("Every output group requires one transfer plan.")
        core_stages = []
        if residency_plan.full_resident and residency_plan.preloads[core_id]:
            loads = tuple(PlannedTransfer(source, BufferRegion(core_id, residency_plan.buffer_locations[core_id][source.storage_key].addr, source.size)) for source in residency_plan.preloads[core_id])
            core_stages.append(StagePlan(core_id, "PRELOAD", None, None, None, loads, (), (), (), ()))
        for group, transfer in zip(groups[core_id], transfers[core_id]):
            if group.core_id != core_id:
                raise ValueError("Output group belongs to a different core.")
            partial_local = scratch.output_local and not (group.first_chunk and group.last_chunk)
            buffered = any(task.output.mem_type == MeshMemoryType.DEVICE_MEMORY for task in group.tasks) or partial_local
            ld_slot = group.index % 2 if not residency_plan.full_resident and transfer.loads else None
            st_slot = group.output_group_id % 2 if not residency_plan.full_resident and buffered else None
            core_stages.append(StagePlan(core_id, "COMPUTE", group, ld_slot, st_slot, transfer.loads, transfer.local_reads, transfer.stores, transfer.outputs, transfer.scratch_regions))
        if residency_plan.full_resident and residency_plan.writebacks[core_id]:
            stores = tuple(PlannedTransfer(destination, BufferRegion(core_id, residency_plan.buffer_locations[core_id][destination.storage_key].addr, destination.size)) for destination in residency_plan.writebacks[core_id])
            core_stages.append(StagePlan(core_id, "WRITEBACK", None, None, None, (), (), stores, (), ()))
        stages[core_id] = core_stages
    return stages


def validate_materialization_plan(core_plans: dict[int, list[StagePlan]], scratch_plans: dict[int, ScratchPlan], residency_plan: ResidencyPlan) -> dict[int, list[StagePlan]]:
    if set(core_plans) != set(scratch_plans):
        raise ValueError("Stage plans must cover every core.")
    output_series = {}
    source_owners = {}
    all_input_keys = set()
    for core_id, stages in core_plans.items():
        scratch = scratch_plans[core_id]
        preloaded = set()
        written_back = set()
        compute_index = 0
        for stage in stages:
            if stage.core_id != core_id or stage.kind not in ("PRELOAD", "COMPUTE", "WRITEBACK"):
                raise ValueError("Invalid stage kind or core ID.")
            if stage.kind != "COMPUTE" and not residency_plan.full_resident:
                raise ValueError("Streaming has no boundary stages.")
            if stage.kind == "PRELOAD":
                if stage.group is not None or stage.stores or stage.outputs or stage.ld_slot is not None or stage.st_slot is not None:
                    raise ValueError("Invalid preload stage.")
                keys = [transfer.tile.storage_key for transfer in stage.loads]
                if len(keys) != len(set(keys)) or preloaded:
                    raise ValueError("Device tile is preloaded more than once on one core.")
                preloaded.update(keys)
                continue
            if stage.kind == "WRITEBACK":
                if stage.group is not None or stage.loads or stage.outputs or stage.ld_slot is not None or stage.st_slot is not None:
                    raise ValueError("Invalid writeback stage.")
                keys = [transfer.tile.storage_key for transfer in stage.stores]
                if len(keys) != len(set(keys)) or written_back:
                    raise ValueError("Device tile is written back more than once on one core.")
                written_back.update(keys)
                continue
            group = stage.group
            if group is None or group.core_id != core_id or group.index != compute_index or len(group.tasks) != len(group.source_tasks) or len(stage.outputs) != len(group.tasks):
                raise ValueError("Compute stage has an invalid output group.")
            compute_index += 1
            if len({transfer.tile.storage_key for transfer in stage.loads}) != len(stage.loads):
                raise ValueError("Input tile is loaded more than once in one stage.")
            available = {transfer.tile.storage_key for transfer in stage.loads} | {location.storage_key for location in stage.local_reads}
            buffered = []
            for fragment, source, output in zip(group.tasks, group.source_tasks, stage.outputs):
                key = source.output.storage_key
                if fragment.output.storage_key != key or output.storage_key != key or fragment.scratch_bytes != source.scratch_bytes:
                    raise ValueError("Partial stage changed its output or scratch requirement.")
                if any(location.storage_key not in available for _, inputs in fragment.inputs for location in inputs):
                    raise ValueError("Compute stage lacks a required input tile.")
                for _, inputs in source.inputs:
                    all_input_keys.update(location.storage_key for location in inputs)
                if key in source_owners and source_owners[key] != core_id:
                    raise ValueError("A physical output tile is assigned to multiple cores.")
                source_owners[key] = core_id
                output_series.setdefault(key, []).append((stage, fragment, source, output))
                if not residency_plan.full_resident and (source.output.mem_type == MeshMemoryType.DEVICE_MEMORY or not (group.first_chunk and group.last_chunk)):
                    buffered.append(output)
            expected_stores = {task.output.storage_key for task in group.tasks if group.last_chunk and not residency_plan.full_resident and (task.output.mem_type == MeshMemoryType.DEVICE_MEMORY or not group.first_chunk)}
            if {transfer.tile.storage_key for transfer in stage.stores} != expected_stores or len(stage.stores) != len(expected_stores):
                raise ValueError("Only a final chunk may store each buffered output.")
            if residency_plan.full_resident and (stage.ld_slot is not None or stage.st_slot is not None or stage.loads or stage.stores):
                raise ValueError("Full residency must not use ping-pong transfers.")
            regions = [transfer.buffer for transfer in stage.loads] + [BufferRegion(core_id, output.addr, output.size) for output in buffered] + list(stage.scratch_regions)
            if any(region.core_id != core_id or region.size <= 0 or not any(parent.addr <= region.addr and region.addr + region.size <= parent.addr + parent.size for parent in scratch.combined_regions) for region in regions):
                raise ValueError("Stage uses memory outside its reserved LD/ST regions.")
            if any(first.addr < second.addr + second.size and second.addr < first.addr + first.size for index, first in enumerate(regions) for second in regions[index + 1:]):
                raise ValueError("Live fixed-buffer regions overlap within a stage.")
            if not residency_plan.full_resident:
                partial_local = scratch.output_local and not (group.first_chunk and group.last_chunk)
                ld_slots = _split_regions(scratch.ld_regions) if partial_local else scratch.ld_slots
                st_slots = _split_regions(scratch.st_regions) if partial_local else scratch.st_slots
                if stage.loads and (stage.ld_slot != group.index % 2 or any(not any(parent.addr <= item.buffer.addr and item.buffer.addr + item.buffer.size <= parent.addr + parent.size for parent in ld_slots[stage.ld_slot]) for item in stage.loads)):
                    raise ValueError("Stage loads exceed one LD slot.")
                if buffered and (stage.st_slot != group.output_group_id % 2 or any(not any(parent.addr <= output.addr and output.addr + output.size <= parent.addr + parent.size for parent in st_slots[stage.st_slot]) for output in buffered)):
                    raise ValueError("Output state exceeds its assigned ST slot.")
        if residency_plan.full_resident:
            if preloaded != {item.storage_key for item in residency_plan.preloads[core_id]} or written_back != {item.storage_key for item in residency_plan.writebacks[core_id]}:
                raise ValueError("Full-residency preload or writeback coverage is incomplete.")
            kinds = [stage.kind for stage in stages]
            if kinds != (["PRELOAD"] if preloaded else []) + ["COMPUTE"] * compute_index + (["WRITEBACK"] if written_back else []):
                raise ValueError("Full-residency stage order is invalid.")
            occupied = [BufferRegion(core_id, location.addr, location.size) for location in residency_plan.buffer_locations[core_id].values()] + list(residency_plan.scratch_regions[core_id])
            if any(not any(parent.addr <= region.addr and region.addr + region.size <= parent.addr + parent.size for parent in scratch.combined_regions) for region in occupied):
                raise ValueError("Resident tile is outside the reserved LD/ST regions.")
            if any(first.addr < second.addr + second.size and second.addr < first.addr + first.size for index, first in enumerate(occupied) for second in occupied[index + 1:]):
                raise ValueError("Resident fixed-buffer regions overlap.")
    if all_input_keys & set(output_series):
        raise ValueError("In-place input/output storage aliases require a kernel-specific plan.")
    for key, series in output_series.items():
        first_stage, _, original, first_output = series[0]
        if any(source != original for _, _, source, _ in series) or [entry[0].group.first_chunk for entry in series] != [True] + [False] * (len(series) - 1) or [entry[0].group.last_chunk for entry in series] != [False] * (len(series) - 1) + [True]:
            raise ValueError("Output chunks must have one first and one final stage.")
        if len({entry[0].group.output_group_id for entry in series}) != 1 or any(series[index][0].group.index + 1 != series[index + 1][0].group.index for index in range(len(series) - 1)):
            raise ValueError("Partial output stages must be consecutive and own one ST slot.")
        for tensor_id, locations in original.inputs:
            expected = {location.ref for location in locations}
            actual = {location.ref for _, fragment, _, _ in series for input_id, fragment_locations in fragment.inputs if input_id == tensor_id for location in fragment_locations}
            if actual != expected:
                raise ValueError("Partial stages omit or add input contributions.")
        if len(series) > 1 and any(output.addr != first_output.addr or output.owner_id != first_output.owner_id or stage.st_slot != first_stage.st_slot for stage, _, _, output in series):
            raise ValueError("Partial output stages do not retain one accumulator region.")
    return core_plans


if __name__ == "__main__":
    import torch
    from neuromta.system.software.utils.compiler import MeshDeviceCompiledTensorStats
    from neuromta.system.software.utils.descriptor import MeshMemoryBankDescriptor
    from neuromta.system.software.utils.kernel import MESH_KERNEL_LINEAR

    core_mesh = np.array([[7]])
    ifm = MeshTensorDescriptor((32, 32), (32, 32), torch.bfloat16)
    wgt = MeshTensorDescriptor((384, 32), (32, 32), torch.bfloat16)
    ofm = MeshTensorDescriptor((32, 384), (32, 32), torch.bfloat16)
    descriptor = MESH_KERNEL_LINEAR(ifm, wgt, ofm)
    output_grid, input_grids, ops = create_linear_mapping_requistes(descriptor)
    mapping = create_mapping(core_mesh, output_grid, input_grids, ops)

    def place_tensor(tensor: MeshTensorDescriptor, local: bool, base_addr: int) -> MeshDeviceCompiledTensorStats:
        size = tensor.get_tile_size()
        banks = {coord: (MeshMemoryBankDescriptor.LOCAL_CACHE(7, base_addr + index * size, size) if local else MeshMemoryBankDescriptor.DEVICE_MEMORY(base_addr + index * size, size)) for index, coord in enumerate(tensor.get_tile_coords())}
        return MeshDeviceCompiledTensorStats(tensor).place(banks)

    def run_case(name: str, local_inputs: bool, local_output: bool, ld_size: int, st_size: int):
        input_placements = [place_tensor(ifm, local_inputs, 80000), place_tensor(wgt, local_inputs, 100000)]
        output_placements = [place_tensor(ofm, local_output, 150000)]
        tasks = collect_core_tasks(mapping)
        located = resolve_tile_locations(tasks, input_placements, output_placements)
        scratch = plan_scratch_regions(core_mesh, {7: (0, ld_size)}, {7: (ld_size, st_size)}, located)
        residency = plan_full_residency(located, scratch)
        groups = {core_id: group_output_tasks(core_tasks, scratch[core_id], residency) for core_id, core_tasks in located.items()}
        transfers = {core_id: [plan_group_transfers(group, residency, scratch[core_id]) for group in core_groups] for core_id, core_groups in groups.items()}
        stages = schedule_core_stages(groups, transfers, scratch, residency)
        validate_materialization_plan(stages, scratch, residency)
        print(f"{name}: full_resident={residency.full_resident}, groups={[len(group.tasks) for group in groups[7]]}, stages={[stage.kind for stage in stages[7]]}")
        return scratch, residency, groups, stages

    scratch, residency, groups, stages = run_case("streaming", False, False, 32768, 16384)
    assert not residency.full_resident and [len(group.tasks) for group in groups[7]] == [4, 4, 4]
    assert all(len(stage.loads) == 5 and len(stage.stores) == 4 for stage in stages[7])
    assert [stage.ld_slot for stage in stages[7]] == [0, 1, 0]
    scratch, residency, groups, stages = run_case("full residency", False, False, 32768, 24576)
    assert residency.full_resident and [stage.kind for stage in stages[7]] == ["PRELOAD", "COMPUTE", "WRITEBACK"]
    assert all(any(region.addr <= location.addr and location.addr + location.size <= region.addr + region.size for region in scratch[7].combined_regions) for location in residency.buffer_locations[7].values())
    scratch, residency, groups, stages = run_case("local output", False, True, 8192, 8192)
    assert not residency.full_resident and scratch[7].output_local and not any(stage.stores for stage in stages[7])
    scratch, residency, groups, stages = run_case("all local", True, True, 8192, 8192)
    assert residency.full_resident and [stage.kind for stage in stages[7]] == ["COMPUTE"]

    partial_inputs = tuple(TileLocation(TileRef(0, (index,)), (200, (index,)), MeshMemoryType.DEVICE_MEMORY, 0, 300000 + index * 2048, 2048) for index in range(7))
    partial_output = TileLocation(TileRef(0, (0,), True), (201, (0,)), MeshMemoryType.DEVICE_MEMORY, 0, 400000, 2048)
    partial_tasks = {7: [LocatedTask(7, partial_output, ((0, partial_inputs),))]}
    partial_scratch = plan_scratch_regions(core_mesh, {7: (0, 8192)}, {7: (8192, 4096)}, partial_tasks)
    partial_residency = plan_full_residency(partial_tasks, partial_scratch)
    partial_groups = {7: group_output_tasks(partial_tasks[7], partial_scratch[7], partial_residency)}
    partial_transfers = {7: [plan_group_transfers(group, partial_residency, partial_scratch[7]) for group in partial_groups[7]]}
    partial_stages = schedule_core_stages(partial_groups, partial_transfers, partial_scratch, partial_residency)
    validate_materialization_plan(partial_stages, partial_scratch, partial_residency)
    assert not partial_residency.full_resident and [len(stage.loads) for stage in partial_stages[7]] == [2, 2, 2, 1]
    assert [len(stage.stores) for stage in partial_stages[7]] == [0, 0, 0, 1]
    assert len({stage.outputs[0].addr for stage in partial_stages[7]}) == 1
    print(f"partial reduction: loads={[len(stage.loads) for stage in partial_stages[7]]}, stores={[len(stage.stores) for stage in partial_stages[7]]}")

    local_partial_output = replace(partial_output, mem_type=MeshMemoryType.LOCAL_CACHE, owner_id=7, addr=500000)
    local_partial_tasks = {7: [replace(partial_tasks[7][0], output=local_partial_output)]}
    local_partial_scratch = plan_scratch_regions(core_mesh, {7: (0, 8192)}, {7: (8192, 4096)}, local_partial_tasks)
    local_partial_residency = plan_full_residency(local_partial_tasks, local_partial_scratch)
    local_partial_groups = {7: group_output_tasks(local_partial_tasks[7], local_partial_scratch[7], local_partial_residency)}
    local_partial_transfers = {7: [plan_group_transfers(group, local_partial_residency, local_partial_scratch[7]) for group in local_partial_groups[7]]}
    local_partial_stages = schedule_core_stages(local_partial_groups, local_partial_transfers, local_partial_scratch, local_partial_residency)
    validate_materialization_plan(local_partial_stages, local_partial_scratch, local_partial_residency)
    assert [len(stage.stores) for stage in local_partial_stages[7]] == [0, 0, 0, 1]
    assert local_partial_stages[7][-1].stores[0].tile.mem_type == MeshMemoryType.LOCAL_CACHE

# if __name__ == "__main__":
#     # Example usage of create_mapping function
#     # LINEAR kernel example

#     import torch
#     from neuromta.system.software.utils.descriptor import MeshTensorDescriptor
#     from neuromta.system.software.utils.kernel import MESH_KERNEL_LINEAR

#     N_PRINTED_SLOTS = 4
    
#     print("\n=== LINEAR kernel mapping example ===")

#     ifm = MeshTensorDescriptor(shape=(32, 32), tile_shape=(32, 32), dtype=torch.bfloat16)
#     wgt = MeshTensorDescriptor(shape=(1024, 32), tile_shape=(32, 32), dtype=torch.bfloat16)
#     bias = MeshTensorDescriptor(shape=(1024,), tile_shape=(32,), dtype=torch.bfloat16)
#     ofm = MeshTensorDescriptor(shape=(32, 1024), tile_shape=(32, 32), dtype=torch.bfloat16)

#     kernel_desc = MESH_KERNEL_LINEAR(ifm=ifm, wgt=wgt, bias=bias, ofm=ofm)
#     output_grid, input_grids, ops = create_linear_mapping_requistes(kernel_desc)
#     mapping = create_mapping(np.arange(4).reshape(2, 2), output_grid, input_grids, ops)

#     for core in np.ndindex(*mapping.mesh_shape):
#         valid_slots = [slot for slot, tile in enumerate(mapping.output_mapping[core]) if np.all(tile >= 0)]
#         print(f"core {core}: {len(valid_slots)} OFM tiles")

#         for slot in valid_slots[:N_PRINTED_SLOTS]:
#             output_tile = tuple(int(index) for index in mapping.output_mapping[core][slot])
#             ifm_tiles = mapping.ops[output_tile][0]
#             wgt_tiles = mapping.ops[output_tile][1]
#             bias_tiles = mapping.ops[output_tile][2]
#             print(f"  OFM {tuple(int(index) for index in output_tile)}: IFM {ifm_tiles}, WGT {wgt_tiles}, BIAS {bias_tiles}")

#         if len(valid_slots) > N_PRINTED_SLOTS:
#             print(f"  ...")

# if __name__ == "__main__":
#     # Example usage of create_mapping function
#     # CONV2D kernel example

#     import torch
#     from neuromta.system.software.utils.descriptor import MeshTensorDescriptor
#     from neuromta.system.software.utils.kernel import MESH_KERNEL_CONV2D

#     N_PRINTED_SLOTS = 4
    
#     print("\n=== CONV2D kernel mapping example ===")

#     ifm = MeshTensorDescriptor(shape=(1, 224, 224, 32), tile_shape=(32, 32), dtype=torch.bfloat16)
#     wgt = MeshTensorDescriptor(shape=(3, 3, 64, 32), tile_shape=(32, 32), dtype=torch.bfloat16)
#     bias = MeshTensorDescriptor(shape=(64,), tile_shape=(32,), dtype=torch.bfloat16)
#     ofm = MeshTensorDescriptor(shape=(1, 224, 224, 64), tile_shape=(32, 32), dtype=torch.bfloat16)

#     kernel_desc = MESH_KERNEL_CONV2D(ifm=ifm, wgt=wgt, bias=bias, ofm=ofm, padding=1, stride=1, dilation=1)
#     output_grid, input_grids, ops = create_conv2d_mapping_requistes(kernel_desc)
#     mapping = create_mapping(np.arange(4).reshape(2, 2), output_grid, input_grids, ops)

#     print(f"OFM tiles: {ofm.get_n_tiles()}, core mesh: {mapping.mesh_shape}")
#     for core in np.ndindex(*mapping.mesh_shape):
#         valid_slots = [slot for slot, tile in enumerate(mapping.output_mapping[core]) if np.all(tile >= 0)]
#         print(f"core {core}: {len(valid_slots)} OFM tiles")

#         for slot in valid_slots[:N_PRINTED_SLOTS]:
#             output_tile = tuple(int(index) for index in mapping.output_mapping[core][slot])
#             ifm_tiles = mapping.ops[output_tile][0]
#             wgt_tiles = mapping.ops[output_tile][1]
#             bias_tiles = mapping.ops[output_tile][2]
#             print(f"  OFM {output_tile}: IFM {ifm_tiles}, WGT {wgt_tiles}, BIAS {bias_tiles}")

#         if len(valid_slots) > N_PRINTED_SLOTS:
#             print(f"  ...")

# if __name__ == "__main__":
#     # Example usage of create_mapping function
#     # ELEMENTWISE kernel example

#     import torch
#     from neuromta.system.software.utils.descriptor import MeshTensorDescriptor
#     from neuromta.system.software.utils.kernel import MESH_KERNEL_ELEMENTWISE
    
#     N_PRINTED_SLOTS = 4
    
#     print("\n=== ELEMENTWISE kernel mapping example ===")

#     ifm = MeshTensorDescriptor(shape=(1024, 1024), tile_shape=(32, 32), dtype=torch.bfloat16)
#     ofm = MeshTensorDescriptor(shape=(1024, 1024), tile_shape=(32, 32), dtype=torch.bfloat16)

#     kernel_desc = MESH_KERNEL_ELEMENTWISE(ifm, ofm)
#     output_grid, input_grids, ops = create_elementwise_mapping_requistes(kernel_desc)
#     mapping = create_mapping(np.arange(4).reshape(2, 2), output_grid, input_grids, ops)

#     print(f"OFM tiles: {ofm.get_n_tiles()}, core mesh: {mapping.mesh_shape}")
#     for core in np.ndindex(*mapping.mesh_shape):
#         valid_slots = [slot for slot, tile in enumerate(mapping.output_mapping[core]) if np.all(tile >= 0)]
#         print(f"core {core}: {len(valid_slots)} OFM tiles")

#         for slot in valid_slots[:N_PRINTED_SLOTS]:
#             output_tile = tuple(int(index) for index in mapping.output_mapping[core][slot])
#             ifm_tiles = mapping.ops[output_tile][0]
#             print(f"  OFM {output_tile}: IFM {ifm_tiles}")

#         if len(valid_slots) > N_PRINTED_SLOTS:
#             print(f"  ...")

# if __name__ == "__main__":
#     # Example usage of create_mapping function
#     # REDUCTION kernel example

#     import torch
#     from neuromta.system.software.utils.descriptor import MeshTensorDescriptor
#     from neuromta.system.software.utils.kernel import MESH_KERNEL_REDUCTION

#     N_PRINTED_SLOTS = 4
    
#     print("\n=== REDUCTION kernel mapping example ===")

#     ifm = MeshTensorDescriptor(shape=(256, 128), tile_shape=(32, 32), dtype=torch.bfloat16)
#     ofm = MeshTensorDescriptor(shape=(256, 1), tile_shape=(32, 1), dtype=torch.bfloat16)

#     kernel_desc = MESH_KERNEL_REDUCTION(ifm=ifm, ofm=ofm, ops_per_input_element=1)
#     output_grid, input_grids, ops = create_reduction_mapping_requistes(kernel_desc)
#     mapping = create_mapping(np.arange(4).reshape(2, 2), output_grid, input_grids, ops)

#     print(f"OFM tiles: {ofm.get_n_tiles()}, core mesh: {mapping.mesh_shape}")
#     for core in np.ndindex(*mapping.mesh_shape):
#         valid_slots = [slot for slot, tile in enumerate(mapping.output_mapping[core]) if np.all(tile >= 0)]
#         print(f"core {core}: {len(valid_slots)} OFM tiles")

#         for slot in valid_slots[:N_PRINTED_SLOTS]:
#             output_tile = tuple(int(index) for index in mapping.output_mapping[core][slot])
#             input_tiles = mapping.ops[output_tile][0]
#             print(f"  OFM {output_tile}: IFM {input_tiles}")

#         if len(valid_slots) > N_PRINTED_SLOTS:
#             print(f"  ...")

# if __name__ == "__main__":
#     # Example usage of create_mapping function
#     # SDPA kernel example

#     import torch
#     from neuromta.system.software.utils.descriptor import MeshTensorDescriptor
#     from neuromta.system.software.utils.kernel import MESH_KERNEL_SDPA

#     N_PRINTED_SLOTS = 4
    
#     print("\n=== SDPA kernel mapping example ===")

#     q = MeshTensorDescriptor(shape=(1, 32, 32, 64), tile_shape=(32, 32), dtype=torch.bfloat16)
#     k = MeshTensorDescriptor(shape=(1, 32, 32, 64), tile_shape=(32, 32), dtype=torch.bfloat16)
#     v = MeshTensorDescriptor(shape=(1, 32, 32, 64), tile_shape=(32, 32), dtype=torch.bfloat16)
#     o = MeshTensorDescriptor(shape=(1, 32, 32, 64), tile_shape=(32, 32), dtype=torch.bfloat16)

#     kernel_desc = MESH_KERNEL_SDPA(q=q, k=k, v=v, ofm=o)
#     output_grid, input_grids, ops = create_sdpa_mapping_requistes(kernel_desc)
#     mapping = create_mapping(np.arange(4).reshape(2, 2), output_grid, input_grids, ops)

#     print(f"OFM tiles: {o.get_n_tiles()}, core mesh: {mapping.mesh_shape}")
#     for core in np.ndindex(*mapping.mesh_shape):
#         valid_slots = [slot for slot, tile in enumerate(mapping.output_mapping[core]) if np.all(tile >= 0)]
#         print(f"core {core}: {len(valid_slots)} OFM tiles")

#         for slot in valid_slots[:N_PRINTED_SLOTS]:
#             output_tile = tuple(int(index) for index in mapping.output_mapping[core][slot])
#             q_tiles = mapping.ops[output_tile][0]
#             k_tiles = mapping.ops[output_tile][1]
#             v_tiles = mapping.ops[output_tile][2]
#             print(f"  OFM {output_tile}: Q {q_tiles}, K {k_tiles}, V {v_tiles}")

#         if len(valid_slots) > N_PRINTED_SLOTS:
#             print(f"  ...")