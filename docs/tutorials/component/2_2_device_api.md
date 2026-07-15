# Device and Hardware APIs

This document summarizes the main hardware-side APIs defined in `srcs/neuromta/component/implementation/hardware.py`. The focus is on how to use device, memory space, and core group abstractions. Internal implementation details are intentionally omitted.


## Core Groups

Core group classes are lightweight containers for NPU core IDs. They are used by device APIs, memory space APIs, and compiler code to describe which physical cores participate in an operation.

### `MCA_CoreGroup`

`MCA_CoreGroup` represents a one-dimensional group of NPU core IDs. It behaves like a list while adding convenience APIs for merging, splitting, and querying core groups.

| API | Description |
| --- | --- |
| `MCA_CoreGroup(core_ids)` | Creates a core group from a sequence of core IDs. |
| `merge(other)` | Returns a new `MCA_CoreGroup` containing the union of this group and `other`. |
| `split(shape)` | Splits the group into subgroups with at most `shape` cores each. |
| `intersection(other)` | Returns a new `MCA_CoreGroup` containing core IDs that appear in both groups. |
| `merge_core_groups(core_groups)` | Class method that merges multiple core groups into one sorted `MCA_CoreGroup`. |
| `core_ids` | Property that returns the core IDs as a list. |
| `n_cores` | Property that returns the number of cores in the group. |

| Argument | Role |
| --- | --- |
| `core_ids` | Sequence of integer NPU core IDs. |
| `other` | Another `MCA_CoreGroup` used for merge or intersection. |
| `shape` | Maximum subgroup size for one-dimensional splitting. |
| `core_groups` | Sequence of `MCA_CoreGroup` objects to merge. |

### `MTA_CoreGrid`

`MTA_CoreGrid` extends `MCA_CoreGroup` with two-dimensional grid metadata. It is useful for multi-tile accelerators where spatial position matters, such as row-wise or column-wise mapping.

| API | Description |
| --- | --- |
| `MTA_CoreGrid(offset, shape, core_ids)` | Creates a two-dimensional view over a list of core IDs. |
| `lower()` | Converts the grid view into a flat `MCA_CoreGroup`. |
| `split(shape)` | Splits the grid into smaller `MTA_CoreGrid` subgrids. |
| `get_core_id(y, x)` | Returns the core ID at the given grid coordinate. |
| `__getitem__(idx)` | Supports list-style indexing and tuple-based grid slicing. |

| Argument | Role |
| --- | --- |
| `offset` | `(row, col)` coordinate of the grid origin in the device-level core mesh. |
| `shape` | `(rows, cols)` shape of the grid or subgrid. |
| `core_ids` | Flattened core ID list stored in row-major order. |
| `y` | Row index inside the grid. |
| `x` | Column index inside the grid. |
| `idx` | Integer index, slice, or tuple-style grid index used to select cores. |


## Memory Spaces

Memory space classes are allocation helpers layered on top of the device `GlobalContext`. They create stack-like allocation regions over main memory channels or NPU L1 banks and return `Pointer` objects for allocated data.

### `MCA_MemorySpace`

`MCA_MemorySpace` is the base abstraction for a logical memory space. It tracks the owner IDs, memory type, per-owner capacity, and stack allocations created in the underlying `GlobalContext`.

| API | Description |
| --- | --- |
| `MCA_MemorySpace(device, mem_type, size_per_owner, owner_ids)` | Creates a memory space over selected memory owners. |
| `empty_space(owner_id)` | Returns the remaining free space for the selected owner. |
| `allocate(owner_id, size)` | Allocates `size` bytes from the selected owner and returns a `Pointer`. |
| `remove()` | Removes the backing allocation stacks from the global context. |
| `override(new_owners)` | Creates an overridden view that reuses the same backing memory space with different visible owners. |
| `device` | Property that returns the owning `MCA_DeviceBase`. |
| `mem_type` | Property that returns `GlobalContextMemType.MAIN` or `GlobalContextMemType.L1`. |
| `owner_ids` | Property that returns the owner IDs visible through this memory space. |
| `size_per_owner` | Property that returns the reserved size per owner. |
| `is_removed` | Property indicating whether the memory space has been removed. |
| `is_main` | Property indicating whether this space targets main memory. |
| `is_l1` | Property indicating whether this space targets L1 memory. |

| Argument | Role |
| --- | --- |
| `device` | `MCA_DeviceBase` that owns the global context and memory handles. |
| `mem_type` | Memory type selected from `GlobalContextMemType`. |
| `size_per_owner` | Number of bytes reserved for each owner. |
| `owner_ids` | Main memory channel IDs or NPU core IDs that own memory regions. |
| `owner_id` | Selected channel ID or core ID for allocation/query. |
| `size` | Number of bytes to allocate. |
| `new_owners` | Replacement visible owner list for an overridden memory-space view. |

### `MCA_MainMemorySpace`

`MCA_MainMemorySpace` is a concrete memory space for main memory channels. If `channel_ids` is not provided, it spans all configured main memory channels.

| API | Description |
| --- | --- |
| `MCA_MainMemorySpace(device, size_per_channel, channel_ids=None)` | Creates a main memory allocation space over selected memory channels. |
| `empty_space(channel_id)` | Returns the remaining free space in the selected channel. |
| `allocate(channel_id, size)` | Allocates memory from the selected channel and returns a `Pointer`. |

| Argument | Role |
| --- | --- |
| `device` | Device that owns the main memory channels. |
| `size_per_channel` | Number of bytes reserved per channel. |
| `channel_ids` | Optional sequence of channel IDs; defaults to all channels. |
| `channel_id` | Channel ID used for allocation or free-space query. |
| `size` | Number of bytes to allocate. |

### `MCA_L1MemorySpace`

`MCA_L1MemorySpace` is a concrete memory space for NPU-local L1 memory banks. It is usually created for a specific `MCA_CoreGroup` or `MTA_CoreGrid`.

| API | Description |
| --- | --- |
| `MCA_L1MemorySpace(device, size_per_bank, core_group)` | Creates an L1 allocation space over the cores in `core_group`. |
| `empty_space(core_id)` | Returns the remaining free space in the selected core's L1 allocation stack. |
| `allocate(core_id, size)` | Allocates L1 memory from the selected core and returns a `Pointer`. |

| Argument | Role |
| --- | --- |
| `device` | Device that owns the NPU cores and L1 banks. |
| `size_per_bank` | Number of bytes reserved per L1 bank. |
| `core_group` | `MCA_CoreGroup` or `MTA_CoreGrid` whose cores participate in the memory space. |
| `core_id` | NPU core ID used for allocation or free-space query. |
| `size` | Number of bytes to allocate. |


## Device Base Classes

Device classes bind global context, interconnect context, NPU cores, DMA cores, optional companion modules, and memory spaces into one simulation target.

### `MCA_DeviceBase`

`MCA_DeviceBase` is the base class for multi-core accelerator devices. It creates `NPUCore` and `DMACore` instances from `GlobalContextConfig`, configures `IcntContext`, and optionally registers `BookSim2` and `DRAMSim3` companion modules.

| API | Description |
| --- | --- |
| `MCA_DeviceBase(global_config, icnt_config, mxu_config, vpu_config)` | Creates a multi-core accelerator device from global, interconnect, MXU, and VPU configs. |
| `mxu_ifm_tile_shape` | Property that returns the configured MXU IFM tile shape. |
| `mxu_wgt_tile_shape` | Property that returns the configured MXU weight tile shape. |
| `mxu_ofm_tile_shape` | Property that returns the configured MXU OFM tile shape. |
| `get_npu_core(core_id)` | Returns the `NPUCore` instance with the given core ID. |
| `get_npu_core_group(offset=None, n_cores=None)` | Returns a one-dimensional `MCA_CoreGroup` selected from NPU core IDs. |
| `create_main_mem_space(size_per_channel, channel_ids=None)` | Creates and registers a `MCA_MainMemorySpace`. |
| `create_l1_mem_space(size_per_bank, core_group)` | Creates and registers a `MCA_L1MemorySpace`. |
| `remove_all_main_mem_space()` | Removes all registered main memory spaces. |
| `remove_all_l1_mem_space()` | Removes all registered L1 memory spaces. |
| `clear_all_mem_spaces()` | Removes all registered L1 and main memory spaces. |
| `mem_get_data(ptr, size, dtype, native_python_type=False)` | Reads data from the memory handle that owns `ptr`. |
| `mem_set_data(ptr, size, data)` | Writes data to the memory handle that owns `ptr`. |
| `summary()` | Returns a dictionary describing device type, core counts, and core configuration. |
| `print_summary()` | Pretty-prints the result of `summary()`. |

| Argument | Role |
| --- | --- |
| `global_config` | `GlobalContextConfig` describing core IDs, memory regions, main memory, and companion settings. |
| `icnt_config` | `IcntConfig` describing interconnect behavior and optional `BookSim2` settings. |
| `mxu_config` | `MXUConfig` used to initialize each `NPUCore` MXU context. |
| `vpu_config` | `VPUConfig` used to initialize each `NPUCore` VPU context. |
| `core_id` | NPU core ID to query. |
| `offset` | Starting index in the one-dimensional NPU core ID list. |
| `n_cores` | Number of cores to include in the returned group. |
| `size_per_channel` | Number of bytes reserved per main memory channel. |
| `channel_ids` | Optional sequence of main memory channel IDs. |
| `size_per_bank` | Number of bytes reserved per NPU L1 bank. |
| `core_group` | `MCA_CoreGroup` or compatible core group used to select L1 owners. |
| `ptr` | `Pointer` whose owning memory handle should be accessed. |
| `size` | Number of bytes to read or write. |
| `dtype` | Data type used when reading memory. |
| `native_python_type` | Whether `mem_get_data` returns native Python values instead of tensor-like data. |
| `data` | Data payload written by `mem_set_data`. |

### `MTA_DeviceBase`

`MTA_DeviceBase` extends `MCA_DeviceBase` with a two-dimensional NPU core grid. It is intended for multi-tile accelerator models where the physical row/column layout of cores is important.

| API | Description |
| --- | --- |
| `MTA_DeviceBase(global_config, icnt_config, mxu_config, vpu_config)` | Creates a multi-tile accelerator device and builds a 2D NPU core grid from interconnect coordinates. |
| `get_npu_core_group(offset=None, shape=None)` | Returns an `MTA_CoreGrid` selected by 2D offset and shape. |
| `summary()` | Returns the base device summary with interconnect configuration included. |

| Argument | Role |
| --- | --- |
| `global_config` | `GlobalContextConfig` used to create core and memory metadata. |
| `icnt_config` | `IcntConfig` used to map core IDs to 2D coordinates. |
| `mxu_config` | `MXUConfig` propagated to each NPU core. |
| `vpu_config` | `VPUConfig` propagated to each NPU core. |
| `offset` | `(row, col)` start coordinate in the NPU core grid. |
| `shape` | `(rows, cols)` shape of the requested core grid. |


## Typical Usage Pattern

The device API is commonly used in three steps:

1. Create an `MCA_DeviceBase` or `MTA_DeviceBase` from hardware configuration objects.
2. Select NPU cores with `get_npu_core_group()` and reserve memory with `create_l1_mem_space()` or `create_main_mem_space()`.
3. Allocate tensors through memory spaces, access raw data through `mem_get_data()` or `mem_set_data()`, and release temporary spaces with `clear_all_mem_spaces()` when the experiment or compiled program is finished.

For one-dimensional accelerators, use `MCA_CoreGroup`. For grid-shaped multi-tile accelerators, use `MTA_CoreGrid` so that software mapping code can reason about row and column positions.
