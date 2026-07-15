# Tensor Buffer APIs

This document summarizes `MCA_TensorBuffer`, defined in `srcs/neuromta/component/implementation/tensor_buffer.py`. The class is the compiler/runtime-side tensor storage abstraction used by NeuroMTA component implementations. It connects tensor shape, memory layout, sharding, tiling, and device memory pointers.


## Role of `MCA_TensorBuffer`

`MCA_TensorBuffer` represents a tensor that is placed in an `MCA_MemorySpace`. The memory space can be main memory or L1 memory, and the buffer allocates one pointer per shard through that memory space.

The buffer is also responsible for converting an N-dimensional tensor into a two-dimensional memory layout. The last tensor dimension is treated as layout width, while all leading dimensions and the second-to-last dimension are flattened into layout height.


## Memory Layout

The logical tensor shape is stored in `shape`. If the input shape is one-dimensional, `MCA_TensorBuffer` internally treats it as `(1, width)` so that the layout is always two-dimensional.

The memory layout is defined as:

| Concept | Meaning |
| --- | --- |
| `layout_shape` | Two-dimensional physical layout `(layout_height, layout_width)`. |
| `layout_height` | Product of all leading dimensions and the tensor height dimension. |
| `layout_width` | The last tensor dimension. |
| `shard_shape` | Two-dimensional shard size `(shard_height, shard_width)`. |
| `shard_grid` | Number of shards along layout height and width. |
| `tile_shape` | Tile size inside each shard. |
| `tile_grid` | Number of tiles across the whole tensor buffer. |

For example, a tensor with shape `(N, H, W)` is flattened into layout shape `(N * H, W)`. Shards are then created over this two-dimensional layout.


## Shard Shape Selection

The `shard_shape` argument determines how large each memory shard is. It can be explicitly provided as `(shard_height, shard_width)`, as a single integer, or as one of the automatic modes.

| Mode | Description |
| --- | --- |
| `MCA_TensorBuffer.AUTO` | Uses the device MXU tile shape when IFM, OFM, and WGT tile shapes are identical. |
| `MCA_TensorBuffer.AUTO_IFM` | Selects shard shape based on `device.mxu_ifm_tile_shape`. |
| `MCA_TensorBuffer.AUTO_OFM` | Selects shard shape based on `device.mxu_ofm_tile_shape`. |
| `MCA_TensorBuffer.AUTO_WGT` | Selects shard shape based on `device.mxu_wgt_tile_shape`. |

For automatic modes, the selected shard dimension is the smallest divisor of the tensor dimension that is greater than or equal to the corresponding MXU tile dimension. If the tensor dimension is smaller than the MXU tile dimension, the tensor dimension itself is used.

Both layout height and layout width must be divisible by the selected shard height and shard width. This keeps shard boundaries aligned with the physical tensor layout.


## Sharding and Pointer Allocation

Each shard is assigned one `Pointer`. The pointer is allocated from the underlying `MCA_MemorySpace` by calling `allocate()`.

`MCA_TensorBuffer` supports three shard-to-owner mapping styles:

| Mapping style | Description |
| --- | --- |
| Round-robin mapping | Default behavior. Shards are assigned to `owner_ids` in round-robin order. |
| Contiguous mapping | Enabled with `contiguous_mapping=True`. Consecutive shards are assigned to the same owner until that owner's shard quota is filled. |
| Blocked mapping | Enabled with `blocked_mapping=True`. Only used for L1 memory spaces whose owners are an `MTA_CoreGrid`; shard `(h, w)` is mapped by grid position. |

`blocked_mapping` is useful when the tensor layout should follow a two-dimensional core grid. If blocked mapping is requested but the memory space is not an L1 memory space or the owners are not an `MTA_CoreGrid`, the buffer falls back to non-blocked mapping.


## Tiling

Shards can be subdivided into tiles with `tiling(tile_shape)`. Tiling does not allocate new memory; it only changes how later tile access APIs compute offsets, row sizes, strides, and padding.

Tile shape does not need to evenly divide shard shape. When a tile crosses a shard boundary, read/write argument helpers reflect the valid edge-tile size through `row_size` and `row_num`, and read helpers include zero-padding information in `dst_row_zero_pad`. This lets kernels read partial edge tiles into fixed-size tile containers.


## Constructor

| API | Description |
| --- | --- |
| `MCA_TensorBuffer(mem_space, shape, dtype, shard_shape=AUTO, blocked_mapping=False, contiguous_mapping=False)` | Creates a tensor buffer metadata object over a memory space. Allocation is not performed until `allocate()` is called. |

| Argument | Role |
| --- | --- |
| `mem_space` | `MCA_MemorySpace` used to allocate shard pointers. |
| `shape` | Logical tensor shape. One-dimensional shapes are internally promoted to two-dimensional layout. |
| `dtype` | Tensor element dtype. |
| `shard_shape` | Explicit shard shape or automatic shard selection mode. |
| `blocked_mapping` | Whether to map shards according to a two-dimensional `MTA_CoreGrid`. |
| `contiguous_mapping` | Whether to assign consecutive shards contiguously to each memory owner. |


## Public Methods

### Memory Requirement and Allocation

| API | Description |
| --- | --- |
| `check_memory_requirement(mem_space, shape, dtype, shard_shape=AUTO, blocked_mapping=False, contiguous_mapping=False)` | Static helper that checks whether the given memory space has enough capacity for the requested tensor buffer. |
| `required_mem_space_per_id()` | Returns the estimated memory required per owner ID under the default per-owner distribution model. |
| `check_mem_vacancy()` | Checks whether the current memory space has enough free capacity for this buffer. |
| `allocate()` | Allocates all shard pointers from the underlying memory space and returns `self`. |

| Argument | Role |
| --- | --- |
| `mem_space` | Memory space to test or allocate from. |
| `shape` | Logical tensor shape. |
| `dtype` | Tensor dtype. |
| `shard_shape` | Explicit or automatic shard shape. |
| `blocked_mapping` | Whether memory requirement should assume blocked grid mapping. |
| `contiguous_mapping` | Whether memory requirement should assume contiguous owner mapping. |

### Shape and Layout

| API | Description |
| --- | --- |
| `copy()` | Creates a shallow copy of the tensor buffer metadata and shard pointers. |
| `tiling(tile_shape=None)` | Updates tile shape used by tile access helper methods and returns `self`. |

| Argument | Role |
| --- | --- |
| `tile_shape` | Tile shape `(tile_height, tile_width)`; defaults to the full shard shape. |

<!-- > Note: In the current implementation, `reshape()` is intended to preserve the existing shard pointers, but the constructor call inside the method passes `shard_grid`, which is not a constructor argument. Treat this API as experimental until that implementation detail is corrected. -->

### Payload Movement

| API | Description |
| --- | --- |
| `update(tensor)` | Writes a PyTorch tensor into the allocated shard memory. |
| `restore()` | Restores the full PyTorch tensor from shard memory. |
| `get_raw_data(y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx)` | Reads one tile from the underlying memory and returns it as a tensor. |

| Argument | Role |
| --- | --- |
| `tensor` | PyTorch tensor whose shape must match the buffer shape. |
| `y_shard_idx` | Shard row index. |
| `x_shard_idx` | Shard column index. |
| `y_tile_in_shard_idx` | Tile row index inside the selected shard. |
| `x_tile_in_shard_idx` | Tile column index inside the selected shard. |

`update()` requires the buffer to be allocated first. In performance mode, payload data is not materialized and `restore()` returns a zero tensor with the correct shape and dtype.

### Pointer and Tile Access

| API | Description |
| --- | --- |
| `get_shard_ptr(y_shard_idx, x_shard_idx)` | Returns the base `Pointer` for a shard. |
| `get_tile_ptr_read_args(y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx)` | Returns pointer and row-copy arguments for reading a tile from a shard. |
| `get_tile_ptr_write_args(y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx)` | Returns pointer and row-copy arguments for writing a tile into a shard. |
| `get_shard_grid_from_tile_grid_idx(y_tile_idx, x_tile_idx)` | Converts global tile-grid indices into shard indices and tile-in-shard indices. |

| Argument | Role |
| --- | --- |
| `y_shard_idx` | Shard row index. |
| `x_shard_idx` | Shard column index. |
| `y_tile_in_shard_idx` | Tile row index inside a shard. |
| `x_tile_in_shard_idx` | Tile column index inside a shard. |
| `y_tile_idx` | Global tile row index across the whole tensor buffer. |
| `x_tile_idx` | Global tile column index across the whole tensor buffer. |

`get_tile_ptr_read_args()` returns `(src_ptr, row_size, row_num, src_row_stride, dst_row_stride, dst_row_zero_pad)`. These values can be passed to core memory-read APIs.

`get_tile_ptr_write_args()` returns `(dst_ptr, row_size, row_num, src_row_stride, dst_row_stride)`. These values can be passed to core memory-write APIs.


## Properties

| Property | Description |
| --- | --- |
| `shape` | Logical tensor shape. |
| `dtype` | Tensor dtype. |
| `numel` | Number of logical tensor elements. |
| `layout_shape` | Two-dimensional memory layout shape. |
| `shard_grid` | Number of shards along layout height and width. |
| `n_outer_shards` | Number of outer-dimension shard groups derived from leading dimensions. |
| `tile_grid` | Number of tiles across the full tensor layout. |
| `tile_grid_per_shard` | Number of tiles inside each shard. |
| `shard_shape` | Shard shape `(shard_height, shard_width)`. |
| `tile_shape` | Tile shape `(tile_height, tile_width)`. |
| `total_size` | Total tensor size in bytes. |
| `shard_size` | Size of one shard in bytes. |
| `tile_size` | Size of one full tile in bytes. |
| `n_tiles` | Total number of tiles in the buffer. |
| `mem_space` | Underlying `MCA_MemorySpace`. |
| `device` | Device that owns the memory space. |
| `mem_type` | Memory type of the underlying memory space. |
| `owner_ids` | Memory owner IDs used for allocation. |
| `is_allocated` | Whether shard pointers have been allocated. |


## Typical Usage Pattern

1. Create an `MCA_MemorySpace` from a device, such as `create_l1_mem_space()` or `create_main_mem_space()`.
2. Create an `MCA_TensorBuffer` with logical shape, dtype, and shard configuration.
3. Optionally call `tiling()` to define tile granularity.
4. Call `allocate()` to assign memory pointers for all shards.
5. Use `update()` and `restore()` for functional payload movement, or use tile pointer helper APIs when generating core memory commands.
