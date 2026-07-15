# Core APIs

This document summarizes the API-level roles of `NPUCore` and `DMACore`. It focuses on what each function is used for and what each argument represents. Detailed implementation behavior, scheduling semantics, and cycle-model internals are intentionally omitted.


## `NPUCore`

`NPUCore` is the programmable compute core in `neuromta.component`. It owns an L1 `MemoryHandle`, communicates with other cores through `IcntContext` or `BookSim2`, and executes MXU/VPU commands through `MXUContext` and `VPUContext`.

### Constructor

| Function | Description |
| --- | --- |
| `NPUCore(core_id, global_context, icnt_context=None, vpu_config=VPUConfig(), mxu_config=MXUConfig())` | Creates an NPU core, binds it to the global hardware context, and initializes its local memory, MXU, and VPU contexts. |

| Argument | Role |
| --- | --- |
| `core_id` | Unique core identifier used by the runtime and interconnect model. |
| `global_context` | `GlobalContext` object that provides core metadata, memory ownership, and device-level configuration. |
| `icnt_context` | Optional `IcntContext` used for inter-core communication modeling. |
| `vpu_config` | `VPUConfig` used to create the core-local `VPUContext`. |
| `mxu_config` | `MXUConfig` used to create the core-local `MXUContext`. |

### Utility Methods

| Function | Description |
| --- | --- |
| `get_buffer_owner(ptr)` | Returns the core ID that owns the memory region addressed by `ptr`. |
| `check_ptr_belonging(ptr)` | Checks whether `ptr` belongs to the local memory region of the current core. |

| Argument | Role |
| --- | --- |
| `ptr` | `Pointer` or integer address to inspect. |


## `DMACore`

`DMACore` is the memory-side core in `neuromta.component`. It owns a DRAM memory bank, handles remote DRAM read/write requests, and can use either the lightweight `MemorySimulator` or an external `DRAMSim3` companion.

### Constructor

| Function | Description |
| --- | --- |
| `DMACore(core_id, global_context, icnt_context)` | Creates a DMA core, binds it to a DRAM memory region, and connects it to the interconnect context. |

| Argument | Role |
| --- | --- |
| `core_id` | Unique DMA core identifier used by the runtime and interconnect model. |
| `global_context` | `GlobalContext` object that provides DMA core metadata and DRAM memory ownership. |
| `icnt_context` | `IcntContext` used to model NPU-to-DMA data transfers. |

### Utility Methods

| Function | Description |
| --- | --- |
| `check_ptr_belonging(ptr)` | Checks whether `ptr` belongs to the DRAM memory region owned by the current DMA core. |

| Argument | Role |
| --- | --- |
| `ptr` | `Pointer` to inspect. |


## Common Memory APIs

The following APIs are implemented by both `NPUCore` and `DMACore`. For `NPUCore`, they operate on local L1 memory. For `DMACore`, they operate on the DRAM bank owned by the DMA core.

### Data Container and Local Memory

| Function | Description |
| --- | --- |
| `local_data_container_init(container, shape, dtype)` | Initializes a `DataContainer` with a zero tensor or metadata for performance-only simulation. |
| `local_mem_init(ptr, size, init_data=None)` | Initializes a local memory region at `ptr`. |
| `local_mem_page_read(ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0, cont_row_zero_pad=0)` | Reads one or more memory rows from local memory into a `DataContainer`. |
| `local_mem_page_write(ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0)` | Writes one or more rows from a `DataContainer` into local memory. |

| Argument | Role |
| --- | --- |
| `container` | `DataContainer[torch.Tensor]` used as the temporary source or destination for data movement. |
| `shape` | Tensor shape used when initializing a `DataContainer`. |
| `dtype` | Tensor dtype used when initializing a `DataContainer`. |
| `ptr` | `Pointer` to the target local memory address. |
| `size` | Number of bytes to initialize or copy. |
| `init_data` | Optional tensor data used to initialize memory; zero-filled data is used when omitted. |
| `row_size` | Number of bytes copied per row. |
| `row_num` | Number of rows to read or write. |
| `mem_row_stride` | Byte stride between source or destination rows in memory; defaults to `row_size`. |
| `cont_row_stride` | Byte stride between rows inside `container`; defaults to `row_size`. |
| `row_pattern` | Optional mapping from container row index to memory row index for non-contiguous row access. |
| `cont_row_offset` | Byte offset inside each container row where copied data begins. |
| `cont_row_zero_pad` | Number of zero bytes appended after each copied row when reading into `container`. |

### Static Memory Space

| Function | Description |
| --- | --- |
| `allocate_static_mem_space(ptr, size)` | Reserves a static memory range in the core-owned `MemoryHandle`. |
| `deallocate_static_mem_space(ptr)` | Releases a previously allocated static memory range. |

| Argument | Role |
| --- | --- |
| `ptr` | Base `Pointer` of the static memory region. |
| `size` | Size of the static memory region in bytes. |


## `NPUCore` Memory Transfer APIs

The following methods are available on `NPUCore` and are usually used by compiled kernels to move data between L1, DRAM, and FIFO buffers.

| Function | Description |
| --- | --- |
| `mem_init(ptr, size, init_data=None)` | Initializes a memory region, dispatching to the owning core when `ptr` is remote. |
| `remote_mem_page_read(dst_core_id, ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0, cont_row_zero_pad=0)` | Reads local memory and models transfer from this core to `dst_core_id`. |
| `remote_mem_page_write(src_core_id, ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0)` | Models transfer from `src_core_id` and writes the received data into local memory. |
| `mem_copy(dst_ptr, src_ptr, row_size, row_num=1, src_row_stride=None, dst_row_stride=None, dst_row_zero_pad=0)` | Copies data between two memory regions when at least one side belongs to the current core. |
| `mem_copy_to_fifo(ptr, fifo_handle, entry_id, size=None, ref_count=1)` | Copies a memory region into a FIFO entry after waiting for vacancy, then pushes the FIFO entry. |
| `mem_copy_from_fifo(ptr, fifo_handle, entry_id, size=None)` | Waits for a valid FIFO entry, copies it into memory, then pops the FIFO entry. |
| `mem_read(ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0, cont_row_zero_pad=0)` | Reads from local or remote memory into `container`. |
| `mem_write(ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0)` | Writes `container` data into local or remote memory. |

| Argument | Role |
| --- | --- |
| `dst_core_id` | Core ID that receives data during a remote read. |
| `src_core_id` | Core ID that sends data during a remote write. |
| `dst_ptr` | Destination `Pointer` for `mem_copy`. |
| `src_ptr` | Source `Pointer` for `mem_copy`. |
| `ptr` | Source or destination `Pointer` for the memory operation. |
| `container` | `DataContainer[torch.Tensor]` used to temporarily hold transferred data. |
| `row_size` | Number of bytes copied per row. |
| `row_num` | Number of rows to copy. |
| `src_row_stride` | Byte stride between source rows. |
| `dst_row_stride` | Byte stride between destination rows. |
| `dst_row_zero_pad` | Number of zero bytes appended per copied row before writing to the destination layout. |
| `mem_row_stride` | Byte stride between memory rows. |
| `cont_row_stride` | Byte stride between container rows. |
| `row_pattern` | Optional row remapping used for sparse or reordered row copies. |
| `cont_row_offset` | Byte offset inside each container row. |
| `cont_row_zero_pad` | Zero padding inserted after each row when reading. |
| `fifo_handle` | `FIFOBufferHandle` that owns the FIFO entry storage and synchronization state. |
| `entry_id` | FIFO entry index, either an integer or `VariableHandle`. |
| `size` | Number of bytes copied through the FIFO; defaults to `fifo_handle.entry_size`. |
| `ref_count` | Number of consumers expected to pop the pushed FIFO entry. |
| `init_data` | Optional initial tensor data. |


## `NPUCore` MXU APIs

MXU APIs model matrix-oriented execution on the NPU core.

| Function | Description |
| --- | --- |
| `mxu_reconfigure(dtype, acc_dtype)` | Reconfigures MXU input and accumulator dtypes. |
| `mxu_load_context(psum_cont)` | Loads partial-sum data from a container into MXU accumulator state. |
| `mxu_store_context(psum_cont)` | Stores MXU accumulator state into a container. |
| `mxu_tiled_gemm(ifm_cont, wgt_cont, psum_cont, ofm_cont, preload_psum=False, flush_ofm=False, ifm_transposed=False, wgt_transposed=False, psum_vectored=False)` | Executes one tiled GEMM operation on MXU inputs. |
| `mxu_tiled_maxpool(ifm_cont, psum_cont, ofm_cont, preload_psum=False, flush_ofm=False, ifm_transposed=False)` | Executes one tiled max-pooling style update using the MXU context. |
| `mxu_tiled_elemwise(op, src, dst, preload_psum=False, flush_ofm=False, ifm_transposed=False)` | Executes one tiled elementwise MXU operation. |
| `mxu_tiled_elemwise_imm(op, imm, dst, flush_ofm=False)` | Executes one tiled elementwise MXU operation with an immediate value. |

| Argument | Role |
| --- | --- |
| `dtype` | MXU input data type. |
| `acc_dtype` | MXU accumulator data type. |
| `psum_cont` | Container holding partial sums to load or store. |
| `ifm_cont` | Container holding input feature-map tile data. |
| `wgt_cont` | Container holding weight tile data. |
| `ofm_cont` | Container that receives output feature-map tile data when `flush_ofm` is enabled. |
| `preload_psum` | Whether to load partial-sum data before execution. |
| `flush_ofm` | Whether to write MXU output state to `ofm_cont` or `dst`. |
| `ifm_transposed` | Whether to interpret the input tile as transposed. |
| `wgt_transposed` | Whether to interpret the weight tile as transposed. |
| `psum_vectored` | Whether to interpret partial sums as a vector-shaped tile. |
| `op` | `MXUElementwiseOp` describing the elementwise operation. |
| `src` | Source container for elementwise MXU input. |
| `dst` | Destination container for elementwise MXU output. |
| `imm` | Immediate scalar value used by `mxu_tiled_elemwise_imm`. |


## `NPUCore` VPU APIs

VPU APIs model vector register execution on the NPU core.

| Function | Description |
| --- | --- |
| `vpu_reconfigure(vlen, vdtype)` | Reconfigures vector register length and dtype. |
| `vpu_load_reg(data_cont, vreg_idx, burst_len=1, offset=0)` | Loads one or more vector registers from a data container. |
| `vpu_store_reg(data_cont, vreg_idx, burst_len=1, offset=0)` | Stores one or more vector registers into a data container. |
| `vpu_execute(opcode, vreg_a, vreg_b=None, vreg_dest=None, inplace=False, burst_len=1)` | Executes a vector operation over one or more vector registers. |

| Argument | Role |
| --- | --- |
| `vlen` | Number of elements in one vector register. |
| `vdtype` | Data type of vector register elements. |
| `data_cont` | `DataContainer[torch.Tensor]` used as vector load/store source or destination. |
| `vreg_idx` | First vector register index used by load/store. |
| `burst_len` | Number of consecutive vector registers or operations. |
| `offset` | Element offset inside `data_cont`. |
| `opcode` | `VPUOperator` identifying the vector operation. |
| `vreg_a` | First source vector register index. |
| `vreg_b` | Optional second source vector register index. |
| `vreg_dest` | Optional destination vector register index. |
| `inplace` | Whether the operation updates a source register in place. |


## `DMACore` Remote Memory APIs

`DMACore` exposes remote memory APIs for DRAM access. These methods combine local DRAM-bank memory operations with interconnect transfer modeling and optional DRAM timing simulation.

| Function | Description |
| --- | --- |
| `remote_mem_page_read(dst_core_id, ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0, cont_row_zero_pad=0)` | Reads a DRAM page into `container`, models DRAM latency, and models transfer to `dst_core_id`. |
| `remote_mem_page_write(src_core_id, ptr, container, row_size, row_num=1, mem_row_stride=None, cont_row_stride=None, row_pattern=None, cont_row_offset=0)` | Models transfer from `src_core_id`, models DRAM latency, and writes `container` data into DRAM. |

| Argument | Role |
| --- | --- |
| `dst_core_id` | Destination core ID that receives read data from the DMA core. |
| `src_core_id` | Source core ID that sends write data to the DMA core. |
| `ptr` | DRAM `Pointer` owned by the DMA core. |
| `container` | `DataContainer[torch.Tensor]` used to hold read or write data. |
| `row_size` | Number of bytes copied per row. |
| `row_num` | Number of rows to access. |
| `mem_row_stride` | Byte stride between DRAM rows. |
| `cont_row_stride` | Byte stride between container rows. |
| `row_pattern` | Optional mapping from container row index to memory row index. |
| `cont_row_offset` | Byte offset inside each container row. |
| `cont_row_zero_pad` | Number of zero bytes appended after each row when reading. |


## Internal Cycle-Model Hooks

The following methods are command hooks used by the cycle models. They are generally invoked by higher-level memory APIs rather than called directly from user code.

| Function | Owner | Description |
| --- | --- | --- |
| `_icnt_data_transfer_handle(src_core_id, dst_core_id, data_size, is_write)` | `NPUCore`, `DMACore` | Accounts for interconnect transfer latency between two cores. |
| `_dma_lightweight_request_handle(addr, size, is_write)` | `DMACore` | Accounts for lightweight DRAM request latency through `MemorySimulator`. |

| Argument | Role |
| --- | --- |
| `src_core_id` | Source core ID of the interconnect transfer. |
| `dst_core_id` | Destination core ID of the interconnect transfer. |
| `data_size` | Transfer size in bytes. |
| `addr` | DRAM byte address for the lightweight memory request. |
| `size` | Request size in bytes. |
| `is_write` | Whether the modeled request is a write. |
