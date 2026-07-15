# Operator Signature APIs

This document explains how to create `MCA_OperatorSignature` objects and how they are converted into `MCA_CompiledOperator` objects. It does not document every method in `srcs/neuromta/component/implementation/operator.py`; instead, it focuses on the APIs involved in operator signature construction.


## Role of `MCA_OperatorSignature`

`MCA_OperatorSignature` is the compiler-facing description of one operator. It records the operator type, kernel template, tensor buffers, tile signatures, tiled operator signatures, and operator-level keyword arguments.

An operator signature is not an executable program by itself. It is first created by an operator helper such as `MCA_OP_LINEAR`, populated by a mapper such as `MCA_MAPPER_LINEAR`, then lowered by `MCA_OperatorGraphCompiler` into `MCA_CompiledOperator`.


## Operator Helper Pattern

Common operator helpers are defined in `srcs/neuromta/system/software/common/operator.py`. Each helper follows the same high-level pattern:

1. Create an `MCA_OperatorSignature`.
2. Configure tensor tiling based on device MXU tile sizes.
3. Register input, parameter, and output buffers with `add_buffer()`.
4. Store operator-specific attributes in `global_kwargs` if needed.
5. Call a mapper that creates tiled operations and uops.
6. Return the populated `MCA_OperatorSignature`.

For example, `MCA_OP_LINEAR` creates a linear operator signature, registers `ifm`, `wgt`, `bias`, and `ofm`, then calls `MCA_MAPPER_LINEAR(op_sig)`.

```python
op_sig = MCA_OperatorSignature(
    op_type="LINEAR",
    kernel_template=common_kernel_lib.MCA_KERNEL_TILED_LINEAR(),
)

m_tile = ifm.mem_space.device.mxu_config.m_tile
k_tile = ifm.mem_space.device.mxu_config.k_tile
n_tile = ofm.mem_space.device.mxu_config.n_tile

op_sig.add_buffer("ifm", ifm.tiling((m_tile, k_tile)), is_input=True)
op_sig.add_buffer("wgt", wgt.tiling((n_tile, k_tile)), is_param=True)
op_sig.add_buffer("bias", bias.tiling((1, n_tile)), is_param=True)
op_sig.add_buffer("ofm", ofm.tiling((m_tile, n_tile)), is_output=True)

op_sig = common_mapping_lib.MCA_MAPPER_LINEAR(op_sig)
```

`MCA_OP_CONV2D` follows the same structure, but stores convolution attributes in `global_kwargs`.

```python
op_sig.global_kwargs["stride"] = stride
op_sig.global_kwargs["padding"] = padding
op_sig.global_kwargs["dilation"] = dilation
op_sig.global_kwargs["groups"] = groups

op_sig = common_mapping_lib.MCA_MAPPER_CONV2D(op_sig, is_conv2d=True)
```


## `mca_operator_method`

Operator helper functions in the common library are decorated with `mca_operator_method`.

| API | Description |
| --- | --- |
| `mca_operator_method(func)` | Marks a function as an MCA operator helper and checks that the function returns `MCA_OperatorSignature`. |
| `mca_operator_method_check(func)` | Returns whether a function was marked by `mca_operator_method`. |

| Argument | Role |
| --- | --- |
| `func` | Python callable that creates and returns an operator signature. |

This decorator does not build the operator by itself. It is a lightweight validation and tagging mechanism used by higher-level software layers.


## Creating `MCA_OperatorSignature`

### Constructor

| API | Description |
| --- | --- |
| `MCA_OperatorSignature(op_type, kernel_template)` | Creates an empty operator signature for one logical operator. |

| Argument | Role |
| --- | --- |
| `op_type` | String identifier for the logical operator type, such as `"LINEAR"` or `"CONV2D"`. |
| `kernel_template` | `MCA_KernelTemplate` used later to convert compiled IR stages into load, execute, store, and barrier kernels. |

The constructor initializes empty dictionaries for buffers and tiles. The initial `op_id` is set to `op_type`, but `MCA_OperatorGraphCompiler.add_op()` later assigns a unique ID such as `LINEAR_1`.

### Buffer Registration

| API | Description |
| --- | --- |
| `add_buffer(buf_name, buffer, is_input=False, is_output=False, is_param=False)` | Registers a tensor buffer and creates `TileSignature` objects for every tile in that buffer. |

| Argument | Role |
| --- | --- |
| `buf_name` | Logical buffer name used by mappers and kernel templates, such as `"ifm"`, `"wgt"`, `"bias"`, or `"ofm"`. |
| `buffer` | `MCA_TensorBuffer` registered for the operator; can be `None` for optional buffers such as missing bias. |
| `is_input` | Marks the buffer as an input buffer. |
| `is_output` | Marks the buffer as the output buffer. Only one output buffer is currently supported. |
| `is_param` | Marks the buffer as a parameter. Parameter buffers are also treated as input buffers. |

When a non-`None` buffer is registered, `add_buffer()` iterates over the buffer's `shard_grid` and `tile_grid_per_shard` and creates `TileSignature` entries in `op_sig.tiles[buf_name]`.

### Operator-Level Keyword Arguments

`global_kwargs` is a dictionary for attributes shared by all tiled operations of an operator. It is commonly used by convolution and pooling operators.

| Usage | Description |
| --- | --- |
| `op_sig.global_kwargs["stride"] = stride` | Stores stride information for mapper and kernel use. |
| `op_sig.global_kwargs["padding"] = padding` | Stores padding information. |
| `op_sig.global_kwargs["dilation"] = dilation` | Stores dilation information. |
| `op_sig.global_kwargs["groups"] = groups` | Stores grouped convolution metadata. |
| `op_sig.global_kwargs["window"] = window` | Stores pooling window metadata. |

| API | Description |
| --- | --- |
| `update_global_kwargs(op_kwargs)` | Updates `global_kwargs` with values from `op_kwargs`. |

| Argument | Role |
| --- | --- |
| `op_kwargs` | Dictionary of operator-level keyword arguments. |


## Mapping to Tiled Operations

After buffers are registered, a mapper converts tile signatures into `TiledOperatorSignature` objects. This is where the operator's mathematical dependency structure is expressed.

The common mapper library provides examples:

| Mapper | Role |
| --- | --- |
| `MCA_MAPPER_LINEAR(op_sig)` | Creates tiled GEMM-style uops for linear operators. |
| `MCA_MAPPER_UNARY(op_sig)` | Creates one-input one-output tiled uops for unary operators such as ReLU. |
| `MCA_MAPPER_CONV2D(op_sig, is_conv2d=True)` | Creates tiled uops for convolution or pooling-style operators. |

### Tiled Operator APIs

| API | Description |
| --- | --- |
| `new_tiled_op()` | Creates a new `TiledOperatorSignature` and appends it to the operator signature. |
| `get_tiled_op(tiled_op_id)` | Returns a previously created tiled operator by ID. |
| `TiledOperatorSignature.add_uop(i_tiles, o_tile, op_kwargs=None)` | Adds one uop dependency to a tiled operator. |

| Argument | Role |
| --- | --- |
| `tiled_op_id` | Integer ID of a tiled operator inside `op_sig.tiled_ops`. |
| `i_tiles` | List of input `TileSignature` objects consumed by one uop. |
| `o_tile` | Output `TileSignature` produced by the tiled operator. |
| `op_kwargs` | Optional uop-level keyword arguments, such as whether bias is used. |

In the linear mapper, each OFM tile creates one `TiledOperatorSignature`. Multiple uops are then added to that tiled operator, one per reduction tile in the K dimension. Each uop consumes IFM/WGT tiles, optionally consumes bias, and accumulates into the same OFM tile.


## Creating a Mapper

A mapper is a function that takes a partially constructed `MCA_OperatorSignature` and returns the same signature after filling `op_sig.tiled_ops`. It should validate tensor shapes, shard grids, and tile shapes, then describe the operator as a set of tiled output dependencies.

The common mapper file `srcs/neuromta/system/software/common/mapping.py` contains useful patterns:

| Mapper | Main pattern |
| --- | --- |
| `MCA_MAPPER_LINEAR(op_sig)` | Iterates over OFM tiles and creates reduction uops over IFM/WGT K tiles. |
| `MCA_MAPPER_UNARY(op_sig)` | Creates one tiled op per IFM/OFM tile pair. |
| `MCA_MAPPER_CONV2D(op_sig, is_conv2d=True)` | Computes convolution or pooling tile dependencies, including halo and row-copy patterns. |

### Mapper Structure

Most mappers follow this structure:

1. Read registered buffers from `op_sig.buffers`.
2. Validate logical tensor shape compatibility.
3. Validate `shard_grid` and `tile_shape` compatibility.
4. Iterate over output tile coordinates.
5. Create one `TiledOperatorSignature` with `op_sig.new_tiled_op()`.
6. Add one or more uops with `tiled_op.add_uop()`.
7. Return `op_sig`.

For a simple unary operator, the mapper can be as small as:

```python
def MCA_MAPPER_MY_UNARY(op_sig: MCA_OperatorSignature) -> MCA_OperatorSignature:
    ifm = op_sig.buffers["ifm"]

    for y_s in range(ifm.shard_grid[0]):
        for x_s in range(ifm.shard_grid[1]):
            for y_t in range(ifm.tile_grid_per_shard[0]):
                for x_t in range(ifm.tile_grid_per_shard[1]):
                    tiled_op = op_sig.new_tiled_op()
                    tiled_op.add_uop(
                        i_tiles=[op_sig.tiles["ifm"][(y_s, x_s, y_t, x_t)]],
                        o_tile=op_sig.tiles["ofm"][(y_s, x_s, y_t, x_t)],
                    )

    return op_sig
```

### Mapper APIs

| API | Description |
| --- | --- |
| `op_sig.buffers[name]` | Returns the `MCA_TensorBuffer` registered under `name`. |
| `op_sig.tiles[name][coords]` | Returns the `TileSignature` for a buffer tile. |
| `op_sig.new_tiled_op()` | Creates one tiled operator, usually corresponding to one output tile. |
| `tiled_op.add_uop(i_tiles, o_tile, op_kwargs=None)` | Adds one executable micro-operation dependency to the tiled operator. |
| `op_sig.global_kwargs` | Provides operator-level attributes such as stride, padding, dilation, groups, or pooling window. |

| Argument | Role |
| --- | --- |
| `name` | Buffer name registered through `add_buffer()`, such as `"ifm"` or `"ofm"`. |
| `coords` | Tile coordinate tuple `(y_shard_idx, x_shard_idx, y_tile_in_shard_idx, x_tile_in_shard_idx)`. |
| `i_tiles` | Input tile signatures consumed by one uop. |
| `o_tile` | Output tile signature produced by the tiled operator. |
| `op_kwargs` | Uop-level metadata consumed later by the kernel template. |

### `op_kwargs` and Kernel Cooperation

`op_kwargs` is how a mapper passes uop-specific metadata to the kernel template. For example:

| Example key | Used by | Meaning |
| --- | --- | --- |
| `"use_bias"` | linear and conv kernels | Whether the current uop should load bias or initial partial sum. |
| `"ifm_tile_count"` | conv and pooling kernels | Number of IFM tile references at the beginning of `i_tiles`. |
| `"memcpy_pattern"` | conv and pooling kernels | Row-copy pattern used to assemble halo or pooling windows. |

When creating a new mapper, define `op_kwargs` only for information that cannot be inferred from `TileSignature` alone.


## Creating a Kernel Template

A kernel template explains how compiled IR is executed on an `NPUCore`. The base `MCA_KernelTemplate` in `srcs/neuromta/component/implementation/kernel.py` already implements load thread, store thread, barrier, reference reads, and reference writes. Most custom templates only need to implement `EXE_UOP()`.

Common examples are defined in `srcs/neuromta/system/software/common/kernel.py`.

| Template | Main behavior |
| --- | --- |
| `MCA_KERNEL_TILED_LINEAR` | Reads IFM/WGT/bias tile refs, runs `core.mxu_tiled_gemm()`, and writes the final OFM tile. |
| `MCA_KERNEL_TILED_RELU` | Reads one tile, executes VPU ReLU, and writes one tile. |
| `MCA_KERNEL_MERGED_LINEAR_RELU` | Runs tiled GEMM and applies VPU ReLU before writing the output tile. |
| `MCA_KERNEL_TILED_CONV2D` | Assembles IFM tiles using mapper-provided row patterns, then runs tiled GEMM. |
| `MCA_KERNEL_TILED_MAXPOOL2D` | Reads IFM window tiles and updates MXU max-pool state. |
| `MCA_KERNEL_TILED_AVGPOOL2D` | Accumulates IFM window tiles and divides by the number of uops at output flush. |
| `MCA_KERNEL_DIRECT_COPY` | Reads one tile reference and writes it to the output reference. |

### Kernel Template Structure

A custom kernel template typically subclasses `MCA_KernelTemplate` or a local base such as `_MCA_KERNEL_BASE`.

```python
class MCA_KERNEL_MY_UNARY(MCA_KernelTemplate):
    @classmethod
    def EXE_UOP(cls, core, env, ir):
        ifm = cls.read_from_ref(core, env, ir.i_tile_refs[0])
        ofm = DataContainer(
            shape=ir.o_tile_ref.tile_sig.tile_shape,
            dtype=ir.o_tile_ref.tile_sig.dtype,
        )

        # Execute core commands here.
        # For example, configure VPU, load registers, execute, and store.

        cls.write_to_ref(core, env, ofm, ir.o_tile_ref)
```

### Kernel Template APIs

| API | Description |
| --- | --- |
| `get_ld_thread_kernel(core, env, ir_seq)` | Creates the load-thread `KernelPrototype` for a compiled IR sequence. |
| `get_ex_thread_kernel(core, env, ir_seq)` | Creates the execute-thread `KernelPrototype`. |
| `get_st_thread_kernel(core, env, ir_seq)` | Creates the store-thread `KernelPrototype`. |
| `get_barrier_kernel(core, env, barrier)` | Creates a barrier `KernelPrototype`. |
| `read_from_ref(core, env, ref, row_pattern=None, inplace_container=None, fifo_sync=True)` | Reads a `MCA_CompiledOperator.IR.Reference` into a `DataContainer`. |
| `write_to_ref(core, env, container, ref, row_pattern=None, fifo_sync=True)` | Writes a `DataContainer` into a compiled IR reference. |
| `EXE_UOP(core, env, ir)` | Executes one compiled uop; custom operator templates usually override this. |
| `EXE_CTX_LOAD(core, env, ir)` | Loads saved MXU context; common `_MCA_KERNEL_BASE` provides an implementation. |
| `EXE_CTX_STORE(core, env, ir)` | Stores MXU context; common `_MCA_KERNEL_BASE` provides an implementation. |

| Argument | Role |
| --- | --- |
| `core` | `NPUCore` executing the kernel. |
| `env` | Compiler environment containing buffers, variables, FIFO handles, and operator metadata. |
| `ir_seq` | List of compiled IR objects assigned to one thread. |
| `barrier` | Tuple describing global barrier variables and participant count. |
| `ref` | Compiled IR reference to a tensor buffer, SPM slot, or FIFO slot. |
| `container` | `DataContainer` used as source or destination for data movement. |
| `row_pattern` | Optional row mapping used for halo, pooling, or irregular tile reads. |
| `inplace_container` | Existing container reused for reads to reduce temporary allocation. |
| `fifo_sync` | Whether `read_from_ref()` or `write_to_ref()` should perform FIFO wait/push/pop itself. |
| `ir` | `MCA_CompiledOperator.IR.EXE_UOP`, `EXE_CTX_LOAD`, or `EXE_CTX_STORE` object. |

### Using `ir` inside `EXE_UOP`

`EXE_UOP()` receives an `MCA_CompiledOperator.IR.EXE_UOP` object generated from the mapper and compiler schedule.

| IR field | Meaning |
| --- | --- |
| `ir.op_id` | Operator ID in the compiler environment. |
| `ir.tiled_op_idx` | Index of the tiled operator inside `op_sig.tiled_ops`. |
| `ir.uop_idx` | Index of the uop inside the tiled operator. |
| `ir.i_tile_refs` | Runtime references for input tiles, already scheduled by the compiler. |
| `ir.o_tile_ref` | Runtime reference for the output tile, or `None` for non-output uops. |
| `ir.dtype` | Input dtype inferred from input tiles. |
| `ir.acc_dtype` | Accumulator/output dtype inferred from the output tile. |

The kernel can retrieve mapper-provided metadata with:

```python
op_sig = env.op_meta[ir.op_id].op_sig
tiled_op = op_sig.tiled_ops[ir.tiled_op_idx]
uop_kwargs = tiled_op.op_kwargs[ir.uop_idx]
```

This is the main contract between mapper and kernel template: the mapper creates tile dependencies and `op_kwargs`; the kernel interprets them and emits `NPUCore` commands.


## Adding a New Operator

To add a new operator to the common software layer:

1. Create a new `MCA_KERNEL_*` class and implement `EXE_UOP()`.
2. Create a new `MCA_MAPPER_*` function that validates tensor layout and fills `op_sig.tiled_ops`.
3. Create a new `MCA_OP_*` helper decorated with `mca_operator_method`.
4. Inside the helper, create `MCA_OperatorSignature`, call `add_buffer()`, set `global_kwargs` if needed, and call the mapper.
5. Use `MCA_OperatorGraphCompiler.add_op()` and `compile()` to lower the signature into an `MCA_CompiledProgram`.

The important design rule is that the mapper should only describe tile dependencies and per-uop metadata, while the kernel template should describe how those dependencies are executed on core APIs.


## Useful `MCA_OperatorSignature` Properties

| Property | Description |
| --- | --- |
| `op_type` | Logical operator type string. |
| `buffers` | Mapping from buffer names to `MCA_TensorBuffer` objects. |
| `tiles` | Mapping from buffer names and tile coordinates to `TileSignature` objects. |
| `tiled_ops` | List of `TiledOperatorSignature` objects created by the mapper. |
| `buffer_names` | Ordered list of registered buffer names. |
| `input_buffer_names` | Names of buffers marked as inputs. |
| `param_buffer_names` | Names of buffers marked as parameters. |
| `output_buffer_name` | Name of the output buffer. |
| `core_group` | Core group assigned during compilation. |
| `is_core_group_initialized` | Whether a compile-time core group has been assigned. |
| `total_buffer_size` | Sum of registered buffer sizes. |
| `total_n_uops` | Total number of uops across all tiled operators. |
| `total_arithmetic_intensity` | `total_n_uops / total_buffer_size`, used as a rough compile-time metric. |


## Buffer Renaming During Compilation

`MCA_OperatorGraphCompiler.Environment.add_op_sig()` renames buffers before compilation. This is done so that multiple operators can share the same tensor buffer object while still keeping unique names in the compiled environment.

| API | Description |
| --- | --- |
| `rename_buffers(rename_map)` | Renames registered buffers, tile signatures, tiled-op references, and buffer-name lists. |

| Argument | Role |
| --- | --- |
| `rename_map` | Mapping from old logical buffer names to new environment-level buffer names. |

Users usually do not call `rename_buffers()` directly when creating an operator helper. It is part of the graph compiler environment setup.


## Compiling Operator Signatures

`MCA_OperatorGraphCompiler` converts operator signatures into compiled operators.

### Compiler APIs

| API | Description |
| --- | --- |
| `MCA_OperatorGraphCompiler()` | Creates an empty operator graph compiler. |
| `add_op(op_sig)` | Adds an `MCA_OperatorSignature` and assigns a unique `op_id`. |
| `clear_ops()` | Removes all registered operator signatures from the compiler. |
| `compile(recipe)` | Compiles all registered operator signatures into an `MCA_CompiledProgram`. |

| Argument | Role |
| --- | --- |
| `op_sig` | Operator signature returned by an operator helper or mapper. |
| `recipe` | `MCA_OperatorGraphCompiler.CompileRecipe` describing target device, core groups, scratchpad size, FIFO depth, and reuse policy. |

### Compile Recipe

| API | Description |
| --- | --- |
| `CompileRecipe(device, core_groups, spad_space_size_per_core, context_buffer_slot_num=16, fifo_buffer_slot_num=16, temporal_reuse_target=ReuseTarget.ALL, spatial_reuse_target=ReuseTarget.SINGLE_MAIN)` | Describes hardware resources and reuse policies used during operator compilation. |

| Argument | Role |
| --- | --- |
| `device` | `MCA_DeviceBase` or `MTA_DeviceBase` used as the compilation target. |
| `core_groups` | List of `MCA_CoreGroup` or `MTA_CoreGrid` objects available for mapping. |
| `spad_space_size_per_core` | L1 scratchpad space reserved per core for compiled operator execution. |
| `context_buffer_slot_num` | Number of context slots used for partial output/context buffering. |
| `fifo_buffer_slot_num` | Number of FIFO slots used for load-execute, execute-store, and spatial reuse channels. |
| `temporal_reuse_target` | Selects which buffers are considered for cache-based temporal reuse. |
| `spatial_reuse_target` | Selects which buffers are considered for FIFO-based spatial reuse. |


## From Signature to `MCA_CompiledOperator`

The compilation flow is:

1. User code or a common operator helper creates `MCA_OperatorSignature`.
2. Buffers are registered with `add_buffer()`, and operator attributes are stored in `global_kwargs`.
3. A mapper creates `TiledOperatorSignature` objects and uops through `new_tiled_op()` and `add_uop()`.
4. `MCA_OperatorGraphCompiler.add_op()` registers the signature and assigns a unique `op_id`.
5. `MCA_OperatorGraphCompiler.compile(recipe)` creates an environment and freezes operator metadata.
6. Operator metadata creates per-core thread mappings, including cache schedule and spatial reuse schedule.
7. `compile_grouped_target_ops()` lowers thread mappings into `MCA_CompiledOperator` IR stages.
8. Each `MCA_CompiledOperator` contains load, execute, and store IR sequences that can later be dispatched through its kernel template.

The resulting `MCA_CompiledProgram` owns the compiler environment and the compiled operators internally. At this point, the operator signature has been transformed from a high-level tiled dependency description into stage-based IR suitable for kernel generation.


## Minimal Example

The following sketch shows the intended usage pattern. It omits device and tensor-buffer construction details.

```python
# 1. Create an operator signature through the common operator helper.
op_sig = MCA_OP_LINEAR(ifm, wgt, bias, ofm)

# 2. Create a compiler and register the operator.
compiler = MCA_OperatorGraphCompiler()
op_id = compiler.add_op(op_sig)

# 3. Create a compile recipe.
recipe = MCA_OperatorGraphCompiler.CompileRecipe(
    device=device,
    core_groups=[device.get_npu_core_group()],
    spad_space_size_per_core=64 * 1024,
)

# 4. Compile signatures into compiled operators.
program = compiler.compile(recipe)

# 5. Inspect or dispatch the compiled program through public APIs.
summary = program.summary()
program.dispatch()
```

In common code, users usually call helpers such as `MCA_OP_LINEAR`, `MCA_OP_RELU`, `MCA_OP_CONV2D`, or pooling variants instead of manually constructing every tile dependency.
