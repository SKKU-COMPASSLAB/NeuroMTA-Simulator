# Mesh Accelerator Software API

## Overview

Import the public API with `from neuromta.system.software import *`. Tensor creation and kernel-producing operations must run inside both a `MeshDeviceRuntimeContext` and an active compiler context. View and placement-preference operations do not emit kernels.

| Module | Role |
| --- | --- |
| `common.py` | Opens the Mesh device runtime, compiles and submits workloads, selects the scheduler, runs simulation, and exposes shared descriptors, enums, and state types. |
| `tensor.py` | Creates tensor descriptors, changes placement preferences, builds zero-copy views, materializes copies, and provides tensor layout/data-movement operations. |
| `blas.py` | Builds linear, matrix, elementwise, activation, reduction, normalization, and rotary-position operators from the base Mesh kernels. |
| `kv_cache.py` | Creates fixed-capacity, persistent per-layer KV caches and provides reservation, append, read/view, reset, and deallocation operations. |

| Basic workflow | Usage |
| --- | --- |
| Open the runtime | `with MeshDeviceRuntimeContext(device, (32, 32), torch.bfloat16) as context:` |
| Build and submit a workload | `with context.new_compiler_context(workload_id="model"):` then create tensors and call operators. The workload is compiled and submitted when the block exits successfully. |
| Execute submitted workloads | `jobs = context.run()` |
| Release persistent memory | Call `context.deallocate_weights()` and `cache.deallocate()` when the corresponding state is no longer needed. |

## Common

| API | Usage | Requirements and behavior |
| --- | --- | --- |
| `MeshDeviceRuntimeContext(device, default_tile_shape, default_dtype, scheduler_type=FRFCFS, enable_debug_log=False)` | Own the compiler/runtime lifecycle for one `MeshAccelerator`. Prefer use as a context manager. | `device` must be a `MeshAccelerator`; only one global context may be open. `default_tile_shape` and `default_dtype` are used when tensor APIs omit them. |
| `context.new_compiler_context(arrival_cycle=0, workload_id=None, dependent_workload_ids=(), scheduling_hint=None, warmup=False)` | Recommended context manager for building one workload. It compiles and submits on normal exit and rolls back KV-cache lengths on failure. | `arrival_cycle` is the runtime submission cycle. A workload becomes ready only after every named dependency completes. Compiler contexts cannot be nested. |
| `context.open_compiler(...)` | Manually start workload construction and return the active compiler. | Uses the same arguments and constraints as `new_compiler_context()`. Pair with `close_compiler()` or `abort_compiler()`. |
| `context.close_compiler()` | Compile and submit the active workload. | Returns `(compiled_workload, submitted_workload_id)`; requires an active compiler. |
| `context.abort_compiler()` | Discard the active compiler and restore tracked KV-cache lengths. | Safe cleanup path for a manually managed compiler context. |
| `context.compile(compiler=None)` | Compile the supplied compiler, or the active compiler when omitted, without submitting it. | Requires a `MeshDeviceCompiler` target. |
| `context.submit(compiled_workload, arrival_cycle=0, workload_id=None, dependent_workload_ids=(), scheduling_hint=None, warmup=False)` | Submit a compiled workload to the runtime. | `arrival_cycle` controls release time; dependencies may reference workloads submitted later but must exist and form an acyclic graph before `run()`. |
| `context.warmup(compiled_workload)` | Allocate resident weights before workload execution. | Reuses compiler-selected logical memory-bank counts. Runtime submission can also request warmup. |
| `context.run()` | Execute all submitted workloads until completion. | Returns completed `HostJob` objects. |
| `context.reserve_state(states)` | Reserve persistent placement for a `MeshKVCache` or tensor descriptors. | Placement uses each descriptor's memory preference, with runtime fallback when necessary. |
| `context.deallocate_state(states=None)` | Release selected persistent state, or all state when omitted. | Accepts a `MeshKVCache` or tensor descriptors; returns the number of released placements. |
| `context.deallocate_weights(compiled_workload=None)` | Release resident weights for one compiled workload, or all workloads when omitted. | Returns the number of released weight tensors. |
| `context.compiler` | Access the currently active compiler. | `None` outside a compiler context. |
| `context.runtime` | Access runtime workloads, decision logs, placements, and execution state. | Intended for execution and inspection after submission. |
| `context.device_descriptor` | Access the descriptor derived from the Mesh device. | Used by compilation and runtime placement. |
| `MeshDeviceSchedulerType` | Select `RR`, `FCFS`, or `FRFCFS` through `scheduler_type`. | `FRFCFS` is the default. |
| `MeshWorkloadSchedulingHint(priority=0, weight=1.0, max_wait_cycles=None)` | Attach workload scheduling metadata through compiler-context or `submit()` arguments. | `priority` is an integer, `weight` is positive, and `max_wait_cycles` is a positive integer or `None`. |
| `MeshDeviceCompiledWorkload` | Result returned by compilation; pass it to `submit()`, `warmup()`, or `deallocate_weights()`. | Contains compiled kernel/tensor tables and unique IDs. Normally created by the compiler, not directly. |
| `MeshTensorDescriptor` | Logical tensor shape, tile layout, dtype, type, views, and placement preference. | Normally create it with `mesh_tensor()` or `mesh_parameter()` rather than calling the constructor. |
| `MeshTensorType` | Mark tensors as `INTERMEDIATE` or `WEIGHT`. | Weights default to device memory; intermediates default to local cache. |
| `MeshMemoryType` | Express a placement preference: `LOCAL_CACHE` or `DEVICE_MEMORY`. | It is a preference; runtime may fall back to device memory when local cache cannot satisfy placement. |
| `MeshKernelState` | Inspect kernel states: `WAITING`, `READY`, `PLACED`, `RUNNING`, `COMPLETED`, or `FAILED`. | Runtime inspection type. |
| `MeshWorkloadState` | Inspect workload states: `PENDING`, `ACTIVE`, `COMPLETED`, or `FAILED`. | Runtime inspection type. |

## Tensor

Unless noted otherwise, kernel-producing functions register their kernels in the active compiler and return only output tensor descriptors.

| API | Usage | Shape and input requirements |
| --- | --- | --- |
| `mesh_tensor(*shape, tile_shape=None, dtype=None, tensor_type=INTERMEDIATE, reserved_shape=None, preferred_mem=None)` | Create a logical tensor descriptor using context defaults when tile shape or dtype is omitted. | Shape dimensions must be positive. `reserved_shape` must have the same rank and be at least `shape` in every dimension. At most two tile dimensions may be non-unit. |
| `mesh_parameter(*shape, tile_shape=None, dtype=None, preferred_mem=DEVICE_MEMORY)` | Create a weight tensor. | Same shape requirements as `mesh_tensor()`; sets `tensor_type=WEIGHT`. |
| `mesh_empty_like(x, shape=None, tile_shape=None, dtype=None, tensor_type=INTERMEDIATE, preferred_mem=None)` | Create a descriptor from `x`, optionally overriding selected attributes. | Inherits `x.shape`, dtype, and compatible tile shape when omitted. |
| `mesh_to_local_cache(x)` | Set `LOCAL_CACHE` as the placement preference and return `x`. | Runtime may fall back to device memory if local cache is unavailable. |
| `mesh_to_device_memory(x)` | Set `DEVICE_MEMORY` as the placement preference and return `x`. | Accepts one tensor descriptor. |
| `mesh_reshape(x, *shape)` | Create a zero-copy reshape; one `-1` dimension may be inferred. | Element count and tile count must remain unchanged. Materialize a compatible layout first when they do not. |
| `mesh_view(x, *shape)` | Alias of `mesh_reshape()`. | Same constraints as `mesh_reshape()`. |
| `mesh_flatten(x, start_dim=0, end_dim=-1)` | Flatten an inclusive dimension range as a zero-copy view. | `start_dim <= end_dim`; the current tile layout and tile count must be preserved. |
| `mesh_squeeze(x, dim=None)` | Remove selected size-one dimensions as a zero-copy view. | Removed dimensions must have logical size and tile size `1`; scalar descriptors are unsupported. A non-size-one selected dimension returns `x`. |
| `mesh_unsqueeze(x, dim)` | Insert a size-one, tile-size-one dimension. | `dim` may address any valid insertion position. |
| `mesh_permute(x, dims)` | Reorder all dimensions as a zero-copy view. | `dims` must be a complete, unique permutation of the input rank. |
| `mesh_transpose(x, dim0, dim1)` | Swap two dimensions through `mesh_permute()`. | Both dimensions must be valid. |
| `mesh_narrow(x, dim, start, length)` | Create a range view along one dimension. | `length > 0`; the range must stay in bounds. Negative `start` is accepted. Unaligned element offsets are supported. |
| `mesh_slice(x, dim, start=0, end=None, step=1)` | Create a non-empty range view. | Only `step=1` is supported; negative start/end values are normalized against the dimension size. |
| `mesh_select(x, dim, index)` | Select one index and remove that dimension. | Index must be in bounds. The selected dimension's tile size must be `1` because selection squeezes it. |
| `mesh_split(x, split_size_or_sections, dim=0)` | Return range views partitioning one dimension. | A split size must be positive; explicit positive sections must sum exactly to the dimension size. |
| `mesh_chunk(x, chunks, dim=0)` | Split one dimension into at most `chunks` range views. | `chunks` must be a positive integer. |
| `mesh_expand(x, *shape)` | Create a broadcast view without copying. | Output rank cannot shrink; each aligned input dimension must equal the target or be `1`. Target `-1` preserves the source size. |
| `mesh_copy(x, preferred_mem=None)` | Materialize `x` into a new tensor, optionally with a new memory preference. | Preserves shape, tile shape, and dtype. Requires an active compiler. |
| `mesh_prefetch(x)` | Emit a read-only memory-copy kernel and return `x`. | Requires an active compiler; does not create a new tensor. |
| `mesh_store(x, dst)` | Copy `x` into `dst` with range-aware addressing and return `dst`. | Source/destination compatibility is validated by the MemCopy kernel. Requires an active compiler. |
| `mesh_contiguous(x, preferred_mem=None)` | Return `x` when already materialized with the requested preference; otherwise emit `mesh_copy()`. | A view always requires materialization. |
| `mesh_cat(tensors, dim=0)` | Concatenate tensors into a materialized output using range-aware stores. | Requires at least one tensor. Rank, dtype, tile shape, and every non-concatenated dimension must match. |
| `mesh_stack(tensors, dim=0)` | Insert a dimension and concatenate tensors along it. | Requires at least one tensor; inputs must satisfy `mesh_cat()` compatibility after unsqueeze. |
| `mesh_head_split(x, num_heads, head_dim=None)` | View `[..., hidden]` as `[..., num_heads, head_dim]`. | `num_heads > 0`; `num_heads * head_dim == hidden`; tile count must be preserved. |
| `mesh_head_merge(x)` | View `[..., num_heads, head_dim]` as `[..., hidden]`. | Input rank must be at least two and tile count must be preserved. |
| `mesh_embedding(indices, weight)` | Gather rows from `weight` and return shape `indices.shape + (embedding_dim,)`. | `indices` dtype must be `torch.int32` or `torch.int64`; `weight` must be `[num_embeddings, embedding_dim]`. |

## BLAS

All BLAS functions require an active compiler context. Broadcast rules follow trailing-dimension broadcasting for descriptor shapes.

| API | Usage | Shape and input requirements |
| --- | --- | --- |
| `mesh_linear(x, weight, bias=None)` | Compute a linear projection with output `x.shape[:-1] + (out_features,)`. | `x` rank must be at least two; `weight` is `[out_features, in_features]`; `x[-1] == in_features`; bias must broadcast to output; tensor dtypes must match. |
| `mesh_matmul(a, b, transpose_a=False, transpose_b=False)` | Matrix-multiply the last two dimensions and broadcast batch dimensions. | Both ranks must be at least two; effective reduction dimensions and dtypes must match. |
| `mesh_bmm(a, b, transpose_a=False, transpose_b=False)` | Batched matrix multiplication through `mesh_matmul()`. | Both inputs must be rank three; remaining matmul requirements apply. |
| `mesh_add(x, y)` | Elementwise addition. | Operands are tensors or real scalars; at least one must be a tensor. Tensor operands must have matching dtypes and broadcastable shapes. |
| `mesh_sub(x, y)` | Elementwise subtraction. | Same requirements as `mesh_add()`. |
| `mesh_mul(x, y)` | Elementwise multiplication. | Same requirements as `mesh_add()`. |
| `mesh_div(x, y)` | Elementwise division. | Same requirements as `mesh_add()`. |
| `mesh_neg(x)` | Elementwise negation. | `x` must be a tensor descriptor. |
| `mesh_abs(x)` | Elementwise absolute value. | `x` must be a tensor descriptor. |
| `mesh_square(x)` | Elementwise square. | `x` must be a tensor descriptor. |
| `mesh_pow(x, exponent)` | Raise every element to a scalar power. | `exponent` must be a real scalar. |
| `mesh_exp(x)` | Elementwise exponential. | `x` must be a tensor descriptor. |
| `mesh_sqrt(x)` | Elementwise square root. | `x` must be a tensor descriptor. |
| `mesh_rsqrt(x)` | Elementwise reciprocal square root. | `x` must be a tensor descriptor. |
| `mesh_scale(x, scale)` | Multiply every element by a scalar. | `scale` must be a real scalar. |
| `mesh_where(condition, x, y)` | Select broadcasted values from `x` and `y`. | `condition` is a tensor. Values are tensors or real scalars; at least one value must be a tensor. Tensor values must share dtype; all tensor shapes must broadcast. |
| `mesh_masked_fill(x, mask, value)` | Fill masked elements with a scalar. | `mask` must broadcast to `x`; `value` must be a real scalar. |
| `mesh_clamp(x, minimum=None, maximum=None)` | Clamp elements to one or both scalar bounds. | At least one bound is required; supplied bounds must be real scalars. |
| `mesh_relu(x)` | Apply ReLU elementwise. | Preserves shape and dtype. |
| `mesh_silu(x)` | Apply SiLU elementwise. | Preserves shape and dtype. |
| `mesh_gelu(x)` | Apply GELU elementwise. | Preserves shape and dtype. |
| `mesh_quick_gelu(x)` | Apply QuickGELU elementwise. | Preserves shape and dtype. |
| `mesh_sigmoid(x)` | Apply sigmoid elementwise. | Preserves shape and dtype. |
| `mesh_swiglu(gate, up)` | Compute `silu(gate) * up`. | `gate` and `up` must have matching dtypes and broadcastable shapes. |
| `mesh_sum(x, dim=-1, keepdim=True)` | Reduce by sum. | Only the last dimension is supported. A rank-one input requires `keepdim=True` because scalar descriptors are unsupported. |
| `mesh_max(x, dim=-1, keepdim=True)` | Reduce by maximum. | Same dimension and scalar-output constraints as `mesh_sum()`. |
| `mesh_mean(x, dim=-1, keepdim=True)` | Reduce by mean. | Same dimension and scalar-output constraints as `mesh_sum()`. |
| `mesh_argmax(x, dim=-1, keepdim=False)` | Return `torch.int64` indices of maxima. | Only the last dimension is supported; a rank-one input requires `keepdim=True`. |
| `mesh_softmax(x, dim=-1, mask=None, scale=None)` | Apply optional scalar scaling and mask, then softmax. | Only the last dimension is supported; `mask` must broadcast to `x`; `scale`, when supplied, must be real. |
| `mesh_rms_norm(x, weight=None, eps=1e-6)` | Normalize over the last dimension and optionally apply `weight`. | `eps > 0`; `weight` must broadcast to `x`. |
| `mesh_layer_norm(x, weight=None, bias=None, eps=1e-5)` | Normalize over the last dimension and optionally apply affine terms. | `eps > 0`; `weight` and `bias` must each broadcast to `x`. |
| `mesh_rope(q, k, cos, sin, position_ids=None, rotary_sections=None)` | Apply rotary-position elementwise work and return `(rotated_q, rotated_k)`. | `q`, `k`, `cos`, and `sin` dtypes must match; `q[-1] == k[-1]`; `cos` and `sin` must broadcast to both. Sections must be positive and total at most the head dimension. |

## KV Cache

| API | Usage | Shape and input requirements |
| --- | --- | --- |
| `mesh_kv_cache(batch_size, max_seq_len, num_layers, num_kv_heads, head_dim, dtype=None, tile_shape=None, preferred_mem=DEVICE_MEMORY, context_buckets=(), overflow_policy="error")` | Create and register a fixed-capacity per-layer KV cache in the open runtime context. | All dimensions must be positive integers. Dtype/tile shape use context defaults when omitted. Only `overflow_policy="error"` is supported. The selected capacity is the smallest configured bucket at least `max_seq_len`, or `max_seq_len` without buckets. |
| `MeshKVCache` | Handle returned by `mesh_kv_cache()`; do not normally instantiate directly. | Each key/value storage has shape `[batch_size, num_kv_heads, capacity, head_dim]` and is persistent intermediate state. |
| `cache.reserve(capacity=None)` | Allocate persistent runtime placement for every layer's key/value storage. | Requested capacity must be positive and no larger than the cache's fixed capacity. |
| `cache.append(layer, key, value)` | Emit range-aware stores at the layer's current end and advance its length. | Key/value must have identical dtype and shape `[batch_size, num_kv_heads, append_length, head_dim]`; dtype must match the cache; append must fit capacity. Requires an active compiler. |
| `cache.view(layer, start=0, end=None)` | Return zero-copy `(key, value)` views for an existing sequence range. | Requires `0 <= start < end <= cache.get_length(layer)`; `end=None` uses the current layer length. |
| `cache.read(layer, start=0, end=None)` | Alias of `cache.view()`. | Same range requirements as `view()`. |
| `cache.get_length(layer)` | Return the active sequence length for one layer. | `layer` must be in `[0, num_layers)`. |
| `cache.reset(layer=None)` | Reset one layer's logical length, or all lengths when omitted. | Does not release persistent storage or erase simulated memory. |
| `cache.deallocate()` | Release all persistent placements owned by the cache. | Returns the number of released placements. |
| `cache.storage_descriptors` | Access all key/value storage descriptors in layer order. | Returns `(layer_0_key, layer_0_value, layer_1_key, ...)`. |
| `cache.current_length` | Inspect the minimum active length across all layers. | Use `layer_lengths` or `get_length()` when layers may differ. |
| `cache.layer_lengths` | Inspect every layer's active length. | Returns a tuple of length `num_layers`. |
| `cache.is_reserved` | Check whether every storage descriptor currently has persistent placement. | Boolean runtime-state property. |
| `cache.placements` | Inspect currently allocated persistent placements by persistent-state ID. | Returns only placements that currently exist. |
| `MeshKVCacheOverflowError` | Catch fixed-capacity creation, reservation, or append overflow. | Increase `max_seq_len` or select a larger `context_buckets` entry and recompile. |
