"""FlashAttention-2-style forward mapping for the fused SDPA runtime.

The mapping describes ordered work for each query block. It is separate from
runtime_utils._Mapping, whose operations are independent output tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

from neuromta.system.software.utils.descriptor import MeshKernelDescriptor


@dataclass(frozen=True)
class SDPAOperation:
    """One graph node for a query block's load, online update, or final store.

    ``query_block`` is (batch, query_head, query_chunk); ``kv_block`` is the
    key/value chunk index, or None outside the inner loop. ``predecessors``
    orders score, softmax, and output accumulation across K/V chunks. Tensor
    references are (input tensor index, tile coordinate) pairs; output tiles
    are listed only by the final store node.
    """

    op_id: int
    kind: Literal[
        "load_q", "init", "load_kv", "score", "softmax_update",
        "accumulate", "finalize", "store",
    ]
    core_id: int
    query_block: tuple[int, int, int]
    kv_block: int | None
    predecessors: tuple[int, ...]
    input_tiles: tuple[tuple[int, tuple[int, ...]], ...] = ()
    output_tiles: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class SDPAMapping:
    """Core placement and dependency graph for fused SDPA forward work.

    Each query block owns its complete head dimension and output accumulator;
    its operations visit valid K/V chunks in order before storing output tiles.
    """

    core_mesh: np.ndarray
    operations: tuple[SDPAOperation, ...]
    q_chunk_size: int
    kv_chunk_size: int


def create_sdpa_mapping(
    kernel_desc: MeshKernelDescriptor,
    core_mesh: np.ndarray,
    ld_slot_bytes: int,
    st_slot_bytes: int,
) -> SDPAMapping:
    """Plan query-block owners and their ordered FlashAttention-2 operations.

    Use the descriptor's query/KV chunk sizes and GQA head mapping, choose
    buffer-fitting chunks, omit fully causal-masked K/V chunks, and account for
    partial boundary chunks. A query block loads Q once, streams each K/V
    chunk once across the full head dimension, updates online softmax and the
    output accumulator, then normalizes and stores the output tiles.
    """
    if core_mesh.ndim != 2 or not core_mesh.size or np.any(core_mesh < 0):
        raise ValueError("SDPA requires a nonempty two-dimensional core mesh.")
    core_ids = tuple(int(core) for core in core_mesh.flat)
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("SDPA core IDs must be unique.")

    q, k = kernel_desc.input_tensors[:2]
    output = kernel_desc.output_tensors[0]
    mask = kernel_desc.input_tensors[3] if len(kernel_desc.input_tensors) == 4 else None
    causal, q_chunk, kv_chunk, group_size = kernel_desc.get_required_kwargs(
        "is_causal", "q_chunk_size", "kv_chunk_size", "head_group_size"
    )
    batch_size, n_heads, q_length, _ = q.shape
    kv_length = k.shape[2]
    padded_dim = q.tile_grid_shape[3] * q.tile_shape[3]

    # STEP 1: Reduce sequence chunks until Q, K/V, scores and the output
    # accumulator fit in one LD and one ST slot, respectively.
    def fits(q_rows: int, kv_rows: int) -> bool:
        ld_bytes = (q_rows + 2 * kv_rows) * padded_dim * q.dtype.itemsize
        ld_bytes += q_rows * kv_rows * (4 + (mask.dtype.itemsize if mask else 0))
        st_bytes = q_rows * (4 * padded_dim + 8)
        return ld_bytes <= ld_slot_bytes and st_bytes <= st_slot_bytes

    while not fits(q_chunk, kv_chunk):
        if kv_chunk > k.tile_shape[2] and (kv_chunk >= q_chunk or q_chunk == q.tile_shape[2]):
            kv_chunk -= k.tile_shape[2]
        elif q_chunk > q.tile_shape[2]:
            q_chunk -= q.tile_shape[2]
        elif kv_chunk > k.tile_shape[2]:
            kv_chunk -= k.tile_shape[2]
        else:
            raise ValueError("The smallest SDPA Q/K/V tiles do not fit the LD/ST buffer slots.")

    # STEP 2: Assign an entire query block, including every head-dimension
    # tile, to one core. Its online softmax state stays on that core.
    operations: list[SDPAOperation] = []

    def append(kind, core_id, query_block, kv_block=None, predecessors=(),
               input_tiles=(), output_tiles=()) -> int:
        op_id = len(operations)
        operations.append(SDPAOperation(
            op_id, kind, core_id, query_block, kv_block, tuple(predecessors),
            tuple(input_tiles), tuple(output_tiles),
        ))
        return op_id

    query_blocks = product(range(batch_size), range(0, q_length, q_chunk), range(n_heads))
    max_cores = kernel_desc.get_required_kwargs("max_cores_per_head")
    cores_per_head = min(max_cores, len(core_ids), (q_length + q_chunk - 1) // q_chunk)
    for batch, q_start, q_head in query_blocks:
        head_index = batch * n_heads + q_head
        chunk_index = q_start // q_chunk
        core_id = core_ids[(head_index * cores_per_head + chunk_index % cores_per_head) % len(core_ids)]
        q_end = min(q_length, q_start + q_chunk)
        q_block = (batch, q_head, q_start // q_chunk)
        kv_head = q_head // group_size
        q_tiles = range(q_start // q.tile_shape[2], (q_end - 1) // q.tile_shape[2] + 1)
        q_inputs = ((0, (batch, q_head, row, dim))
                    for row in q_tiles for dim in range(q.tile_grid_shape[3]))

        # STEP 3: Load Q only once and initialize per-row online softmax.
        previous = append("load_q", core_id, q_block, input_tiles=q_inputs)
        previous = append("init", core_id, q_block, predecessors=(previous,))

        # STEP 4: Stream valid K/V chunks in order, updating the running
        # maximum, denominator and output accumulator after each score block.
        for kv_start in range(0, kv_length, kv_chunk):
            kv_end = min(kv_length, kv_start + kv_chunk)
            if causal and kv_start >= kv_length - q_length + q_end:
                break
            kv_block = kv_start // kv_chunk
            kv_tiles = range(kv_start // k.tile_shape[2], (kv_end - 1) // k.tile_shape[2] + 1)
            inputs = [(tensor_id, (batch, kv_head, row, dim))
                      for tensor_id in (1, 2) for row in kv_tiles
                      for dim in range(k.tile_grid_shape[3])]
            if mask is not None:
                for q_tile, kv_tile in product(q_tiles, kv_tiles):
                    score_coord = (batch, q_head, q_tile, kv_tile)[-len(mask.tile_grid_shape):]
                    coord = tuple(0 if dim == 1 else index for dim, index in
                                  zip(mask.tile_grid_shape, score_coord))
                    ref = (3, coord)
                    if ref not in inputs:
                        inputs.append(ref)
            previous = append("load_kv", core_id, q_block, kv_block, (previous,), inputs)
            previous = append("score", core_id, q_block, kv_block, (previous,))
            previous = append("softmax_update", core_id, q_block, kv_block, (previous,))
            previous = append("accumulate", core_id, q_block, kv_block, (previous,))

        # STEP 5: Normalize the rows and store every output tile once.
        previous = append("finalize", core_id, q_block, predecessors=(previous,))
        output_tiles = ((batch, q_head, row, dim) for row in q_tiles
                        for dim in range(output.tile_grid_shape[3]))
        append("store", core_id, q_block, predecessors=(previous,), output_tiles=output_tiles)

    return SDPAMapping(np.asarray(core_mesh, dtype=int).copy(), tuple(operations), q_chunk, kv_chunk)
