import math
import sqlite3
from dataclasses import dataclass

from transformers import AutoConfig
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()


@dataclass
class StageEstimate:
    flops: float
    bytes: float


def parse_range_fields(name: str) -> dict[str, str]:
    fields = {}
    for piece in name.split("|")[1:]:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key] = value
    return fields


def scalar(cur, query, params=()):
    return cur.execute(query, params).fetchone()[0]


def named_ranges(cur, exact=None, prefix=None, start=None, end=None):
    filters = ["n.end IS NOT NULL"]
    params = []
    if exact is not None:
        filters.append("COALESCE(n.text, s.value) = ?")
        params.append(exact)
    if prefix is not None:
        filters.append("COALESCE(n.text, s.value) LIKE ?")
        params.append(prefix + "%")
    if start is not None:
        filters.append("n.start >= ?")
        params.append(start)
    if end is not None:
        filters.append("n.end <= ?")
        params.append(end)
    query = f"""
        SELECT n.start, n.end, COALESCE(n.text, s.value) AS name
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE {' AND '.join(filters)}
        ORDER BY n.start
    """
    return cur.execute(query, params).fetchall()


def overlap_kernel_ns(cur, ranges):
    total = 0
    for start, end, _ in ranges:
        total += scalar(
            cur,
            """
            SELECT COALESCE(SUM(k.end - k.start), 0)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            WHERE k.start < ? AND k.end > ?
            """,
            (end, start),
        )
    return total


def overlap_memcpy_bytes(cur, ranges):
    total = 0
    for start, end, _ in ranges:
        total += scalar(
            cur,
            """
            SELECT COALESCE(SUM(m.bytes), 0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            WHERE m.start < ? AND m.end > ?
            """,
            (end, start),
        )
    return total


def range_duration_ms(ranges):
    return sum(end - start for start, end, _ in ranges) / 1e6


def get_qwen_configs(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    if hasattr(cfg, "thinker_config"):
        return cfg.thinker_config.vision_config, cfg.thinker_config.text_config
    return cfg.vision_config, cfg.text_config


def matmul_flops(m, k, n):
    return 2.0 * m * k * n


def linear_flops(tokens, in_dim, out_dim):
    return matmul_flops(tokens, in_dim, out_dim)


def linear_weight_bytes(in_dim, out_dim, dtype_bytes):
    return in_dim * out_dim * dtype_bytes


def estimate_vision(vision_cfg, patch_tokens, dtype_bytes):
    d = vision_cfg.hidden_size
    i = vision_cfg.intermediate_size
    depth = vision_cfg.depth
    heads = vision_cfg.num_heads
    head_dim = d // heads
    merge = vision_cfg.spatial_merge_size**2
    merged_tokens = max(1, math.ceil(patch_tokens / merge))
    patch_kernel = 3 * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size
    flops = matmul_flops(patch_tokens, patch_kernel, d)
    bytes_ = patch_kernel * d * dtype_bytes
    fullatt = set(getattr(vision_cfg, "fullatt_block_indexes", []) or [])
    window_size = getattr(vision_cfg, "window_size", 112) or 112
    window_tokens = max(1, (window_size // vision_cfg.patch_size) ** 2)

    for layer in range(depth):
        flops += 4 * linear_flops(patch_tokens, d, d)
        bytes_ += 4 * linear_weight_bytes(d, d, dtype_bytes)
        attn_len = patch_tokens if layer in fullatt else min(patch_tokens, window_tokens)
        flops += 4.0 * patch_tokens * attn_len * head_dim * heads
        flops += 3 * linear_flops(patch_tokens, d, i)
        bytes_ += 3 * linear_weight_bytes(d, i, dtype_bytes)

    merger_hidden = d * merge
    flops += linear_flops(merged_tokens, merger_hidden, merger_hidden)
    flops += linear_flops(merged_tokens, merger_hidden, vision_cfg.out_hidden_size)
    bytes_ += linear_weight_bytes(merger_hidden, merger_hidden, dtype_bytes)
    bytes_ += linear_weight_bytes(merger_hidden, vision_cfg.out_hidden_size, dtype_bytes)
    return StageEstimate(flops=flops, bytes=bytes_)


def estimate_prefill(text_cfg, seq_tokens, dtype_bytes):
    d = text_cfg.hidden_size
    i = text_cfg.intermediate_size
    layers = text_cfg.num_hidden_layers
    q_heads = text_cfg.num_attention_heads
    kv_heads = text_cfg.num_key_value_heads
    head_dim = d // q_heads
    flops = 0.0
    bytes_ = 0.0
    for _ in range(layers):
        flops += linear_flops(seq_tokens, d, q_heads * head_dim)
        flops += 2 * linear_flops(seq_tokens, d, kv_heads * head_dim)
        flops += linear_flops(seq_tokens, q_heads * head_dim, d)
        bytes_ += linear_weight_bytes(d, q_heads * head_dim, dtype_bytes)
        bytes_ += 2 * linear_weight_bytes(d, kv_heads * head_dim, dtype_bytes)
        bytes_ += linear_weight_bytes(q_heads * head_dim, d, dtype_bytes)
        flops += 4.0 * seq_tokens * seq_tokens * head_dim * q_heads
        flops += 3 * linear_flops(seq_tokens, d, i)
        bytes_ += 3 * linear_weight_bytes(d, i, dtype_bytes)
        bytes_ += 2 * seq_tokens * kv_heads * head_dim * dtype_bytes
    flops += linear_flops(1, d, text_cfg.vocab_size)
    bytes_ += linear_weight_bytes(d, text_cfg.vocab_size, dtype_bytes)
    return StageEstimate(flops=flops, bytes=bytes_)


def estimate_decode(text_cfg, context_tokens, generated_tokens, dtype_bytes):
    d = text_cfg.hidden_size
    i = text_cfg.intermediate_size
    layers = text_cfg.num_hidden_layers
    q_heads = text_cfg.num_attention_heads
    kv_heads = text_cfg.num_key_value_heads
    head_dim = d // q_heads
    per_flops = 0.0
    per_bytes = 0.0
    for _ in range(layers):
        per_flops += linear_flops(1, d, q_heads * head_dim)
        per_flops += 2 * linear_flops(1, d, kv_heads * head_dim)
        per_flops += linear_flops(1, q_heads * head_dim, d)
        per_bytes += linear_weight_bytes(d, q_heads * head_dim, dtype_bytes)
        per_bytes += 2 * linear_weight_bytes(d, kv_heads * head_dim, dtype_bytes)
        per_bytes += linear_weight_bytes(q_heads * head_dim, d, dtype_bytes)
        per_flops += 4.0 * context_tokens * head_dim * q_heads
        per_bytes += 2 * context_tokens * kv_heads * head_dim * dtype_bytes
        per_bytes += 2 * kv_heads * head_dim * dtype_bytes
        per_flops += 3 * linear_flops(1, d, i)
        per_bytes += 3 * linear_weight_bytes(d, i, dtype_bytes)
    per_flops += linear_flops(1, d, text_cfg.vocab_size)
    per_bytes += linear_weight_bytes(d, text_cfg.vocab_size, dtype_bytes)
    return StageEstimate(flops=per_flops * generated_tokens, bytes=per_bytes * generated_tokens)


def add_stage_metrics(row, prefix, duration_ms, kernel_ms, memcpy_bytes, estimate):
    row[f"{prefix}_ms"] = duration_ms
    row[f"{prefix}_kernel_ms"] = kernel_ms
    row[f"{prefix}_measured_memcpy_bytes"] = memcpy_bytes
    row[f"{prefix}_estimated_flops"] = estimate.flops
    row[f"{prefix}_estimated_bytes"] = estimate.bytes
    seconds = duration_ms / 1e3
    row[f"{prefix}_avg_tops"] = estimate.flops / seconds / 1e12 if seconds > 0 else 0.0
    row[f"{prefix}_avg_bandwidth_gbps"] = estimate.bytes / seconds / 1e9 if seconds > 0 else 0.0


def analyze_sqlite(sqlite_path, model_id, dtype_bytes=2):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    vision_cfg, text_cfg = get_qwen_configs(model_id)
    rows = []

    for sample_start, sample_end, sample_name in named_ranges(cur, prefix="SAMPLE|"):
        fields = parse_range_fields(sample_name)
        meta_ranges = named_ranges(cur, prefix="SAMPLE_META|", start=sample_start, end=sample_end)
        if meta_ranges:
            fields.update(parse_range_fields(meta_ranges[-1][2]))
        patch_tokens = int(fields.get("vision_patch_tokens", "0"))
        prefill_tokens = int(fields.get("prefill_tokens", "0"))
        context_tokens = int(fields.get("context_tokens", prefill_tokens))
        decode_tokens = int(fields.get("decode_tokens", "0"))

        row = {
            "sample_id": fields.get("sample_id", ""),
            "question_id": fields.get("question_id", ""),
            "sample_index": int(fields.get("sample_index", "0")),
            "task_type": fields.get("task_type", ""),
            "timestamp_sec": float(fields.get("timestamp_sec", "0")),
            "frame_index": int(fields.get("frame_index", "0")),
            "sample_video_frames": int(fields.get("sample_video_frames", "0")),
            "total_video_frames_seen": int(fields.get("frame_index", "0")),
            "vision_patch_tokens": patch_tokens,
            "merged_visual_tokens": int(fields.get("merged_visual_tokens", "0")),
            "prefill_tokens": prefill_tokens,
            "context_tokens_before_decode": context_tokens,
            "decode_tokens": decode_tokens,
            "sample_wall_ms": (sample_end - sample_start) / 1e6,
        }

        stage_specs = {
            "vision_encode": (named_ranges(cur, exact="VISION_ENCODING", start=sample_start, end=sample_end),
                              estimate_vision(vision_cfg, patch_tokens, dtype_bytes)),
            "prefill": (named_ranges(cur, exact="LLM_PREFILL_VISION_TEXT", start=sample_start, end=sample_end),
                        estimate_prefill(text_cfg, prefill_tokens, dtype_bytes)),
            "decode": (named_ranges(cur, exact="LLM_DECODE_LOOP", start=sample_start, end=sample_end),
                       estimate_decode(text_cfg, context_tokens, decode_tokens, dtype_bytes)),
        }

        for stage, (ranges, estimate) in stage_specs.items():
            duration_ms = range_duration_ms(ranges)
            kernel_ms = overlap_kernel_ns(cur, ranges) / 1e6 if ranges else 0.0
            memcpy_bytes = overlap_memcpy_bytes(cur, ranges) if ranges else 0
            row[f"{stage}_range_count"] = len(ranges)
            add_stage_metrics(row, stage, duration_ms, kernel_ms, memcpy_bytes, estimate)

        rows.append(row)

    con.close()
    rows.sort(key=lambda r: (r["sample_id"], r["sample_index"]))
    return rows
