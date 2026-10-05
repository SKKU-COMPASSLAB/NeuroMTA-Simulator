import csv
import math
import os
import sqlite3
from dataclasses import dataclass

from transformers import AutoConfig
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()

STAGES = ("encode", "prefill", "decode")


@dataclass
class StageEstimate:
    flops: float
    bytes: float


@dataclass
class TimelineEvent:
    sample_id: str
    question_id: str
    sample_index: int
    task_type: str
    stage: str
    event_name: str
    step_index: int
    start_ns: int
    end_ns: int
    fields: dict
    flops: float = 0.0
    bytes: float = 0.0
    kernel_ms: float = 0.0
    memcpy_bytes: float = 0.0

    @property
    def duration_ms(self):
        return (self.end_ns - self.start_ns) / 1e6


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


def table_exists(cur, name):
    return bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def table_columns(cur, name):
    if not table_exists(cur, name):
        return []
    return [row[1] for row in cur.execute(f"PRAGMA table_info({name})")]


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


def overlap_kernel_ns(cur, start, end):
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return 0
    return scalar(
        cur,
        """
        SELECT COALESCE(SUM(MIN(k.end, ?) - MAX(k.start, ?)), 0)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE k.start < ? AND k.end > ?
        """,
        (end, start, end, start),
    )


def overlap_memcpy_bytes(cur, start, end):
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return 0
    return scalar(
        cur,
        """
        SELECT COALESCE(SUM(m.bytes), 0)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.start < ? AND m.end > ?
        """,
        (end, start),
    )


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


def event_estimate(stage, fields, vision_cfg, text_cfg, dtype_bytes):
    if stage == "encode":
        return estimate_vision(vision_cfg, int(float(fields.get("patch_tokens", 0) or 0)), dtype_bytes)
    if stage == "prefill":
        return estimate_prefill(text_cfg, int(float(fields.get("prefill_tokens", 0) or 0)), dtype_bytes)
    context_tokens = int(float(fields.get("context_tokens", 0) or 0))
    generated_tokens = int(float(fields.get("generated_tokens", fields.get("decode_tokens", 1)) or 1))
    return estimate_decode(text_cfg, context_tokens, generated_tokens, dtype_bytes)


def make_event(cur, start, end, name, stage, vision_cfg, text_cfg, dtype_bytes, fallback_step=0):
    fields = parse_range_fields(name)
    estimate = event_estimate(stage, fields, vision_cfg, text_cfg, dtype_bytes)
    if stage == "encode":
        step = int(float(fields.get("chunk_index", fallback_step) or fallback_step))
    elif stage == "decode":
        step = int(float(fields.get("decode_step", fallback_step) or fallback_step))
    else:
        step = int(float(fields.get("chunk_index", fallback_step) or fallback_step))
    event = TimelineEvent(
        sample_id=fields.get("sample_id", ""),
        question_id=fields.get("question_id", ""),
        sample_index=int(float(fields.get("sample_index", 0) or 0)),
        task_type=fields.get("task_type", ""),
        stage=stage,
        event_name=name.split("|", 1)[0],
        step_index=step,
        start_ns=start,
        end_ns=end,
        fields=fields,
        flops=estimate.flops,
        bytes=estimate.bytes,
        kernel_ms=overlap_kernel_ns(cur, start, end) / 1e6,
        memcpy_bytes=overlap_memcpy_bytes(cur, start, end),
    )
    return event


def extract_kernel_timeline(cur, origin_ns, sample_id):
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    columns = table_columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    select_cols = ["start", "end"]
    if "streamId" in columns:
        select_cols.append("streamId")
    elif "streamId" not in select_cols:
        select_cols.append("0 AS streamId")
    query = f"SELECT {', '.join(select_cols)} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    rows = []
    for idx, row in enumerate(cur.execute(query)):
        start, end, stream_id = row
        rows.append({
            "sample_id": sample_id,
            "kernel_index": idx,
            "stream_id": stream_id,
            "start_ms": (start - origin_ns) / 1e6,
            "end_ms": (end - origin_ns) / 1e6,
            "duration_ms": (end - start) / 1e6,
        })
    return rows


def timeline_row(event, origin_ns, schedule_name, start_ms=None, end_ms=None, peak_tops=312.0, peak_bandwidth_gbps=1555.0):
    if start_ms is None:
        start_ms = (event.start_ns - origin_ns) / 1e6
    if end_ms is None:
        end_ms = (event.end_ns - origin_ns) / 1e6
    duration_ms = max(end_ms - start_ms, 0.0)
    seconds = duration_ms / 1000.0
    avg_tops = event.flops / seconds / 1e12 if seconds > 0 else 0.0
    avg_bandwidth = event.bytes / seconds / 1e9 if seconds > 0 else 0.0
    return {
        "schedule": schedule_name,
        "sample_id": event.sample_id,
        "question_id": event.question_id,
        "sample_index": event.sample_index,
        "task_type": event.task_type,
        "stage": event.stage,
        "step_index": event.step_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "kernel_ms": event.kernel_ms,
        "estimated_flops": event.flops,
        "estimated_bytes": event.bytes,
        "avg_tops": avg_tops,
        "avg_bandwidth_gbps": avg_bandwidth,
        "compute_util_pct": 100.0 * avg_tops / peak_tops if peak_tops > 0 else 0.0,
        "bandwidth_util_pct": 100.0 * avg_bandwidth / peak_bandwidth_gbps if peak_bandwidth_gbps > 0 else 0.0,
        "name": event.event_name,
    }


def aggregate_event(events, stage, start_ns=None, end_ns=None):
    if not events:
        return None
    first = events[0]
    return TimelineEvent(
        sample_id=first.sample_id,
        question_id=first.question_id,
        sample_index=first.sample_index,
        task_type=first.task_type,
        stage=stage,
        event_name=f"{stage.upper()}_AGGREGATE",
        step_index=0,
        start_ns=min(e.start_ns for e in events) if start_ns is None else start_ns,
        end_ns=max(e.end_ns for e in events) if end_ns is None else end_ns,
        fields=dict(first.fields),
        flops=sum(e.flops for e in events),
        bytes=sum(e.bytes for e in events),
        kernel_ms=sum(e.kernel_ms for e in events),
        memcpy_bytes=sum(e.memcpy_bytes for e in events),
    )


def schedule_replay(sample_infos, mode, peak_tops, peak_bandwidth_gbps):
    rows = []
    if not sample_infos:
        return rows
    encode_available = prefill_available = decode_available = 0.0
    monolithic_available = 0.0

    for info in sample_infos:
        encode_events = sorted(info["encode_events"], key=lambda event: event.step_index)
        prefill_events = sorted(info["prefill_events"], key=lambda event: event.step_index)
        decode_event = info["decode_event"]

        if mode == "sequential":
            cursor = monolithic_available
            for index, encode_event in enumerate(encode_events):
                start = cursor
                end = start + encode_event.duration_ms
                rows.append(timeline_row(encode_event, 0, mode, start, end, peak_tops, peak_bandwidth_gbps))
                cursor = end

                if index < len(prefill_events):
                    prefill_event = prefill_events[index]
                    start = cursor
                    end = start + prefill_event.duration_ms
                    rows.append(timeline_row(prefill_event, 0, mode, start, end, peak_tops, peak_bandwidth_gbps))
                    cursor = end

            if decode_event is not None:
                start = cursor
                end = start + decode_event.duration_ms
                rows.append(timeline_row(decode_event, 0, mode, start, end, peak_tops, peak_bandwidth_gbps))
                cursor = end
            monolithic_available = cursor
            continue

        sample_last_prefill_end = encode_available
        for index, encode_event in enumerate(encode_events):
            encode_start = encode_available
            encode_end = encode_start + encode_event.duration_ms
            rows.append(timeline_row(encode_event, 0, mode, encode_start, encode_end, peak_tops, peak_bandwidth_gbps))
            encode_available = encode_end

            if index < len(prefill_events):
                prefill_event = prefill_events[index]
                prefill_start = max(prefill_available, encode_end)
                prefill_end = prefill_start + prefill_event.duration_ms
                rows.append(timeline_row(prefill_event, 0, mode, prefill_start, prefill_end, peak_tops, peak_bandwidth_gbps))
                prefill_available = prefill_end
                sample_last_prefill_end = prefill_end
            else:
                sample_last_prefill_end = max(sample_last_prefill_end, encode_end)

        if decode_event is not None:
            decode_start = max(decode_available, sample_last_prefill_end)
            decode_end = decode_start + decode_event.duration_ms
            rows.append(timeline_row(decode_event, 0, mode, decode_start, decode_end, peak_tops, peak_bandwidth_gbps))
            decode_available = decode_end
    return rows

def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator > 0 else 0.0


def sum_duration(events):
    return sum(e.duration_ms for e in events)


def compute_summary(sample_infos, sequential_rows, pipelined_rows, peak_tops, peak_bandwidth_gbps):
    seq_end = max((row["end_ms"] for row in sequential_rows), default=0.0)
    pipe_end = max((row["end_ms"] for row in pipelined_rows), default=0.0)
    total_stage = sum(row["duration_ms"] for row in sequential_rows)
    stage_busy = {stage: sum(row["duration_ms"] for row in pipelined_rows if row["stage"] == stage) for stage in STAGES}
    stage_flops = {stage: sum(row["estimated_flops"] for row in pipelined_rows if row["stage"] == stage) for stage in STAGES}
    stage_bytes = {stage: sum(row["estimated_bytes"] for row in pipelined_rows if row["stage"] == stage) for stage in STAGES}
    pipe_seconds = pipe_end / 1000.0
    return {
        "sequential_makespan_ms": seq_end,
        "pipelined_makespan_ms": pipe_end,
        "speedup": seq_end / pipe_end if pipe_end > 0 else 0.0,
        "overlap_ratio": (total_stage - pipe_end) / total_stage if total_stage > 0 else 0.0,
        "encode_resource_util_pct": pct(stage_busy["encode"], pipe_end),
        "prefill_resource_util_pct": pct(stage_busy["prefill"], pipe_end),
        "decode_resource_util_pct": pct(stage_busy["decode"], pipe_end),
        "avg_pipeline_resource_util_pct": pct(total_stage, 3.0 * pipe_end),
        "pipeline_resource_waste_pct": max(0.0, 100.0 - pct(total_stage, 3.0 * pipe_end)),
        "encode_compute_util_pct": pct(stage_flops["encode"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_tops * 1e12),
        "prefill_compute_util_pct": pct(stage_flops["prefill"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_tops * 1e12),
        "decode_compute_util_pct": pct(stage_flops["decode"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_tops * 1e12),
        "encode_bandwidth_util_pct": pct(stage_bytes["encode"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_bandwidth_gbps * 1e9),
        "prefill_bandwidth_util_pct": pct(stage_bytes["prefill"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_bandwidth_gbps * 1e9),
        "decode_bandwidth_util_pct": pct(stage_bytes["decode"] / pipe_seconds if pipe_seconds > 0 else 0.0, peak_bandwidth_gbps * 1e9),
        "bottleneck_stage": max(stage_busy, key=stage_busy.get) if stage_busy else "",
    }


def analyze_sqlite(sqlite_path, model_id, dtype_bytes=2, peak_tops=312.0, peak_bandwidth_gbps=1555.0):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    vision_cfg, text_cfg = get_qwen_configs(model_id)

    sample_ranges = named_ranges(cur, prefix="SAMPLE|")
    origin_ns = sample_ranges[0][0] if sample_ranges else 0
    sample_infos = []
    observed_rows = []

    for sample_start, sample_end, sample_name in sample_ranges:
        sample_fields = parse_range_fields(sample_name)
        sample_id = sample_fields.get("sample_id", "")
        sample_index = int(float(sample_fields.get("sample_index", 0) or 0))
        meta_ranges = named_ranges(cur, prefix="SAMPLE_META|", start=sample_start, end=sample_end)
        meta_fields = parse_range_fields(meta_ranges[-1][2]) if meta_ranges else {}

        encode_events = []
        for i, (start, end, name) in enumerate(named_ranges(cur, prefix="ENCODE_STEP|", start=sample_start, end=sample_end)):
            event = make_event(cur, start, end, name, "encode", vision_cfg, text_cfg, dtype_bytes, i)
            encode_events.append(event)
            observed_rows.append(timeline_row(event, origin_ns, "observed", peak_tops=peak_tops, peak_bandwidth_gbps=peak_bandwidth_gbps))

        prefill_events = []
        for i, (start, end, name) in enumerate(named_ranges(cur, prefix="PREFILL_STAGE|", start=sample_start, end=sample_end)):
            event = make_event(cur, start, end, name, "prefill", vision_cfg, text_cfg, dtype_bytes, i)
            prefill_events.append(event)
            observed_rows.append(timeline_row(event, origin_ns, "observed", peak_tops=peak_tops, peak_bandwidth_gbps=peak_bandwidth_gbps))

        decode_step_events = []
        for i, (start, end, name) in enumerate(named_ranges(cur, prefix="DECODE_STEP|", start=sample_start, end=sample_end)):
            event = make_event(cur, start, end, name, "decode", vision_cfg, text_cfg, dtype_bytes, i)
            decode_step_events.append(event)
            observed_rows.append(timeline_row(event, origin_ns, "observed", peak_tops=peak_tops, peak_bandwidth_gbps=peak_bandwidth_gbps))

        prefill_event = aggregate_event(prefill_events, "prefill")
        decode_event = aggregate_event(decode_step_events, "decode")
        if decode_event is not None:
            stage_ranges = named_ranges(cur, prefix="DECODE_STAGE|", start=sample_start, end=sample_end)
            if stage_ranges:
                decode_event.start_ns = stage_ranges[0][0]
                decode_event.end_ns = stage_ranges[-1][1]
                decode_event.kernel_ms = overlap_kernel_ns(cur, decode_event.start_ns, decode_event.end_ns) / 1e6
                decode_event.memcpy_bytes = overlap_memcpy_bytes(cur, decode_event.start_ns, decode_event.end_ns)

        sample_infos.append({
            "sample_id": sample_id,
            "sample_index": sample_index,
            "question_id": sample_fields.get("question_id", ""),
            "task_type": sample_fields.get("task_type", ""),
            "timestamp_sec": float(sample_fields.get("timestamp_sec", 0) or 0),
            "frame_index": int(float(sample_fields.get("frame_index", 0) or 0)),
            "sample_video_frames": int(float(sample_fields.get("sample_video_frames", 0) or 0)),
            "vision_patch_tokens": int(float(meta_fields.get("vision_patch_tokens", 0) or 0)),
            "merged_visual_tokens": int(float(meta_fields.get("merged_visual_tokens", 0) or 0)),
            "prefill_tokens": int(float(meta_fields.get("prefill_tokens", 0) or 0)),
            "context_tokens_before_decode": int(float(meta_fields.get("context_tokens", 0) or 0)),
            "decode_tokens": int(float(sample_fields.get("decode_tokens", 0) or 0)),
            "sample_wall_ms": (sample_end - sample_start) / 1e6,
            "encode_events": encode_events,
            "prefill_events": prefill_events,
            "decode_step_events": decode_step_events,
            "prefill_event": prefill_event,
            "decode_event": decode_event,
        })

    sample_infos.sort(key=lambda item: item["sample_index"])
    sequential_rows = schedule_replay(sample_infos, "sequential", peak_tops, peak_bandwidth_gbps)
    pipelined_rows = schedule_replay(sample_infos, "pipelined", peak_tops, peak_bandwidth_gbps)
    summary = compute_summary(sample_infos, sequential_rows, pipelined_rows, peak_tops, peak_bandwidth_gbps)

    result_rows = []
    for info in sample_infos:
        encode_ms = sum_duration(info["encode_events"])
        prefill_ms = sum_duration(info["prefill_events"])
        decode_ms = info["decode_event"].duration_ms if info["decode_event"] is not None else 0.0
        stage_sum = encode_ms + prefill_ms + decode_ms
        result_rows.append({
            "row_type": "sample",
            "sample_id": info["sample_id"],
            "question_id": info["question_id"],
            "sample_index": info["sample_index"],
            "task_type": info["task_type"],
            "timestamp_sec": info["timestamp_sec"],
            "frame_index": info["frame_index"],
            "sample_video_frames": info["sample_video_frames"],
            "vision_patch_tokens": info["vision_patch_tokens"],
            "merged_visual_tokens": info["merged_visual_tokens"],
            "prefill_tokens": info["prefill_tokens"],
            "context_tokens_before_decode": info["context_tokens_before_decode"],
            "decode_tokens": info["decode_tokens"],
            "sample_wall_ms": info["sample_wall_ms"],
            "encode_step_count": len(info["encode_events"]),
            "prefill_range_count": len(info["prefill_events"]),
            "decode_step_count": len(info["decode_step_events"]),
            "encode_ms": encode_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "stage_sum_ms": stage_sum,
            "local_ideal_speedup": stage_sum / max(encode_ms, prefill_ms, decode_ms) if max(encode_ms, prefill_ms, decode_ms) > 0 else 0.0,
        })

    if sample_infos:
        first = sample_infos[0]
        result_rows.append({
            "row_type": "summary",
            "sample_id": first["sample_id"],
            "question_id": "",
            "sample_index": len(sample_infos),
            "task_type": "video_summary",
            "timestamp_sec": "",
            "frame_index": max(info["frame_index"] for info in sample_infos),
            "sample_video_frames": sum(info["sample_video_frames"] for info in sample_infos),
            "vision_patch_tokens": sum(info["vision_patch_tokens"] for info in sample_infos),
            "merged_visual_tokens": sum(info["merged_visual_tokens"] for info in sample_infos),
            "prefill_tokens": sum(info["prefill_tokens"] for info in sample_infos),
            "context_tokens_before_decode": max(info["context_tokens_before_decode"] for info in sample_infos),
            "decode_tokens": sum(info["decode_tokens"] for info in sample_infos),
            "sample_wall_ms": sum(info["sample_wall_ms"] for info in sample_infos),
            "encode_step_count": sum(len(info["encode_events"]) for info in sample_infos),
            "prefill_range_count": sum(len(info["prefill_events"]) for info in sample_infos),
            "decode_step_count": sum(len(info["decode_step_events"]) for info in sample_infos),
            "encode_ms": sum(sum_duration(info["encode_events"]) for info in sample_infos),
            "prefill_ms": sum(sum_duration(info["prefill_events"]) for info in sample_infos),
            "decode_ms": sum(info["decode_event"].duration_ms for info in sample_infos if info["decode_event"] is not None),
            "stage_sum_ms": sum(row["duration_ms"] for row in sequential_rows),
            "sequential_makespan_ms": summary["sequential_makespan_ms"],
            "pipelined_makespan_ms": summary["pipelined_makespan_ms"],
            "pipeline_speedup": summary["speedup"],
            "pipeline_overlap_ratio": summary["overlap_ratio"],
            "encode_resource_util_pct": summary["encode_resource_util_pct"],
            "prefill_resource_util_pct": summary["prefill_resource_util_pct"],
            "decode_resource_util_pct": summary["decode_resource_util_pct"],
            "avg_pipeline_resource_util_pct": summary["avg_pipeline_resource_util_pct"],
            "pipeline_resource_waste_pct": summary["pipeline_resource_waste_pct"],
            "encode_compute_util_pct": summary["encode_compute_util_pct"],
            "prefill_compute_util_pct": summary["prefill_compute_util_pct"],
            "decode_compute_util_pct": summary["decode_compute_util_pct"],
            "encode_bandwidth_util_pct": summary["encode_bandwidth_util_pct"],
            "prefill_bandwidth_util_pct": summary["prefill_bandwidth_util_pct"],
            "decode_bandwidth_util_pct": summary["decode_bandwidth_util_pct"],
            "bottleneck_stage": summary["bottleneck_stage"],
        })

    kernel_rows = extract_kernel_timeline(cur, origin_ns, sample_infos[0]["sample_id"] if sample_infos else "")
    con.close()
    timelines = {
        "observed": observed_rows,
        "sequential": sequential_rows,
        "pipelined": pipelined_rows,
        "kernel": kernel_rows,
    }
    return result_rows, timelines


def write_dict_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
