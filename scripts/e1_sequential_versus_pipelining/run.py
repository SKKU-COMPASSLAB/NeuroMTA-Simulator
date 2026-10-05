import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

from scripts.e1_sequential_versus_pipelining.analysis import analyze_sqlite, write_dict_csv
from scripts.e1_sequential_versus_pipelining.dataset import list_sample_ids


CSV_FIELDS = [
    "row_type",
    "sample_id",
    "question_id",
    "sample_index",
    "task_type",
    "timestamp_sec",
    "frame_index",
    "sample_video_frames",
    "vision_patch_tokens",
    "merged_visual_tokens",
    "prefill_tokens",
    "context_tokens_before_decode",
    "decode_tokens",
    "sample_wall_ms",
    "encode_step_count",
    "prefill_range_count",
    "decode_step_count",
    "encode_ms",
    "prefill_ms",
    "decode_ms",
    "stage_sum_ms",
    "local_ideal_speedup",
    "sequential_makespan_ms",
    "pipelined_makespan_ms",
    "pipeline_speedup",
    "pipeline_overlap_ratio",
    "encode_resource_util_pct",
    "prefill_resource_util_pct",
    "decode_resource_util_pct",
    "avg_pipeline_resource_util_pct",
    "pipeline_resource_waste_pct",
    "encode_compute_util_pct",
    "prefill_compute_util_pct",
    "decode_compute_util_pct",
    "encode_bandwidth_util_pct",
    "prefill_bandwidth_util_pct",
    "decode_bandwidth_util_pct",
    "bottleneck_stage",
    "profile_video_id",
]

TIMELINE_FIELDS = [
    "schedule",
    "sample_id",
    "question_id",
    "sample_index",
    "task_type",
    "stage",
    "step_index",
    "start_ms",
    "end_ms",
    "duration_ms",
    "kernel_ms",
    "estimated_flops",
    "estimated_bytes",
    "avg_tops",
    "avg_bandwidth_gbps",
    "compute_util_pct",
    "bandwidth_util_pct",
    "name",
]

KERNEL_FIELDS = ["sample_id", "kernel_index", "stream_id", "start_ms", "end_ms", "duration_ms"]


def run_checked(cmd, env=None, stdout_path=None):
    print("[run]", " ".join(cmd))
    if stdout_path is None:
        subprocess.run(cmd, check=True, env=env)
        return
    with open(stdout_path, "w") as f:
        subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)


def remove_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)
        print(f"[cleanup] removed {path}")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_sample_ids(args):
    if args.sample_ids:
        return args.sample_ids
    sample_ids = list_sample_ids(args.data_dir)
    if args.max_videos is not None:
        sample_ids = sample_ids[: args.max_videos]
    return sample_ids


def write_timeline_outputs(timeline_dir, sample_id, timelines):
    os.makedirs(timeline_dir, exist_ok=True)
    for name in ["observed", "sequential", "pipelined"]:
        path = os.path.join(timeline_dir, f"{sample_id}_{name}_timeline.csv")
        write_dict_csv(path, timelines.get(name, []), TIMELINE_FIELDS)
        print(f"[timeline] wrote {path}")
    kernel_rows = timelines.get("kernel", [])
    if kernel_rows:
        path = os.path.join(timeline_dir, f"{sample_id}_kernel_timeline.csv")
        write_dict_csv(path, kernel_rows, KERNEL_FIELDS)
        print(f"[timeline] wrote {path}")


def profile_one_video(args, sample_id, work_dir):
    base = os.path.join(work_dir, f"profile_{sample_id}")
    rep_path = base + ".nsys-rep"
    sqlite_path = base + ".sqlite"
    log_path = base + ".log"

    for path in [rep_path, sqlite_path, log_path]:
        remove_if_exists(path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    profile_cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx",
        "-o",
        base,
        "--force-overwrite",
        "true",
        "--cuda-um-cpu-page-faults",
        "false",
        "--cuda-um-gpu-page-faults",
        "false",
        "--cpuctxsw=none",
        sys.executable,
        "-m",
        "scripts.e1_sequential_versus_pipelining.worker",
        "--model",
        args.model,
        "--data-dir",
        args.data_dir,
        "--sample-id",
        sample_id,
        "--fps",
        str(args.fps),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--attn-implementation",
        args.attn_implementation,
    ]

    stats_cmd = [
        "nsys",
        "stats",
        "--force-export",
        "true",
        "--force-overwrite",
        "true",
        "--sqlite",
        sqlite_path,
        rep_path,
    ]

    success = False
    try:
        run_checked(profile_cmd, env=env)
        run_checked(stats_cmd, env=env, stdout_path=log_path)
        rows, timelines = analyze_sqlite(
            sqlite_path,
            model_id=args.model,
            dtype_bytes=args.dtype_bytes,
            peak_tops=args.peak_tops,
            peak_bandwidth_gbps=args.peak_bandwidth_gbps,
        )
        for row in rows:
            row["profile_video_id"] = sample_id
        write_timeline_outputs(args.timeline_dir, sample_id, timelines)
        success = True
        return rows
    finally:
        if not args.keep_profiles:
            for path in [rep_path, sqlite_path]:
                remove_if_exists(path)
        if success and not args.keep_logs:
            remove_if_exists(log_path)
        elif os.path.exists(log_path):
            print(f"[debug] kept log at {log_path}")


def main():
    default_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=default_gpu, help="GPU UUID or MIG UUID used as CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--peak-tops", type=float, default=312.0)
    parser.add_argument("--peak-bandwidth-gbps", type=float, default=1555.0)
    parser.add_argument("--work-dir", default=".tmp/e1_sequential_versus_pipelining")
    parser.add_argument("--output-csv", default="scripts/e1_sequential_versus_pipelining/results.csv")
    parser.add_argument("--timeline-dir", default="scripts/e1_sequential_versus_pipelining/timelines")
    parser.add_argument("--keep-profiles", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        raise ValueError("GPU UUID is required. Pass --gpu or set CUDA_VISIBLE_DEVICES before running.")

    os.makedirs(args.work_dir, exist_ok=True)
    sample_ids = resolve_sample_ids(args)
    if not sample_ids:
        raise RuntimeError("No StreamingBench videos were selected.")

    print(f"[config] videos={sample_ids}")
    print(f"[config] gpu={args.gpu}")
    print(f"[config] output_csv={args.output_csv}")
    print(f"[config] timeline_dir={args.timeline_dir}")

    all_rows = []
    for index, sample_id in enumerate(sample_ids):
        print(f"\n[video {index + 1}/{len(sample_ids)}] {sample_id}")
        rows = profile_one_video(args, sample_id, args.work_dir)
        all_rows.extend(rows)
        write_csv(args.output_csv, all_rows)
        print(f"[csv] wrote {len(all_rows)} rows -> {args.output_csv}")

    print(f"\n[done] {len(all_rows)} rows written to {args.output_csv}")


if __name__ == "__main__":
    main()
