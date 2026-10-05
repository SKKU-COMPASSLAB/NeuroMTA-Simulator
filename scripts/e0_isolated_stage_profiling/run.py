import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

from scripts.e0_isolated_stage_profiling.analysis import analyze_sqlite
from scripts.e0_isolated_stage_profiling.dataset import list_sample_ids


CSV_FIELDS = [
    "sample_id",
    "question_id",
    "sample_index",
    "task_type",
    "timestamp_sec",
    "frame_index",
    "sample_video_frames",
    "total_video_frames_seen",
    "vision_patch_tokens",
    "merged_visual_tokens",
    "prefill_tokens",
    "context_tokens_before_decode",
    "decode_tokens",
    "sample_wall_ms",
    "vision_encode_range_count",
    "vision_encode_ms",
    "vision_encode_kernel_ms",
    "vision_encode_measured_memcpy_bytes",
    "vision_encode_estimated_flops",
    "vision_encode_estimated_bytes",
    "vision_encode_avg_tops",
    "vision_encode_avg_bandwidth_gbps",
    "prefill_range_count",
    "prefill_ms",
    "prefill_kernel_ms",
    "prefill_measured_memcpy_bytes",
    "prefill_estimated_flops",
    "prefill_estimated_bytes",
    "prefill_avg_tops",
    "prefill_avg_bandwidth_gbps",
    "decode_range_count",
    "decode_ms",
    "decode_kernel_ms",
    "decode_measured_memcpy_bytes",
    "decode_estimated_flops",
    "decode_estimated_bytes",
    "decode_avg_tops",
    "decode_avg_bandwidth_gbps",
]


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
        "scripts.e0_isolated_stage_profiling.worker",
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
        rows = analyze_sqlite(sqlite_path, model_id=args.model, dtype_bytes=args.dtype_bytes)
        for row in rows:
            row["profile_video_id"] = sample_id
        success = True
        return rows
    finally:
        # The server has limited storage; consume each profile immediately and
        # delete all large profiler artifacts before moving to the next video.
        for path in [rep_path, sqlite_path]:
            remove_if_exists(path)
        if success:
            remove_if_exists(log_path)
        elif os.path.exists(log_path):
            print(f"[debug] kept failed-run log at {log_path}")


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
    parser.add_argument("--work-dir", default=".tmp/e0_isolated_stage_profiling")
    parser.add_argument("--output-csv", default="scripts/e0_isolated_stage_profiling/results.csv")
    args = parser.parse_args()

    if not args.gpu:
        raise ValueError(
            "GPU UUID is required. Pass --gpu MIG-002bc020-db7a-5424-905c-dd94835c8802 "
            "or set CUDA_VISIBLE_DEVICES before running."
        )

    os.makedirs(args.work_dir, exist_ok=True)
    sample_ids = resolve_sample_ids(args)
    if not sample_ids:
        raise RuntimeError("No StreamingBench videos were selected.")

    print(f"[config] videos={sample_ids}")
    print(f"[config] gpu={args.gpu}")
    print(f"[config] output_csv={args.output_csv}")

    all_rows = []
    for index, sample_id in enumerate(sample_ids):
        print(f"\n[video {index + 1}/{len(sample_ids)}] {sample_id}")
        rows = profile_one_video(args, sample_id, args.work_dir)
        all_rows.extend(rows)
        write_csv(args.output_csv, all_rows)
        print(f"[csv] wrote {len(all_rows)} sample rows -> {args.output_csv}")

    print(f"\n[done] {len(all_rows)} sample-level rows written to {args.output_csv}")


if __name__ == "__main__":
    main()
