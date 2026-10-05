import argparse
import csv
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import CACHE_DIR, DEFAULT_CCG_TOPS, DEFAULT_DURATION_CYCLES, DEFAULT_WARMUP_CYCLES

EXPERIMENTS = ("run_sequential", "run_virtual", "run_preemptive", "run_spatial")
RUNTIME_TO_EXPERIMENT = {name.removeprefix("run_"): name for name in EXPERIMENTS}
SUMMARY_FIELDS = (
    "scheduler",
    "job_count",
    "deadline_misses",
    "deadline_miss_rate",
    "job_response_time_mean_cycles",
    "job_response_time_p95_cycles",
    "job_response_time_p99_cycles",
    "job_response_time_max_cycles",
    "job_queue_time_mean_cycles",
    "job_queue_time_p95_cycles",
    "job_execution_time_mean_cycles",
    "job_execution_time_p95_cycles",
    "minimum_slack_cycles",
    "kernel_count",
    "total_kernel_execution_time_cycles",
    "kernel_execution_time_mean_cycles",
    "core_count_average",
    "core_count_stddev",
    "scheduler_decision_count",
    "multi_kernel_decision_count",
    "multi_kernel_decision_rate",
    "max_kernels_per_decision",
    "positive_benefit_decision_count",
    "predicted_benefit_mean",
    "predicted_benefit_mean_positive",
    "simulation_completion_cycle",
    "simulator_wall_time_seconds",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("all",) + tuple(RUNTIME_TO_EXPERIMENT), default="all")
    parser.add_argument("--duration-cycles", type=int, default=DEFAULT_DURATION_CYCLES)
    parser.add_argument("--warmup-cycles", type=int, default=DEFAULT_WARMUP_CYCLES)
    parser.add_argument("--ccg-tops", type=float, default=DEFAULT_CCG_TOPS)
    parser.add_argument("--max-parallel", type=int, default=len(EXPERIMENTS))
    parser.add_argument("--debug-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _import_experiment(module_name: str):
    return importlib.import_module(module_name)


def _validate_experiment(module_name: str, run_arguments: dict) -> None:
    module = _import_experiment(module_name)
    if not hasattr(module, "run") or not callable(module.run):
        raise RuntimeError(f"Experiment module '{module_name}' does not define a callable run().")
    inspect.signature(module.run).bind(**run_arguments)


def _run_experiment(module_name: str, run_arguments: dict) -> None:
    _import_experiment(module_name).run(**run_arguments)


def _run_experiments(experiments: list[str], run_arguments: dict, max_parallel: int) -> None:
    context = mp.get_context("spawn")
    for offset in range(0, len(experiments), max_parallel):
        processes = []
        try:
            for module_name in experiments[offset:offset + max_parallel]:
                process = context.Process(name=module_name, target=_run_experiment, args=(module_name, run_arguments))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
        failures = [f"{process.name} (exitcode={process.exitcode})" for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"Experiments failed: {', '.join(failures)}")


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _summarize_experiment(experiment: str) -> dict:
    profile_dir = os.path.join(CACHE_DIR, experiment)
    with open(os.path.join(profile_dir, "metadata.json")) as file:
        metadata = json.load(file)
    jobs = _read_csv(os.path.join(profile_dir, "workload_profile.csv"))
    if not jobs:
        raise RuntimeError(f"Experiment '{experiment}' has no measured jobs to summarize.")
    response_times = [int(job["response_time"]) for job in jobs]
    queue_times = [int(job["start_cycle"]) - int(job["arrival_cycle"]) for job in jobs]
    execution_times = [int(job["execution_time"]) for job in jobs]
    slacks = [int(job["slack"]) for job in jobs]
    deadline_misses = sum(job["deadline_met"] != "True" for job in jobs)
    kernel_rows = []
    kernel_profile_dir = os.path.join(profile_dir, "kernel_profile")
    for filename in sorted(os.listdir(kernel_profile_dir)):
        if filename.endswith(".csv"):
            kernel_rows.extend(_read_csv(os.path.join(kernel_profile_dir, filename)))
    kernel_count = sum(int(row["count"]) for row in kernel_rows)
    total_kernel_execution_time = sum(int(row["count"]) * float(row["average_latency"]) for row in kernel_rows)
    core_count_sum = sum(int(row["count"]) * float(row["core_count_average"]) for row in kernel_rows)
    core_count_square_sum = sum(int(row["count"]) * (float(row["core_count_stddev"]) ** 2 + float(row["core_count_average"]) ** 2) for row in kernel_rows)
    core_count_average = core_count_sum / kernel_count
    core_count_stddev = math.sqrt(max(0.0, core_count_square_sum / kernel_count - core_count_average ** 2))
    decisions = [row for row in _read_csv(os.path.join(profile_dir, "scheduler_profile.csv")) if int(row["cycle"]) >= int(metadata["warmup_cycles"])]
    kernels_per_decision = [len(row["kernel_ids"].split("|")) for row in decisions]
    benefits = [float(row["benefit"]) for row in decisions]
    positive_benefits = [benefit for benefit in benefits if benefit > 0]
    multi_kernel_decisions = sum(count > 1 for count in kernels_per_decision)
    return {
        "scheduler": experiment.removeprefix("run_"),
        "job_count": len(jobs),
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / len(jobs),
        "job_response_time_mean_cycles": sum(response_times) / len(response_times),
        "job_response_time_p95_cycles": _percentile(response_times, 0.95),
        "job_response_time_p99_cycles": _percentile(response_times, 0.99),
        "job_response_time_max_cycles": max(response_times),
        "job_queue_time_mean_cycles": sum(queue_times) / len(queue_times),
        "job_queue_time_p95_cycles": _percentile(queue_times, 0.95),
        "job_execution_time_mean_cycles": sum(execution_times) / len(execution_times),
        "job_execution_time_p95_cycles": _percentile(execution_times, 0.95),
        "minimum_slack_cycles": min(slacks),
        "kernel_count": kernel_count,
        "total_kernel_execution_time_cycles": total_kernel_execution_time,
        "kernel_execution_time_mean_cycles": total_kernel_execution_time / kernel_count,
        "core_count_average": core_count_average,
        "core_count_stddev": core_count_stddev,
        "scheduler_decision_count": len(decisions),
        "multi_kernel_decision_count": multi_kernel_decisions,
        "multi_kernel_decision_rate": multi_kernel_decisions / len(decisions) if decisions else 0.0,
        "max_kernels_per_decision": max(kernels_per_decision, default=0),
        "positive_benefit_decision_count": len(positive_benefits),
        "predicted_benefit_mean": sum(benefits) / len(benefits) if benefits else 0.0,
        "predicted_benefit_mean_positive": sum(positive_benefits) / len(positive_benefits) if positive_benefits else 0.0,
        "simulation_completion_cycle": metadata["simulation_completion_cycle"],
        "simulator_wall_time_seconds": metadata["wall_time_seconds"],
    }


def summarize_results(experiments: list[str]) -> str:
    path = os.path.join(CACHE_DIR, "summarized_result.csv")
    summaries = [_summarize_experiment(experiment) for experiment in experiments]
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", *(summary["scheduler"] for summary in summaries)))
        for metric in SUMMARY_FIELDS[1:]:
            writer.writerow((metric, *(summary[metric] for summary in summaries)))
    return path


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.duration_cycles <= 0:
        raise ValueError("--duration-cycles must be positive.")
    if args.warmup_cycles < 0 or args.warmup_cycles >= args.duration_cycles:
        raise ValueError("--warmup-cycles must be non-negative and smaller than --duration-cycles.")
    if args.ccg_tops <= 0:
        raise ValueError("--ccg-tops must be positive.")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive.")
    experiments = list(EXPERIMENTS) if args.runtime == "all" else [RUNTIME_TO_EXPERIMENT[args.runtime]]
    run_arguments = {"duration_cycles": args.duration_cycles, "warmup_cycles": args.warmup_cycles, "ccg_tops": args.ccg_tops, "enable_debug_log": args.debug_log}
    for experiment in experiments:
        _validate_experiment(experiment, run_arguments)
    if args.dry_run:
        print(f"Validated experiments: {', '.join(experiments)}")
        return
    parallelism = min(args.max_parallel, len(experiments))
    print(f"Running {len(experiments)} experiments with up to {parallelism} parallel processes: {', '.join(experiments)}")
    start_time = time.perf_counter()
    _run_experiments(experiments, run_arguments, parallelism)
    summary_path = summarize_results(experiments)
    print(f"Summary: {summary_path}")
    print(f"Completed all experiments; total wall time: {time.perf_counter() - start_time:.6f} s")


if __name__ == "__main__":
    main()
