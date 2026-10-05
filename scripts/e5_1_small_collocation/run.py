import argparse
import importlib
import inspect
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

EXPERIMENTS = ("run_sequential", "run_virtual", "run_preemptive", "run_spatial")
RUNTIME_TO_EXPERIMENT = {name.removeprefix("run_"): name for name in EXPERIMENTS}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("all",) + tuple(RUNTIME_TO_EXPERIMENT), default="all")
    parser.add_argument("--duration-cycles", type=int, default=4_000_000_000)
    parser.add_argument("--warmup-cycles", type=int, default=1_000_000_000)
    parser.add_argument("--ccg-tops", type=float, default=0.5)
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
            for module_name in experiments[offset : offset + max_parallel]:
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
    run_arguments = {
        "duration_cycles": args.duration_cycles,
        "warmup_cycles": args.warmup_cycles,
        "ccg_tops": args.ccg_tops,
        "enable_debug_log": args.debug_log,
    }
    for experiment in experiments:
        _validate_experiment(experiment, run_arguments)
    if args.dry_run:
        print(f"Validated experiments: {', '.join(experiments)}")
        return
    parallelism = min(args.max_parallel, len(experiments))
    print(f"Running {len(experiments)} experiments with up to {parallelism} parallel processes: {', '.join(experiments)}")
    start_time = time.perf_counter()
    _run_experiments(experiments, run_arguments, parallelism)
    print(f"Completed all experiments; total wall time: {time.perf_counter() - start_time:.6f} s")


if __name__ == "__main__":
    main()
