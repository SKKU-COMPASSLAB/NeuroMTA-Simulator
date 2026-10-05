import os
import json
import inspect
import argparse
import multiprocessing as mp
from importlib import import_module


ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, ".cache")
EXPERIMENTS = {
    ("spatial", "prefill"): "run_spatial_encode_prefill",
    ("spatial", "decode"): "run_spatial_encode_decode",
    ("sequential", "prefill"): "run_seq_encode_prefill",
    ("sequential", "decode"): "run_seq_encode_decode",
}


def _nonnegative_int(value: str):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _positive_int(value: str):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run scheduling efficiency experiments.")
    parser.add_argument("-n", "--n-video-requests", type=_nonnegative_int, default=10, help="Number of video requests.", dest="n_video_requests")
    parser.add_argument("-p", "--prompt-length", type=_positive_int, default=1024, help="Length of the text prompt.", dest="prompt_length")
    parser.add_argument("-d", "--decode-tokens", type=_positive_int, default=16, help="Number of tokens to decode.", dest="decode_tokens")
    parser.add_argument("-c", "--mesh-accel-config-name", type=str, default="medium", choices=["small", "medium", "large"], help="Mesh accelerator configuration name.", dest="mesh_accel_config_name")
    parser.add_argument("--runtime", nargs="+", default=["spatial", "sequential"], choices=["spatial", "sequential"], dest="runtimes", help="Runtime implementations to evaluate.")
    parser.add_argument("--text-stage", nargs="+", default=["prefill", "decode"], choices=["prefill", "decode"], dest="text_stages", help="Text stages to evaluate with Encode.")
    parser.add_argument("--max-parallel", type=_positive_int, default=len(EXPERIMENTS), help="Maximum number of experiment processes to run concurrently.", dest="max_parallel")
    parser.add_argument("--debug-log", action="store_true", help="Enable runtime debug logging.", dest="enable_debug_log")
    parser.add_argument("--dry-run", action="store_true", help="Validate experiment entry points without running simulations.", dest="dry_run")
    return parser.parse_args(argv)


def _load_experiment(module_name: str):
    return import_module(f".{module_name}", package=__package__) if __package__ else import_module(module_name)


def _run_arguments(args):
    return {
        "n_video_requests": args.n_video_requests,
        "prompt_length": args.prompt_length,
        "decode_tokens": args.decode_tokens,
        "enable_debug_log": args.enable_debug_log,
        "mesh_accel_config_name": args.mesh_accel_config_name,
    }


def _selected_experiments(args):
    return [EXPERIMENTS[(runtime, text_stage)] for runtime in args.runtimes for text_stage in args.text_stages]


def _validate_experiment(module_name: str, run_arguments: dict):
    module = _load_experiment(module_name)
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"{module_name} does not define a callable run()")
    inspect.signature(run).bind(**run_arguments)


def _run_experiment(module_name: str, run_arguments: dict):
    module = _load_experiment(module_name)
    module.run(**run_arguments)


def _run_experiments(experiments: list[str], run_arguments: dict, max_parallel: int):
    context = mp.get_context("spawn")
    processes = []
    for offset in range(0, len(experiments), max_parallel):
        batch = []
        try:
            for experiment in experiments[offset:offset + max_parallel]:
                process = context.Process(target=_run_experiment, args=(experiment, run_arguments), name=experiment)
                process.start()
                processes.append(process)
                batch.append(process)
            for process in batch:
                process.join()
        finally:
            for process in batch:
                if process.is_alive():
                    process.terminate()
            for process in batch:
                process.join()
        failures = [f"{process.name} (exitcode={process.exitcode})" for process in batch if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"Experiments failed: {', '.join(failures)}")
    return processes


def main(argv=None):
    args = parse_args(argv)
    experiments = _selected_experiments(args)
    run_arguments = _run_arguments(args)
    for experiment in experiments:
        _validate_experiment(experiment, run_arguments)
    if args.dry_run:
        print(f"Validated experiments: {', '.join(experiments)}")
        return
    _run_experiments(experiments, run_arguments, min(args.max_parallel, len(experiments)))
    os.makedirs(CACHE_DIR, exist_ok=True)
    summary_file = os.path.join(CACHE_DIR, "summary.json")
    summary = {**run_arguments, "experiments": experiments, "max_parallel": args.max_parallel}
    with open(summary_file, "w") as f:
        json.dump(summary, f)
    print(f"Summary written to '{summary_file}'")


if __name__ == "__main__":
    main()
