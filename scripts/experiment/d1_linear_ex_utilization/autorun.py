import os
import json
import argparse
import multiprocessing as mp

from neuromta.framework import logger
from neuromta.framework.parser_utils import parse_mem_cap_str


def run_command(cmd: str) -> None:
    os.system(f"{cmd} > /dev/null 2>&1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run d1 linear utilization experiments with bounded parallel workers.")
    parser.add_argument(
        "-n", "--max-procs",
        type=int,
        default=mp.cpu_count(),
        help="Maximum number of processes to run concurrently.",
        dest="max_procs",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.max_procs < 1:
        raise ValueError(f"--max-procs must be >= 1, got {args.max_procs}")

    ROOT = os.path.abspath(os.path.dirname(__file__))
    LOGDIR = os.path.join(ROOT, ".logs")
    output_dir_fmt = os.path.join(LOGDIR, "{prefix}_l1_buf_{l1_buf_size}")
    
    commands = []
    
    M, N, K = 512, 512, 256
    l1_buf_sizes = list(range(32, 1024+32, 32))
    
    def get_prefix(use_l1_cache, use_bcast):
        if use_l1_cache:
            prefix = "l1"
        else:
            prefix = "main"
        if use_bcast:
            prefix += "_with_bcast"
        else:
            prefix += "_without_bcast"
        return prefix
    
    def get_additional_options(use_l1_cache, use_bcast):
        additional_options = ""
        if use_l1_cache:
            additional_options += " --use-l1-cache"
        if use_bcast:
            additional_options += " --use-bcast"
        return additional_options

    # Run with PP
    cmd_fmt = f"python3 {ROOT}/main.py -m {M} -n {N} -k {K} --l1-buf-size {{l1_buf_size}} -o {{output_dir}}"
    for use_l1_cache in [True, False]:
        for use_bcast in [True, False]:
            for l1_buf_size in l1_buf_sizes:
                prefix = get_prefix(use_l1_cache, use_bcast)
                additional_options = get_additional_options(use_l1_cache, use_bcast)
                
                output_dir = output_dir_fmt.format(prefix=prefix, l1_buf_size=l1_buf_size)
                cmd = cmd_fmt.format(l1_buf_size=l1_buf_size * parse_mem_cap_str("1KB"), output_dir=output_dir) + additional_options
                commands.append(cmd)
    
    max_procs = args.max_procs
    processes: list[mp.Process] = []
    active_processes: list[mp.Process] = []

    for cmd in commands:
        while len(active_processes) >= max_procs:
            oldest = active_processes.pop(0)
            oldest.join()

        p = mp.Process(target=run_command, args=(cmd,))
        p.start()
        processes.append(p)
        active_processes.append(p)
        logger.info(f"Started process for command: {cmd}")

    for p in active_processes:
        p.join()
        
    logger.info("All experiments completed.")
    
    summarized_results = ["l1_buffer_size,use_l1_cache,use_bcast,core_id,thread,active_time,total_time"]
    for use_l1_cache in [True, False]:
        for use_bcast in [True, False]:
            for l1_buf_size in l1_buf_sizes:
                prefix = get_prefix(use_l1_cache, use_bcast)
                output_dir = output_dir_fmt.format(prefix=prefix, l1_buf_size=l1_buf_size)
                
                exe_time_profile_path = os.path.join(output_dir, "execution_time_profile.json")
                with open(exe_time_profile_path, "r") as f:
                    exe_time_profile = json.load(f)
                
                for core_id, thread_profile in exe_time_profile.items():
                    for thread_id, p in thread_profile.items():
                        summarized_results.append(f"{l1_buf_size},{use_l1_cache},{use_bcast},{core_id},{thread_id},{p['active_time_cycles']},{p['final_commit_cycles']}")
                
    summarized_results_path = os.path.join(LOGDIR, "summarized_results.csv")
    with open(summarized_results_path, "w") as f:
        f.write("\n".join(summarized_results))
        logger.info(f"Summarized results saved to '{summarized_results_path}'")