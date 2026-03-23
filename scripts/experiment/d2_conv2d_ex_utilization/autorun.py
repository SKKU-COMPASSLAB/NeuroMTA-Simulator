import os
import sys
import json
import argparse
import multiprocessing as mp
import time

from neuromta.framework import logger
from neuromta.framework.parser_utils import parse_mem_cap_str

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visualize import visualize_monitoring_data
    VISUALIZE_ENBALED = True
except ImportError as e:
    logger.error(f"Failed to import visualize_monitoring_data from visualize.py: {e}")
    VISUALIZE_ENBALED = False


def run_command(cmd: str) -> None:
    os.system(f"{cmd} > /dev/null 2>&1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run d2 conv2d utilization experiments with bounded parallel workers.")
    parser.add_argument(
        "-n", "--max-procs",
        type=int,
        default=mp.cpu_count(),
        help="Maximum number of processes to run concurrently.",
        dest="max_procs",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Whether to show real-time monitoring window during simulation.",
        dest="monitor",
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

    # AlexNet first Conv2d defaults.
    batch, in_channels, out_channels = 1, 3, 64
    input_height, input_width = 224, 224
    kernel_height, kernel_width = 11, 11
    stride_height, stride_width = 4, 4
    padding_height, padding_width = 2, 2
    dilation_height, dilation_width = 1, 1
    groups = 1

    l1_buf_sizes = list(range(128, 1024+128, 128))

    def get_prefix(use_l1_cache, use_bcast):
        if use_l1_cache:
            prefix = "l1"
        else:
            prefix = "main"
        if use_bcast:
            prefix += "_bcast"
        else:
            prefix += "_wobcast"
        return prefix

    def get_additional_options(use_l1_cache, use_bcast, monitor):
        additional_options = ""
        if use_l1_cache:
            additional_options += " --use-l1-cache"
        if use_bcast:
            additional_options += " --use-bcast"
        if monitor:
            additional_options += " --monitor"
        return additional_options

    cmd_fmt = (
        f"NEUROMTA_MONITOR_SIM_NAME={{prefix}}_{{l1_buf_size}} python3 {ROOT}/main.py "
        f"--batch {batch} --in-channels {in_channels} --out-channels {out_channels} "
        f"--input-height {input_height} --input-width {input_width} "
        f"--kernel-height {kernel_height} --kernel-width {kernel_width} "
        f"--stride-height {stride_height} --stride-width {stride_width} "
        f"--padding-height {padding_height} --padding-width {padding_width} "
        f"--dilation-height {dilation_height} --dilation-width {dilation_width} "
        f"--groups {groups} "
        f"--l1-buf-size {{l1_buf_size}} -o {{output_dir}}"
    )

    for use_l1_cache in [True, False]:
        for use_bcast in [True, False]:
            for l1_buf_size in l1_buf_sizes:
                prefix = get_prefix(use_l1_cache, use_bcast)
                additional_options = get_additional_options(use_l1_cache, use_bcast, args.monitor)

                output_dir = output_dir_fmt.format(prefix=prefix, l1_buf_size=l1_buf_size)
                cmd = cmd_fmt.format(
                    prefix=prefix,
                    l1_buf_size=l1_buf_size * parse_mem_cap_str("1KB"),
                    output_dir=output_dir,
                ) + additional_options
                commands.append(cmd)

    max_procs = args.max_procs
    processes: list[mp.Process] = []
    active_processes: list[mp.Process] = []

    def reap_finished(active: list[mp.Process]) -> list[mp.Process]:
        remaining: list[mp.Process] = []
        for proc in active:
            if proc.is_alive():
                remaining.append(proc)
            else:
                proc.join()
        return remaining

    # for cmd in commands:
    #     while len(active_processes) >= max_procs:
    #         active_processes = reap_finished(active_processes)
    #         if len(active_processes) >= max_procs:
    #             time.sleep(0.05)

    #     p = mp.Process(target=run_command, args=(cmd,))
    #     p.start()
    #     processes.append(p)
    #     active_processes.append(p)
    #     logger.info(f"Started process for command: {cmd}")

    # while len(active_processes) > 0:
    #     active_processes = reap_finished(active_processes)
    #     if len(active_processes) > 0:
    #         time.sleep(0.05)

    # logger.info("All experiments completed.")

    # summarized_results = ["l1_buffer_size,use_l1_cache,use_bcast,core_id,thread,active_time,total_time"]
    # for use_l1_cache in [True, False]:
    #     for use_bcast in [True, False]:
    #         for l1_buf_size in l1_buf_sizes:
    #             prefix = get_prefix(use_l1_cache, use_bcast)
    #             output_dir = output_dir_fmt.format(prefix=prefix, l1_buf_size=l1_buf_size)

    #             exe_time_profile_path = os.path.join(output_dir, "execution_time_profile.json")
    #             with open(exe_time_profile_path, "r") as f:
    #                 exe_time_profile = json.load(f)

    #             for core_id, thread_profile in exe_time_profile.items():
    #                 for thread_id, p in thread_profile.items():
    #                     summarized_results.append(
    #                         f"{l1_buf_size},{use_l1_cache},{use_bcast},{core_id},{thread_id},{p['active_time_cycles']},{p['final_commit_cycles']}"
    #                     )

    # summarized_results_path = os.path.join(LOGDIR, "summarized_results.csv")
    # with open(summarized_results_path, "w") as f:
    #     f.write("\n".join(summarized_results))
    #     logger.info(f"Summarized results saved to '{summarized_results_path}'")
    
    if VISUALIZE_ENBALED:
        for use_l1_cache in [True, False]:
            for use_bcast in [True, False]:
                for l1_buf_size in l1_buf_sizes:
                    prefix = get_prefix(use_l1_cache, use_bcast)
                    output_dir = output_dir_fmt.format(prefix=prefix, l1_buf_size=l1_buf_size)
                    profile_dir = os.path.join(output_dir, "profiles")
                    visualize_monitoring_data(profile_dir, os.path.join(output_dir, "visualizations"))