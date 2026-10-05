import argparse
import csv
import itertools
import os
import time

from common import WORKLOADS, create_device, execute_workloads


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS), default=tuple(WORKLOADS))
    return parser.parse_args(argv)


def run_without_collocation(path: str, workload_names: tuple[str, ...], use_full_mesh: bool) -> float:
    wall_start = time.perf_counter()
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("workload", "start_cycle", "completion_cycle", "execution_cycles"))
        for index, workload_name in enumerate(workload_names, start=1):
            device, partitions = create_device()
            ccg_tile_mesh = partitions.full_ccg_tile_mesh if use_full_mesh else partitions.lhs_ccg_tile_mesh
            profiles, _, _ = execute_workloads(device, ((workload_name, ccg_tile_mesh, partitions.full_dma_ids),))
            profile = profiles[0]
            writer.writerow((workload_name, profile.start_cycle, profile.completion_cycle, profile.execution_cycles))
            print(f"[{os.path.basename(path)} {index}/{len(workload_names)}] {workload_name}: {profile.execution_cycles} cycles")
    return time.perf_counter() - wall_start


def run_with_collocation(path: str, workload_names: tuple[str, ...]) -> float:
    pairs = tuple(itertools.product(workload_names, repeat=2))
    wall_start = time.perf_counter()
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("lhs_workload", "rhs_workload", "lhs_execution_cycles", "rhs_execution_cycles", "collocation_cycles"))
        for index, (lhs_name, rhs_name) in enumerate(pairs, start=1):
            device, partitions = create_device()
            profiles, elapsed_cycles, _ = execute_workloads(device, ((lhs_name, partitions.lhs_ccg_tile_mesh, partitions.full_dma_ids), (rhs_name, partitions.rhs_ccg_tile_mesh, partitions.full_dma_ids)))
            lhs_profile, rhs_profile = profiles
            writer.writerow((lhs_name, rhs_name, lhs_profile.execution_cycles, rhs_profile.execution_cycles, elapsed_cycles))
            print(f"[{os.path.basename(path)} {index}/{len(pairs)}] {lhs_name} + {rhs_name}: {elapsed_cycles} cycles")
    return time.perf_counter() - wall_start


def main(argv=None):
    args = parse_args(argv)
    workload_names = tuple(args.workloads)
    os.makedirs(CACHE_DIR, exist_ok=True)
    lhs_path = os.path.join(CACHE_DIR, "wo_collocation_lhs.csv")
    full_path = os.path.join(CACHE_DIR, "wo_collocation_full.csv")
    collocation_path = os.path.join(CACHE_DIR, "with_collocation.csv")
    total_start = time.perf_counter()
    lhs_time = run_without_collocation(lhs_path, workload_names, use_full_mesh=False)
    full_time = run_without_collocation(full_path, workload_names, use_full_mesh=True)
    collocation_time = run_with_collocation(collocation_path, workload_names)
    print(f"LHS profile: {lhs_path} ({lhs_time:.6f} s)")
    print(f"Full profile: {full_path} ({full_time:.6f} s)")
    print(f"Collocation profile: {collocation_path} ({collocation_time:.6f} s)")
    print(f"Completed {len(workload_names) * 2 + len(workload_names) ** 2} runs in {time.perf_counter() - total_start:.6f} s")


if __name__ == "__main__":
    main()
