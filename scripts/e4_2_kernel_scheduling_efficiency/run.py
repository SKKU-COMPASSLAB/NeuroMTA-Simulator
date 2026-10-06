import argparse
import csv
import itertools
import json
import multiprocessing
import os
import time
from contextlib import nullcontext
from typing import Callable

from neuromta.system.hardware.mesh_accelerator import MeshAccelerator

from common import (
    WORKLOADS,
    create_device,
    execute_workloads_with_sequential,
    execute_workloads_with_virtual,
    execute_workloads_with_preemptive,
    execute_workloads_with_spatial,
    KernelExecutionProfile,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
TARGETS: dict[str, Callable[[MeshAccelerator, str, str], tuple[tuple[KernelExecutionProfile, ...], int, float]]] = {
    "sequential": execute_workloads_with_sequential,
    "virtual": execute_workloads_with_virtual,
    "preemptive": execute_workloads_with_preemptive,
    "spatial": execute_workloads_with_spatial,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS), default=tuple(WORKLOADS))
    return parser.parse_args(argv)

def run_with_collocation(path: str, target_name: str, workload_names: tuple[str, ...]) -> float:
    execute_workloads = TARGETS[target_name]

    pairs = tuple(itertools.product(workload_names, repeat=2))
    wall_start = time.perf_counter()
    decision_path = os.path.join(CACHE_DIR, "spatial_scheduling_decisions.csv")
    decision_file_context = open(decision_path, "w", newline="") if target_name == "spatial" else nullcontext()
    with open(path, "w", newline="") as file, decision_file_context as decision_file:
        writer = csv.writer(file)
        writer.writerow(("workload1", "workload2", "core_count1", "core_count2", "execution_cycles1", "execution_cycles2", "collocation_cycles"))
        decision_writer = csv.writer(decision_file) if decision_file is not None else None
        if decision_writer is not None:
            decision_writer.writerow(("workload1", "workload2", "decision_index", "decision_cycle", "candidate_index", "phase", "reason", "selection_reason", "candidate_id", "kernel_ids", "workload_ids", "mesh_shapes", "core_ids", "predicted_kernel_cycles", "sequential_cycles", "makespan_cycles", "benefit", "peak_dma_utilization"))

        for index, (workload_id_1, workload_id_2) in enumerate(pairs, start=1):
            device = create_device()
            decisions = []
            if target_name == "spatial":
                profiles, elapsed_cycles, _ = execute_workloads(device, workload_id_1, workload_id_2, decision_hook=lambda context, records: decisions.append((context, records)))
            else:
                profiles, elapsed_cycles, _ = execute_workloads(device, workload_id_1, workload_id_2)
            lhs_profile, rhs_profile = profiles
            writer.writerow((workload_id_1, workload_id_2, lhs_profile.core_count, rhs_profile.core_count, lhs_profile.execution_cycles, rhs_profile.execution_cycles, elapsed_cycles))
            if decision_writer is not None:
                for decision_index, (context, records) in enumerate(decisions):
                    for candidate_index, record in enumerate(records):
                        decision_writer.writerow((workload_id_1, workload_id_2, decision_index, context.cycle, candidate_index, record["phase"], record["reason"], record["selection_reason"], record["candidate_id"], json.dumps(record["kernel_ids"]), json.dumps(record["workload_ids"]), json.dumps(record["mesh_shapes"]), json.dumps(record["core_ids"]), json.dumps(record["predicted_kernel_cycles"]), record["sequential_cycles"], record["makespan_cycles"], record["benefit"], record["peak_dma_utilization"]))
    
    print(f"Generated collocation results for {target_name} at {path}")
    if target_name == "spatial":
        print(f"Generated spatial scheduling decisions at {decision_path}")
     
    return time.perf_counter() - wall_start


def main(argv=None):
    args = parse_args(argv)
    
    workload_names = tuple(args.workloads)
    result_file_path_fmt = os.path.join(CACHE_DIR, "collocation_{target}.csv")
    
    os.makedirs(CACHE_DIR, exist_ok=True)

    total_start = time.perf_counter()
    
    processes = {
        target_name: multiprocessing.Process(target=run_with_collocation, args=(result_file_path_fmt.format(target=target_name), target_name, workload_names), name=f"run_{target_name}")
        for target_name in TARGETS
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    failures = tuple(f"{target_name} (exitcode={process.exitcode})" for target_name, process in processes.items() if process.exitcode != 0)
    if failures:
        raise RuntimeError(f"Experiments failed: {', '.join(failures)}")

    print(f"Completed {len(TARGETS) * len(workload_names) ** 2} runs in {time.perf_counter() - total_start:.6f} s")


if __name__ == "__main__":
    main()
