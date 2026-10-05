import argparse
import csv
import functools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRCS_ROOT = os.path.join(REPO_ROOT, "srcs")
if SRCS_ROOT not in sys.path:
    sys.path.insert(0, SRCS_ROOT)

import pybooksim2

from neuromta.component.context.icnt_context import IcntConfig, IcntSimulator


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
SHAPE = (4, 10)
FLIT_SIZE = 64
MAX_PAYLOAD_FLITS = 32
SUBNETS = 2
RESULT_FIELDS = [
    "test",
    "case",
    "model",
    "request_count",
    "total_flits",
    "pattern",
    "completion_cycles",
    "first_completion_cycles",
    "mean_latency_cycles",
    "p50_latency_cycles",
    "p95_latency_cycles",
    "effective_flits_per_cycle",
    "relative_error",
]


@dataclass(frozen=True)
class Request:
    request_id: int
    issue_cycle: int
    src_id: int
    dst_id: int
    n_flits: int
    is_write: bool = True


@dataclass
class TraceResult:
    completion_cycle: int
    completion_cycles: dict[int, int]
    latencies: dict[int, int]


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def coord_to_node_id(coord: tuple[int, int]) -> int:
    return coord[0] * SHAPE[1] + coord[1]


def create_config(booksim2_enable: bool) -> IcntConfig:
    config = IcntConfig(
        processor_clock_freq=1_000_000_000,
        shape=SHAPE,
        flit_size=FLIT_SIZE,
        max_payload_size=MAX_PAYLOAD_FLITS,
        subnets=SUBNETS,
        booksim2_enable=booksim2_enable,
        lightweight_router_latency_cycles=1,
        lightweight_link_latency_cycles=1,
        lightweight_flits_per_cycle_per_channel=4,
        lightweight_injection_flits_per_cycle=4,
        lightweight_egress_flits_per_cycle=4,
        lightweight_router_allocation_cycles=1,
        lightweight_packet_startup_cycles=1,
        lightweight_min_packet_cycles=1,
        lightweight_payload_issue_gap_cycles=1,
        booksim2_kwargs={
            "routing_delay": 1,
            "vc_alloc_delay": 1,
            "sw_alloc_delay": 1,
            "st_prepare_delay": 0,
            "st_final_delay": 1,
            "input_speedup": 1,
            "output_speedup": 1,
            "internal_speedup": 1.0,
            "num_vcs": 16,
            "vc_buf_size": 8,
        },
    )
    for row in range(SHAPE[0]):
        for column in range(SHAPE[1]):
            node_id = coord_to_node_id((row, column))
            config.update_core_map((row, column), node_id)
    return config


def request_payloads(request: Request) -> list[tuple[int, int, int]]:
    payload_count = math.ceil(request.n_flits / MAX_PAYLOAD_FLITS)
    payloads = []
    for payload_index in range(payload_count):
        n_flits = min(MAX_PAYLOAD_FLITS, request.n_flits - payload_index * MAX_PAYLOAD_FLITS)
        subnet = (request.src_id + request.dst_id + payload_index) % SUBNETS
        payloads.append((payload_index, subnet, n_flits))
    return payloads


def run_lightweight_trace(requests: list[Request]) -> TraceResult:
    simulator = IcntSimulator(create_config(booksim2_enable=False))
    completions = {}
    latencies = {}
    for request in sorted(requests, key=lambda item: (item.issue_cycle, item.request_id)):
        result = simulator.send_request(
            src_core_id=request.src_id,
            dst_core_id=request.dst_id,
            data_size=request.n_flits * FLIT_SIZE,
            is_write=request.is_write,
            current_cycle=request.issue_cycle,
        )
        completions[request.request_id] = result["finish_cycle"]
        latencies[request.request_id] = result["latency_cycles"]
    return TraceResult(max(completions.values(), default=0), completions, latencies)


def run_booksim_trace_worker(requests: list[Request]) -> TraceResult:
    config = create_config(booksim2_enable=True)
    interconnect = config.booksim2_config.create_icnt()
    cycle = 0
    pending = sorted(requests, key=lambda item: (item.issue_cycle, item.request_id))
    pending_index = 0
    payload_completions = {}
    request_completions = {}
    commands = []

    def complete(request_id: int, payload_index: int, _capsule) -> None:
        payload_completions[(request_id, payload_index)] = cycle
        expected = len(request_payloads(next(item for item in requests if item.request_id == request_id)))
        completed = [value for (rid, _), value in payload_completions.items() if rid == request_id]
        if len(completed) == expected:
            request_completions[request_id] = max(completed)

    max_cycles = max((request.issue_cycle for request in requests), default=0) + 10_000_000
    while len(request_completions) < len(requests):
        while pending_index < len(pending) and pending[pending_index].issue_cycle <= cycle:
            request = pending[pending_index]
            for payload_index, subnet, n_flits in request_payloads(request):
                command = pybooksim2.create_icnt_cmd_data_packet(
                    src_id=request.src_id,
                    dst_id=request.dst_id,
                    subnet=subnet,
                    size=n_flits,
                    is_write=request.is_write,
                    is_response=not request.is_write,
                )
                pybooksim2.icnt_dispatch_cmd(
                    icnt=interconnect,
                    cmd=command,
                    dispatch_callback=None,
                    execute_callback=functools.partial(complete, request.request_id, payload_index),
                )
                commands.append(command)
            pending_index += 1
        pybooksim2.icnt_cycle_step(icnt=interconnect, cycles=1)
        cycle += 1
        if cycle > max_cycles:
            raise TimeoutError(f"BookSim2 trace did not finish within {max_cycles} cycles")
    latencies = {
        request.request_id: request_completions[request.request_id] - request.issue_cycle
        for request in requests
    }
    return TraceResult(max(request_completions.values(), default=0), request_completions, latencies)


def run_booksim_trace(requests: list[Request]) -> TraceResult:
    with tempfile.TemporaryDirectory(prefix="v4_booksim_") as directory:
        input_path = os.path.join(directory, "input.json")
        output_path = os.path.join(directory, "output.json")
        with open(input_path, "w") as output:
            json.dump([request.__dict__ for request in requests], output)
        process = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--booksim-worker", input_path, output_path],
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"BookSim2 worker failed with code {process.returncode}\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        with open(output_path) as source:
            result = json.load(source)
    return TraceResult(
        completion_cycle=result["completion_cycle"],
        completion_cycles={int(key): value for key, value in result["completion_cycles"].items()},
        latencies={int(key): value for key, value in result["latencies"].items()},
    )


def booksim_worker(input_path: str, output_path: str) -> None:
    with open(input_path) as source:
        requests = [Request(**item) for item in json.load(source)]
    result = run_booksim_trace_worker(requests)
    with open(output_path, "w") as output:
        json.dump({
            "completion_cycle": result.completion_cycle,
            "completion_cycles": result.completion_cycles,
            "latencies": result.latencies,
        }, output)


def make_summary_row(test: str, case: str, model: str, requests: list[Request], result: TraceResult, pattern: str) -> dict:
    latencies = list(result.latencies.values())
    total_flits = sum(request.n_flits for request in requests)
    first_issue = min((request.issue_cycle for request in requests), default=0)
    duration = max(1, result.completion_cycle - first_issue)
    return {
        "test": test,
        "case": case,
        "model": model,
        "request_count": len(requests),
        "total_flits": total_flits,
        "pattern": pattern,
        "completion_cycles": result.completion_cycle,
        "first_completion_cycles": min(result.completion_cycles.values(), default=0),
        "mean_latency_cycles": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_cycles": percentile(latencies, 0.50),
        "p95_latency_cycles": percentile(latencies, 0.95),
        "effective_flits_per_cycle": total_flits / duration,
        "relative_error": "",
    }


def compare_trace(
    rows: list[dict],
    test: str,
    case: str,
    requests: list[Request],
    pattern: str,
) -> tuple[TraceResult, TraceResult]:
    lightweight = run_lightweight_trace(requests)
    booksim = run_booksim_trace(requests)
    lightweight_row = make_summary_row(test, case, "lightweight", requests, lightweight, pattern)
    booksim_row = make_summary_row(test, case, "booksim2", requests, booksim, pattern)
    lightweight_row["relative_error"] = (
        lightweight.completion_cycle - booksim.completion_cycle
    ) / max(1, booksim.completion_cycle)
    booksim_row["relative_error"] = 0.0
    rows.extend([lightweight_row, booksim_row])
    return lightweight, booksim


def test_packet_size_scaling(rows: list[dict], quick: bool) -> None:
    sizes = [1, 2, 4, 8, 16, 32, 64]
    if not quick:
        sizes.extend([128, 512])
    src = coord_to_node_id((0, 0))
    dst = coord_to_node_id((0, 4))
    for n_flits in sizes:
        request = Request(0, 0, src, dst, n_flits)
        compare_trace(rows, "packet_size", f"{n_flits}_flits", [request], "zero_load_4_hops")


def test_hop_scaling(rows: list[dict]) -> None:
    src = coord_to_node_id((0, 0))
    destinations = [(0, 1), (0, 2), (0, 4), (1, 4), (3, 9)]
    for destination in destinations:
        dst = coord_to_node_id(destination)
        hop_count = abs(destination[0]) + abs(destination[1])
        request = Request(0, 0, src, dst, 8)
        compare_trace(rows, "hop_count", f"mesh_hops_{hop_count}_{src}_to_{dst}", [request], "zero_load_8_flits")


def test_same_path_contention(rows: list[dict], quick: bool) -> None:
    src = coord_to_node_id((0, 0))
    dst = coord_to_node_id((0, 6))
    counts = [1, 2, 4, 8, 16]
    if not quick:
        counts.extend([32, 64])
    for count in counts:
        requests = [Request(index, 0, src, dst, 8) for index in range(count)]
        compare_trace(rows, "same_path", f"simultaneous_{count}", requests, "same_source_same_destination")


def test_arrival_rate(rows: list[dict], quick: bool) -> None:
    src = coord_to_node_id((1, 1))
    dst = coord_to_node_id((1, 8))
    count = 16 if quick else 64
    for gap in [0, 1, 2, 4, 8, 16, 64]:
        requests = [Request(index, index * gap, src, dst, 8) for index in range(count)]
        compare_trace(rows, "arrival_rate", f"gap_{gap}", requests, "fixed_gap_same_path")


def test_path_contention_patterns(rows: list[dict], quick: bool) -> None:
    repeat = 4 if quick else 16
    patterns = {
        "shared_first_link": [((0, 0), (0, 8)), ((0, 0), (2, 8))],
        "shared_last_link": [((0, 0), (1, 8)), ((3, 0), (1, 8))],
        "crossing": [((0, 0), (3, 8)), ((3, 0), (0, 8))],
        "disjoint": [((0, 0), (0, 4)), ((3, 5), (3, 9))],
        "opposite_direction": [((1, 1), (1, 8)), ((1, 8), (1, 1))],
    }
    for name, flows in patterns.items():
        requests = []
        request_id = 0
        for _ in range(repeat):
            for source, destination in flows:
                requests.append(Request(request_id, 0, coord_to_node_id(source), coord_to_node_id(destination), 8))
                request_id += 1
        compare_trace(rows, "path_pattern", name, requests, name)


def test_hotspot_traffic(rows: list[dict], quick: bool) -> None:
    destination = coord_to_node_id((1, 9))
    sources = [coord_to_node_id((row, column)) for row in range(SHAPE[0]) for column in [0, 2, 4, 6, 8]]
    if quick:
        sources = sources[:8]
    requests = [Request(index, 0, source, destination, 8) for index, source in enumerate(sources) if source != destination]
    compare_trace(rows, "hotspot", "many_to_one", requests, "many_sources_one_destination")


def test_subnet_isolation(rows: list[dict]) -> None:
    src = coord_to_node_id((0, 0))
    dst_a = coord_to_node_id((0, 4))
    dst_b = coord_to_node_id((0, 5))
    requests = [
        Request(0, 0, src, dst_a, 32),
        Request(1, 0, src, dst_b, 32),
    ]
    compare_trace(rows, "subnet", "hashed_subnets", requests, "request_hash_subnet_selection")
    multi_payload = [Request(0, 0, src, dst_a, 128)]
    compare_trace(rows, "subnet", "multi_payload_round_robin", multi_payload, "payload_subnet_round_robin")


def test_traffic_classes(rows: list[dict]) -> None:
    src = coord_to_node_id((2, 1))
    dst = coord_to_node_id((2, 8))
    patterns = {
        "all_write": [True] * 16,
        "all_response": [False] * 16,
        "alternating": [index % 2 == 0 for index in range(16)],
    }
    for name, directions in patterns.items():
        requests = [Request(index, 0, src, dst, 8, is_write) for index, is_write in enumerate(directions)]
        compare_trace(rows, "traffic_class", name, requests, name)


def test_dispatch_order_sensitivity(rows: list[dict]) -> None:
    flows = [
        (coord_to_node_id((row, 0)), coord_to_node_id((1, 9)))
        for row in range(SHAPE[0])
    ] * 4
    orders = {
        "forward": flows,
        "reverse": list(reversed(flows)),
        "shuffled": random.Random(19).sample(flows, len(flows)),
    }
    for name, ordered_flows in orders.items():
        requests = [Request(index, 0, src, dst, 8) for index, (src, dst) in enumerate(ordered_flows)]
        compare_trace(rows, "dispatch_order", name, requests, "same_flow_multiset")


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        if row["model"] != "lightweight" or row["relative_error"] == "":
            continue
        grouped.setdefault(row["test"], []).append(float(row["relative_error"]))
    return {
        test: {
            "mean_absolute_relative_error": statistics.fmean(abs(value) for value in values),
            "mean_signed_relative_error": statistics.fmean(values),
            "max_absolute_relative_error": max(abs(value) for value in values),
            "sample_count": len(values),
        }
        for test, values in grouped.items()
    }


def print_summary(summary: dict) -> None:
    print("\nAccuracy summary")
    print("test                       samples    MARE      signed     max")
    for test, values in summary.items():
        print(
            f"{test:<26} {values['sample_count']:>7d} "
            f"{values['mean_absolute_relative_error']:>9.2%} "
            f"{values['mean_signed_relative_error']:>9.2%} "
            f"{values['max_absolute_relative_error']:>9.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the lightweight interconnect model against pybooksim2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--booksim-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument(
        "--output",
        default=os.path.join(CACHE_DIR, "v4_lightweight_icntsim_validation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.booksim_worker:
        booksim_worker(*args.booksim_worker)
        return
    rows = []
    tests = [
        lambda: test_packet_size_scaling(rows, args.quick),
        lambda: test_hop_scaling(rows),
        lambda: test_same_path_contention(rows, args.quick),
        lambda: test_arrival_rate(rows, args.quick),
        lambda: test_path_contention_patterns(rows, args.quick),
        lambda: test_hotspot_traffic(rows, args.quick),
        lambda: test_subnet_isolation(rows),
        lambda: test_traffic_classes(rows),
        lambda: test_dispatch_order_sensitivity(rows),
    ]
    for test in tests:
        test()
    write_csv(args.output, rows)
    summary = summarize_results(rows)
    summary_path = os.path.splitext(args.output)[0] + ".json"
    with open(summary_path, "w") as output:
        json.dump({"shape": SHAPE, "flit_size": FLIT_SIZE, "subnets": SUBNETS, "accuracy": summary}, output, indent=2)
    print_summary(summary)
    print(f"\nCSV: {args.output}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
