import argparse
import csv
import functools
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRCS_ROOT = os.path.join(REPO_ROOT, "srcs")
if SRCS_ROOT not in sys.path:
    sys.path.insert(0, SRCS_ROOT)

import pydramsim3

from neuromta.framework import parse_freq_str, parse_mem_cap_str
from neuromta.component.context.mem_context import MemoryConfig, MemorySimulator


CACHE_DIR = os.environ.get("NEUROMTA_VALIDATION_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache"))
RESULT_FIELDS = [
    "test",
    "case",
    "model",
    "request_count",
    "size_bytes",
    "is_write",
    "pattern",
    "completion_cycles",
    "first_completion_cycles",
    "mean_latency_cycles",
    "p50_latency_cycles",
    "p95_latency_cycles",
    "effective_bytes_per_cycle",
    "relative_error",
]


@dataclass(frozen=True)
class Request:
    request_id: int
    issue_cycle: int
    address: int
    size: int
    is_write: bool = False


@dataclass
class TraceResult:
    completion_cycle: int
    completion_cycles: dict[int, int]
    latencies: dict[int, int]
    first_data_cycles: dict[int, int]


@dataclass(frozen=True)
class MemoryPreset:
    name: str
    source_config: str
    channel_size: int
    channels: int
    read_latency: int
    write_latency: int
    write_accept_latency: int
    write_completion_policy: str
    bandwidth_bytes_per_cycle: int
    burst_size: int
    interleave_size: int
    address_mapping: str
    address_mapping_scheme: str
    dram_rows: int
    dram_columns: int
    dram_burst_length: int
    instance_command_gap: int
    command_gap: int
    read_to_write_turnaround: int
    write_to_read_turnaround: int
    startup: int
    bank_groups: int
    banks_per_group: int
    row_size: int
    row_miss: int
    row_conflict: int
    bank_group_penalty: int
    max_outstanding: int
    channel_max_outstanding: int
    request_queue_depth: int


PRESETS = {
    "hbm": MemoryPreset(
        name="hbm",
        source_config="HBM2_8Gb_x128",
        channel_size=parse_mem_cap_str("1GB"),
        channels=8,
        read_latency=17,
        write_latency=34,
        write_accept_latency=2,
        write_completion_policy="accept",
        bandwidth_bytes_per_cycle=64,
        burst_size=parse_mem_cap_str("64B"),
        interleave_size=parse_mem_cap_str("64B"),
        address_mapping="dramsim3",
        address_mapping_scheme="rorabgbachco",
        dram_rows=32768,
        dram_columns=64,
        dram_burst_length=4,
        instance_command_gap=1,
        command_gap=1,
        read_to_write_turnaround=6,
        write_to_read_turnaround=8,
        startup=0,
        bank_groups=4,
        banks_per_group=4,
        row_size=parse_mem_cap_str("2KB"),
        row_miss=14,
        row_conflict=14,
        bank_group_penalty=1,
        max_outstanding=64,
        channel_max_outstanding=16,
        request_queue_depth=32,
    ),
    "lpddr": MemoryPreset(
        name="lpddr",
        source_config="LPDDR4_8Gb_x16_2400.ini",
        channel_size=parse_mem_cap_str("2GB"),
        channels=2,
        read_latency=15,
        write_latency=37,
        write_accept_latency=1,
        write_completion_policy="retire",
        bandwidth_bytes_per_cycle=19,
        burst_size=parse_mem_cap_str("128B"),
        interleave_size=parse_mem_cap_str("128B"),
        address_mapping="dramsim3",
        address_mapping_scheme="rochrababgco",
        dram_rows=65536,
        dram_columns=1024,
        dram_burst_length=16,
        instance_command_gap=0,
        command_gap=7,
        read_to_write_turnaround=4,
        write_to_read_turnaround=25,
        startup=0,
        bank_groups=2,
        banks_per_group=4,
        row_size=parse_mem_cap_str("8KB"),
        row_miss=13,
        row_conflict=25,
        bank_group_penalty=0,
        max_outstanding=64,
        channel_max_outstanding=32,
        request_queue_depth=32,
    ),
}


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


def create_lightweight_simulator(preset: MemoryPreset) -> MemorySimulator:
    config = MemoryConfig(
        processor_clock_freq=parse_freq_str("1GHz"),
        n_instance=1,
        channel_size=preset.channel_size,
        n_channel_per_instance=preset.channels,
        dramsim3_enable=False,
        lightweight_read_latency_cycles=preset.read_latency,
        lightweight_write_latency_cycles=preset.write_latency,
        lightweight_write_accept_latency_cycles=preset.write_accept_latency,
        lightweight_write_completion_policy=preset.write_completion_policy,
        lightweight_enable_latency_amortization=False,
        lightweight_channel_bandwidth_bytes_per_cycle=preset.bandwidth_bytes_per_cycle,
        lightweight_dma_granularity=parse_mem_cap_str("256B"),
        lightweight_address_mapping=preset.address_mapping,
        lightweight_address_mapping_scheme=preset.address_mapping_scheme,
        lightweight_channel_interleave_bytes=preset.interleave_size,
        lightweight_dram_rows=preset.dram_rows,
        lightweight_dram_columns=preset.dram_columns,
        lightweight_dram_burst_length=preset.dram_burst_length,
        lightweight_instance_command_issue_gap_cycles=preset.instance_command_gap,
        lightweight_command_issue_gap_cycles=preset.command_gap,
        lightweight_read_to_write_turnaround_cycles=preset.read_to_write_turnaround,
        lightweight_write_to_read_turnaround_cycles=preset.write_to_read_turnaround,
        lightweight_request_startup_latency_cycles=preset.startup,
        lightweight_burst_size_bytes=preset.burst_size,
        lightweight_n_rank_per_channel=1,
        lightweight_n_bank_group_per_rank=preset.bank_groups,
        lightweight_n_bank_per_bank_group=preset.banks_per_group,
        lightweight_row_size_bytes=preset.row_size,
        lightweight_row_hit_latency_cycles=0,
        lightweight_row_miss_penalty_cycles=preset.row_miss,
        lightweight_row_conflict_penalty_cycles=preset.row_conflict,
        lightweight_bank_group_penalty_cycles=preset.bank_group_penalty,
        lightweight_dma_max_outstanding_bursts=preset.max_outstanding,
        lightweight_channel_max_outstanding_bursts=preset.channel_max_outstanding,
        lightweight_request_queue_depth=preset.request_queue_depth,
        lightweight_concurrent_request_command_gap_cycles=0,
        lightweight_concurrent_request_command_gap_threshold=4,
        lightweight_concurrent_request_command_gap_limit=1,
    )
    return MemorySimulator(config, mem_addr_offset=0)


def create_dramsim_config(preset: MemoryPreset) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"v3_{preset.name}_dramsim.ini")
    pydramsim3.create_new_dramsim_config_file(
        src_config_path=preset.source_config,
        new_config_path=path,
        system_params={
            "channel_size": preset.channel_size // (1024 * 1024),
            "channels": preset.channels,
            "cmd_queue_size": 8,
            "trans_queue_size": 32,
            "row_buf_policy": "OPEN_PAGE",
        },
    )
    return path


def run_lightweight_trace(requests: list[Request], preset: MemoryPreset) -> TraceResult:
    simulator = create_lightweight_simulator(preset)
    completions = {}
    latencies = {}
    first_data = {}
    for request in sorted(requests, key=lambda item: (item.issue_cycle, item.request_id)):
        result = simulator.send_request(
            addr=request.address,
            size=request.size,
            is_write=request.is_write,
            current_cycle=request.issue_cycle,
            profile=True,
        )
        completions[request.request_id] = result["finish_cycle"]
        latencies[request.request_id] = result["latency_cycles"]
        chunks = result["chunks"]
        first_data[request.request_id] = min((chunk["finish_cycle"] for chunk in chunks), default=request.issue_cycle)
    return TraceResult(
        completion_cycle=max(completions.values(), default=0),
        completion_cycles=completions,
        latencies=latencies,
        first_data_cycles=first_data,
    )


def run_dramsim_trace(requests: list[Request], config_path: str) -> TraceResult:
    memory = pydramsim3.create_msys(
        config_file=config_path,
        output_dir=CACHE_DIR,
        max_issue_per_cmd_q_per_cycle=1,
    )
    tck_ns = pydramsim3.msys_get_tck(memory)
    reference_tck_ns = 1.0
    remainder_ns = 0.0
    cycle = 0
    pending = sorted(requests, key=lambda item: (item.issue_cycle, item.request_id))
    pending_index = 0
    completions = {}

    def complete(request: Request, _capsule) -> None:
        completions[request.request_id] = cycle

    max_cycles = max((request.issue_cycle for request in requests), default=0) + 100_000_000
    while len(completions) < len(requests):
        while pending_index < len(pending) and pending[pending_index].issue_cycle <= cycle:
            request = pending[pending_index]
            command = pydramsim3.create_msys_cmd(
                addr=request.address,
                size=request.size,
                is_write=request.is_write,
            )
            accepted = pydramsim3.msys_dispatch_cmd(
                msys=memory,
                cmd=command,
                dispatch_callback=None,
                execute_callback=functools.partial(complete, request),
            )
            if not accepted:
                break
            pending_index += 1

        remainder_ns += reference_tck_ns
        dram_cycles = math.floor(remainder_ns / tck_ns)
        remainder_ns -= dram_cycles * tck_ns
        if dram_cycles > 0:
            pydramsim3.msys_cycle_step(msys=memory, cycles=dram_cycles)
        cycle += 1
        if cycle > max_cycles:
            raise TimeoutError(f"DRAMSim3 trace did not finish within {max_cycles} cycles")

    latencies = {
        request.request_id: completions[request.request_id] - request.issue_cycle
        for request in requests
    }
    return TraceResult(
        completion_cycle=max(completions.values(), default=0),
        completion_cycles=completions,
        latencies=latencies,
        first_data_cycles=dict(completions),
    )


def make_summary_row(
    test: str,
    case: str,
    model: str,
    requests: list[Request],
    result: TraceResult,
    pattern: str,
) -> dict:
    latencies = list(result.latencies.values())
    total_bytes = sum(request.size for request in requests)
    first_issue = min((request.issue_cycle for request in requests), default=0)
    duration = max(1, result.completion_cycle - first_issue)
    return {
        "test": test,
        "case": case,
        "model": model,
        "request_count": len(requests),
        "size_bytes": requests[0].size if requests and len({request.size for request in requests}) == 1 else total_bytes,
        "is_write": requests[0].is_write if requests and len({request.is_write for request in requests}) == 1 else "mixed",
        "pattern": pattern,
        "completion_cycles": result.completion_cycle,
        "first_completion_cycles": min(result.completion_cycles.values(), default=0),
        "mean_latency_cycles": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_cycles": percentile(latencies, 0.50),
        "p95_latency_cycles": percentile(latencies, 0.95),
        "effective_bytes_per_cycle": total_bytes / duration,
        "relative_error": "",
    }


def compare_trace(
    rows: list[dict],
    test: str,
    case: str,
    requests: list[Request],
    preset: MemoryPreset,
    dramsim_config: str,
    pattern: str,
) -> tuple[TraceResult, TraceResult]:
    lightweight = run_lightweight_trace(requests, preset)
    dramsim = run_dramsim_trace(requests, dramsim_config)
    lightweight_row = make_summary_row(test, case, "lightweight", requests, lightweight, pattern)
    dramsim_row = make_summary_row(test, case, "dramsim3", requests, dramsim, pattern)
    denominator = max(1, dramsim.completion_cycle)
    lightweight_row["relative_error"] = (lightweight.completion_cycle - dramsim.completion_cycle) / denominator
    dramsim_row["relative_error"] = 0.0
    rows.extend([lightweight_row, dramsim_row])
    return lightweight, dramsim


def test_request_size_scaling(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    sizes = [64, 256, 1024, 4096, 32768]
    if not quick:
        sizes.extend([131072, 524288])
    for size in sizes:
        request = Request(0, 0, 0, size, False)
        compare_trace(rows, "request_size", f"read_{size}B", [request], preset, dramsim_config, "isolated_read")
        write = Request(0, 0, 0, size, True)
        compare_trace(rows, "request_size", f"write_{size}B", [write], preset, dramsim_config, "isolated_write")


def test_first_data_latency(rows: list[dict], preset: MemoryPreset, dramsim_config: str) -> None:
    reference = Request(0, 0, 0, 64, False)
    lightweight_reference, dramsim_reference = compare_trace(
        rows,
        "first_data",
        "64B_reference",
        [reference],
        preset,
        dramsim_config,
        "isolated_first_burst_proxy",
    )
    for size in [256, 4096, 32768]:
        request = Request(0, 0, 0, size, False)
        lightweight = run_lightweight_trace([request], preset)
        first_cycle = lightweight.first_data_cycles[0]
        rows.append({
            "test": "first_data",
            "case": f"lightweight_first_chunk_{size}B",
            "model": "lightweight",
            "request_count": 1,
            "size_bytes": size,
            "is_write": False,
            "pattern": "first_chunk_vs_dramsim_64B_completion_proxy",
            "completion_cycles": lightweight.completion_cycle,
            "first_completion_cycles": first_cycle,
            "mean_latency_cycles": lightweight.latencies[0],
            "p50_latency_cycles": lightweight.latencies[0],
            "p95_latency_cycles": lightweight.latencies[0],
            "effective_bytes_per_cycle": size / max(1, lightweight.completion_cycle),
            "relative_error": (first_cycle - dramsim_reference.completion_cycle) / max(1, dramsim_reference.completion_cycle),
        })
    if lightweight_reference.first_data_cycles[0] <= 0:
        raise AssertionError("Lightweight first-data latency must be positive")


def build_address_pattern(pattern: str, count: int, size: int, preset: MemoryPreset, seed: int = 7) -> list[int]:
    if pattern == "sequential":
        return [index * size for index in range(count)]
    if pattern == "row_local":
        return [(index * 64) % preset.row_size for index in range(count)]
    if pattern == "row_conflict":
        stride = preset.row_size * preset.bank_groups * preset.banks_per_group
        return [index * stride for index in range(count)]
    if pattern == "channel_spread":
        return [(index % preset.channels) * preset.channel_size for index in range(count)]
    if pattern == "random":
        generator = random.Random(seed)
        capacity = preset.channel_size * preset.channels
        return [generator.randrange(0, capacity - size, 64) for _ in range(count)]
    raise ValueError(f"Unknown address pattern: {pattern}")


def test_access_patterns(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    count = 16 if quick else 64
    size = 256
    for pattern in ["sequential", "row_local", "row_conflict", "channel_spread", "random"]:
        addresses = build_address_pattern(pattern, count, size, preset)
        requests = [Request(index, 0, address, size, False) for index, address in enumerate(addresses)]
        compare_trace(rows, "access_pattern", pattern, requests, preset, dramsim_config, pattern)


def test_outstanding_load(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    counts = [1, 2, 4, 8, 16]
    if not quick:
        counts.extend([32, 64])
    for count in counts:
        addresses = build_address_pattern("sequential", count, 256, preset)
        requests = [Request(index, 0, address, 256, False) for index, address in enumerate(addresses)]
        compare_trace(rows, "outstanding", f"simultaneous_{count}", requests, preset, dramsim_config, "simultaneous_sequential")


def test_arrival_patterns(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    count = 16 if quick else 64
    for gap in [0, 1, 4, 16, 64]:
        requests = [Request(index, index * gap, index * 256, 256, False) for index in range(count)]
        compare_trace(rows, "arrival_pattern", f"gap_{gap}", requests, preset, dramsim_config, "fixed_gap_sequential")


def build_streaming_requests(
    count: int,
    size: int,
    issue_gap: int,
    stream_count: int,
    write_mode: str,
) -> list[Request]:
    stream_span = parse_mem_cap_str("1MB")
    requests = []
    stream_offsets = [0 for _ in range(stream_count)]
    for request_id in range(count):
        stream_id = request_id % stream_count
        address = stream_id * stream_span + stream_offsets[stream_id]
        stream_offsets[stream_id] += size
        if write_mode == "write":
            is_write = True
        elif write_mode == "alternating":
            is_write = request_id % 2 == 1
        elif write_mode == "stream_mixed":
            is_write = stream_id % 2 == 1
        else:
            is_write = False
        requests.append(Request(request_id, request_id * issue_gap, address, size, is_write))
    return requests


def test_streaming_patterns(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    count = 32 if quick else 128
    cases = [
        ("sequential_read_256B_gap1", 256, 1, 1, "read"),
        ("sequential_write_256B_gap1", 256, 1, 1, "write"),
        ("sequential_read_4KB_gap8", 4096, 8, 1, "read"),
        ("sequential_alternating_256B_gap1", 256, 1, 1, "alternating"),
        ("four_stream_read_256B_gap1", 256, 1, 4, "read"),
        ("four_stream_mixed_256B_gap1", 256, 1, 4, "stream_mixed"),
    ]
    for name, size, issue_gap, stream_count, write_mode in cases:
        requests = build_streaming_requests(count, size, issue_gap, stream_count, write_mode)
        compare_trace(
            rows,
            "streaming",
            name,
            requests,
            preset,
            dramsim_config,
            f"streams={stream_count},gap={issue_gap},mode={write_mode}",
        )


def test_read_write_patterns(rows: list[dict], preset: MemoryPreset, dramsim_config: str, quick: bool) -> None:
    count = 16 if quick else 64
    patterns = {
        "all_read": lambda index: False,
        "all_write": lambda index: True,
        "alternating": lambda index: index % 2 == 1,
        "read_then_write": lambda index: index >= count // 2,
    }
    for name, is_write in patterns.items():
        requests = [Request(index, 0, index * 256, 256, is_write(index)) for index in range(count)]
        compare_trace(rows, "read_write", name, requests, preset, dramsim_config, name)


def test_dispatch_order_sensitivity(rows: list[dict], preset: MemoryPreset, dramsim_config: str) -> None:
    addresses = build_address_pattern("row_conflict", 16, 256, preset)
    orders = {
        "forward": addresses,
        "reverse": list(reversed(addresses)),
        "shuffled": random.Random(11).sample(addresses, len(addresses)),
    }
    for name, ordered_addresses in orders.items():
        requests = [Request(index, 0, address, 256, False) for index, address in enumerate(ordered_addresses)]
        compare_trace(rows, "dispatch_order", name, requests, preset, dramsim_config, "same_set_row_conflict")


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: list[dict]) -> dict:
    lightweight_rows = [row for row in rows if row["model"] == "lightweight" and row["relative_error"] != ""]
    request_types = {
        "overall": lightweight_rows,
        "read": [row for row in lightweight_rows if row["is_write"] is False],
        "write": [row for row in lightweight_rows if row["is_write"] is True],
        "mixed": [row for row in lightweight_rows if row["is_write"] == "mixed"],
    }
    summary = {}
    for request_type, selected_rows in request_types.items():
        grouped = {}
        for row in selected_rows:
            grouped.setdefault(row["test"], []).append(float(row["relative_error"]))
        summary[request_type] = {
            test: {
                "mean_absolute_relative_error": statistics.fmean(abs(value) for value in values),
                "mean_signed_relative_error": statistics.fmean(values),
                "max_absolute_relative_error": max(abs(value) for value in values),
                "sample_count": len(values),
            }
            for test, values in grouped.items()
        }
    return summary


def print_summary(summary: dict) -> None:
    for request_type, results in summary.items():
        if not results:
            continue
        print(f"\nAccuracy summary: {request_type}")
        print("test                       samples    MARE      signed     max")
        for test, values in results.items():
            print(
                f"{test:<26} {values['sample_count']:>7d} "
                f"{values['mean_absolute_relative_error']:>9.2%} "
                f"{values['mean_signed_relative_error']:>9.2%} "
                f"{values['max_absolute_relative_error']:>9.2%}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the lightweight DRAM model against pydramsim3 traces")
    parser.add_argument("--memory", choices=sorted(PRESETS), default="hbm")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(CACHE_DIR, "v3_lightweight_dramsim_validation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.memory]
    dramsim_config = create_dramsim_config(preset)
    rows = []
    tests = [
        lambda: test_request_size_scaling(rows, preset, dramsim_config, args.quick),
        lambda: test_first_data_latency(rows, preset, dramsim_config),
        lambda: test_access_patterns(rows, preset, dramsim_config, args.quick),
        lambda: test_outstanding_load(rows, preset, dramsim_config, args.quick),
        lambda: test_arrival_patterns(rows, preset, dramsim_config, args.quick),
        lambda: test_streaming_patterns(rows, preset, dramsim_config, args.quick),
        lambda: test_read_write_patterns(rows, preset, dramsim_config, args.quick),
        lambda: test_dispatch_order_sensitivity(rows, preset, dramsim_config),
    ]
    for test in tests:
        test()
    write_csv(args.output, rows)
    summary = summarize_results(rows)
    summary_path = os.path.splitext(args.output)[0] + ".json"
    with open(summary_path, "w") as output:
        json.dump({"preset": asdict(preset), "accuracy": summary}, output, indent=2)
    print_summary(summary)
    print(f"\nCSV: {args.output}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
