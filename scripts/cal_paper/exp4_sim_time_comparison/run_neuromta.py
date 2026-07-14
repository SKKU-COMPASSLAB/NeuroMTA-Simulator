import argparse
import csv
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / ".logs"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_DIR / "matplotlib"))

import torch
import torch.nn as nn
import torchvision

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import *

_WORKER_GRAPH: MCA_CompiledNetworkGraph | None = None
_WORKER_DUMMY_INPUTS: list[Any] | None = None
_WORKER_ARGS: dict[str, Any] = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Compile ResNet18 on Tenstorrent Blackhole and time compiled entries.")
    parser.add_argument("--print-autorun-config", action="store_true", help="Print the autorun configuration for the compiled graph.")
    parser.add_argument("--print-graph-summary", action="store_true", help="Print graph summary after compiling.")
    parser.add_argument("--print-compile-summary", action="store_true", help="Print per-entry compilation summary.")
    parser.add_argument("--run-compiled-graph", action="store_true", help="Run the selected entry or graph in the main process.")
    parser.add_argument("--run-entries-parallel", action="store_true", help="Run selected entries in multiprocessing workers and write a timing CSV.")
    parser.add_argument("--group-idx", type=int, default=None, help="Group index to run. Defaults to all groups.")
    parser.add_argument("--entry-idx", type=int, default=None, help="Entry index to run. Defaults to all entries in selected groups.")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of worker processes.")
    parser.add_argument("--entry-timing-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "neuromta_entry_timings.csv")
    parser.add_argument("--profiler-output-dir", type=str, default=None, help="Optional profiler output root directory.")
    parser.add_argument("--monitoring-window", action="store_true", help="Enable monitoring window during execution.")
    parser.add_argument("--pcc-check", action="store_true", help="Enable PCC check after running each entry.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load torchvision default weights.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model_and_inputs(pretrained: bool, seed: int) -> tuple[nn.Module, list[Any]]:
    torch.manual_seed(seed)
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    module = torchvision.models.resnet18(weights=weights).eval()
    dummy_inputs = [torch.randn(1, 3, 224, 224)]
    return module, dummy_inputs


def create_device_and_recipe() -> tuple[TenstorrentDevice, MCA_NetworkRecipe]:
    config = TenstorrentConfig.BLACKHOLE(use_pydramsim3=False, use_pybooksim2=False)
    device = TenstorrentDevice(**config).initialize()
    device.set_command_debug_verbosity(verbose=False)
    device.set_simulation_mode(mode=SimulationMode.PERFORMANCE)
    graph_recipe = MCA_NetworkRecipe(
        device=device,
        core_groups=device.get_npu_core_group((0, 0), (12, 14)),
        spad_space_size_per_core=parse_mem_cap_str("1MB"),
        dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    return device, graph_recipe


def compile_graph(
    module: nn.Module,
    graph_recipe: MCA_NetworkRecipe,
    dummy_inputs: list[Any],
    print_graph_summary: bool=False,
    print_compile_summary: bool=False,
) -> MCA_CompiledNetworkGraph:
    graph = MCA_CompiledNetworkGraph.from_trace(module, graph_recipe, *dummy_inputs)

    if print_graph_summary:
        graph.print_graph()

    if print_compile_summary:
        for group_idx, group in enumerate(graph.grouped_compiled_entries):
            for entry_idx, entry in enumerate(group):
                print(f"Group {group_idx} Entry {entry_idx}:")
                print(f"  - entry type: {entry.node.kind()}")
                print(f"  - input nodes: {[input_node.debugName() for input_node in entry.node.inputs()]}")
                print(f"  - output nodes: {[output_node.debugName() for output_node in entry.node.outputs()]}")
                print(f"  - context entries:")
                for ctx_name, ctx_entry in entry._ctx_entries.items():
                    print(f"    * {ctx_name}: {ctx_entry}")

    return graph


def print_autorun_config(graph: MCA_CompiledNetworkGraph):
    print({
        group_idx: {
            entry_idx: {
                "node_type": entry.node.kind(),
                "op_type": entry._op_method.__name__,
            }
            for entry_idx, entry in enumerate(group)
        }
        for group_idx, group in enumerate(graph.grouped_compiled_entries)
    })


def iter_entry_metadata(graph: MCA_CompiledNetworkGraph, group_idx: int=None, entry_idx: int=None) -> list[dict[str, Any]]:
    rows = []
    for g_idx, group in enumerate(graph.grouped_compiled_entries):
        if group_idx is not None and g_idx != group_idx:
            continue
        for e_idx, entry in enumerate(group):
            if entry_idx is not None and e_idx != entry_idx:
                continue
            rows.append({
                "group_idx": g_idx,
                "entry_idx": e_idx,
                "node_kind": entry.node.kind(),
                "op_type": entry._op_method.__name__,
                "input_nodes": "|".join(input_node.debugName() for input_node in entry.node.inputs()),
                "output_nodes": "|".join(output_node.debugName() for output_node in entry.node.outputs()),
                "context_entry_count": len(entry._ctx_entries),
            })
    return rows


def create_profilers(device: TenstorrentDevice, profiler_output_dir: str | None):
    if profiler_output_dir is None:
        return None
    os.makedirs(profiler_output_dir, exist_ok=True)
    return [
        DRAMBandwidthProfiler(device, record_type="BOTH"),
        InterconnectBandwidthProfiler(device),
    ]


def run_compiled_graph(
    graph: MCA_CompiledNetworkGraph,
    dummy_inputs: list[Any],
    group_idx: int=None,
    entry_idx: int=None,
    monitoring_window=False,
    profilers: list[ProfilerTemplate | GroupedProfilerTemplate]=None,
    profiler_output_dir: str=None,
    pcc_check: bool=False,
    pcc_check_atol: float=1e-4,
    pcc_check_rtol: float=1e-4,
):
    result_dict = graph.run_compiled_graph(
        *dummy_inputs,
        group_idx=group_idx,
        entry_idx=entry_idx,
        monitoring_window=monitoring_window,
        profilers=profilers,
        profiler_output_dir=profiler_output_dir,
        pcc_check=pcc_check,
        pcc_check_atol=pcc_check_atol,
        pcc_check_rtol=pcc_check_rtol,
    )

    for sim_name, timestamp in result_dict.items():
        print(f"{sim_name}: {timestamp} cycles")


def init_entry_worker(worker_args: dict[str, Any]) -> None:
    global _WORKER_GRAPH, _WORKER_DUMMY_INPUTS, _WORKER_ARGS
    logger.set_print_options(log_level=LogLevel.INFO)
    _WORKER_ARGS = worker_args
    device, graph_recipe = create_device_and_recipe()
    device.set_simulation_mode(mode=SimulationMode.PERFORMANCE)
    module, dummy_inputs = build_model_and_inputs(
        pretrained=worker_args["pretrained"],
        seed=worker_args["seed"],
    )
    _WORKER_GRAPH = compile_graph(module, graph_recipe, dummy_inputs)
    _WORKER_DUMMY_INPUTS = dummy_inputs


def run_entry_worker(task: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER_GRAPH is not None
    assert _WORKER_DUMMY_INPUTS is not None

    group_idx = task["group_idx"]
    entry_idx = task["entry_idx"]
    profiler_output_dir = None
    if _WORKER_ARGS.get("profiler_output_dir") is not None:
        profiler_output_dir = os.path.join(
            _WORKER_ARGS["profiler_output_dir"],
            f"group_{group_idx:03d}_entry_{entry_idx:03d}",
        )

    profilers = create_profilers(_WORKER_GRAPH.graph_recipe.device, profiler_output_dir)
    start = time.perf_counter()
    try:
        result_dict = _WORKER_GRAPH.run_compiled_graph(
            *_WORKER_DUMMY_INPUTS,
            group_idx=group_idx,
            entry_idx=entry_idx,
            monitoring_window=_WORKER_ARGS["monitoring_window"],
            profilers=profilers,
            profiler_output_dir=profiler_output_dir,
            pcc_check=_WORKER_ARGS["pcc_check"],
            pcc_check_atol=1e-4,
            pcc_check_rtol=1e-4,
        )
        wall_time_sec = time.perf_counter() - start
        return {
            **task,
            "status": "ok",
            "timestamp_cycles": max(result_dict.values()) if result_dict else "",
            "wall_time_sec": wall_time_sec,
            "result_timestamps": "|".join(f"{name}:{timestamp}" for name, timestamp in result_dict.items()),
            "profiler_output_dir": profiler_output_dir or "",
            "error": "",
        }
    except Exception:
        wall_time_sec = time.perf_counter() - start
        return {
            **task,
            "status": "error",
            "timestamp_cycles": "",
            "wall_time_sec": wall_time_sec,
            "result_timestamps": "",
            "profiler_output_dir": profiler_output_dir or "",
            "error": traceback.format_exc(),
        }


def run_entries_parallel(entry_metadata: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if len(entry_metadata) == 0:
        return []

    worker_count = max(1, min(args.num_workers, len(entry_metadata)))
    worker_args = {
        "pretrained": not args.no_pretrained,
        "seed": args.seed,
        "monitoring_window": args.monitoring_window,
        "profiler_output_dir": args.profiler_output_dir,
        "pcc_check": args.pcc_check,
    }
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count, initializer=init_entry_worker, initargs=(worker_args,)) as pool:
        return list(pool.imap_unordered(run_entry_worker, entry_metadata))


def write_entry_timing_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_idx",
        "entry_idx",
        "node_kind",
        "op_type",
        "input_nodes",
        "output_nodes",
        "context_entry_count",
        "status",
        "timestamp_cycles",
        "wall_time_sec",
        "result_timestamps",
        "profiler_output_dir",
        "error",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["group_idx"], item["entry_idx"])):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.INFO)
    args = parse_args()

    device, graph_recipe = create_device_and_recipe()
    device.set_simulation_mode(mode=SimulationMode.PERFORMANCE)
    module, dummy_inputs = build_model_and_inputs(pretrained=not args.no_pretrained, seed=args.seed)
    graph = compile_graph(
        module,
        graph_recipe,
        dummy_inputs,
        print_graph_summary=args.print_graph_summary,
        print_compile_summary=args.print_compile_summary,
    )

    if args.print_autorun_config:
        print_autorun_config(graph)

    profilers = create_profilers(device, args.profiler_output_dir)
    entry_metadata = iter_entry_metadata(graph, group_idx=args.group_idx, entry_idx=args.entry_idx)

    if args.run_compiled_graph:
        run_compiled_graph(
            graph,
            dummy_inputs,
            group_idx=args.group_idx,
            entry_idx=args.entry_idx,
            monitoring_window=args.monitoring_window,
            profilers=profilers,
            profiler_output_dir=args.profiler_output_dir,
            pcc_check=args.pcc_check,
            pcc_check_atol=1e-4,
            pcc_check_rtol=1e-4,
        )

    if args.run_entries_parallel:
        rows = run_entries_parallel(entry_metadata, args)
        write_entry_timing_csv(rows, args.entry_timing_csv)
        print(f"entry timing CSV written to {args.entry_timing_csv}")
