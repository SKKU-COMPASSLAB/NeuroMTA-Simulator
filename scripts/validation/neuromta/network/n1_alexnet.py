import argparse
import os

import torch
import torchvision
import torch.nn as nn
from typing import Any

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for compiling and running a network graph on Tenstorrent hardware.")
    parser.add_argument("--monitoring-window", action="store_true", help="Enable monitoring window during graph execution.")
    parser.add_argument("--print-autorun-config", action="store_true", help="Print the autorun configuration for the compiled graph.")
    parser.add_argument("--print-graph-summary", action="store_true", help="Print compilation summary after compiling the graph.")
    parser.add_argument("--print-compile-summary", action="store_true", help="Print compilation summary.")
    parser.add_argument("--run-compiled-graph", action="store_true", help="Run the compiled graph after compilation.")
    parser.add_argument("--group-idx", type=int, default=None, help="Index of the group to run. If not specified, all groups will be run.")
    parser.add_argument("--entry-idx", type=int, default=None, help="Index of the entry to run within the specified group. If not specified, all entries in the group will be run.")
    parser.add_argument("--max-timestamp", type=int, default=None, help="Maximum timestamp to run the kernels for. If not specified, kernels will run until completion.")
    parser.add_argument("--profiler-output-dir", type=str, default=None, help="Directory to save profiler outputs. If not specified, profiler outputs will not be saved.")
    parser.add_argument("--pcc-check", action="store_true", help="Enable PCC check after running each entry.")
    return parser.parse_args()


def compile_graph(module: nn.Module, graph_recipe: MCA_NetworkRecipe, dummy_inputs: list[Any], print_graph_summary: bool=False, print_compile_summary: bool=False) -> MCA_CompiledNetworkGraph:
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

def run_compiled_graph(
    graph: MCA_CompiledNetworkGraph, dummy_inputs: list[Any], 
    group_idx: int=None, entry_idx: int=None, 
    monitoring_window=False,
    profilers: list[ProfilerTemplate | GroupedProfilerTemplate]=None, profiler_output_dir: str=None,
    pcc_check: bool=False, pcc_check_atol: float=1e-4, pcc_check_rtol: float=1e-4,    
):
    result_dict = graph.run_compiled_graph(
        *dummy_inputs, group_idx=group_idx, entry_idx=entry_idx, monitoring_window=monitoring_window,
        profilers=profilers, profiler_output_dir=profiler_output_dir,
        pcc_check=pcc_check, pcc_check_atol=pcc_check_atol, pcc_check_rtol=pcc_check_rtol
    )
    
    for sim_name, timestamp in result_dict.items():
        print(f"{sim_name}: {timestamp} cycles")


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.INFO)
    
    args = parse_args()
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config).initialize()
    device.set_command_debug_verbosity(verbose=False)

    module = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).eval()
    dummy_inputs = [torch.randn(1, 3, 224, 224)]
    
    graph_recipe = MCA_NetworkRecipe(
        device=device,
        core_groups=device.get_npu_core_group((0, 0), (12, 14)),
        spad_space_size_per_core=parse_mem_cap_str("1MB"),
        dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    
    graph = compile_graph(
        module, graph_recipe, dummy_inputs, 
        print_graph_summary=args.print_graph_summary, print_compile_summary=args.print_compile_summary)
        
    if args.print_autorun_config:
        print_autorun_config(graph)
        
    if args.profiler_output_dir is not None:
        os.makedirs(args.profiler_output_dir, exist_ok=True)
        
        profilers = [
            DRAMBandwidthProfiler(device, record_type="BOTH"),
            InterconnectBandwidthProfiler(device),
        ]
    else:
        profilers = None
        
    if args.run_compiled_graph:
        run_compiled_graph(
            graph, dummy_inputs,
            
            # target entries
            group_idx=args.group_idx,
            entry_idx=args.entry_idx, 
            
            # monitoring
            monitoring_window=args.monitoring_window,
            
            # profiling
            profilers=profilers,
            profiler_output_dir=args.profiler_output_dir,
            
            # validation (PCC checking)
            pcc_check=args.pcc_check, 
            pcc_check_atol=1e-4, 
            pcc_check_rtol=1e-4,
        )
