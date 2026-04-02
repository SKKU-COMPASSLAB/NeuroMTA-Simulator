import os
import json
import time
import torch
import torchvision

from neuromta.framework import *
from neuromta.component.implementation import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import *


FILEROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
LOGDIR = os.path.join(FILEROOT, ".logs")


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config).initialize()
    device.set_command_debug_verbosity(verbose=False)

    module = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    core_groups = device.get_npu_core_group().split((2, 2))
    
    graph_recipe  = MCA_NetworkRecipe(
        device=device,
        core_groups=core_groups,
        
        main_data_mem_space_size_per_channel=parse_mem_cap_str("30GB"),
        l1_data_mem_space_size_per_core=parse_mem_cap_str("256KB"),
        spad_mem_space_size_per_core=parse_mem_cap_str("1MB"),
        
        pipeline_granularity=16,
        broadcast_optimize_queue_depth=32,
        operator_pipelining=True,
        
        dtype=torch.float16, 
        acc_dtype=torch.float16,
    )
    
    graph = MCA_CompiledNetworkGraph.from_trace(module, graph_recipe, dummy_input)
    graph.print_graph()
    
    def save_compiled_entry_summary(graph: MCA_CompiledNetworkGraph, dirname: str, filename: str):
        for entry_idx, entry in enumerate(graph.graph_entries):
            if entry.is_subgraph_available:
                save_compiled_entry_summary(entry.subgraph, dirname, f"{filename}_{entry.node.debugName()}")
            if not entry.is_compiled:
                continue
            
            summary = entry.summary()
            os.makedirs(dirname, exist_ok=True)
            
            entry_structure = []
            
            for op_id, op_summary in summary.items():
                filepath = os.path.join(dirname, f"{filename}_entry{entry_idx}_{op_id}.json")
                with open(filepath, "w") as f:
                    json.dump(op_summary, f, indent=4)
                    logger.info(f"saved compiled summary for op {op_id} to {filepath}")
                entry_structure.append({
                    "op_id": op_id,
                    "summary_filepath": filepath,
                })
                
            structure_filepath = os.path.join(dirname, f"{filename}_entry{entry_idx}_structure.json")
            with open(structure_filepath, "w") as f:
                json.dump(entry_structure, f, indent=4)
                logger.info(f"saved compiled entry structure for entry {entry_idx} to {structure_filepath}")
                    
    save_compiled_entry_summary(graph, os.path.join(LOGDIR, FILENAME), "summary")
    
    with MonitoringWindow(device, core_groups) as monitor:
        reference = module(dummy_input)
        
        st = time.time()
        try:
            simulated = graph.run_graph(dummy_input, pcc_check=True)  # run the graph with pre-run for compiled-entry validation
        except Exception as e:
            logger.error(f"simulation terminated early due to compiled-entry validation failure: {e}")
            raise
        ed = time.time()
        
    logger.info(f"reference computation time: {ed - st:.4f} sec")
        
    if torch.allclose(reference, simulated, atol=1e-5):
        logger.info("simulation successful: outputs match reference.")
    else:
        logger.error("simulation failed: outputs do not match reference.")
        logger.error(f"normalized error: {torch.norm(reference - simulated) / torch.norm(reference) * 100:.4f}%")
        # print("reference:\n", reference)
        # print("simulated:\n", simulated)