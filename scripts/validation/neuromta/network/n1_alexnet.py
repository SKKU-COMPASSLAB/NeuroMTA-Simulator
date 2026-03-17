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

    module = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    global_core_group = device.get_npu_core_group()
    core_group_shape = (2, 2)  # each core group has 4 cores
    core_groups = global_core_group.split(core_group_shape)
    
    graph_recipe  = MCA_NetworkRecipe(
        device=device,
        global_core_group=global_core_group,
        core_group_shape=core_group_shape, 
        
        main_data_mem_space_size_per_channel=parse_mem_cap_str("30GB"),
        l1_data_mem_space_size_per_core=parse_mem_cap_str("256KB"),
        spad_mem_space_size_per_core=parse_mem_cap_str("1.2MB"),
        
        dtype=torch.float32, 
        acc_dtype=torch.float32,
    )
    
    graph = MCA_NetworkGraphCompiler.from_trace(module, graph_recipe, dummy_input)
    graph.print_graph()
    
    def save_compiled_entry_summary(graph: MCA_NetworkGraphCompiler, dirname: str, filename: str):
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
    
    with MonitoringWindow() as monitor:
        for core_group_idx, core_group in enumerate(core_groups):
            for core_id in core_group.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar_idx = monitor.add_core_pbar(desc=f"G{core_group_idx+1:<2d} {core_id:<3d}", ncols=40)
                monitor.pbar_handles[pbar_idx].bind_core(core)
        
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