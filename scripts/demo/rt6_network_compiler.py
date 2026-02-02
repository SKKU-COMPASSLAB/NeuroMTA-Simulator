import time
import torch
import torchvision

from neuromta.framework import *
from neuromta.component.implementation import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import *


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config).initialize()
    device.set_command_debug_verbosity(verbose=True)

    module = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    core_groups = device.get_npu_core_group().split(shape=(2, 2))[:8]  # use only 8 core groups for this demo
    
    graph_recipe  = MCA_NETWORK_COMPILE_RECIPE(
        device=device,
        core_groups=core_groups,
        
        dtype=torch.float16, 
        acc_dtype=torch.float16,
        
        main_data_mem_space_size=parse_mem_cap_str("30GB"),
        l1_mem_space_size_per_core=parse_mem_cap_str("1.5MB"),
        l1_spad_ld_pp_space_ratio=0.3,
        l1_spad_st_pp_space_ratio=0.1,
        
        pipelining_window=5,
    )
    
    graph = NetworkGraphCompiler.from_trace(module, graph_recipe, dummy_input)
    graph.print_graph()
    
    with MonitoringWindow() as monitor:
        for core_group_idx, core_group in enumerate(core_groups):
            for core_id in core_group.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar_idx = monitor.add_core_pbar(desc=f"G{core_group_idx+1:<2d} {core_id:<3d}", ncols=40)
                monitor.pbar_handles[pbar_idx].bind_core(core)
        
        reference = module(dummy_input)
        
        st = time.time()
        simulated = graph.run_graph(dummy_input)
        # simulated = graph.run_graph_compiled_parallel(dummy_input)
        ed = time.time()
        
    logger.info(f"reference computation time: {ed - st:.4f} sec")
        
    if torch.allclose(reference, simulated, atol=1e-5):
        logger.info("simulation successful: outputs match reference.")
    else:
        logger.error("simulation failed: outputs do not match reference.")
        logger.error(f"normalized error: {torch.norm(reference - simulated) / torch.norm(reference) * 100:.4f}%")
        print("reference:\n", reference)
        print("simulated:\n", simulated)