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

    module = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    graph_recipe  = MCA_NETWORK_COMPILE_RECIPE(
        device=device,
        
        dtype=torch.float32, 
        acc_dtype=torch.float32,
        
        main_data_mem_space_size=parse_mem_cap_str("30GB"),
        l1_data_mem_space_size_per_core=parse_mem_cap_str("1MB"),
        spad_ld_pp_space_size_per_core=parse_mem_cap_str("256KB"),
        spad_st_pp_space_size_per_core=parse_mem_cap_str("256KB"),
    )
    
    graph = NetworkGraphCompiler.from_trace(module, graph_recipe, dummy_input)
    graph.print_graph()
    
    reference = module(dummy_input)
    simulated = graph.run_graph(dummy_input)
    
    if torch.allclose(reference, simulated, atol=1e-5):
        logger.info("simulation successful: outputs match reference.")
    else:
        logger.error("simulation failed: outputs do not match reference.")
        logger.error(f"normalized error: {torch.norm(reference - simulated) / torch.norm(reference) * 100:.4f}%")
        print("reference:\n", reference)
        print("simulated:\n", simulated)