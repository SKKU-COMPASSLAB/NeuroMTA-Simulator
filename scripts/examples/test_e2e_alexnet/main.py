import torch
import torchvision

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.hardware.implementation.mta.network import *
from neuromta.ip.mta.tenstorrent import *


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    torch.set_printoptions(precision=4, linewidth=1024)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    device.initialize()
    device.set_command_debug_verbosity(verbose=False)
    
    core_grid = device.get_npu_core_grid(offset=(0 , 0), shape=(8, 8))
    network_context = MTA_NETWORK_CONTEXT(device=device, core_grid=core_grid)

    input_shape = (1, 3, 224, 224)
    model = torchvision.models.alexnet(num_classes=1024).eval()
    
    with torch.no_grad():
        with network_context:
            print(f"=== Trace Graph ===")
            dummy_input = torch.randn(*input_shape)
            graph = NetworkGraph.from_trace(model, dummy_input)
            graph.print_graph()
        
            # with MonitoringWindow() as monitor:
            #     for core_id in core_grid.core_ids:
            #         core = device.get_npu_core(core_id=core_id)
            #         pbar_idx = monitor.add_core_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            #         monitor.pbar_handles[pbar_idx].bind_core(core)
            #     network_context.attach_monitoring_window()

            #     dummy_input = torch.randn(*input_shape)
            #     reference = model(dummy_input)
            #     simulated = graph.run_graph(dummy_input)
            
            # print(f"=== Check Integrity ===")
            # print(f"reference: {reference}")
            # print(f"simulated: {simulated}")
            # print(f"simulation terminated at timestamp {device.timestamp}")
