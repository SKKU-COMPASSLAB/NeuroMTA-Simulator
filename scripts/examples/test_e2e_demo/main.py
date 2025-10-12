import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent import *


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    torch.set_printoptions(precision=4, linewidth=1024)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    device.initialize()
    
    core_grid = device.get_npu_core_grid(offset=(0 , 0), shape=(4, 4))
    host_context = TT_HOST_CONTEXT(device=device, core_ids=core_grid)

    model = CNN().eval()
    
    print(f"=== Trace Graph ===")
    dummy_input = torch.randn(1, 1, 28, 28)
    graph = NetworkGraph.from_trace(model, dummy_input, host_context=host_context)
    graph.print_graph()
    
    with torch.no_grad():
        with MonitoringWindow() as monitor:
            for core_id in core_grid.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar = monitor.add_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
                pbar.bind_core(core)
                
            dummy_input = torch.randn(1, 1, 28, 28)
            reference = model(dummy_input)
            simulated = graph.run_graph(dummy_input)
        
        print(f"=== Check Integrity ===")
        print(f"reference: {reference}")
        print(f"simulated: {simulated}")
