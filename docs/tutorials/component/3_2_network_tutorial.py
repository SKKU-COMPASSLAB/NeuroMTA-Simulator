import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.software.common.network import MCA_NetworkRecipe


main_mem_config = MainMemoryConfig(
    dramsim3_enable=False
)

global_config = GlobalContextConfig(
    n_npu_core=4,
    n_dma_core=2,
    l1_mem_bank_size=parse_mem_cap_str("1MB"),
    l1_mem_dynamic_space_size_per_bank=0,
    main_mem_config=main_mem_config,
)

icnt_config = IcntConfig(
    processor_clock_freq=parse_freq_str("1GHz"),
    shape=(2, 3),
    booksim2_enable=False
)
icnt_config.update_core_map((0, 0), global_config.dma_core_ids[0])
icnt_config.update_core_map((0, 1), global_config.npu_core_ids[0])
icnt_config.update_core_map((0, 2), global_config.npu_core_ids[1])
icnt_config.update_core_map((1, 0), global_config.dma_core_ids[1])
icnt_config.update_core_map((1, 1), global_config.npu_core_ids[2])
icnt_config.update_core_map((1, 2), global_config.npu_core_ids[3])

mxu_config = MXUConfig(
    pe_arr_height=32,
    pe_arr_width=32,
    seq_len=32,
)

vpu_config = VPUConfig(
    # use default config
)

device = MTA_DeviceBase(
    global_config=global_config,
    icnt_config=icnt_config,
    mxu_config=mxu_config,
    vpu_config=vpu_config
)

device.initialize()
device.set_command_debug_verbosity(False)
logger.set_print_options(LogLevel.DEBUG)

class CustomMNISTCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

module = CustomMNISTCNN().eval()
dummy_inputs = [torch.randn(1, 1, 28, 28)]

core_group = device.get_npu_core_group((0, 0), (2, 2))

recipe = MCA_NetworkRecipe(
    device, core_group,
    main_space_size_per_channel=parse_mem_cap_str("1GB"),
    data_space_size_per_core=parse_mem_cap_str("512KB"),
    spad_space_size_per_core=parse_mem_cap_str("512KB"),
    context_buffer_slot_num=16,
    fifo_buffer_slot_num=16,
    temporal_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN,
    spatial_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_L1,
)

graph = MCA_CompiledNetworkGraph.from_trace(module, recipe, *dummy_inputs)

result_dict = graph.run_compiled_graph(*dummy_inputs)

for sim_name, result in result_dict.items():
    timestamp = result['timestamp']
    print(f"simulation of {sim_name} terminated with timestamp: {timestamp} cycles")
