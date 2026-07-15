# Tutorial for NeuroMTA Component


## Operator Simulation Example

### STEP 1: Import necessary library

```python
import torch

from neuromta.framework import *
from neuromta.component import *
```

### STEP 2: Create configs for the device

```python
# Main Memory Config
main_mem_config = MainMemoryConfig(
    dramsim3_enable=False
)

# Global Config
#   - main memory configuration
#   - l1 memory configuration
#   - number of NPUCores and DMACores
global_config = GlobalContextConfig(
    n_npu_core=4,
    n_dma_core=2,
    l1_mem_bank_size=parse_mem_cap_str("1MB"),
    l1_mem_dynamic_space_size_per_bank=0,
    main_mem_config=main_mem_config,
)

# Interconnect Config
#   - determine each node with `update_core_map`
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

# Compute Unit Configs
mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32,)
vpu_config = VPUConfig()
```

### STEP 3: Create device instance

```python
# Create device instance
device.initialize()     # initialize the device (`Device` identifies the member `Core`s)
device.set_command_debug_verbosity(False)   # do not print command logs while running kernels
logger.set_print_options(LogLevel.DEBUG)    # print DEBUG messages

# Create core group: 2x2 mesh from (0, 0) core
core_group = device.get_npu_core_group((0, 0), (2, 2))
```

### STEP 4: Create memory space and tensor buffers

```python
# Create memory space
main_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
l1_mem_space = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group)

# Create original tensor
M, N, K = 256, 256, 256

torch_ifm = torch.randint(0, 10, (M, K), dtype=torch.int32)
torch_wgt = torch.randint(0, 10, (N, K), dtype=torch.int32)
torch_bias = torch.randint(0, 10, (N,), dtype=torch.int32)

# Create tensor buffers
ifm = MCA_TensorBuffer(l1_mem_space, shape=(M, K), dtype=torch.int32)
wgt = MCA_TensorBuffer(main_mem_space, shape=(N, K), dtype=torch.int32)
bias = MCA_TensorBuffer(main_mem_space, shape=(N,), dtype=torch.int32)
ofm = MCA_TensorBuffer(l1_mem_space, shape=(M, N), dtype=torch.int32)

# Allocate tensor buffers to the memory
#   - update the tensor buffer for IFM, WGT, BIAS
#   - OFM tensor buffer is allocated, but not initialized
ifm.allocate().update(torch_ifm)
wgt.allocate().update(torch_wgt)
bias.allocate().update(torch_bias)
ofm.allocate()
```

### STEP 5: Create operator signature and compile

```python
# We are going to use the predefined operator instead of the custom one.
# See more details in `2_4_operator_api.md` to check how can we create custom operators.
from neuromta.system.software.common.operator import MCA_OP_LINEAR

# Create operator signature
op = MCA_OP_LINEAR(ifm, wgt, bias, ofm)

# Create operator compiler
compiler = MCA_OperatorGraphCompiler()
recipe = MCA_OperatorGraphCompiler.CompileRecipe(
    device,         # device instance
    core_group,     # core group
    parse_mem_cap_str("512KB")  # available L1 memory space for the compiler (e.g., cache, FIFOs ...)
)

# Compile operator signature
compiler.add_op(op)
program = compiler.compile(recipe)
```

### STEP 6: Dispatch the program and run kernels

```python
program.dispatch()
device.run_kernels()

print(f"simulation terminated with timestamp: {device.timestamp} cycles")
```

### STEP 7: Check whether the program is properly compiled

```python
reference_ofm = torch_ifm @ torch_wgt.T + torch_bias
simulated_ofm = ofm.restore()   # restore tensor from the tensor buffer

if torch.equal(reference_ofm, simulated_ofm):
    print("simulation successful: the output matches the reference.")
else:
    print("simulation failed: the output does not match the reference.")
    print("reference OFM:")
    print(reference_ofm)
    print("simulated OFM:")
    print(simulated_ofm)
```


## Network Simulation Example

To simulate DNN model, you should follow the workflow specified below after creating the device instance.

### STEP 1: Create DNN model

```python
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

module = CustomMNISTCNN().eval()            # network module to simulate
dummy_inputs = [torch.randn(1, 1, 28, 28)]  # dummy input tensors
```

### STEP 2: Create network recipe

```python
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
```

### STEP 3: Compiled graph and run simulation

```python
graph = MCA_CompiledNetworkGraph.from_trace(module, recipe, *dummy_inputs)

result_dict = graph.run_compiled_graph(*dummy_inputs)

for sim_name, result in result_dict.items():
    timestamp = result['timestamp']
    print(f"simulation of {sim_name} terminated with timestamp: {timestamp} cycles")
```