# Tutorial 2: Customizing Device and Model Presets

## Make a custom device preset

Make a repository to store custom device presets.

```bash
mkdir custom_devices
```

Create a python source code that includes several predefined variables specifying the device architecture and compilation recipe. 

* `DEVICE`: an initialized device instance that inherits `MTA_DeviceBase` or `MCA_DeviceBase`
* `CORE_GROUP_OFFSET`: the offset of the core group (`tuple` for `MTA_DeviceBase` / `int` for `MCA_DeviceBase`)
* `CORE_GROUP_SHAPE`: the shape of the core group (`tuple` for `MTA_DeviceBase` / `int` for `MCA_DeviceBase`)
* `RECIPE`: a dictionary specifiying other variables related to the compilation recipe 

In this tutorial, we are going to use Tenstorrent Blackhole architecture and $8 \times 8$ core grid. We will change the datatype from `float16` into `float32`. 

```python
# custom_devices/custom_device.py

import torch
from neuromta.runner.device_presets.tenstorrent_bh import *     # import base device preset

CORE_GROUP_SHAPE  = (8, 8)      # originally, it was (12, 14)

RECIPE = dict(
    dtype=torch.float32,        # originally, it was torch.float16
    acc_dtype=torch.float32,    # originally, it was torch.float16
)
```

## Make a custom model preset

Make a repository to store custom model presets.

```bash
mkdir custom_models
```

Create a python source code that includes several predefined variables specifying the DNN model and dummy inputs. 

* `MODULE`: `torch.nn.Module` that will be given to the model compiler
* `INPUTS`: list of arguments given to the model for further simulation

In this tutorial, we are going to use customized MLP model for MNIST dataset.

```python
# custom_models/custom_model.py

import torch

class CustomMNISTMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 2048)
        self.fc2 = torch.nn.Linear(2048, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

MODULE = CustomMNISTMLP().eval()
INPUTS = [torch.randn(1, 1, 28, 28)]

```

## Run simulation with NeuroMTA Runner

Use `open_repo` command to open custom device and model preset repositories. After opening all the repositories, you can see that the custom presets are now available for further use.

```
>>> open_repo device ./custom_devices/
[INFO] [Runner] Added './custom_devices/' to device presets directories.
>>> open_repo model ./custom_models/
[INFO] [Runner] Added './custom_models/' to model presets directories.
>>> list
[INFO] [Runner] Models:
[INFO] [Runner]  - mnist
[INFO] [Runner]  - alexnet
[INFO] [Runner]  - resnet18
[INFO] [Runner]  - llama2_attn_decode
[INFO] [Runner]  - llama2_attn_prefill
[INFO] [Runner]  - custom_model
[INFO] [Runner] Devices:
[INFO] [Runner]  - tenstorrent_bh
[INFO] [Runner]  - google_tpuv4
[INFO] [Runner]  - tenstorrent_wh
[INFO] [Runner]  - custom_device
```

Open a session with the given custom device and model presets.

```
>>> open_session custom_device custom_model
[INFO] [Runner] Opening session...
[INFO] [Runner]   Device preset: custom_device
[INFO] [Runner]   Model preset: custom_model
[INFO] [Runner]   Number of workers: 1
[INFO] [Runner] Session 0 initialization succeeded.
[INFO] [Runner] All sessions initialized successfully.
>>> compile_graph
[INFO] [Runner] Compiling graph on all sessions...
[INFO] [Runner] Graph compiled successfully for session 0.
[INFO] [Runner] Compilation Summary:
[INFO] [Runner]   GROUP 0:
[INFO] [Runner]     ENTRY 0: node=aten::linear, op_method=MCA_OP_LINEAR
[INFO] [Runner]     ENTRY 1: node=aten::relu, op_method=MCA_OP_RELU
[INFO] [Runner]     ENTRY 2: node=aten::linear, op_method=MCA_OP_LINEAR
[INFO] [Runner]     ENTRY 3: node=aten::relu, op_method=MCA_OP_RELU
[INFO] [Runner]     ENTRY 4: node=aten::linear, op_method=MCA_OP_LINEAR
>>> run_graph
[INFO] [Runner] Running graph on all sessions...
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 0 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 1 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 2 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 3 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 4 on session 0
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 0
[INFO] [Runner]     CustomMNISTMLP::group0::entry0: {'timestamp': 26119}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 1
[INFO] [Runner]     CustomMNISTMLP::group0::entry1: {'timestamp': 113}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 2
[INFO] [Runner]     CustomMNISTMLP::group0::entry2: {'timestamp': 11401}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 3
[INFO] [Runner]     CustomMNISTMLP::group0::entry3: {'timestamp': 113}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 4
[INFO] [Runner]     CustomMNISTMLP::group0::entry4: {'timestamp': 468}
[INFO] [Runner] All scheduled graph entries have been executed.
>>> exit
[INFO] [Runner] Closing active sessions before exiting...
[INFO] [Runner] Session 0 closed successfully.
[INFO] [Runner] Exiting NeuroMTA Runner...
```

