# Tutorial 1: Quick Start with NeuroMTA Runner

## Prerequisites

We assume that you have created `neuromta` conda environment and properly installed NeuroMTA simulator as specified in the installation section of top readme document. Before getting into the further tutorial process, you should activate the environment.

```bash
conda activate neuromta
```

## Execute simulation with NeuroMTA Runner

The next step is to execute the `neuromta runner`. This application helps users easily access to the predefined simulation profiles including device and DNN workload presets. Execute the command below to spawn a `neuromta runner` shell.

```bash
neuromta_runner
```

Check the available device and model presets by simply typing `list` command.

```
>>> list
[INFO] [Runner] Models:
[INFO] [Runner]  - mnist
[INFO] [Runner]  - alexnet
[INFO] [Runner]  - resnet18
[INFO] [Runner]  - llama2_attn_decode
[INFO] [Runner]  - llama2_attn_prefill
[INFO] [Runner] Devices:
[INFO] [Runner]  - tenstorrent_bh
[INFO] [Runner]  - google_tpuv4
[INFO] [Runner]  - tenstorrent_wh
```

Then, open a session by using `open_session` command. We are going to create a session with Tenstorrent Blackhole architecture (`tenstorrent_bh`) with a simple CNN model (`mnist`).

```
>>> open_session tenstorrent_bh mnist
[INFO] [Runner] Opening session...
[INFO] [Runner]   Device preset: tenstorrent_bh
[INFO] [Runner]   Model preset: mnist
[INFO] [Runner]   Number of workers: 1
[INFO] [Runner] Session 0 initialization succeeded.
[INFO] [Runner] All sessions initialized successfully.
```

Compile the graph (or workload) with the given device preset.

```
>>> compile_graph
[INFO] [Runner] Compiling graph on all sessions...
[INFO] [Runner] Graph compiled successfully for session 0.
[INFO] [Runner] Compilation Summary:
[INFO] [Runner]   GROUP 0:
[INFO] [Runner]     ENTRY 0: node=aten::_convolution, op_method=MCA_OP_CONV2D
[INFO] [Runner]     ENTRY 1: node=aten::relu, op_method=MCA_OP_RELU
[INFO] [Runner]     ENTRY 2: node=aten::max_pool2d, op_method=MCA_OP_MAXPOOL2D
[INFO] [Runner]     ENTRY 3: node=aten::_convolution, op_method=MCA_OP_CONV2D
[INFO] [Runner]     ENTRY 4: node=aten::relu, op_method=MCA_OP_RELU
[INFO] [Runner]     ENTRY 5: node=aten::max_pool2d, op_method=MCA_OP_MAXPOOL2D
[INFO] [Runner]   GROUP 1:
[INFO] [Runner]     ENTRY 0: node=aten::linear, op_method=MCA_OP_LINEAR
[INFO] [Runner]     ENTRY 1: node=aten::relu, op_method=MCA_OP_RELU
[INFO] [Runner]     ENTRY 2: node=aten::linear, op_method=MCA_OP_LINEAR
```

Finaly, run simulation with the given compiled graph.

```
>>> run_graph
[INFO] [Runner] Running graph on all sessions...
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 0 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 1 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 2 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 3 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 4 on session 0
[INFO] [Runner] Scheduling execution for GROUP 0 ENTRY 5 on session 0
[INFO] [Runner] Scheduling execution for GROUP 1 ENTRY 0 on session 0
[INFO] [Runner] Scheduling execution for GROUP 1 ENTRY 1 on session 0
[INFO] [Runner] Scheduling execution for GROUP 1 ENTRY 2 on session 0
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 0
[INFO] [Runner]     CustomMNISTCNN::group0::entry0: {'timestamp': 1462}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 1
[INFO] [Runner]     CustomMNISTCNN::group0::entry1: {'timestamp': 113}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 2
[INFO] [Runner]     CustomMNISTCNN::group0::entry2: {'timestamp': 230}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 3
[INFO] [Runner]     CustomMNISTCNN::group0::entry3: {'timestamp': 1734}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 4
[INFO] [Runner]     CustomMNISTCNN::group0::entry4: {'timestamp': 113}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 0 ENTRY 5
[INFO] [Runner]     CustomMNISTCNN::group0::entry5: {'timestamp': 204}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 1 ENTRY 0
[INFO] [Runner]     CustomMNISTCNN::group1::entry0: {'timestamp': 14528}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 1 ENTRY 1
[INFO] [Runner]     CustomMNISTCNN::group1::entry1: {'timestamp': 113}
[INFO] [Runner] Graph entry executed successfully for session 0.
[INFO] [Runner]   Result for GROUP 1 ENTRY 2
[INFO] [Runner]     CustomMNISTCNN::group1::entry2: {'timestamp': 437}
[INFO] [Runner] All scheduled graph entries have been executed.
```

You can save the simulation results and compilation summary as specified below.

```
>>> mkdir output
[INFO] [Runner] Directory created: output
>>> save_results ./output/results.json
[INFO] [Runner] Logs saved successfully to: ./output/results.json
>>> save_compile_summary ./output/summary.json
[INFO] [Runner] Compilation summary saved successfully to: ./output/summary.json
>>> ls output
[INFO] [Runner] Directory contents:
-rw-rw-r--  1 results.json
-rw-rw-r--  1 summary.json
```

Terminate the `neuromta runner` with `exit` command.

```
>>> exit
[INFO] [Runner] Closing active sessions before exiting...
[INFO] [Runner] Session 0 closed successfully.
[INFO] [Runner] Exiting NeuroMTA Runner...
```