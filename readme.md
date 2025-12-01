# NeuroMTA Simulator

## Introduction

NeuroMTA is a highly programmable cycle-level multi-tile deep learning accelerator simulator. This simulator provides a fundamental framework to implement various multi-tile accelerator architectures and programming API to create test workload with inter-core spatial dataflow. The simulator is implemented as a Python library and easy to be extended by the hardware and software developers. 

## Installation

### NeuroMTA Simulator

```bash
conda create -n neuromta python=3.11    # python >= 3.11
conda activate neuromta
pip install -r requirements.txt

# install pytorch with CPU backend (unnecessary if PyTorch is already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install NeuroMTA simulator
pip install -e .
```

### NeuroMTA Simulator Extension Modules

```bash
# Common Initialization
git submodule update --init --recursive
conda activate neuromta
pip install cython  # extension modules are built upon Cython!

# Install PyBookSim (python extension of booksim2)
sudo apt update
sudo apt install flex bison
pip install ./externals/pybooksim2    # pybooksim2 (cycle-level NoC simulator)

# Install PyDRAMSim (python extension of dramsim3)
pip install ./externals/pydramsim3    # pydramsim3 (cycle-level DRAM simulator)
```

## Deep Dive into NeuroMTA

### NeuroMTA Framework

NeuroMTA simulator provides a comprehensive framework `neuromta/framework` to implement behavioral and cycle-level model of the deep learning accelerator. The framework includes several metaclasses to create cores, memory space, and device instances. You can create your own cores and hardware components by defining command-level interface of them.

### NeuroMTA Component (under-development)

NeuroMTA simulator provides `neuromta/component`, which contains the actual implementation of predetermined hardware architectures including multi-tile accelerator. You can check details of each hardware architecture including NPU core, MXU (Matrix Multiplication Unit) and DMA (Direct Memory Access) engines. In this subproject, you can see several softwares that determines the memory layout of the tensors, dynamically maps tiled operators to each NPU core, and compiled DNN operator considering the on-chip memory usage and data transfer between NPU and DMA cores.

### NeuroMTA System (under-development)

NeuroMTA simulator provides `neuromta/system`, which contains the preset of the commercial NPU architectures.