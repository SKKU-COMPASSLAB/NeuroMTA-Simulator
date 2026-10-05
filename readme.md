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
# Initialize submodules
git submodule update --init --recursive
conda activate neuromta
pip install cython  # extension modules are built upon Cython!

# Install PyBookSim (python extension of booksim2)
sudo apt update
sudo apt install flex bison           # BookSim2 dependency
pip install ./externals/pybooksim2    # pybooksim2 (cycle-level NoC simulator)

# Install PyDRAMSim (python extension of dramsim3)
pip install ./externals/pydramsim3    # pydramsim3 (cycle-level DRAM simulator)
```

## Simulator Architecture

### NeuroMTA Framework

NeuroMTA simulator provides a comprehensive framework `neuromta.framework` to implement behavioral and cycle-level model of the deep learning accelerator. The framework includes several metaclasses to create cores, memory space, and device instances. You can create your own cores and hardware components by defining command-level interface of them.

### NeuroMTA Component

NeuroMTA simulator provides `neuromta.component`, which contains the actual implementation of predetermined hardware architectures including multi-tile accelerator. You can check details of each hardware architecture including Compute Tile (`CCG Tile`) and DMA (Direct Memory Access) engines (`DMA Tile`).

### NeuroMTA System

NeuroMTA simulator provides `neuromta.system`, which contains the preset of the NPU architecture and full-stack softwares including `compiler`, `runtime`, `scheduler`, and `programming API`. It also provides model library called `nn`, which is built upon the programming API.

## Documentation

### Tutorials

TBD

### API Introduction

TBD

### Versions

#### NeuroMTA v2.0

* Provide general MTA architecture template called `mesh_accelerator`
    - CCG (Compute Core Group) and DMA (Direct Memory Access) tiles, interconnected via dedicated NoC
    - Job dispatcher for runtime scheduling
* Rearchitecture software stack including runtime scheduling features
    - Compiler: generates kernel descriptors with 6 different types (`LINEAR`, `CONV2D`, `SDPA`, `ELEMENTWISE`, `REDUCTION`, `MEMCOPY`)
    - Scheduler: assigns hardware resources (CCG mesh) to each kernel and generates kernel schedules
    - Runtime: dispatches kernels to the CCG and schedules multiple workloads
* Deprecated features
    - NeuroMTA Runner
    - NeuroMTA Monitor

#### NeuroMTA v1.0

* Initial simulator development version

## Citation

Please cite the following [paper (NeuroMTA v1.0)](https://ieeexplore.ieee.org/abstract/document/11617314).

```
@article{kim2026neuromta,
  title={NeuroMTA: Programmable Simulation Framework for Multi-Tile NPU Architectures},
  author={Kim, Seongwook and Hong, Seokin},
  journal={IEEE Computer Architecture Letters},
  year={2026},
  publisher={IEEE}
}
```