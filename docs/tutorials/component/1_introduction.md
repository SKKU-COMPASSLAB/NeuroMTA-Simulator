# Introduction to NeuroMTA Component

NeuroMTA simulator provides `neuromta.component`, which contains the actual implementation of predetermined hardware architectures including multi-tile accelerator. You can check details of each hardware architecture including NPU core, MXU (Matrix Multiplication Unit) and DMA (Direct Memory Access) engines. In this subproject, you can see several softwares that determines the memory layout of the tensors, dynamically maps tiled operators to each NPU core, and compiled DNN operator considering the on-chip memory usage and data transfer between NPU and DMA cores.

## Cores

### NPU Core

`NPUCore` class is a user-programmable core module that has its own MXU (Matrix Multiplication Unit) and VPU (Vector Processing Unit). All NPU cores can directly trasfer data to other NPU and DMA cores through a dedicated interconnect network. Each NPU core has its own memory context called `mem_handle`, which corresponds to the 

### DMA Core