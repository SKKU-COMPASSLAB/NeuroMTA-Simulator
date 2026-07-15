# Introduction to NeuroMTA Component

## Overview

NeuroMTA simulator provides `neuromta.component`, which contains the actual implementation of predetermined hardware architectures including multi-tile accelerator. You can check details of each hardware architecture including NPU core, MXU (Matrix Multiplication Unit), VPU (Vector Processing Unit), interconnect network, and DMA (Direct Memory Access) engines. In this subproject, you can also see software components that determine tensor memory layout, map tiled operators to NPU cores, and compile DNN operators while considering on-chip memory usage and data transfer between NPU and DMA cores.


## Cores

Cores provided by the `neuromta.component` are basic building blocks to create accelerator simulation model.

### NPU Core

`NPUCore` is a user-programmable core module that has its own MXU (Matrix Multiplication Unit) and VPU (Vector Processing Unit). All NPU cores can directly transfer data to other `NPUCore` and `DMACore` through a dedicated interconnect network. Each `NPUCore` has its own memory context called `mem_handle`, which corresponds to the local L1 memory. The remote L1 and DRAM memory access can be implemented with RPC (Remote Procedure Call) protocol API. `NPUCore` provides command interfaces related to the MXU and VPU operations, where each command preloads data stored in `DataContainer`, executes computation, and flushes the output to the `DataContainer`.

#### (with BookSim2)

If `pybooksim2` is enabled with the `neuromta.component`, your device instance may integrate the `BookSim2` companion module. In that case, `NPUCore` sends RPC message to the `BookSim2` companion module and waits until the message is properly handled by the `CompanionCore`.

#### (with lightweight interconnect simulation model)

If `IcntSimulator` is enabled with the `IcntContext`, `NPUCore` can use a lightweight `IcntSimulator` to estimate the latency of each interconnect packet transfer request.

### DMA Core

`DMACore` is a core module which is responsible for controlling the DRAM channel (or bank). Each `DMACore` has its own memory context called `mem_handle`, which corresponds to the DRAM bank associated with the given `DMACore`. Once `NPUCore` transfers an RPC message corresponding to DRAM read/write requests, the `DMACore` updates the `mem_handle` context and sends the response through the `DataContainer` if necessary.

#### (with DRAMSim3)

If `pydramsim3` is enabled with the `neuromta.component`, your device instance may integrate the `DRAMSim3` companion module. In that case, `DMACore` sends RPC message to the `DRAMSim3` companion module and waits until the message is properly handled by the `CompanionCore`.

#### (with lightweight DRAM simulation model)

If `MemorySimulator` is enabled with the `GlobalContext`, `DMACore` can use a lightweight `MemorySimulator` to estimate the latency of each DRAM read/write request.


## Companions

Companions provided by the `neuromta.component` are cycle-level simulator extensions to support accurate simulation of interconnect network and device memory (DRAM).

### BookSim2

`BookSim2` is a companion module that connects NeuroMTA cores to the external `pybooksim2` network simulator. It is used when the simulation needs a more detailed interconnect model than the lightweight internal interconnect context. `BookSim2Config` describes the network configuration used to initialize the companion.

### DRAMSim3

`DRAMSim3` is a companion module that connects NeuroMTA DMA cores to the external `pydramsim3` memory simulator. It is used when the simulation needs a detailed DRAM timing model. `DRAMSim3Config` describes the DRAM system configuration used to initialize the companion.


## Contexts

Contexts in `neuromta.component.context` describe shared architectural state used by cores and compiled kernels.

### Global Context

`GlobalContext` is the central hardware context of a device. It records core information, memory regions, memory type, and device-wide simulation resources. It also owns the optional lightweight `MemorySimulator`, which can be used to estimate DRAM access latency without invoking an external DRAM simulator.

### Interconnect Context

`IcntContext` and `IcntSimulator` describe the interconnect state used by core-to-core and core-to-DMA communication. They provide the internal model for routing and timing communication between accelerator cores when an external `BookSim2` companion is not used.

### MXU Context

`MXUContext` describes the state of the matrix computation unit used by `NPUCore`. `MXUConfig` stores the shape and timing parameters of the matrix unit, and `MXUElementwiseOp` identifies elementwise operations that can be associated with MXU execution.

### VPU Context

`VPUContext` describes the vector computation unit used by `NPUCore`. `VPUConfig` stores the VPU configuration, and `VPUOperator` identifies supported vector-style operations.


## Hardware Implementation

The `neuromta.component.implementation.hardware` module builds concrete accelerator devices from the core and context layers.

`MCA_DeviceBase` is the base class for multi-core accelerator devices. It combines NPU cores, DMA cores, memory spaces, and device-level context into one simulation target. `MTA_DeviceBase` extends this idea for multi-tile accelerator configurations.

`MCA_CoreGroup` and `MTA_CoreGrid` represent groups or grids of cores. These classes are used to organize the spatial structure of the accelerator and to map software work onto physical core locations.

`MCA_MemorySpace`, `MCA_MainMemorySpace`, and `MCA_L1MemorySpace` describe the logical memory spaces visible to operators and kernels. They connect high-level tensor placement decisions to the underlying `MemoryHandle` objects used by the simulation runtime.


## Software Implementation

The implementation package also contains compiler-facing software abstractions that translate neural network operations into executable simulation programs.

### Tensor Buffer

`MCA_TensorBuffer` represents tensor storage managed by the compiler and runtime. It is used to describe tensor shape, memory location, and tile-level access information before the data is consumed by compiled kernels.

### Mapping

`TileSignature` and `TiledOperatorSignature` describe tiled work units. They are the intermediate representation used when an operator is divided into smaller pieces that can be assigned to different cores.

### Operator

`MCA_OperatorSignature` describes an operator before compilation. `MCA_OperatorGraphCompiler` converts one or more operator signatures into compiled programs, considering tiling, memory reuse, spatial dataflow, and synchronization. `MCA_CompiledOperator` and `MCA_CompiledProgram` represent the result of this compilation step before dispatching to executable kernels.

### Kernel

`MCA_KernelTemplate` is the bridge between compiled operator IR and executable simulation kernels. Kernel templates define how compiled load, execute, and store stages are materialized into core programs.

### Network

`NetworkGraphEntry`, `CompiledGraphEntry`, `NetworkRecipe`, `NetworkGraphContext`, and `MCA_CompiledNetworkGraph` describe model-level compilation. These classes are used to trace a neural network, split it into compilable entries, compile each entry, and run or profile the resulting graph.


## Utilities

Utilities in `neuromta.component.utils` provide profiling support for experiments and debugging.

`DRAMBandwidthProfiler`, `InterconnectBandwidthProfiler`, and `ThreadUtilizationProfiler` collect high-level statistics from device execution. `ProfilerTemplate` and `GroupedProfilerTemplate` provide reusable profiler structure for per-core, per-group, or per-thread measurements.


## Relationship with `neuromta.framework`

`neuromta.component` is implemented on top of the generic runtime abstractions in `neuromta.framework`.

`Core`, `Kernel`, `Command`, `RPCMessage`, and `Program` define the generic execution model used by component-level cores. `Device` provides the base device container. `MemoryHandle`, `MemoryBankHandle`, `Pointer`, and `ReferencePointer` provide the memory abstraction used by both NPU and DMA cores. `FIFOBufferHandle` and `VariableHandle` provide synchronization primitives used by compiled kernels. `CompanionModule` and `CompanionCore` provide the generic interface used by `BookSim2` and `DRAMSim3`.

In short, `neuromta.framework` provides the simulation runtime, while `neuromta.component` provides concrete accelerator components, hardware contexts, compiler objects, and profiling utilities built on top of that runtime.
