import torch

from neuromta.framework import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.network import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.system.software.common.operator import *


__all__ = [
    "MCA_NETWORK_COMPILE_RECIPE",
]


class MCA_NETWORK_COMPILE_RECIPE(NetworkGraphCompilationRecipe):
    def __init__(
        self, 
        
        device: MCA_DeviceBase,
        
        dtype: torch.dtype, 
        acc_dtype: torch.dtype,
        
        main_data_mem_space_size: int,
        l1_data_mem_space_size_per_core: int,   # TODO: make this configurable by the compiler
        spad_ld_pp_space_size_per_core: int,    # TODO: make this configurable by the compiler
        spad_st_pp_space_size_per_core: int,    # TODO: make this configurable by the compiler
    ):
        super().__init__(device=device)
    
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        
        self._main_data_mem_space_size = main_data_mem_space_size
        self._l1_data_mem_space_size_per_core = l1_data_mem_space_size_per_core
        self._spad_ld_pp_space_size_per_core = spad_ld_pp_space_size_per_core
        self._spad_st_pp_space_size_per_core = spad_st_pp_space_size_per_core
        
        self._cached_main_data_mem_space = None
    
    @NetworkGraphCompilationRecipe.recipe
    def Conv2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Conv2d) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        wgt: torch.Tensor = submodule.weight.data
        
        if ifm.dtype == wgt.dtype:
            ofm_dtype = ifm.dtype
        else:
            raise Exception(f"Incompatible dtypes between ifm ({ifm.dtype}) and wgt ({wgt.dtype})")  # TODO: support mixed precision
        
        N, C, H, W = ifm.shape
        K, C, FH, FW = wgt.shape
        stride = submodule.stride
        padding = submodule.padding
        dilation = submodule.dilation
        
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        W_SHARD  = 32 if (W  % 32 == 0) else W
        OW_SHARD = 32 if (OW % 32 == 0) else OW
        C_SHARD  = 32 if (C  % 32 == 0) else C
        K_SHARD  = 32 if (K  % 32 == 0) else K
        
        buffer_signatures = {
            "ifm": NetworkGraphCompiledEntry.BufferSignature(
                shape=(N, H, W, C),
                dtype=self.dtype,
                shard_shape=(W_SHARD, C_SHARD),
                blocked_mapping=False,
                orig_dtype=ifm.dtype,
            ),
            
            "wgt": NetworkGraphCompiledEntry.BufferSignature(
                shape=(FH, FW, K, C),
                dtype=self.dtype,
                shard_shape=(K_SHARD, C_SHARD),
                blocked_mapping=False,
                orig_dtype=wgt.dtype,
            ),
            
            "bias": NetworkGraphCompiledEntry.BufferSignature(
                shape=(1, K,),
                dtype=self.acc_dtype,
                shard_shape=(1, K_SHARD),
                blocked_mapping=False,
                orig_dtype=ofm_dtype,
            ),
            
            "ofm": NetworkGraphCompiledEntry.BufferSignature(
                shape=(N, OH, OW, K),
                dtype=self.acc_dtype,
                shard_shape=(OW_SHARD, K_SHARD),
                blocked_mapping=False,
                orig_dtype=ofm_dtype,
            ),
        }
        
        runtime_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": submodule.groups,
        }
        
        def entry_compile_method(graph_context: NetworkGraphContext, entry: NetworkGraphCompiledEntry):
            if not entry.is_compilation_ready:
                raise RuntimeError("Entry is not ready for compilation.")
            
            entry._mem_space = {}
            entry._buffers   = {}
            entry._runtime_ops = []
            
            device = entry.device
            core_group = entry.core_group
            
            if self._cached_main_data_mem_space is None:
                self._cached_main_data_mem_space = device.create_main_mem_space(self._main_data_mem_space_size)
            
            main_data_mem_space = self._cached_main_data_mem_space
            l1_data_mem_space   = device.create_l1_mem_space(self._l1_data_mem_space_size_per_core, core_group=core_group)
            spad_ld_mem_space   = device.create_l1_mem_space(self._spad_ld_pp_space_size_per_core, core_group=core_group)
            spad_st_mem_space   = device.create_l1_mem_space(self._spad_st_pp_space_size_per_core, core_group=core_group)
            
            entry._mem_space["main_data_mem_space"] = main_data_mem_space
            entry._mem_space["l1_data_mem_space"]   = l1_data_mem_space
            entry._mem_space["spad_ld_mem_space"]   = spad_ld_mem_space
            entry._mem_space["spad_st_mem_space"]   = spad_st_mem_space
            
            n_cores = len(core_group.core_ids)
            remaining_l1_size = self._l1_data_mem_space_size_per_core
            
            buffer_names = ["ofm", "ifm", "wgt", "bias"]
            size_per_core = [
                entry.buffer_signatures[n].get_shard_size() * ((entry.buffer_signatures[n].get_shard_num() + n_cores - 1) // n_cores)
                for n in buffer_names
            ]
            buf_sigs = [
                entry.buffer_signatures[n] 
                for n in buffer_names
            ]
            
            for buf_sig, size, buf_name in zip(buf_sigs, size_per_core, buffer_names):
                if size <= remaining_l1_size:
                    load_l1 = True
                    remaining_l1_size -= size
                else:
                    load_l1 = False
                
                if buf_name not in entry._buffers.keys():
                    # create buffer iff it does not exist
                    #   - this allows pre-allocation of buffers outside of compile method
                    #   - this feature will be used in advance for implementing pipelining support
                    
                    entry._buffers[buf_name] = MCA_TensorBuffer(
                        mem_space=l1_data_mem_space if load_l1 else main_data_mem_space,
                        shape=buf_sig.shape,
                        dtype=buf_sig.dtype,
                        shard_shape=buf_sig.shard_shape,
                        blocked_mapping=buf_sig.blocked_mapping,
                    ).allocate()
                
            entry._runtime_ops.append(
                MCA_OP_CONV2D(
                    device, core_group, spad_ld_mem_space, spad_st_mem_space,
                    **entry._buffers,
                    **runtime_kwargs,
                    broadcast_optimize=True,
                    mapping_strategy=MCA_OperatorMapper.OUTPUT_STATIONARY,
                )
            )
            
            device.remove_all_l1_mem_space()    # clean up SPAD memory spaces after compilation
            
        def entry_buffer_init_method(graph_context: NetworkGraphContext, entry: NetworkGraphCompiledEntry):
            ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
            wgt: torch.Tensor = submodule.weight.data
            bias: torch.Tensor = submodule.bias.data if submodule.bias is not None else torch.zeros((submodule.out_channels,), dtype=self.acc_dtype)
            
            entry._buffers["ifm"].update(ifm.to(self.dtype).permute(0, 2, 3, 1).contiguous())   # NCHW -> NHWC
            entry._buffers["wgt"].update(wgt.to(self.dtype).permute(2, 3, 0, 1).contiguous())   # KCHW -> HWKC
            entry._buffers["bias"].update(bias.to(self.acc_dtype))                              # K -> 1K
        
        def entry_buffer_finalize_method(graph_context: NetworkGraphContext, entry: NetworkGraphCompiledEntry):
            ofm_buffer: MCA_TensorBuffer = entry._buffers["ofm"]
            ofm_tensor = ofm_buffer.restore().permute(0, 3, 1, 2).contiguous()   # NHWC -> NCHW
            
            graph_context[node.outputsAt(0).debugName()] = ofm_tensor.to(entry.buffer_signatures["ofm"].orig_dtype)
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            entry_compile_method=entry_compile_method,
            entry_buffer_init_method=entry_buffer_init_method,
            entry_buffer_finalize_method=entry_buffer_finalize_method,
        )