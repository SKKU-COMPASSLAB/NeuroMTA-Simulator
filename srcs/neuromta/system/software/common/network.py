from ast import operator
import functools
import functools
from typing import Iterable, List
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


def _find_smallest_divisor_above(num: int, threshold: int) -> int:
    for i in range(threshold, num + 1):
        if num % i == 0:
            return i
    return num


class MCA_NETWORK_COMPILE_RECIPE(NetworkGraphCompilationRecipe):
    DEFAULT = "__DEFAULT"
    
    def __init__(
        self, 
        device: MCA_DeviceBase,
        core_groups: List[MCA_CoreGroup],
        
        dtype: torch.dtype, 
        acc_dtype: torch.dtype,
        
        main_data_mem_space_size: int,
        l1_mem_space_size_per_core: int,
        l1_spad_ld_pp_space_ratio: float,
        l1_spad_st_pp_space_ratio: float,
        
        pipelining_window: int,
    ):
        if core_groups == MCA_NETWORK_COMPILE_RECIPE.DEFAULT:
            default_core_ids = device.npu_core_ids
            core_groups = [MCA_CoreGroup([core_id]) for core_id in default_core_ids]
        
        super().__init__(
            device=device, 
            core_groups=core_groups,
            main_data_mem_space_size=main_data_mem_space_size, 
            l1_mem_space_size_per_core=l1_mem_space_size_per_core,
            l1_spad_ld_pp_space_ratio=l1_spad_ld_pp_space_ratio,
            l1_spad_st_pp_space_ratio=l1_spad_st_pp_space_ratio,
            max_pipeline_window=pipelining_window,
        )
    
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        
        self._cached_main_data_mem_space = None
        
    @NetworkGraphCompilationRecipe.recipe
    def Linear(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Linear) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        wgt: torch.Tensor = submodule.weight.data
        
        if ifm.dtype == wgt.dtype:
            ofm_dtype = ifm.dtype
        else:
            raise Exception(f"Incompatible dtypes between ifm ({ifm.dtype}) and wgt ({wgt.dtype})")  # TODO: support mixed precision
        
        M, K = ifm.shape
        N, K = wgt.shape

        M_SHARD = 32 if (M % 32 == 0) else _find_smallest_divisor_above(M, 32)
        N_SHARD = 32 if (N % 32 == 0) else _find_smallest_divisor_above(N, 32)
        K_SHARD = 32 if (K % 32 == 0) else _find_smallest_divisor_above(K, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        wgt_src  = NetworkGraphCompiledEntry.BufferSource("weight", submodule)
        bias_src = NetworkGraphCompiledEntry.BufferSource("bias", submodule)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(ifm_src,  None,     (M, K),  self.dtype,     (M_SHARD, K_SHARD), False, ifm.dtype, buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT),
            "wgt":  NetworkGraphCompiledEntry.BufferSignature(wgt_src,  None,     (N, K),  self.dtype,     (N_SHARD, K_SHARD), False, wgt.dtype, buffer_usage=NetworkGraphCompiledEntry.BufferUsage.PARAMS),            
            "bias": NetworkGraphCompiledEntry.BufferSignature(bias_src, None,     (1, N,), self.acc_dtype, (1, N_SHARD),       False, ofm_dtype, buffer_usage=NetworkGraphCompiledEntry.BufferUsage.PARAMS),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(None,     ofm_src,  (M, N),  self.acc_dtype, (M_SHARD, N_SHARD), False, ofm_dtype, buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT),
        }
        
        runtime_kwargs = {}
        
        target_op_method = MCA_OP_LINEAR
        total_ops = 2 * M * N * K
            
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    
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
        
        W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
        OW_SHARD = 32 if (OW % 32 == 0) else _find_smallest_divisor_above(OW, 32)
        C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
        K_SHARD  = 32 if (K  % 32 == 0) else _find_smallest_divisor_above(K, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        wgt_src  = NetworkGraphCompiledEntry.BufferSource("weight", submodule)
        bias_src = NetworkGraphCompiledEntry.BufferSource("bias", submodule)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, None,  (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT, preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]),
            "wgt":  NetworkGraphCompiledEntry.BufferSignature(
                wgt_src, None,  (FH, FW, K, C), self.dtype,     (K_SHARD, C_SHARD),   False, wgt.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.PARAMS, preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(2, 3, 0, 1)]),
            "bias": NetworkGraphCompiledEntry.BufferSignature(
                bias_src, None, (1, K,),        self.acc_dtype, (1, K_SHARD),         False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.PARAMS,),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(
                None, ofm_src,  (N, OH, OW, K), self.acc_dtype, (OW_SHARD, K_SHARD),  False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT, postprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]),
        }
        
        runtime_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": submodule.groups,
        }
        
        target_op_method = MCA_OP_CONV2D
        total_ops = 2 * N * K * OH * OW * C * FH * FW
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    @NetworkGraphCompilationRecipe.recipe
    def ReLU(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.ReLU) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        if len(ifm.shape) == 2:
            M = ifm.shape[-2]
            N = ifm.shape[-1]
            
            M_SHARD = 32 if (M % 32 == 0) else _find_smallest_divisor_above(M, 32)
            N_SHARD = 32 if (N % 32 == 0) else _find_smallest_divisor_above(N, 32)
            
            buffer_shape = (M, N)
            shard_shape = (M_SHARD, N_SHARD)
            preprocessing = []
            postprocessing = []
        elif len(ifm.shape) == 4:
            N, C, H, W = ifm.shape
            
            W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
            C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
            
            buffer_shape = (N, H, W, C)
            shard_shape = (W_SHARD, C_SHARD)
            preprocessing = [NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]
            postprocessing = [NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]
        else:
            raise Exception(f"Unsupported tensor shape for ReLU operator: {ifm.shape}")
        
        ifm_src = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        ifm_dst = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm": NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, ifm_dst, buffer_shape, self.dtype, shard_shape, False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPLACE,
                preprocessings=preprocessing,
                postprocessings=postprocessing,
            ),
        }
        
        runtime_kwargs = {}
        
        target_op_method = MCA_OP_RELU_INPLACE
        total_ops = ifm.numel()
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    @NetworkGraphCompilationRecipe.recipe
    def MaxPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.MaxPool2d) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        stride = (submodule.stride, submodule.stride) if not isinstance(submodule.stride, Iterable) else submodule.stride
        padding = (submodule.padding, submodule.padding) if not isinstance(submodule.padding, Iterable) else submodule.padding
        dilation = (1, 1)
        FH = FW = submodule.kernel_size  # Assuming kernel_size is an int for MaxPool2d
        
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
        OW_SHARD = 32 if (OW % 32 == 0) else _find_smallest_divisor_above(OW, 32)
        C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, None,  (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT,  preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(
                None, ofm_src,  (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT, postprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]),
        }
        
        runtime_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        target_op_method = MCA_OP_MAXPOOL2D
        total_ops = 2 * N * C * OH * OW * FH * FW
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    @NetworkGraphCompilationRecipe.recipe
    def AvgPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AvgPool2d) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        stride = submodule.stride
        padding = submodule.padding
        dilation = (1, 1)  # AvgPool2d in PyTorch does not have dilation parameter
        FH, FW = submodule.kernel_size
        
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
        OW_SHARD = 32 if (OW % 32 == 0) else _find_smallest_divisor_above(OW, 32)
        C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, None,  (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT,  preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(
                None, ofm_src,  (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT, postprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]),
        }
        
        runtime_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        target_op_method = MCA_OP_AVGPOOL2D
        total_ops = 2 * N * C * OH * OW * FH * FW
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    @NetworkGraphCompilationRecipe.recipe
    def AdaptiveMaxPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AdaptiveMaxPool2d) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        OH, OW = submodule.output_size
        FH, FW = H // OH, W // OW
        stride = (FH, FW)
        padding = (0, 0)
        dilation = (1, 1)
        
        W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
        OW_SHARD = 32 if (OW % 32 == 0) else _find_smallest_divisor_above(OW, 32)
        C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, None,  (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT,  preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(
                None, ofm_src,  (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT, postprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]),
        }
        
        runtime_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        target_op_method = MCA_OP_MAXPOOL2D
        total_ops = 2 * N * C * OH * OW * FH * FW
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
        
    @NetworkGraphCompilationRecipe.recipe
    def AdaptiveAvgPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AdaptiveAvgPool2d) -> NetworkGraphCompiledEntry:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        OH, OW = submodule.output_size
        FH, FW = H // OH, W // OW
        stride = (FH, FW)
        padding = (0, 0)
        dilation = (1, 1)
        
        W_SHARD  = 32 if (W  % 32 == 0) else _find_smallest_divisor_above(W, 32)
        OW_SHARD = 32 if (OW % 32 == 0) else _find_smallest_divisor_above(OW, 32)
        C_SHARD  = 32 if (C  % 32 == 0) else _find_smallest_divisor_above(C, 32)
        
        ifm_src  = NetworkGraphCompiledEntry.BufferSource(node.inputsAt(1), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        ofm_src  = NetworkGraphCompiledEntry.BufferSource(node.outputsAt(0), NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT)
        
        buffer_signatures = {
            "ifm":  NetworkGraphCompiledEntry.BufferSignature(
                ifm_src, None,  (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.INPUT,  preprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 2, 3, 1)]),
            "ofm":  NetworkGraphCompiledEntry.BufferSignature(
                None, ofm_src,  (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                buffer_usage=NetworkGraphCompiledEntry.BufferUsage.OUTPUT, postprocessings=[NetworkGraphCompiledEntry.TensorProcessing.permute(0, 3, 1, 2)]),
        }
        
        runtime_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        target_op_method = MCA_OP_AVGPOOL2D
        total_ops = 2 * N * C * OH * OW * FH * FW
        
        return NetworkGraphCompiledEntry(
            node=node,
            submodule=submodule,
            buffer_signatures=buffer_signatures,
            runtime_kwargs=runtime_kwargs,
            target_op_method=target_op_method,
            total_ops=total_ops,
        )
