from ast import operator
import functools
import functools
from typing import Iterable, List
import torch

from neuromta.framework import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.network import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *
from neuromta.component.implementation.tensor_buffer import *
from neuromta.system.software.common.operator import *


__all__ = [
    "MCA_NetworkRecipe",
]


def _find_smallest_divisor_above(num: int, threshold: int) -> int:
    for i in range(threshold, num + 1):
        if num % i == 0:
            return i
    return num


CONTEXT = NetworkGraphEntryCompileTarget.BufferSignature.CONTEXT


class MCA_NetworkRecipe(MCA_CompiledNetworkGraph.NetworkRecipe):
    DEFAULT = "__DEFAULT"
    
    def __init__(
        self, 
        device: MCA_DeviceBase, 
        core_groups: list[MCA_CoreGroup],
        
        main_data_mem_space_size_per_channel: int,
        l1_data_mem_space_size_per_core: int,
        spad_mem_space_size_per_core: int,
        
        pipeline_granularity: int=8,
        broadcast_optimize_queue_depth: int=32,
        operator_pipelining: bool=False,
        context_buffer_slot_num: int=4,
        
        dtype: torch.dtype=torch.float16,
        acc_dtype: torch.dtype=torch.float16,
    ):
        super().__init__(
            device, 
            core_groups, 
        
            main_data_mem_space_size_per_channel, 
            l1_data_mem_space_size_per_core, 
            spad_mem_space_size_per_core, 
            
            pipeline_granularity, 
            broadcast_optimize_queue_depth, 
            operator_pipelining, 
            context_buffer_slot_num
        )
        
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def Linear(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Linear) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        wgt: torch.Tensor = submodule.weight.data
        
        if ifm.dtype == wgt.dtype:
            ofm_dtype = ifm.dtype
        else:
            raise Exception(f"Incompatible dtypes between ifm ({ifm.dtype}) and wgt ({wgt.dtype})")  # TODO: support mixed precision
        
        M, K = ifm.shape
        N, K = wgt.shape
        
        mxu_config = self.compiler_recipe.device.mxu_config

        M_SHARD = mxu_config.m_tile if (M % mxu_config.m_tile == 0) else _find_smallest_divisor_above(M, mxu_config.m_tile)
        N_SHARD = mxu_config.n_tile if (N % mxu_config.n_tile == 0) else _find_smallest_divisor_above(N, mxu_config.n_tile)
        K_SHARD = mxu_config.k_tile if (K % mxu_config.k_tile == 0) else _find_smallest_divisor_above(K, mxu_config.k_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature((M, K),  self.dtype,     (M_SHARD, K_SHARD), False, ifm.dtype).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature((N, K),  self.dtype,     (N_SHARD, K_SHARD), False, wgt.dtype).load_from("weight", submodule),            
            NetworkGraphEntryCompileTarget.BufferSignature((1, N,), self.acc_dtype, (1, N_SHARD),       False, ofm_dtype).load_from("bias", submodule),
            NetworkGraphEntryCompileTarget.BufferSignature((M, N),  self.acc_dtype, (M_SHARD, N_SHARD), False, ofm_dtype).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {}
        
        op_method = MCA_OP_LINEAR
        
        _total_ops = M * N * K * 2  # 2 for MAC
        _total_buf_bytes = (M * K + N * K) * (self.dtype.itemsize) + (M * N) * (self.acc_dtype.itemsize)
        arith_intensity = _total_ops / _total_buf_bytes
            
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )
        
    
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def Conv2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Conv2d) -> NetworkGraphEntryCompileTarget:
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
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
        OW_SHARD = mxu_config.m_tile if (OW % mxu_config.m_tile == 0) else _find_smallest_divisor_above(OW, mxu_config.m_tile)
        C_SHARD  = mxu_config.k_tile if (C  % mxu_config.k_tile == 0) else _find_smallest_divisor_above(C, mxu_config.k_tile)
        K_SHARD  = mxu_config.n_tile if (K  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(K, mxu_config.n_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (FH, FW, K, C), self.dtype,     (K_SHARD, C_SHARD),   False, wgt.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(2, 3, 0, 1)]).load_from("weight", submodule),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (1, K,),        self.acc_dtype, (1, K_SHARD),         False, ofm_dtype).load_from("bias", submodule),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, OH, OW, K), self.acc_dtype, (OW_SHARD, K_SHARD),  False, ofm_dtype, 
                postprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": submodule.groups,
        }
        
        op_method = MCA_OP_CONV2D
        
        _total_ops = N * K * OH * OW * C * FH * FW * 2  # 2 for MAC
        _total_buf_bytes = (N * H * W * C + FH * FW * K * C) * (self.dtype.itemsize) + (N * OH * OW * K) * (self.acc_dtype.itemsize)
        arith_intensity = _total_ops / _total_buf_bytes
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )
    
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def ReLU(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.ReLU) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        if len(ifm.shape) == 2:
            M = ifm.shape[-2]
            N = ifm.shape[-1]
            
            M_SHARD = mxu_config.m_tile if (M % mxu_config.m_tile == 0) else _find_smallest_divisor_above(M, mxu_config.m_tile)
            N_SHARD = mxu_config.n_tile if (N % mxu_config.n_tile == 0) else _find_smallest_divisor_above(N, mxu_config.n_tile)
            
            buffer_shape = (M, N)
            shard_shape = (M_SHARD, N_SHARD)
            preprocessing = []
            postprocessing = []
        elif len(ifm.shape) == 4:
            N, C, H, W = ifm.shape
            
            W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
            C_SHARD  = mxu_config.n_tile if (C  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(C, mxu_config.n_tile)
            
            buffer_shape = (N, H, W, C)
            shard_shape = (W_SHARD, C_SHARD)
            preprocessing = [NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]
            postprocessing = [NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]
        else:
            raise Exception(f"Unsupported tensor shape for ReLU operator: {ifm.shape}")
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                buffer_shape, self.dtype, shard_shape, False, ifm.dtype,
                preprocessings=preprocessing,
            ).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                buffer_shape, self.dtype, shard_shape, False, ifm.dtype,
                postprocessings=postprocessing,
            ).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {}
        
        op_method = MCA_OP_RELU
        
        _total_ops = ifm.numel()  # 1 for ReLU
        _total_buf_bytes = ifm.numel() * (self.dtype.itemsize) * 2  # load and store
        arith_intensity = _total_ops / _total_buf_bytes
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )
        
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def MaxPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.MaxPool2d) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        stride = (submodule.stride, submodule.stride) if not isinstance(submodule.stride, Iterable) else submodule.stride
        padding = (submodule.padding, submodule.padding) if not isinstance(submodule.padding, Iterable) else submodule.padding
        dilation = (1, 1)
        FH = FW = submodule.kernel_size  # Assuming kernel_size is an int for MaxPool2d
        
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
        OW_SHARD = mxu_config.m_tile if (OW % mxu_config.m_tile == 0) else _find_smallest_divisor_above(OW, mxu_config.m_tile)
        C_SHARD  = mxu_config.n_tile if (C  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(C, mxu_config.n_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                postprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        op_method = MCA_OP_MAXPOOL2D
        
        _total_ops = N * C * OH * OW * FH * FW  # 1 for max, but multiplied by kernel size for the number of comparisons
        _total_buf_bytes = (N * H * W * C) * (self.dtype.itemsize) + (N * OH * OW * C) * (self.acc_dtype.itemsize)
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=_total_ops / _total_buf_bytes,
            max_n_cores=buf_sigs[-1].n_tiles
        )
        
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def AvgPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AvgPool2d) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        stride = submodule.stride
        padding = submodule.padding
        dilation = (1, 1)  # AvgPool2d in PyTorch does not have dilation parameter
        FH, FW = submodule.kernel_size
        
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
        OW_SHARD = mxu_config.m_tile if (OW % mxu_config.m_tile == 0) else _find_smallest_divisor_above(OW, mxu_config.m_tile)
        C_SHARD  = mxu_config.n_tile if (C  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(C, mxu_config.n_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                postprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        op_method = MCA_OP_AVGPOOL2D
        
        _total_ops = N * C * OH * OW * FH * FW  # 1 for avg, but multiplied by kernel size for the number of additions
        _total_buf_bytes = (N * H * W * C) * (self.dtype.itemsize) + (N * OH * OW * C) * (self.acc_dtype.itemsize) 
        arith_intensity = _total_ops / _total_buf_bytes
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )
        
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def AdaptiveMaxPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AdaptiveMaxPool2d) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        OH, OW = submodule.output_size
        FH, FW = H // OH, W // OW
        stride = (FH, FW)
        padding = (0, 0)
        dilation = (1, 1)
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
        OW_SHARD = mxu_config.m_tile if (OW % mxu_config.m_tile == 0) else _find_smallest_divisor_above(OW, mxu_config.m_tile)
        C_SHARD  = mxu_config.n_tile if (C  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(C, mxu_config.n_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                postprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        op_method = MCA_OP_MAXPOOL2D
        
        _total_ops = N * C * OH * OW * FH * FW  # 1 for max, but multiplied by kernel size for the number of comparisons
        _total_buf_bytes = (N * H * W * C) * (self.dtype.itemsize) + (N * OH * OW * C) * (self.acc_dtype.itemsize)
        arith_intensity = _total_ops / _total_buf_bytes
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )
        
    @MCA_CompiledNetworkGraph.NetworkRecipe.recipe
    def AdaptiveAvgPool2d(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.AdaptiveAvgPool2d) -> NetworkGraphEntryCompileTarget:
        ifm: torch.Tensor = graph_context[node.inputsAt(1).debugName()]
        
        ofm_dtype = ifm.dtype
        
        N, C, H, W = ifm.shape
        OH, OW = submodule.output_size
        FH, FW = H // OH, W // OW
        stride = (FH, FW)
        padding = (0, 0)
        dilation = (1, 1)
        
        mxu_config = self.compiler_recipe.device.mxu_config
        
        W_SHARD  = mxu_config.m_tile if (W  % mxu_config.m_tile == 0) else _find_smallest_divisor_above(W, mxu_config.m_tile)
        OW_SHARD = mxu_config.m_tile if (OW % mxu_config.m_tile == 0) else _find_smallest_divisor_above(OW, mxu_config.m_tile)
        C_SHARD  = mxu_config.n_tile if (C  % mxu_config.n_tile == 0) else _find_smallest_divisor_above(C, mxu_config.n_tile)
        
        buf_sigs = [
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, H, W, C),   self.dtype,     (W_SHARD, C_SHARD),   False, ifm.dtype, 
                preprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 2, 3, 1)]).load_from(node.inputsAt(1), CONTEXT),
            NetworkGraphEntryCompileTarget.BufferSignature(
                (N, OH, OW, C), self.acc_dtype, (OW_SHARD, C_SHARD),  False, ofm_dtype, 
                postprocessings=[NetworkGraphEntryCompileTarget.TensorProcessing.permute(0, 3, 1, 2)]).store_to(node.outputsAt(0), CONTEXT),
        ]
        
        op_kwargs = {
            "window": (FH, FW),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        
        op_method = MCA_OP_AVGPOOL2D
        
        _total_ops = N * C * OH * OW * FH * FW  # 1 for avg, but multiplied by kernel size for the number of additions
        _total_buf_bytes = (N * H * W * C) * (self.dtype.itemsize) + (N * OH * OW * C) * (self.acc_dtype.itemsize) 
        arith_intensity = _total_ops / _total_buf_bytes
        
        return NetworkGraphEntryCompileTarget(
            op_method=op_method,
            buf_sigs=buf_sigs,
            op_kwargs=op_kwargs,
            arith_intensity=arith_intensity,
            max_n_cores=buf_sigs[-1].n_tiles
        )