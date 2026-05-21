from typing import Any
import torch

from neuromta.framework import *
from neuromta.component.implementation.hardware import *
from neuromta.component.implementation.network import *
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *
from neuromta.component.implementation.tensor_buffer import *
# from neuromta.system.software.common.operator import *
import neuromta.system.software.common.operator as mca_op_lib


__all__ = [
    "MCA_NetworkRecipe",
]


class MCA_NetworkRecipe(NetworkRecipe):
    def __init__(
        self,
        device: MCA_DeviceBase,
        core_groups: list[MCA_CoreGroup],
        main_space_size_per_channel: int=parse_mem_cap_str("4GB"),
        data_space_size_per_core: int=parse_mem_cap_str("1MB"),
        spad_space_size_per_core: int=parse_mem_cap_str("512KB"),
        broadcast_optimize_queue_depth: int=8,
        broadcast_optimize_max_ref_cnt: int=4,
        context_buffer_slot_num: int=16,
        ld_ex_buffer_slot_num: int=16,
        ex_st_buffer_slot_num: int=8,
        concurrent_load_num: int=1,
        temporal_reuse_type: MCA_OperatorGraphCompiler.CompileRecipe.ReuseType=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.ALL,
        spatial_reuse_type: MCA_OperatorGraphCompiler.CompileRecipe.ReuseType=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_MAIN,
        greedy_temporal_reuse: bool=True,
        
        dtype: torch.dtype=torch.float16,
        acc_dtype: torch.dtype=torch.float16,
    ):
        super().__init__(
            device=device,
            core_groups=core_groups,
            main_space_size_per_channel=main_space_size_per_channel,
            data_space_size_per_core=data_space_size_per_core,
            spad_space_size_per_core=spad_space_size_per_core,
            broadcast_optimize_queue_depth=broadcast_optimize_queue_depth,
            broadcast_optimize_max_ref_cnt=broadcast_optimize_max_ref_cnt,
            context_buffer_slot_num=context_buffer_slot_num,
            ld_ex_buffer_slot_num=ld_ex_buffer_slot_num,
            ex_st_buffer_slot_num=ex_st_buffer_slot_num,
            concurrent_load_num=concurrent_load_num,
            temporal_reuse_type=temporal_reuse_type,
            spatial_reuse_type=spatial_reuse_type,
            greedy_temporal_reuse=greedy_temporal_reuse,
        )
        
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        
    @NetworkRecipe.recipe("aten::linear")
    def _linear(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        wgt: torch.Tensor  = i_args[1]
        bias: torch.Tensor = i_args[2] if len(i_args) > 2 else None
        ofm: torch.Tensor  = o_args[0]
        
        M, K = ifm.shape
        N, K = wgt.shape
        
        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (M, K), self.dtype).set_orig_dtype(ifm.dtype)
        compiled_entry.add_param_buffer_context("wgt", compiled_entry.node.inputsAt(1).debugName(), (N, K), self.dtype).set_orig_dtype(wgt.dtype)
        if bias is not None:
            compiled_entry.add_param_buffer_context("bias", compiled_entry.node.inputsAt(2).debugName(), (1, N), self.dtype).set_orig_dtype(bias.dtype)
        else:
            compiled_entry.add_constant_context("bias", None)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (M, N), self.acc_dtype).set_orig_dtype(ofm.dtype)
        
        compiled_entry.set_op_method(mca_op_lib.MCA_OP_LINEAR)
        
        return compiled_entry

    @NetworkRecipe.recipe("aten::_convolution")
    def _conv2d(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        wgt: torch.Tensor  = i_args[1]
        bias: torch.Tensor = i_args[2] if len(i_args) > 2 else None
        ofm: torch.Tensor  = o_args[0]
        stride = i_args[3]
        padding = i_args[4]
        dilation = i_args[5]
        groups = i_args[8]
        
        N, C, H, W = ifm.shape
        K, C, FH, FW = wgt.shape
        OH = (H + 2 * padding[0] - dilation[0] * (FH - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (FW - 1) - 1) // stride[1] + 1
        
        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (N, H, W, C), self.dtype).permute(0, 2, 3, 1).set_orig_dtype(ifm.dtype)
        compiled_entry.add_param_buffer_context("wgt", compiled_entry.node.inputsAt(1).debugName(), (FH, FW, K, C), self.dtype).permute(2, 3, 0, 1).set_orig_dtype(wgt.dtype)
        if bias is not None:
            compiled_entry.add_param_buffer_context("bias", compiled_entry.node.inputsAt(2).debugName(), (1, K), self.dtype).set_orig_dtype(bias.dtype)
        else:
            compiled_entry.add_constant_context("bias", None)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (N, OH, OW, K), self.acc_dtype).permute(0, 2, 3, 1).set_orig_dtype(ofm.dtype)
        
        compiled_entry.add_constant_context("stride", stride)
        compiled_entry.add_constant_context("padding", padding)
        compiled_entry.add_constant_context("dilation", dilation)
        compiled_entry.add_constant_context("groups", groups)
        
        compiled_entry.set_op_method(mca_op_lib.MCA_OP_CONV2D)
        
        return compiled_entry
    
    @NetworkRecipe.recipe("aten::relu")
    def _relu(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        ofm: torch.Tensor  = o_args[0]
        
        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), ifm.shape, self.dtype).set_orig_dtype(ifm.dtype)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), ofm.shape, self.acc_dtype).set_orig_dtype(ofm.dtype)
        
        compiled_entry.set_op_method(mca_op_lib.MCA_OP_RELU)
        
        return compiled_entry
    
    @NetworkRecipe.recipe("aten::max_pool2d")
    def _max_pool2d(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        ofm: torch.Tensor  = o_args[0]
        
        kernel_size = i_args[1]
        stride = i_args[2]
        padding = i_args[3]
        dilation = i_args[4]
        
        N, C, H, W = ifm.shape
        OH = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (N, H, W, C), self.dtype).permute(0, 2, 3, 1).set_orig_dtype(ifm.dtype)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (N, OH, OW, C), self.acc_dtype).permute(0, 2, 3, 1).set_orig_dtype(ofm.dtype)

        compiled_entry.add_constant_context("window", kernel_size)
        compiled_entry.add_constant_context("stride", stride)
        compiled_entry.add_constant_context("padding", padding)
        compiled_entry.add_constant_context("dilation", dilation)
        
        compiled_entry.set_op_method(mca_op_lib.MCA_OP_MAXPOOL2D)

        return compiled_entry
    
    @NetworkRecipe.recipe("aten::avg_pool2d")
    def _avg_pool2d(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        ofm: torch.Tensor  = o_args[0]
        
        kernel_size = i_args[1]
        stride = i_args[2]
        padding = i_args[3]
        dilation = i_args[4]
        
        N, C, H, W = ifm.shape
        OH = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        OW = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (N, H, W, C), self.dtype).permute(0, 2, 3, 1).set_orig_dtype(ifm.dtype)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (N, OH, OW, C), self.acc_dtype).permute(0, 2, 3, 1).set_orig_dtype(ofm.dtype)

        compiled_entry.add_constant_context("window", kernel_size)
        compiled_entry.add_constant_context("stride", stride)
        compiled_entry.add_constant_context("padding", padding)
        compiled_entry.add_constant_context("dilation", dilation)

        compiled_entry.set_op_method(mca_op_lib.MCA_OP_AVGPOOL2D)

        return compiled_entry
    
    @NetworkRecipe.recipe("aten::adaptive_max_pool2d")
    def _adaptive_max_pool2d(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        ofm: torch.Tensor  = o_args[0]
        
        output_size = i_args[1]
        
        N, C, H, W = ifm.shape
        OH, OW = output_size

        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (N, H, W, C), self.dtype).permute(0, 2, 3, 1).set_orig_dtype(ifm.dtype)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (N, OH, OW, C), self.acc_dtype).permute(0, 2, 3, 1).set_orig_dtype(ofm.dtype)

        compiled_entry.set_op_method(mca_op_lib.MCA_OP_ADAPTIVE_MAXPOOL2D)

        return compiled_entry
    
    @NetworkRecipe.recipe("aten::adaptive_avg_pool2d")
    def _adaptive_avg_pool2d(self, compiled_entry: CompiledGraphEntry, i_args: list[Any], o_args: list[Any]) -> CompiledGraphEntry:
        ifm: torch.Tensor  = i_args[0]
        ofm: torch.Tensor  = o_args[0]
        
        output_size = i_args[1]
        
        N, C, H, W = ifm.shape
        OH, OW = output_size

        compiled_entry.add_input_buffer_context("ifm", compiled_entry.node.inputsAt(0).debugName(), (N, H, W, C), self.dtype).permute(0, 2, 3, 1).set_orig_dtype(ifm.dtype)
        compiled_entry.add_output_buffer_context("ofm", compiled_entry.node.outputsAt(0).debugName(), (N, OH, OW, C), self.acc_dtype).permute(0, 2, 3, 1).set_orig_dtype(ofm.dtype)

        compiled_entry.set_op_method(mca_op_lib.MCA_OP_ADAPTIVE_AVGPOOL2D)

        return compiled_entry
