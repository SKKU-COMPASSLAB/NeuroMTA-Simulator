import math
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *
from neuromta.ip.tenstorrent.runtime_operator import *
from neuromta.ip.common.runtime_operator import *
from neuromta.ip.common.network import *


__all__ = [
    "TT_HOST_CONTEXT",
    "TT_HRT_CONV2D",
    "TT_HRT_LINEAR",
    "TT_HRT_RELU",
    "TT_HRT_MAXPOOL2D",
]


class TT_HOST_CONTEXT(HostContext):
    def __init__(self, device: Device, core_ids: list[int]):
        super().__init__(device=device, core_ids=core_ids)
        
        self.rt_register("Conv2d",      TT_HRT_CONV2D())
        self.rt_register("Linear",      TT_HRT_LINEAR())
        self.rt_register("ReLU",        TT_HRT_RELU())
        self.rt_register("MaxPool2d",   TT_HRT_MAXPOOL2D())


class TT_HRT_CONV2D(HostRuntime):
    def __init__(self):
        super().__init__(
            n_inputs=1,
            n_outputs=1,
            input_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
            output_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
        )
        
        self.l1_wgt_p = None
        self.l1_bias_p = None
    
    def main_process(self, module: torch.nn.Conv2d, ifm_p: Placeholder, ofm_p: Placeholder):
        self.l1_wgt_p = self.host_context.new_placeholder(f"_tmp_wgt")
        if module.bias is not None:
            self.l1_bias_p = self.host_context.new_placeholder(f"_tmp_bias")
        
        wgt = module.weight.detach().clone()
        wgt = wgt.permute(2, 3, 0, 1)
        wgt_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))  # TODO: originally, MAIN
        wgt_buffer = MCA_TensorBuffer(shape=wgt.shape, dtype=wgt.dtype, layout=wgt_layout, device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(self.l1_wgt_p, wgt_buffer)
        HostAction.init_buffer(self.l1_wgt_p, wgt)
        
        if module.bias is not None:
            bias = module.bias.detach().clone()
            bias_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(1, 32))  # TODO: originally, MAIN
            bias_buffer = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=bias_layout, device=self.host_context.device, core_ids=self.host_context.core_ids)

            HostAction.alloc_buffer(self.l1_bias_p, bias_buffer)
            HostAction.init_buffer(self.l1_bias_p, bias)
            
        ifm: torch.Tensor = self.host_context[ifm_p].detach().clone()
        ofm: torch.Tensor = self.host_context[ofm_p].detach().clone()
        
        ifm = ifm.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        ofm = ofm.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        ifm_buffer = MCA_TensorBuffer(shape=ifm.shape, dtype=ifm.dtype, layout=self.input_layouts[0], device=self.host_context.device, core_ids=self.host_context.core_ids)
        ofm_buffer = MCA_TensorBuffer(shape=ofm.shape, dtype=ofm.dtype, layout=self.output_layouts[0], device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(ifm_p, ifm_buffer)
        HostAction.init_buffer(ifm_p, ifm)

        HostAction.alloc_buffer(ofm_p, ofm_buffer)
        HostAction.init_buffer(ofm_p, ofm)
        
        HostAction.dispatch_kernel(
            TT_RT_CONV2D,

            device=self.host_context.device,
            core_grid=self.host_context.core_ids,
            
            buf_ifm  = ifm_p,
            buf_wgt  = self.l1_wgt_p,
            buf_bias = self.l1_bias_p if module.bias is not None else None,
            buf_ofm  = ofm_p,

            stride   = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride),
            padding  = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding),
            dilation = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation),
            
            dtype = ifm.dtype,
            acc_dtype = ofm.dtype,
        )

    def post_process(self, module: torch.nn.Conv2d, ifm_p: Placeholder, ofm_p: Placeholder):
        self.host_context[ofm_p] = self.host_context.get_buffer(ofm_p).restore().permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        HostAction.dealloc_buffer(ifm_p)
        HostAction.dealloc_buffer(ofm_p)
        HostAction.dealloc_buffer(self.l1_wgt_p)
        if module.bias is not None:
            HostAction.dealloc_buffer(self.l1_bias_p)
            

class TT_HRT_LINEAR(HostRuntime):
    def __init__(self):
        super().__init__(
            n_inputs=1,
            n_outputs=1,
            input_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (1, 32))],
            output_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (1, 32))],
        )
        
        self.l1_wgt_p = None
        self.l1_bias_p = None
    
    def main_process(self, module: torch.nn.Linear, ifm_p: Placeholder, ofm_p: Placeholder):
        self.l1_wgt_p = self.host_context.new_placeholder(f"_tmp_wgt")
        if module.bias is not None:
            self.l1_bias_p = self.host_context.new_placeholder(f"_tmp_bias")
        
        wgt = module.weight.detach().clone()
        wgt_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(32, 32))  # TODO: originally, MAIN
        wgt_buffer = MCA_TensorBuffer(shape=wgt.shape, dtype=wgt.dtype, layout=wgt_layout, device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(self.l1_wgt_p, wgt_buffer)
        HostAction.init_buffer(self.l1_wgt_p, wgt)
        
        if module.bias is not None:
            bias = module.bias.detach().clone()
            bias_layout = MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=(1, 32))  # TODO: originally, MAIN
            bias_buffer = MCA_TensorBuffer(shape=bias.shape, dtype=bias.dtype, layout=bias_layout, device=self.host_context.device, core_ids=self.host_context.core_ids)

            HostAction.alloc_buffer(self.l1_bias_p, bias_buffer)
            HostAction.init_buffer(self.l1_bias_p, bias)
        
        ifm: torch.Tensor = self.host_context[ifm_p].detach().clone()
        ofm: torch.Tensor = self.host_context[ofm_p].detach().clone()
        
        batch_size = ifm.shape[0]
        if batch_size < 32:
            ii_page_shape = (batch_size, 32)
        else:
            ii_page_shape = (32, 32)
            
        ifm_buffer = MCA_TensorBuffer(shape=ifm.shape, dtype=ifm.dtype, layout=self.input_layouts[0].overrides(page_shape=ii_page_shape), device=self.host_context.device, core_ids=self.host_context.core_ids)
        ofm_buffer = MCA_TensorBuffer(shape=ofm.shape, dtype=ofm.dtype, layout=self.output_layouts[0].overrides(page_shape=ii_page_shape), device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(ifm_p, ifm_buffer)
        HostAction.init_buffer(ifm_p, ifm)

        HostAction.alloc_buffer(ofm_p, ofm_buffer)
        HostAction.init_buffer(ofm_p, ofm)
        
        HostAction.dispatch_kernel(
            TT_RT_LINEAR,

            device=self.host_context.device,
            core_grid=self.host_context.core_ids,
            
            buf_ifm  = ifm_p,
            buf_wgt  = self.l1_wgt_p,
            buf_bias = self.l1_bias_p if module.bias is not None else None,
            buf_ofm  = ofm_p,
            
            dtype = ifm.dtype,
            acc_dtype = ofm.dtype,
        )

    def post_process(self, module: torch.nn.Conv2d, ifm_p: Placeholder, ofm_p: Placeholder):
        self.host_context[ofm_p] = self.host_context.get_buffer(ofm_p).restore()
        
        HostAction.dealloc_buffer(ifm_p)
        HostAction.dealloc_buffer(ofm_p)
        HostAction.dealloc_buffer(self.l1_wgt_p)
        if module.bias is not None:
            HostAction.dealloc_buffer(self.l1_bias_p)
            
            
class TT_HRT_MAXPOOL2D(HostRuntime):
    def __init__(self):
        super().__init__(
            n_inputs=1,
            n_outputs=1,
            input_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
            output_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
        )
        
    def main_process(self, module: torch.nn.MaxPool2d, ifm_p: Placeholder, ofm_p: Placeholder):
        ifm: torch.Tensor = self.host_context[ifm_p].detach().clone()
        ofm: torch.Tensor = self.host_context[ofm_p].detach().clone()
        
        ifm = ifm.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        ofm = ofm.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        ifm_buffer = MCA_TensorBuffer(shape=ifm.shape, dtype=ifm.dtype, layout=self.input_layouts[0], device=self.host_context.device, core_ids=self.host_context.core_ids)
        ofm_buffer = MCA_TensorBuffer(shape=ofm.shape, dtype=ofm.dtype, layout=self.output_layouts[0], device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(ifm_p, ifm_buffer)
        HostAction.init_buffer(ifm_p, ifm)

        HostAction.alloc_buffer(ofm_p, ofm_buffer)
        HostAction.init_buffer(ofm_p, ofm)

        HostAction.dispatch_kernel(
            TT_RT_MAXPOOL2D,

            device=self.host_context.device,
            core_grid=self.host_context.core_ids,
            
            buf_ifm  = ifm_p,
            buf_ofm  = ofm_p,

            kernel   = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size),
            stride   = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride),
            padding  = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding),
            dilation = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation),
            
            dtype = ifm.dtype,
            acc_dtype = ofm.dtype,
        )

    def post_process(self, module: torch.nn.Conv2d, ifm_p: Placeholder, ofm_p: Placeholder):
        self.host_context[ofm_p] = self.host_context.get_buffer(ofm_p).restore().permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        HostAction.dealloc_buffer(ifm_p)
        HostAction.dealloc_buffer(ofm_p)
        

class TT_HRT_RELU(HostRuntime):
    def __init__(self):
        super().__init__(
            n_inputs=1,
            n_outputs=1,
            input_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
            output_layouts=[MCA_TensorMemoryLayout(MCA_TensorMemoryType.L1, (32, 32))],
        )
        
    def main_process(self, module: torch.nn.MaxPool2d, ifm_p: Placeholder, ofm_p: Placeholder):
        ifm: torch.Tensor = self.host_context[ifm_p].detach().clone()
        ofm: torch.Tensor = self.host_context[ofm_p].detach().clone()
        
        batch_size = math.prod(ifm.shape[:-1])
        if batch_size < 32:
            ii_page_shape = (batch_size, 32)
        else:
            ii_page_shape = (32, 32)
            
        ifm_buffer = MCA_TensorBuffer(shape=ifm.shape, dtype=ifm.dtype, layout=self.input_layouts[0].overrides(page_shape=ii_page_shape), device=self.host_context.device, core_ids=self.host_context.core_ids)
        ofm_buffer = MCA_TensorBuffer(shape=ofm.shape, dtype=ofm.dtype, layout=self.output_layouts[0].overrides(page_shape=ii_page_shape), device=self.host_context.device, core_ids=self.host_context.core_ids)
        
        HostAction.alloc_buffer(ifm_p, ifm_buffer)
        HostAction.init_buffer(ifm_p, ifm)

        HostAction.alloc_buffer(ofm_p, ofm_buffer)
        HostAction.init_buffer(ofm_p, ofm)

        HostAction.dispatch_kernel(
            TT_RT_RELU,

            device=self.host_context.device,
            core_grid=self.host_context.core_ids,
            
            buf_src  = ifm_p,
            buf_dst  = ofm_p,
            
            dtype = ifm.dtype,
            inplace = False,  # TODO: support inplace (need to modify HostContext / SSA representation)
        )

    def post_process(self, module: torch.nn.Conv2d, ifm_p: Placeholder, ofm_p: Placeholder):
        self.host_context[ofm_p] = self.host_context.get_buffer(ofm_p).restore()
        
        HostAction.dealloc_buffer(ifm_p)
        HostAction.dealloc_buffer(ofm_p)
