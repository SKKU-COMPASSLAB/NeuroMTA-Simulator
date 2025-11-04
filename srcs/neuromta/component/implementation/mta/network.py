import math
import torch

from neuromta.framework import *
from neuromta.component.implementation.common import *
from neuromta.component.implementation.mca.runtime_operator import *
from neuromta.component.implementation.mta.runtime_operator import *


__all__ = [
    "MTA_NETWORK_CONTEXT",
    "MTA_LINEAR_RECIPE",
    "MTA_CONV2D_RECIPE",
    "MTA_MAXPOOL2D_RECIPE",
    "MTA_RELU_RECIPE",
]


class MTA_NETWORK_CONTEXT(NetworkGraphContext):
    def __init__(self, device: MTA_DeviceBase, core_grid: MTA_CoreGrid):
        super().__init__(device, core_grid)
        
        self.device:   MTA_DeviceBase  # override with more specific type
        self.core_ids: MTA_CoreGrid    # override with more specific type
        
        self.add_module_compilation_recipe(torch.nn.Conv2d,    MTA_CONV2D_RECIPE)
        self.add_module_compilation_recipe(torch.nn.Linear,    MTA_LINEAR_RECIPE)
        self.add_module_compilation_recipe(torch.nn.MaxPool2d, MTA_MAXPOOL2D_RECIPE)
        self.add_module_compilation_recipe(torch.nn.ReLU,      MTA_RELU_RECIPE)
        
        
class MTA_LINEAR_RECIPE(CompilationRecipe):
    def __init__(self, node: torch.Node):
        super().__init__(node=node)
        
        #########################################################################
        # STEP 1: Read context and extract variables
        #########################################################################
        
        context: MTA_NETWORK_CONTEXT = get_global_network_context()
        assert isinstance(context, MTA_NETWORK_CONTEXT), f"{type(self).__name__} requires an active MTA_NETWORK_CONTEXT."

        try:
            module:   torch.nn.Linear   = context.variables[self.node.inputsAt(0)]
            ifm_t:    torch.Tensor      = context.variables[self.node.inputsAt(1)]
            wgt_t:    torch.Tensor      = module.weight
            bias_t:   torch.Tensor      = module.bias if module.bias is not None else torch.zeros(module.out_features)
            ofm_t:    torch.Tensor      = context.variables[self.node.outputsAt(0)]
        except Exception as e:
            logger.error(f"Failed to extract variables because of the faultly implementation of the trace mode. Please check if the prerun with dummy inputs is correctly done during compilation.")
            raise Exception(f"failed to extract variables for node: {self.node}") from e
        
        #########################################################################
        # STEP 2: Define layouts, buffers, and tiling strategy
        #########################################################################
        
        self.dims.add("M", "N", "K")
        
        self.layouts.add("ifm",  dims=self.dims.get("M", "K"), initial_tensor=ifm_t, is_input=True )
        self.layouts.add("wgt",  dims=self.dims.get("N", "K"), initial_tensor=wgt_t)
        self.layouts.add("bias", dims=self.dims.get("N"),      initial_tensor=bias_t)
        self.layouts.add("ofm",  dims=self.dims.get("M", "N"), shape=ofm_t.shape, dtype=ofm_t.dtype, is_output=True)
        
        device    = context.device
        core_ids  = context.core_ids  # TODO: should we partition core_ids for multiple operators?
        npu_cores = [device.get_npu_core(core_id=core_id) for core_id in core_ids]
        
        total_onc_mem = sum(c.mem_handle.size for c in npu_cores)
        empty_onc_mem = sum(c.mem_handle.empty_space() for c in npu_cores)
        max_mem_usage = math.floor((total_onc_mem * 0.94) / 2)  # use up to 94% of total ONC memory, divide by 2 for double buffering

        flag = self.layouts.set_tiling_factor_with_mem_usage(max_mem_usage=max_mem_usage)
        if not flag:
            raise Exception(f"Failed to set tiling factor within memory budget for node: {self.node}")
        
        self._layout_names    = list(self.layouts.keys())
        self._main_buf_fmt    = "main_{layout_name}_{cursor}"
        self._l1_buf_fmt      = "l1_{layout_name}_{pingpong}"
        self._l1_buf_pingpong = 2  # double buffering  
        
        for layout_name in self._layout_names:
            if layout_name == "ifm" or layout_name == "ofm":
                page_shape = (min(self.layouts.input_layouts[0].tensor_shape[-2], 32), 32)
                # page_shape = (32, 32)
            elif layout_name == "bias":
                page_shape = (32,)
            else:
                page_shape = (32, 32)
            
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=page_shape),
                    device=device,
                    # no core ids for main memory buffer
                )
            
            for pp in range(self._l1_buf_pingpong):
                if pp > 0 and layout_name == "ofm":
                    continue  # only need one L1 buffer for ofm (TODO: no double buffering for ofm)
                
                full_buf_name = self._l1_buf_fmt.format(layout_name=layout_name, pingpong=pp)
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=page_shape),
                    device=device,
                    core_ids=core_ids,
                )
                
        for layout_name in self._layout_names:
            for pp in range(self._l1_buf_pingpong):
                if pp > 0 and layout_name == "ofm":
                    continue  # only need one L1 buffer for ofm (TODO: no double buffering for ofm)
                
                full_buf_name = self._l1_buf_fmt.format(layout_name=layout_name, pingpong=pp)
                
        #########################################################################
        # STEP 3: Compilation recipe
        #########################################################################
        
        step = self.new_step()
        
        step.layout_update("ifm",  self.node.inputsAt(1))
        step.layout_update("wgt",  wgt_t)
        step.layout_update("bias", bias_t)
        
        for buffer_name in self.buffers.keys():
            step.buffer_alloc(buffer_name=buffer_name)
        
        for layout_name in self._layout_names:
            for cursor in self.layouts[layout_name].get_cursors():
                step.buffer_load(
                    buffer_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string()),
                    layout_name = layout_name,
                    cursor      = cursor,
                )
        
        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            for i, cursor in enumerate(self.dims.get_cursors(fixed=ofm_cursor)):
                pingpong = i % self._l1_buf_pingpong
                
                for layout_name, layout in self.layouts.items():
                    if layout_name == "ofm":
                        continue  # no need to load ofm buffer
                    
                    step.op("DMA", MCA_RT_DMA_LOAD,
                        device = device,
                        src_buf = self.buffers[self._main_buf_fmt.format(layout_name=layout_name,  cursor=cursor.filter(layout).to_string())],
                        dst_buf = self.buffers[self._l1_buf_fmt.format(layout_name=layout_name,  pingpong=pingpong)],
                    )
                
                step = self.new_step()
                
                step.op("COMPUTE", MTA_RT_LINEAR,        
                    device = device,
                    core_grid = core_ids,

                    buf_ifm  = self.buffers[self._l1_buf_fmt.format(layout_name="ifm",  pingpong=pingpong)],
                    buf_wgt  = self.buffers[self._l1_buf_fmt.format(layout_name="wgt",  pingpong=pingpong)],
                    buf_bias = self.buffers[self._l1_buf_fmt.format(layout_name="bias", pingpong=pingpong)] if i == 0 else None,  # only load bias for the first accumulation
                    buf_ofm  = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                    
                    accumulate_psum = (i != 0),
                )
            
            step = self.new_step()

            step.op("DMA", MCA_RT_DMA_STORE,
                device = device,
                src_buf = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                dst_buf = self.buffers[self._main_buf_fmt.format(layout_name="ofm",  cursor=ofm_cursor.to_string())],
            )

        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            step.buffer_store(
                buffer_name = self._main_buf_fmt.format(layout_name="ofm", cursor=ofm_cursor.to_string()),
                layout_name = "ofm",
                cursor      = ofm_cursor,
            )
        
        for buffer_name in self.buffers.keys():
            step.buffer_dealloc(buffer_name=buffer_name)
            
        step.layout_restore("ofm", self.node.outputsAt(0))
        

class MTA_CONV2D_RECIPE(CompilationRecipe):
    def __init__(self, node: torch.Node):
        super().__init__(node=node)
        
        #########################################################################
        # STEP 1: Read context and extract variables
        #########################################################################
        
        context: MTA_NETWORK_CONTEXT = get_global_network_context()
        assert isinstance(context, MTA_NETWORK_CONTEXT), f"{type(self).__name__} requires an active MTA_NETWORK_CONTEXT."

        try:
            module:   torch.nn.Conv2d   = context.variables[self.node.inputsAt(0)]
            ifm_t:    torch.Tensor      = context.variables[self.node.inputsAt(1)]
            wgt_t:    torch.Tensor      = module.weight
            bias_t:   torch.Tensor      = module.bias if module.bias is not None else torch.zeros(module.out_channels)
            ofm_t:    torch.Tensor      = context.variables[self.node.outputsAt(0)]
        except Exception as e:
            logger.error(f"Failed to extract variables because of the faultly implementation of the trace mode. Please check if the prerun with dummy inputs is correctly done during compilation.")
            raise Exception(f"failed to extract variables for node: {self.node}") from e
        
        self.stride   = (module.stride, module.stride) if isinstance(module.stride, int) else module.stride
        self.padding  = (module.padding, module.padding) if isinstance(module.padding, int) else module.padding
        self.dilation = (module.dilation, module.dilation) if isinstance(module.dilation, int) else module.dilation
        
        #########################################################################
        # STEP 2: Define layouts, buffers, and tiling strategy
        #########################################################################
        
        self.dims.add("N", "K", "C", "H", "W", "OH", "OW", "FH", "FW")
        self.dims.disable_tiling("H", "OH", "FH")  # TODO: Disable tiling on height-related dimensions (image height cannot be simply tiled due to halo regions)
        self.dims.disable_tiling("W", "OW", "FW")  # TODO: Disable tiling on width-related dimensions (image width cannot be simply tiled due to halo regions)
        
        self.layouts.add("ifm",  dims=self.dims.get("N",  "H",  "W",  "C"), preprocessing=[TensorPermute([0, 2, 3, 1])], initial_tensor=ifm_t, is_input=True )
        self.layouts.add("wgt",  dims=self.dims.get("FH", "FW", "K",  "C"), preprocessing=[TensorPermute([2, 3, 0, 1])], initial_tensor=wgt_t)
        self.layouts.add("bias", dims=self.dims.get("K"),                                                                initial_tensor=bias_t)
        self.layouts.add("ofm",  dims=self.dims.get("N",  "OH", "OW", "K"), preprocessing=[TensorPermute([0, 2, 3, 1])], shape=ofm_t.shape, dtype=ofm_t.dtype, is_output=True)
        
        device    = context.device
        core_ids  = context.core_ids  # TODO: should we partition core_ids for multiple operators?
        npu_cores = [device.get_npu_core(core_id=core_id) for core_id in core_ids]
        
        total_onc_mem = sum(c.mem_handle.size for c in npu_cores)
        empty_onc_mem = sum(c.mem_handle.empty_space() for c in npu_cores)
        max_mem_usage = math.floor((total_onc_mem * 0.94) / 2)  # use up to 94% of total ONC memory, divide by 2 for double buffering

        flag = self.layouts.set_tiling_factor_with_mem_usage(max_mem_usage=max_mem_usage)
        if not flag:
            raise Exception(f"Failed to set tiling factor within memory budget for node: {self.node}")

        self._layout_names    = list(self.layouts.keys())
        self._main_buf_fmt    = "main_{layout_name}_{cursor}"
        self._l1_buf_fmt      = "l1_{layout_name}_{pingpong}"
        self._l1_buf_pingpong = 2  # double buffering  
        
        for layout_name in self._layout_names:
            if layout_name == "bias":
                page_shape = (32,)
            else:
                page_shape = (32, 32)
            
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=page_shape),
                    device=device,
                    # no core ids for main memory buffer
                )
            
            for pp in range(self._l1_buf_pingpong):
                if pp > 0 and layout_name == "ofm":
                    continue  # only need one L1 buffer for ofm (TODO: no double buffering for ofm)
                
                full_buf_name = self._l1_buf_fmt.format(layout_name=layout_name, pingpong=pp)
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=page_shape),
                    device=device,
                    core_ids=core_ids,
                )
        
        #########################################################################
        # STEP 3: Compilation recipe
        #########################################################################
        
        step = self.new_step()
        
        step.layout_update("ifm",  self.node.inputsAt(1))
        step.layout_update("wgt",  wgt_t)
        step.layout_update("bias", bias_t)
        
        for buffer_name in self.buffers.keys():
            step.buffer_alloc(buffer_name=buffer_name)
        
        for layout_name in self._layout_names:
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                step.buffer_load(
                    buffer_name = full_buf_name,
                    layout_name = layout_name,
                    cursor      = cursor,
                )
        
        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            for i, cursor in enumerate(self.dims.get_cursors(fixed=ofm_cursor)):
                pingpong = i % self._l1_buf_pingpong
                
                for layout_name, layout in self.layouts.items():
                    if layout_name == "ofm":
                        continue  # no need to load ofm buffer
                    step.op("DMA", MCA_RT_DMA_LOAD,
                        device = device,
                        src_buf = self.buffers[self._main_buf_fmt.format(layout_name=layout_name,  cursor=cursor.filter(layout).to_string())],
                        dst_buf = self.buffers[self._l1_buf_fmt.format(layout_name=layout_name,  pingpong=pingpong)],
                    )
                
                step = self.new_step()
                
                step.op("COMPUTE", MTA_RT_CONV2D,        
                    device = device,
                    core_grid = core_ids,

                    buf_ifm  = self.buffers[self._l1_buf_fmt.format(layout_name="ifm",  pingpong=pingpong)],
                    buf_wgt  = self.buffers[self._l1_buf_fmt.format(layout_name="wgt",  pingpong=pingpong)],
                    buf_bias = self.buffers[self._l1_buf_fmt.format(layout_name="bias", pingpong=pingpong)] if i == 0 else None,  # only load bias for the first accumulation
                    buf_ofm  = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],

                    stride   = self.stride,
                    padding  = self.padding,
                    dilation = self.dilation,
                    
                    accumulate_psum = (i != 0),
                )
            
            step = self.new_step()

            step.op("DMA", MCA_RT_DMA_STORE,
                device = device,
                src_buf = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                dst_buf = self.buffers[self._main_buf_fmt.format(layout_name="ofm",  cursor=ofm_cursor.to_string())],
            )

        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            step.buffer_store(
                buffer_name = self._main_buf_fmt.format(layout_name="ofm", cursor=ofm_cursor.to_string()),
                layout_name = "ofm",
                cursor      = ofm_cursor,
            )
        
        for buffer_name in self.buffers.keys():
            step.buffer_dealloc(buffer_name=buffer_name)
            
        step.layout_restore("ofm", self.node.outputsAt(0))
        
    
class MTA_MAXPOOL2D_RECIPE(CompilationRecipe):
    def __init__(self, node: torch.Node):
        super().__init__(node=node)
        
        #########################################################################
        # STEP 1: Read context and extract variables
        #########################################################################
        
        context: MTA_NETWORK_CONTEXT = get_global_network_context()
        assert isinstance(context, MTA_NETWORK_CONTEXT), f"{type(self).__name__} requires an active MTA_NETWORK_CONTEXT."

        try:
            module:   torch.nn.MaxPool2d = context.variables[self.node.inputsAt(0)]
            ifm_t:    torch.Tensor       = context.variables[self.node.inputsAt(1)]
            ofm_t:    torch.Tensor       = context.variables[self.node.outputsAt(0)]
        except Exception as e:
            logger.error(f"Failed to extract variables because of the faultly implementation of the trace mode. Please check if the prerun with dummy inputs is correctly done during compilation.")
            raise Exception(f"failed to extract variables for node: {self.node}") from e

        self.kernel_size = (module.kernel_size, module.kernel_size) if isinstance(module.kernel_size, int) else module.kernel_size
        self.stride   = (module.stride, module.stride) if isinstance(module.stride, int) else module.stride
        self.padding  = (module.padding, module.padding) if isinstance(module.padding, int) else module.padding
        self.dilation = (module.dilation, module.dilation) if isinstance(module.dilation, int) else module.dilation
        
        #########################################################################
        # STEP 2: Define layouts, buffers, and tiling strategy
        #########################################################################
        
        self.dims.add("N", "K", "C", "H", "W", "OH", "OW")
        self.dims.disable_tiling("H", "OH")  # TODO: Disable tiling on height-related dimensions (image height cannot be simply tiled due to halo regions)
        self.dims.disable_tiling("W", "OW")  # TODO: Disable tiling on width-related dimensions (image width cannot be simply tiled due to halo regions)
        
        self.layouts.add("ifm",  dims=self.dims.get("N",  "H",  "W",  "C"), preprocessing=[TensorPermute([0, 2, 3, 1])], initial_tensor=ifm_t, is_input=True )
        self.layouts.add("ofm",  dims=self.dims.get("N",  "OH", "OW", "K"), preprocessing=[TensorPermute([0, 2, 3, 1])], shape=ofm_t.shape, dtype=ofm_t.dtype, is_output=True)
        
        device    = context.device
        core_ids  = context.core_ids  # TODO: should we partition core_ids for multiple operators?
        npu_cores = [device.get_npu_core(core_id=core_id) for core_id in core_ids]
        
        total_onc_mem = sum(c.mem_handle.size for c in npu_cores)
        empty_onc_mem = sum(c.mem_handle.empty_space() for c in npu_cores)
        max_mem_usage = math.floor((total_onc_mem * 0.94) / 2)  # use up to 94% of total ONC memory, divide by 2 for double buffering

        flag = self.layouts.set_tiling_factor_with_mem_usage(max_mem_usage=max_mem_usage)
        if not flag:
            raise Exception(f"Failed to set tiling factor within memory budget for node: {self.node}")

        self._layout_names    = list(self.layouts.keys())
        self._main_buf_fmt    = "main_{layout_name}_{cursor}"
        self._l1_buf_fmt      = "l1_{layout_name}_{pingpong}"
        self._l1_buf_pingpong = 2  # double buffering  
        
        for layout_name in self._layout_names:
            page_shape = (32, 32)
            
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=page_shape),
                    device=device,
                    # no core ids for main memory buffer
                )
            
            for pp in range(self._l1_buf_pingpong):
                if pp > 0 and layout_name == "ofm":
                    continue  # only need one L1 buffer for ofm (TODO: no double buffering for ofm)
                
                full_buf_name = self._l1_buf_fmt.format(layout_name=layout_name, pingpong=pp)
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=page_shape),
                    device=device,
                    core_ids=core_ids,
                )
        
        #########################################################################
        # STEP 3: Compilation recipe
        #########################################################################
        
        step = self.new_step()
        
        step.layout_update("ifm",  self.node.inputsAt(1))
        
        for buffer_name in self.buffers.keys():
            step.buffer_alloc(buffer_name=buffer_name)
        
        for layout_name in self._layout_names:
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                step.buffer_load(
                    buffer_name = full_buf_name,
                    layout_name = layout_name,
                    cursor      = cursor,
                )
        
        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            for i, cursor in enumerate(self.dims.get_cursors(fixed=ofm_cursor)):
                pingpong = i % self._l1_buf_pingpong
                
                for layout_name, layout in self.layouts.items():
                    if layout_name == "ofm":
                        continue  # no need to load ofm buffer
                    step.op("DMA", MCA_RT_DMA_LOAD,
                        device = device,
                        src_buf = self.buffers[self._main_buf_fmt.format(layout_name=layout_name,  cursor=cursor.filter(layout).to_string())],
                        dst_buf = self.buffers[self._l1_buf_fmt.format(layout_name=layout_name,  pingpong=pingpong)],
                    )
                
                step = self.new_step()
                
                step.op("COMPUTE", MTA_RT_MAXPOOL2D,        
                    device = device,
                    core_grid = core_ids,

                    buf_ifm  = self.buffers[self._l1_buf_fmt.format(layout_name="ifm",  pingpong=pingpong)],
                    buf_ofm  = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],

                    kernel   = self.kernel_size,
                    stride   = self.stride,
                    padding  = self.padding,
                    dilation = self.dilation,
                    
                    accumulate_psum = (i != 0),
                )
            
            step = self.new_step()

            step.op("DMA", MCA_RT_DMA_STORE,
                device = device,
                src_buf = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                dst_buf = self.buffers[self._main_buf_fmt.format(layout_name="ofm",  cursor=ofm_cursor.to_string())],
            )

        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            step.buffer_store(
                buffer_name = self._main_buf_fmt.format(layout_name="ofm", cursor=ofm_cursor.to_string()),
                layout_name = "ofm",
                cursor      = ofm_cursor,
            )
        
        for buffer_name in self.buffers.keys():
            step.buffer_dealloc(buffer_name=buffer_name)
            
        step.layout_restore("ofm", self.node.outputsAt(0))
        
        
class MTA_RELU_RECIPE(CompilationRecipe):
    def __init__(self, node: torch.Node):
        super().__init__(node=node)
        
        #########################################################################
        # STEP 1: Read context and extract variables
        #########################################################################
        
        context: MTA_NETWORK_CONTEXT = get_global_network_context()
        assert isinstance(context, MTA_NETWORK_CONTEXT), f"{type(self).__name__} requires an active MTA_NETWORK_CONTEXT."

        try:
            ifm_t:    torch.Tensor    = context.variables[self.node.inputsAt(1)]
            ofm_t:    torch.Tensor    = context.variables[self.node.outputsAt(0)]
        except Exception as e:
            logger.error(f"Failed to extract variables because of the faultly implementation of the trace mode. Please check if the prerun with dummy inputs is correctly done during compilation.")
            raise Exception(f"failed to extract variables for node: {self.node}") from e
        
        #########################################################################
        # STEP 2: Define layouts, buffers, and tiling strategy
        #########################################################################
        
        _dim_names = list(map(lambda x: f"dim{x}", range(len(ifm_t.shape))))
        self.dims.add(*_dim_names)

        self.layouts.add("ifm",  dims=self.dims.get(*_dim_names), initial_tensor=ifm_t, is_input=True )
        self.layouts.add("ofm",  dims=self.dims.get(*_dim_names), shape=ofm_t.shape, dtype=ofm_t.dtype, is_output=True)

        device    = context.device
        core_ids  = context.core_ids  # TODO: should we partition core_ids for multiple operators?
        npu_cores = [device.get_npu_core(core_id=core_id) for core_id in core_ids]
        
        total_onc_mem = sum(c.mem_handle.size for c in npu_cores)
        empty_onc_mem = sum(c.mem_handle.empty_space() for c in npu_cores)
        max_mem_usage = math.floor((total_onc_mem * 0.94) / 2)  # use up to 94% of total ONC memory, divide by 2 for double buffering

        flag = self.layouts.set_tiling_factor_with_mem_usage(max_mem_usage=max_mem_usage)
        if not flag:
            raise Exception(f"Failed to set tiling factor within memory budget for node: {self.node}")

        self._layout_names    = list(self.layouts.keys())
        self._main_buf_fmt    = "main_{layout_name}_{cursor}"
        self._l1_buf_fmt      = "l1_{layout_name}_{pingpong}"
        self._l1_buf_pingpong = 2  # double buffering  
        
        for layout_name in self._layout_names:
            page_shape = (32, 32)
            
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.MAIN, page_shape=page_shape),
                    device=device,
                    # no core ids for main memory buffer
                )
            
            for pp in range(self._l1_buf_pingpong):
                if pp > 0 and layout_name == "ofm":
                    continue  # only need one L1 buffer for ofm (TODO: no double buffering for ofm)
                
                full_buf_name = self._l1_buf_fmt.format(layout_name=layout_name, pingpong=pp)
                
                self.buffers[full_buf_name] = MCA_TensorBuffer(
                    shape=self.layouts[layout_name].tile_shape,
                    dtype=self.layouts[layout_name].tensor_dtype,
                    layout=MCA_TensorMemoryLayout(mem_type=MCA_TensorMemoryType.L1, page_shape=page_shape),
                    device=device,
                    core_ids=core_ids,
                )
        
        #########################################################################
        # STEP 3: Compilation recipe
        #########################################################################
        
        step = self.new_step()
        
        step.layout_update("ifm",  self.node.inputsAt(1))
        
        for buffer_name in self.buffers.keys():
            step.buffer_alloc(buffer_name=buffer_name)
        
        for layout_name in self._layout_names:
            for cursor in self.layouts[layout_name].get_cursors():
                full_buf_name = self._main_buf_fmt.format(layout_name=layout_name, cursor=cursor.to_string())
                
                step.buffer_load(
                    buffer_name = full_buf_name,
                    layout_name = layout_name,
                    cursor      = cursor,
                )
        
        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            for i, cursor in enumerate(self.dims.get_cursors(fixed=ofm_cursor)):
                pingpong = i % self._l1_buf_pingpong
                
                for layout_name, layout in self.layouts.items():
                    if layout_name == "ofm":
                        continue  # no need to load ofm buffer
                    step.op("DMA", MCA_RT_DMA_LOAD,
                        device = device,
                        src_buf = self.buffers[self._main_buf_fmt.format(layout_name=layout_name,  cursor=cursor.filter(layout).to_string())],
                        dst_buf = self.buffers[self._l1_buf_fmt.format(layout_name=layout_name,  pingpong=pingpong)],
                    )
                
                step = self.new_step()
                
                step.op("COMPUTE", MTA_RT_RELU,        
                    device = device,
                    core_grid = core_ids,

                    buf_src  = self.buffers[self._l1_buf_fmt.format(layout_name="ifm",  pingpong=pingpong)],
                    buf_dst  = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                    
                    inplace = False,
                )
            
            step = self.new_step()

            step.op("DMA", MCA_RT_DMA_STORE,
                device = device,
                src_buf = self.buffers[self._l1_buf_fmt.format(layout_name="ofm",  pingpong=0)],
                dst_buf = self.buffers[self._main_buf_fmt.format(layout_name="ofm",  cursor=ofm_cursor.to_string())],
            )

        step = self.new_step()
        
        for ofm_cursor in self.layouts["ofm"].get_cursors():
            step.buffer_store(
                buffer_name = self._main_buf_fmt.format(layout_name="ofm", cursor=ofm_cursor.to_string()),
                layout_name = "ofm",
                cursor      = ofm_cursor,
            )
        
        for buffer_name in self.buffers.keys():
            step.buffer_dealloc(buffer_name=buffer_name)
            
        step.layout_restore("ofm", self.node.outputsAt(0))
