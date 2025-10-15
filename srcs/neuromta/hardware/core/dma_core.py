import torch
from neuromta.framework import *

from neuromta.hardware.context.mem_context import MemContext
from neuromta.hardware.context.cmap_context import CmapContext
from neuromta.hardware.context.icnt_context import IcntContext

__all__ = [
    "DMACore",
]


class DMACore(Core):
    def __init__(
        self, 
        core_id: int,
        mem_context: MemContext, 
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=core_id, 
            cycle_model=DMACoreCycleModel(core=self)
        )
        
        self.mem_context = mem_context
        self.cmap_context = cmap_context
        
    # @core_command_method
    # def local_mem_read_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer, offset: int=0, cont_offset: int=0, size: int=None, shape: tuple[int, ...]=(-1,), dtype: torch.dtype=torch.uint8):
    #     if isinstance(ptr, BufferPointer):
    #         handle = ptr.resolve(is_read=True)

    #     if size is None:
    #         size = handle.size - offset
        
    #     raw_data: torch.Tensor = self.mem_handle.get_content(handle, shape=(-1,), dtype=torch.uint8)[offset:offset+size]
    #     rd_data:  torch.Tensor = raw_data.flatten().view(dtype)
    #     rd_data[cont_offset:cont_offset+size] = raw_data.view(dtype=dtype).reshape(shape)
        
    #     container.data = rd_data
        
    # @core_command_method
    # def local_mem_write_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer, offset: int=0, cont_offset: int=0, size: int=None):
    #     if isinstance(ptr, BufferPointer):
    #         rd_handle = ptr.resolve(is_read=True)
    #         wr_handle = ptr.resolve(is_read=False)
    #     else:
    #         rd_handle = ptr
    #         wr_handle = ptr
        
    #     if size is None:
    #         size = rd_handle.size - cont_offset
            
    #     raw_data: torch.Tensor = self.mem_handle.get_content(rd_handle, shape=(-1,), dtype=torch.uint8)
    #     wr_data:  torch.Tensor = container.data.reshape(-1).view(torch.uint8)
        
    #     if cont_offset != 0:
    #         wr_data = wr_data[cont_offset:cont_offset+size]
    
    #     raw_data[offset:offset+wr_data.numel()] = wr_data
    #     self.mem_handle.set_content(wr_handle, raw_data)
    
    @jit_prototype
    def mem_page_read(self, ptr: BufferPointer | Pointer, container: DataContainer):
        msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            cmd_id="mem_load_page_to_container"
        ).with_args(
            ptr=ptr,
            container=container
        )
        
        self.async_rpc_send_req_msg(msg)
        self.async_rpc_wait_rsp_msg(msg)
    
    @jit_prototype
    def mem_page_write(self, ptr: BufferPointer | Pointer, container: DataContainer):
        msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            cmd_id="mem_store_page_from_container"
        ).with_args(
            ptr=ptr,
            container=container
        )
        
        self.async_rpc_send_req_msg(msg)
        self.async_rpc_wait_rsp_msg(msg)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core