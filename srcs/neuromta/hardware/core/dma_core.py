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
    
    @core_kernel_method
    def mem_load_page(self, ptr: Pointer, container: DataContainer):
        msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            cmd_id="mem_load_page"
        ).with_args(
            ptr=ptr,
            container=container
        )
        
        self.async_rpc_send_req_msg(msg)
        self.async_rpc_wait_rsp_msg(msg)
        
    @core_kernel_method
    def mem_store_page(self, ptr: Pointer, container: DataContainer):
        msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            cmd_id="mem_store_page"
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