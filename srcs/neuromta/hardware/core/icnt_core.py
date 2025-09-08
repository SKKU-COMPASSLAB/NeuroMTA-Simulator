import math

from neuromta.framework import *

from neuromta.hardware.context.cmap_context import CmapContext
from neuromta.hardware.context.icnt_context import IcntContext

from neuromta.hardware.companions.booksim import PYBOOKSIM2_AVAILABLE, BookSim2


__all__ = [
    "IcntCore",
]


class IcntCore(Core):
    def __init__(
        self,
        icnt_context: IcntContext,
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=cmap_context.icnt_core_id,
            cycle_model=IcntCoreCycleModel(core=self)
        )
        
        self.icnt_context = icnt_context
        self.cmap_context = cmap_context
    
    @property
    def is_booksim2_enabled(self) -> bool:
        return PYBOOKSIM2_AVAILABLE and self.icnt_context.booksim2_enable

    @core_kernel_method
    def noc_create_data_read_transaction(self, src_id: int, dst_id: int, data_size: int):
        if self.is_booksim2_enabled:
            n_flits = math.ceil(data_size / self.icnt_context.flit_size)
            
            data_req_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=COMPANION_CORE_ID,
                cmd_id="send_companion_command",
            ).with_args(
                self.cmap_context.config.booksim_module_id,
                src_id, dst_id, 
                subnet=0, n_flits=n_flits, 
                is_write=False, is_response=False
            )
            
            data_rsq_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=COMPANION_CORE_ID,
                cmd_id="send_companion_command",
            ).with_args(
                self.cmap_context.config.booksim_module_id,
                dst_id, src_id, 
                subnet=0, n_flits=n_flits, 
                is_write=False, is_response=True
            )
            
            self.async_rpc_send_req_msg(data_req_msg)
            self.async_rpc_wait_rsp_msg(data_req_msg)
            
            self.async_rpc_send_req_msg(data_rsq_msg)
            self.async_rpc_wait_rsp_msg(data_rsq_msg)
            
        else:
            self._static_noc_create_data_read_transaction(src_id, src_id, data_size)
            
    @core_kernel_method
    def noc_create_data_write_transaction(self, src_id: int, dst_id: int, data_size: int):
        if self.is_booksim2_enabled:
            n_flits = math.ceil(data_size / self.icnt_context.flit_size)
            
            data_req_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=COMPANION_CORE_ID,
                cmd_id="send_companion_command",
            ).with_args(
                self.cmap_context.config.booksim_module_id,
                src_id, dst_id, 
                subnet=0, n_flits=n_flits, 
                is_write=True, is_response=False
            )
            
            data_rsq_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=COMPANION_CORE_ID,
                cmd_id="send_companion_command",
            ).with_args(
                self.cmap_context.config.booksim_module_id,
                dst_id, src_id, 
                subnet=0, n_flits=n_flits, 
                is_write=True, is_response=True
            )
            
            self.async_rpc_send_req_msg(data_req_msg)
            self.async_rpc_wait_rsp_msg(data_req_msg)
            
            self.async_rpc_send_req_msg(data_rsq_msg)
            self.async_rpc_wait_rsp_msg(data_rsq_msg)
            
        else:
            self._static_noc_create_data_write_transaction(src_id, dst_id, data_size)

    @core_command_method
    def _static_noc_create_data_read_transaction(self, src_id: int, dst_id: int, data_size: int):
        pass

    @core_command_method
    def _static_noc_create_data_write_transaction(self, src_id: int, dst_id: int, data_size: int):
        pass

class IcntCoreCycleModel(CoreCycleModel):
    def __init__(self, core: IcntCore):
        super().__init__()
        
        self.core = core

    def _static_noc_create_data_read_transaction(self, src_id: int, dst_id: int, data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_id, dst_id, data_size)

    def _static_noc_create_data_write_transaction(self, src_id: int, dst_id: int, data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_id, dst_id, data_size)
