import torch
from neuromta.framework import *

from neuromta.component.context.global_context import *
from neuromta.component.context.icnt_context import IcntContext

__all__ = [
    "DMACore",
]


class DMACore(Core):
    def __init__(
        self, 
        core_id: int,
        global_context: GlobalContext,
        icnt_context: IcntContext,
    ):
        super().__init__(
            core_id=core_id,
            cycle_model=DMACoreCycleModel(core=self)
        )
        
        self.global_context = global_context
        self.icnt_context = icnt_context
        
        self.core_info  = self.global_context.get_core_info(GlobalContextCoreType.DMA, core_id)
        self.mem_info   = self.core_info.owned_mem_info  # Assume that each DMA core owns only one memory
        
        self.set_mem_handle(mem_handle=self.mem_info.mem_handle)
        
    @jit_prototype
    def remote_mem_page_read(self, dst_core_id: int, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0, cont_row_zero_pad: int=0):
        src_core_id = self.core_id
        
        self.local_mem_page_read(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset, cont_row_zero_pad)
        
        with new_parallel_thread("NOC"):
            noc_msgs = [
                RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.config.booksim_module_id,
                    **arg
                )
                for arg in self.icnt_context.get_icnt_data_transfer_args(src_core_id, dst_core_id, row_size * row_num, is_write=True)
            ]
            
            for msg in noc_msgs:
                self.async_rpc_send_req_msg(msg)
            for msg in noc_msgs:
                self.async_rpc_wait_rsp_msg(msg)
                    
        with new_parallel_thread("DMA"):
            dram_msgs = [
                RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.dramsim_module_id,
                    **arg
                )
                for arg in self.global_context.get_main_mem_strided_access_args(ptr, row_size, row_num, mem_row_stride, is_write=False)
            ]
            
            for msg in dram_msgs:
                self.async_rpc_send_req_msg(msg)
            for msg in dram_msgs:
                self.async_rpc_wait_rsp_msg(msg)
                
        self.parallel_merge()
            
    @jit_prototype
    def remote_mem_page_write(self, src_core_id: int, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        dst_core_id = self.core_id
        
        with new_parallel_thread("NOC"):
            noc_msgs = [
                RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.config.booksim_module_id,
                    **arg
                )
                for arg in self.icnt_context.get_icnt_data_transfer_args(src_core_id, dst_core_id, row_size * row_num, is_write=False)
            ]
            
            for msg in noc_msgs:
                self.async_rpc_send_req_msg(msg)
            for msg in noc_msgs:
                self.async_rpc_wait_rsp_msg(msg)
                
        with new_parallel_thread("DMA"):
            dram_msgs = [
                RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.dramsim_module_id,
                    **arg
                )
                for arg in self.global_context.get_main_mem_strided_access_args(ptr, row_size, row_num, mem_row_stride, is_write=True)
            ]
            
            for msg in dram_msgs:
                self.async_rpc_send_req_msg(msg)
            for msg in dram_msgs:
                self.async_rpc_wait_rsp_msg(msg)
                
        self.parallel_merge()

        self.local_mem_page_write(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core