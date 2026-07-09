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
        
    @core_command_method
    def _dma_lightweight_request_handle(self, addr: int, size: int, is_write: bool):
        pass
    
    @core_command_method
    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        pass
        
    @jit_prototype
    def remote_mem_page_read(self, dst_core_id: int, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0, cont_row_zero_pad: int=0):
        src_core_id = self.core_id
        
        self.local_mem_page_read(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset, cont_row_zero_pad)
        
        with new_parallel_thread("NOC"):
            if self.icnt_context.is_icnt_simulator_enabled:
                if mem_row_stride is None or mem_row_stride == row_size:
                    size = row_size * row_num
                    self._icnt_data_transfer_handle(src_core_id, dst_core_id, size, is_write=False)
                else:
                    for row_idx in range(row_num):
                        size = row_size
                        self._icnt_data_transfer_handle(src_core_id, dst_core_id, size, is_write=False)
            else:
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
            if self.global_context.is_main_mem_simulator_enabled:
                if mem_row_stride is None or mem_row_stride == row_size:
                    addr = ptr.addr
                    size = row_size * row_num
                    self._dma_lightweight_request_handle(addr=addr, size=size, is_write=False)
                else:
                    for row_idx in range(row_num):
                        addr = ptr.addr + row_idx * (mem_row_stride if mem_row_stride is not None else row_size)
                        size = row_size
                        self._dma_lightweight_request_handle(addr=addr, size=size, is_write=False)
            else:
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
            if self.icnt_context.is_icnt_simulator_enabled:
                if mem_row_stride is None or mem_row_stride == row_size:
                    size = row_size * row_num
                    self._icnt_data_transfer_handle(src_core_id, dst_core_id, size, is_write=True)
                else:
                    for row_idx in range(row_num):
                        size = row_size
                        self._icnt_data_transfer_handle(src_core_id, dst_core_id, size, is_write=True)
            else:
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
            if self.global_context.is_main_mem_simulator_enabled:
                if mem_row_stride is None or mem_row_stride == row_size:
                    addr = ptr.addr
                    size = row_size * row_num
                    self._dma_lightweight_request_handle(addr=addr, size=size, is_write=True)
                else:
                    for row_idx in range(row_num):
                        addr = ptr.addr + row_idx * (mem_row_stride if mem_row_stride is not None else row_size)
                        size = row_size
                        self._dma_lightweight_request_handle(addr=addr, size=size, is_write=True)
            else:
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
        
    def _dma_lightweight_request_handle(self, addr: int, size: int, is_write: bool) -> int:
        if not self.core.global_context.is_main_mem_simulator_enabled:
            raise RuntimeError("Memory simulator is not available. Please use the DRAMSim3 companion module is properly installed and configured.")
        
        result = self.core.global_context.main_mem_simulator.send_request(
            addr=addr,
            size=size,
            is_write=is_write,
            current_cycle=self.core.timestamp
        )
        
        latency_cycles = result["latency_cycles"]
        return latency_cycles
    
    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        if not self.core.icnt_context.is_icnt_simulator_enabled:
            raise RuntimeError("ICNT simulator is not available. Please use the BookSim2 companion module is properly installed and configured.")
        
        result = self.core.icnt_context.icnt_simulator.send_request(
            src_core_id=src_core_id,
            dst_core_id=dst_core_id,
            data_size=data_size,
            is_write=is_write,
            current_cycle=self.core.timestamp
        )
        
        latency_cycles = result["latency_cycles"]
        return latency_cycles
    