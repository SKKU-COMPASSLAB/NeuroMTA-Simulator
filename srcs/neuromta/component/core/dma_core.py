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
        
        self.mem_handle: MemoryHandle = self.mem_info.mem_handle
        
        # # ticket lock for local memory read/write operations, ensuring mutual exclusion
        # self._mem_handle_curr_lock   = VariableHandle(f"core_{core_id}_mem_handle_lock",   initial_value=0)  # binary lock for memory handle access
        # self._mem_handle_curr_ticket = VariableHandle(f"core_{core_id}_mem_handle_ticket", initial_value=0)  # ticket for memory handle access
    
    
    ###########################################################################
    # Memory Handle Management
    ###########################################################################

    def check_ptr_belonging(self, ptr: Pointer) -> bool:
        mem_st = self.mem_handle.base_addr
        mem_ed = self.mem_handle.base_addr + self.mem_handle.size
        
        if isinstance(ptr, Pointer):
            addr = ptr.addr
            
        return mem_st <= addr < mem_ed
    
    @core_command_method
    def local_data_container_init(self, container: DataContainer[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype):
        if self.is_performance_mode:
            container.set_metadata(shape=shape, dtype=dtype)
            return
        data = torch.zeros(shape, dtype=dtype)
        container.data = data
        
    @core_command_method
    def local_mem_init(self, ptr: Pointer, size: int, init_data: torch.Tensor=None):    
        # tmp_lock = VariableHandle.tmp(initial_value=0)
        # self.var_atomic_copy_and_increment(tmp_lock, self._mem_handle_curr_ticket, 1)
        # self.var_conditional_wait(self._mem_handle_curr_lock, self._mem_handle_curr_lock.equals_to(tmp_lock))
        
        if self.is_performance_mode:
            return
        
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_init' method.")
        
        if init_data is None:
            init_data = torch.zeros((size,), dtype=torch.uint8)
        self.mem_handle.set_data(ptr, size=size, data=init_data)
        
        # if not self.is_performance_mode:
        #     if init_data is None:
        #         init_data = torch.zeros((size,), dtype=torch.uint8)
        #     self.mem_handle.set_data(ptr, size=size, data=init_data)
        
        # self.var_atomic_increase(self._mem_handle_curr_lock, 1)
        
    @core_command_method
    def local_mem_page_read(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0, cont_row_zero_pad: int=0):
        if self.is_performance_mode:
            if container.shape is None or container.dtype is None:
                container.set_metadata(shape=(row_num, cont_row_stride), dtype=torch.uint8)
            else:
                container.set_metadata(shape=container.shape, dtype=container.dtype)
            return
        
        # tmp_lock = VariableHandle.tmp(initial_value=0)
        # self.var_atomic_copy_and_increment(tmp_lock, self._mem_handle_curr_ticket, 1)
        # self.var_conditional_wait(self._mem_handle_curr_lock, self._mem_handle_curr_lock.equals_to(tmp_lock))
        
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_read' method.")
        
        if mem_row_stride is None:
            mem_row_stride = row_size
        if cont_row_stride is None:
            cont_row_stride = row_size
            
        # if self.is_performance_mode:
        #     if container.shape is None or container.dtype is None:
        #         container.set_metadata(shape=(row_num, cont_row_stride), dtype=torch.uint8)
        #     else:
        #         container.set_metadata(shape=container.shape, dtype=container.dtype)
        #     self.var_atomic_increase(self._mem_handle_curr_lock, 1)
        #     return
            
        if container.is_mem_segment:
            container.data = container.data.flatten().view(torch.uint8).reshape(-1, cont_row_stride)
        else:
            container.data = torch.zeros((row_num * cont_row_stride,), dtype=torch.uint8).reshape(row_num, cont_row_stride)

        row_slice = slice(cont_row_offset, cont_row_offset + row_size)
        zero_slice = slice(cont_row_offset + row_size, cont_row_offset + row_size + cont_row_zero_pad)
        base_ptr = ptr

        if row_pattern is None:
            # Fast path: contiguous row copy can be served by a single get_data call.
            if mem_row_stride == row_size and row_num > 0:
                bulk_size = row_num * row_size
                src_data = self.mem_handle.get_data(base_ptr, size=bulk_size, dtype=torch.uint8).reshape(row_num, row_size)
                container.data[:row_num, row_slice] = src_data
            else:
                for r in range(row_num):
                    src_data = self.mem_handle.get_data(base_ptr + (r * mem_row_stride), size=row_size, dtype=torch.uint8)
                    container.data[r, row_slice] = src_data

            if cont_row_zero_pad > 0 and row_num > 0:
                container.data[:row_num, zero_slice] = 0
        else:
            for d, s in row_pattern.items():
                src_data = self.mem_handle.get_data(base_ptr + (s * mem_row_stride), size=row_size, dtype=torch.uint8)
                container.data[d, row_slice] = src_data

                if cont_row_zero_pad > 0:
                    container.data[d, zero_slice] = 0
                    
        # self.var_atomic_increase(self._mem_handle_curr_lock, 1)
        
    @core_command_method
    def local_mem_page_write(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        if self.is_performance_mode:
            return
        
        # tmp_lock = VariableHandle.tmp(initial_value=0)
        # self.var_atomic_copy_and_increment(tmp_lock, self._mem_handle_curr_ticket, 1)
        # self.var_conditional_wait(self._mem_handle_curr_lock, self._mem_handle_curr_lock.equals_to(tmp_lock))
        
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_write' method.")

        if mem_row_stride is None:
            mem_row_stride = row_size
        if cont_row_stride is None:
            cont_row_stride = row_size

        # if self.is_performance_mode:
        #     self.var_atomic_increase(self._mem_handle_curr_lock, 1)
        #     return

        if not container.is_mem_segment:
            raise ValueError("container.data must be a Tensor for local_mem_page_write.")

        cont_data = container.data.flatten().view(torch.uint8)[:row_num * cont_row_stride].reshape(row_num, cont_row_stride)
        row_slice = slice(cont_row_offset, cont_row_offset + row_size)
        base_ptr = ptr

        if row_pattern is None:
            # Fast path: contiguous rows can be committed with a single set_data call.
            if mem_row_stride == row_size and row_num > 0:
                bulk_data = cont_data[:row_num, row_slice].reshape(-1)
                self.mem_handle.set_data(base_ptr, size=row_num * row_size, data=bulk_data)
            else:
                for r in range(row_num):
                    dst_data = cont_data[r, row_slice]
                    self.mem_handle.set_data(base_ptr + (r * mem_row_stride), size=row_size, data=dst_data)
        else:
            for d, s in row_pattern.items():
                dst_data = cont_data[d, row_slice]
                self.mem_handle.set_data(base_ptr + (s * mem_row_stride), size=row_size, data=dst_data)
                
        # self.var_atomic_increase(self._mem_handle_curr_lock, 1)

    ###########################################################################
    # Static Memory Space Management
    ###########################################################################
                
    @core_command_method
    def allocate_static_mem_space(self, ptr: Pointer, size: int):
        self.mem_handle.allocate_static_mem_space(ptr=ptr, size=size)

    @core_command_method
    def deallocate_static_mem_space(self, ptr: Pointer):
        self.mem_handle.deallocate_static_mem_space(ptr=ptr)
        
    ###########################################################################
    # DMA
    ###########################################################################
        
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
    