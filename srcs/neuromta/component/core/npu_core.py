import torch

from neuromta.framework import *

from neuromta.component.context.global_context import GlobalContext, GlobalContextMemType, GlobalContextCoreType
from neuromta.component.context.icnt_context import IcntContext
from neuromta.component.context.vpu_context import VPUConfig, VPUOperator
from neuromta.component.context.mxu_context import MXUConfig, MXUElementwiseOp


__all__ = [
    "NPUCore",
] 


class NPUCore(Core):
    def __init__(
        self,
        core_id: int,
        global_context: GlobalContext,
        icnt_context: IcntContext = None,  # optional: None if no NoC is used
        vpu_config: VPUConfig = VPUConfig(),
        mxu_config: MXUConfig = MXUConfig(),
    ):
        super().__init__(
            core_id=core_id,
            cycle_model=NPUCoreCycleModel(core=self),
        )
        
        self.global_context = global_context
        self.icnt_context = icnt_context
        
        self.mxu_context = mxu_config.create_context()
        self.vpu_context = vpu_config.create_context()
        
        self.core_info  = self.global_context.get_core_info(GlobalContextCoreType.NPU, core_id)
        self.mem_info   = self.core_info.owned_mem_info  # Assume that each NPU core owns only one memory
        self.mem_handle = self.mem_info.mem_handle
        
        # self._dma_engine_idx = self.core_id % self.global_context.n_main_mem_cmd_q_per_instance  # Assume that each NPU core is connected to one DMA engine in a round-robin manner
        # self._dma_engine_idx = 0  # Assume that each NPU core is connected to one DMA engine in a round-robin manner
        self._dma_engine_idx = self.core_id % self.global_context.n_dma_engine_per_channel  # Assume that each NPU core is connected to one DMA engine in a round-robin manner
        
        # synchronization variables
        self.ongoing_core_sync_msg: list[int] = []
    
    def get_buffer_owner(self, ptr: Pointer | int) -> int:
        if isinstance(ptr, Pointer):
            addr = ptr.addr
        else:
            addr = ptr
            
        mem_info = self.global_context.get_mem_info_by_address(addr)
        if mem_info.mem_type == GlobalContextMemType.L1:
            return mem_info.owner_core_ids[0]
        else:
            return mem_info.owner_core_ids[self._dma_engine_idx]
        
    def check_ptr_belonging(self, ptr: Pointer) -> bool:
        return self.get_buffer_owner(ptr) == self.core_id
        
    #############################################################
    # Inter-Core Synchronization
    #############################################################
    
    @core_command_method
    def _static_inter_core_sync_send_msg(self, dst_core_id: int):
        pass
    
    def inter_core_sync_send_msg(self, dst_core_id: int):
        if self.check_rpc_inbox(self.global_context.icnt_core_id):  # check if it is possible to send NOC transaction request (if not, the )
            noc_write_msg = RPCMessage(self.core_id, self.global_context.icnt_core_id, cmd_id="noc_create_data_write_transaction").with_args(src_id=self.core_id, dst_id=dst_core_id, data_size=2)
            self.async_rpc_send_req_msg(noc_write_msg)
            self.async_rpc_wait_rsp_msg(noc_write_msg)
        else:
            self._static_inter_core_sync_send_msg(dst_core_id)
        
        sync_send_msg = RPCMessage(self.core_id, dst_core_id, cmd_id="inter_core_sync_recv_msg").with_args(src_core_id=self.core_id)
        self.async_rpc_send_req_msg(sync_send_msg)
        self.async_rpc_wait_rsp_msg(sync_send_msg)
        
    @core_conditional_command_method
    def inter_core_sync_recv_msg(self, src_core_id: int):
        if src_core_id not in self.ongoing_core_sync_msg:
            return False
        self.ongoing_core_sync_msg.remove(src_core_id)
        return True
    
    @core_conditional_command_method
    def inter_core_sync_trigger(self, slave_core_ids: list[int]):
        for slave_core_id in slave_core_ids:
            if slave_core_id in self.ongoing_core_sync_msg:
                return False
            
        self.ongoing_core_sync_msg = slave_core_ids.copy()
        return True
    
    @core_conditional_command_method
    def inter_core_sync_wait(self):
        return len(self.ongoing_core_sync_msg) == 0
    
    def inter_core_sync_barrier(self, core_ids: list[int]):
        master_core_id = core_ids[0]
        slave_core_ids = core_ids[1:]
        
        if self.core_id == master_core_id:
            self.inter_core_sync_trigger(slave_core_ids)
            self.inter_core_sync_wait()
            for slave_core_id in slave_core_ids:
                with new_parallel_thread():
                    self.inter_core_sync_send_msg(slave_core_id)
            self.parallel_merge()
        else:
            self.inter_core_sync_trigger([master_core_id,])
            self.inter_core_sync_send_msg(master_core_id)
            self.inter_core_sync_wait()
        
    #############################################################
    # Memory Copy Commands
    #############################################################
    
    @core_command_method
    def local_data_container_init(self, container: DataContainer[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype):
        data = torch.zeros(shape, dtype=dtype)
        container.data = data
        
    @core_command_method
    def local_mem_init(self, ptr: Pointer, size: int, init_data: torch.Tensor=None):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_init' method.")
            
        if init_data is None:
            init_data = torch.zeros((size,), dtype=torch.uint8)
        
        self.mem_handle.set_data(ptr, size=size, data=init_data)
            
    def mem_init(self, ptr: Pointer, size: int, init_data: torch.Tensor=None):
        if self.check_ptr_belonging(ptr):
            self.local_mem_init(ptr, size, init_data)
        else:
            owner_id = self.get_buffer_owner(ptr)
            
            msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_mem_init"
            ).with_args(
                ptr=ptr,
                size=size,
                init_data=init_data
            )
            
            self.async_rpc_send_req_msg(msg)
            self.async_rpc_wait_rsp_msg(msg)
    
    @core_command_method
    def local_mem_page_read(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_read' method.")
        
        if mem_row_stride is None:
            mem_row_stride = row_size
        if cont_row_stride is None:
            cont_row_stride = row_size
        if row_pattern is None:
            row_pattern = {i: i for i in range(row_num)}
            
        if container.is_mem_segment:
            container.data = container.data.flatten().view(torch.uint8).reshape(-1, cont_row_stride)
        else:
            container.data = torch.zeros((row_num * cont_row_stride,), dtype=torch.uint8).reshape(row_num, cont_row_stride)
        
        for d, s in row_pattern.items():
            src_data = self.mem_handle.get_data(ptr + (s * mem_row_stride), size=row_size, dtype=torch.uint8)
            container.data[d, cont_row_offset:cont_row_offset+row_size] = src_data
        
    @core_command_method
    def local_mem_page_write(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_write' method.")
        
        if mem_row_stride is None:
            mem_row_stride = row_size
        if cont_row_stride is None:
            cont_row_stride = row_size
        if row_pattern is None:
            row_pattern = {i: i for i in range(row_num)}
            
        if container.is_mem_segment:
            container.data = container.data.flatten().view(torch.uint8).reshape(row_num, cont_row_stride)
        else:
            container.data = torch.zeros((row_num * cont_row_stride,), dtype=torch.uint8).reshape(row_num, cont_row_stride)
        
        for d, s in row_pattern.items():
            dst_data = container.data[d, cont_row_offset:cont_row_offset+row_size]
            self.mem_handle.set_data(ptr + (s * mem_row_stride), size=row_size, data=dst_data)
        
    def local_mem_copy(self, dst_ptr: Pointer, src_ptr: Pointer, row_size: int, row_num: int=1, src_row_stride: int=None, dst_row_stride: int=None, nowait: bool=False):
        if not isinstance(dst_ptr, Pointer) or not isinstance(src_ptr, Pointer):
            raise ValueError("dst_ptr and src_ptr must be Pointer instances.")
        if dst_ptr.addr is None or src_ptr.addr is None:
            raise ValueError("dst_ptr and src_ptr must have valid addresses before 'mem_copy' method is compiled.")
        
        dst_mem_info = self.global_context.get_mem_info_by_address(dst_ptr.addr)
        src_mem_info = self.global_context.get_mem_info_by_address(src_ptr.addr)
        
        if dst_mem_info.mem_type == src_mem_info.mem_type == GlobalContextMemType.MAIN:
            raise Exception("Direct memory copy between MAIN memories is not supported.")  # TODO: Implement DMA-based MAIN-to-MAIN memory copy if needed.
        
        if dst_mem_info.mem_type == GlobalContextMemType.L1:
            dst_owner_core_id = dst_mem_info.owner_core_ids[0]
        else:
            dst_owner_core_id = dst_mem_info.owner_core_ids[self._dma_engine_idx]
            
        if src_mem_info.mem_type == GlobalContextMemType.L1:
            src_owner_core_id = src_mem_info.owner_core_ids[0]
        else:
            src_owner_core_id = src_mem_info.owner_core_ids[self._dma_engine_idx]
            
        if src_row_stride is None:
            src_row_stride = row_size
        if dst_row_stride is None:
            dst_row_stride = row_size
            
        # THREAD: Data Read & Write
        with new_parallel_thread("DATA_RD_WR"):
            container = DataContainer()

            if src_owner_core_id == self.core_id:
                self.local_mem_page_read(src_ptr, container, row_size, row_num, src_row_stride, row_size)
            else:
                data_rd_request = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=src_owner_core_id,
                    cmd_id="local_mem_page_read"
                ).with_args(
                    ptr=src_ptr,
                    container=container,
                    row_size=row_size,
                    row_num=row_num,
                    mem_row_stride=src_row_stride,
                    cont_row_stride=row_size,
                )
                
                self.async_rpc_send_req_msg(data_rd_request)
                self.async_rpc_wait_rsp_msg(data_rd_request)
            
            if dst_owner_core_id == self.core_id:
                self.local_mem_page_write(dst_ptr, container, row_size, row_num, dst_row_stride, row_size)
            else:
                data_wr_request = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=dst_owner_core_id,
                    cmd_id="local_mem_page_write"
                ).with_args(
                    ptr=dst_ptr,
                    container=container,
                    row_size=row_size,
                    row_num=row_num,
                    mem_row_stride=dst_row_stride,
                    cont_row_stride=row_size,
                )
                
                self.async_rpc_send_req_msg(data_wr_request)
                self.async_rpc_wait_rsp_msg(data_wr_request)
                
        # THREAD: DRAM Transactions
        with new_parallel_thread("DRAM_TX"):
            dram_msgs = []
            
            if src_mem_info.mem_type == GlobalContextMemType.MAIN:
                dram_rd_transaction = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.dramsim_module_id,
                    **self.global_context.get_main_mem_access_args(src_ptr, row_size * row_num, is_write=False)
                )
                
                dram_msgs.append(dram_rd_transaction)
                
            if dst_mem_info.mem_type == GlobalContextMemType.MAIN:
                dram_wr_transaction = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.dramsim_module_id,
                    **self.global_context.get_main_mem_access_args(dst_ptr, row_size * row_num, is_write=True)
                )
                
                dram_msgs.append(dram_wr_transaction)
            
            for msg in dram_msgs:
                self.async_rpc_send_req_msg(msg)
            
            if not nowait:
                for msg in dram_msgs:
                    self.async_rpc_wait_rsp_msg(msg)
                    
        # THREAD: NOC Transactions
        if self.icnt_context is not None:
            with new_parallel_thread("NOC_TX"):
                noc_msgs = []
                
                if src_owner_core_id != self.core_id:
                    noc_msgs += [
                        RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=COMPANION_CORE_ID,
                            cmd_id="send_companion_command",
                        ).with_args(
                            self.global_context.config.booksim_module_id,
                            **arg
                        )
                        for arg in self.icnt_context.get_icnt_data_transfer_args(src_owner_core_id, self.core_id, row_size * row_num, is_write=False)
                    ]
                    
                if dst_owner_core_id != self.core_id:
                    noc_msgs += [
                        RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=COMPANION_CORE_ID,
                            cmd_id="send_companion_command",
                        ).with_args(
                            self.global_context.config.booksim_module_id,
                            **arg
                        )
                        
                        for arg in self.icnt_context.get_icnt_data_transfer_args(self.core_id, dst_owner_core_id, row_size * row_num, is_write=True)
                    ]
                    
                for msg in noc_msgs:
                    self.async_rpc_send_req_msg(msg)
                
                if not nowait:
                    for msg in noc_msgs:
                        self.async_rpc_wait_rsp_msg(msg)
        
        self.parallel_merge()

    def local_mem_broadcast(self, dst_ptrs: list[Pointer], src_ptr: Pointer,  row_size: int, row_num: int, src_row_stride: int=None, dst_row_stride: int=None, nowait: bool=False):
        if not isinstance(src_ptr, Pointer):
            raise ValueError("dst_ptr and src_ptr must be Pointer instances.")
        if src_ptr.addr is None:
            raise ValueError("dst_ptr and src_ptr must have valid addresses before 'mem_copy' method is compiled.")
        
        src_mem_info = self.global_context.get_mem_info_by_address(src_ptr.addr)
        
        if src_mem_info.mem_type == GlobalContextMemType.L1:
            src_owner_core_id = src_mem_info.owner_core_ids[0]
        else:
            src_owner_core_id = src_mem_info.owner_core_ids[self._dma_engine_idx]
            
        if src_row_stride is None:
            src_row_stride = row_size
        if dst_row_stride is None:
            dst_row_stride = row_size
            
        # THREAD: Data Read & Write
        with new_parallel_thread("DATA_RD_WR"):
            container = DataContainer()

            if src_owner_core_id == self.core_id:
                self.local_mem_page_read(src_ptr, container, row_size, row_num, src_row_stride, row_size)
            else:
                data_rd_request = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=src_owner_core_id,
                    cmd_id="local_mem_page_read"
                ).with_args(
                    ptr=src_ptr,
                    container=container,
                    row_size=row_size,
                    row_num=row_num,
                    mem_row_stride=src_row_stride,
                    cont_row_stride=row_size,
                )
                
                self.async_rpc_send_req_msg(data_rd_request)
                self.async_rpc_wait_rsp_msg(data_rd_request)
                
            for dst_ptr in dst_ptrs:
                with new_parallel_thread("BCAST"):
                    dst_mem_info = self.global_context.get_mem_info_by_address(dst_ptr.addr)
                    
                    if dst_mem_info.mem_type == GlobalContextMemType.L1:
                        dst_owner_core_id = dst_mem_info.owner_core_ids[0]
                    else:
                        dst_owner_core_id = dst_mem_info.owner_core_ids[self._dma_engine_idx]
            
                    if dst_owner_core_id == self.core_id:
                        self.local_mem_page_write(dst_ptr, container, row_size, row_num, dst_row_stride, row_size)
                    else:
                        data_wr_request = RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=dst_owner_core_id,
                            cmd_id="local_mem_page_write"
                        ).with_args(
                            ptr=dst_ptr,
                            container=container,
                            row_size=row_size,
                            row_num=row_num,
                            mem_row_stride=dst_row_stride,
                            cont_row_stride=row_size,
                        )
                        
                        self.async_rpc_send_req_msg(data_wr_request)
                        self.async_rpc_wait_rsp_msg(data_wr_request)
                        
        # THREAD: DRAM Transactions
        with new_parallel_thread("DRAM_TX"):
            dram_msgs = []
            
            if src_mem_info.mem_type == GlobalContextMemType.MAIN:
                dram_rd_transaction = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self.global_context.dramsim_module_id,
                    **self.global_context.get_main_mem_access_args(src_ptr, row_size * row_num, is_write=False)
                )
                
                dram_msgs.append(dram_rd_transaction)
            
            for dst_ptr in dst_ptrs: 
                dst_mem_info = self.global_context.get_mem_info_by_address(dst_ptr.addr)
                    
                if dst_mem_info.mem_type == GlobalContextMemType.L1:
                    dst_owner_core_id = dst_mem_info.owner_core_ids[0]
                else:
                    dst_owner_core_id = dst_mem_info.owner_core_ids[self._dma_engine_idx]
                        
                if dst_mem_info.mem_type == GlobalContextMemType.MAIN:
                    dram_wr_transaction = RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self.global_context.dramsim_module_id,
                        **self.global_context.get_main_mem_access_args(dst_ptr, row_size * row_num, is_write=True)
                    )
                    
                    dram_msgs.append(dram_wr_transaction)
            
            for msg in dram_msgs:
                self.async_rpc_send_req_msg(msg)
            
            if not nowait:
                for msg in dram_msgs:
                    self.async_rpc_wait_rsp_msg(msg)
                    
        # THREAD: NOC Transactions
        if self.icnt_context is not None:
            with new_parallel_thread("NOC_TX"):
                noc_msgs = []
                
                if src_owner_core_id != self.core_id:
                    noc_msgs += [
                        RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=COMPANION_CORE_ID,
                            cmd_id="send_companion_command",
                        ).with_args(
                            self.global_context.config.booksim_module_id,
                            **arg
                        )
                        for arg in self.icnt_context.get_icnt_data_transfer_args(src_owner_core_id, self.core_id, row_size * row_num, is_write=False)
                    ]
                
                for dst_ptr in dst_ptrs:
                    dst_mem_info = self.global_context.get_mem_info_by_address(dst_ptr.addr)
                    
                    if dst_mem_info.mem_type == GlobalContextMemType.L1:
                        dst_owner_core_id = dst_mem_info.owner_core_ids[0]
                    else:
                        dst_owner_core_id = dst_mem_info.owner_core_ids[self._dma_engine_idx]
                        
                    if dst_owner_core_id != self.core_id:
                        noc_msgs += [
                            RPCMessage(
                                src_core_id=self.core_id,
                                dst_core_id=COMPANION_CORE_ID,
                                cmd_id="send_companion_command",
                            ).with_args(
                                self.global_context.config.booksim_module_id,
                                **arg
                            )
                            
                            for arg in self.icnt_context.get_icnt_data_transfer_args(self.core_id, dst_owner_core_id, row_size * row_num, is_write=True)
                        ]
                    
                for msg in noc_msgs:
                    self.async_rpc_send_req_msg(msg)
                
                if not nowait:
                    for msg in noc_msgs:
                        self.async_rpc_wait_rsp_msg(msg)
                
        self.parallel_merge()

    #############################################################
    # MXU Commands
    #############################################################
        
    @core_command_method
    def mxu_reconfigure(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self.mxu_context.reconfigure_dtype(dtype=dtype, acc_dtype=acc_dtype)
        
    @core_command_method
    def mxu_load_context(self, psum_cont: DataContainer[torch.Tensor]):
        psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
        self.mxu_context.load_tile_pe_arr(psum_tile)
        
    @core_command_method
    def mxu_store_context(self, psum_cont: DataContainer[torch.Tensor]):
        psum_tile = self.mxu_context.get_pe_arr_regs()
        psum_cont.data = psum_tile
    
    @core_command_method
    def mxu_tiled_gemm(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        wgt_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        wgt_transposed: bool=False,
        psum_vectored:  bool=False,
    ):  
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if psum_cont.data is None:
                psum_tile = torch.zeros(self.mxu_context.config.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype)
                
                if psum_vectored:
                    psum_tile = psum_tile.flatten().unsqueeze(0)  # (1, N)
                else:
                    psum_tile = psum_tile.reshape(self.mxu_context.config.ofm_tile_shape)
                    
                if ifm_transposed: 
                    psum_tile = psum_tile.T
                
            self.mxu_context.load_tile_pe_arr(psum_tile)  

        ifm_tile = ifm_cont.data.view(self.mxu_context.dtype)  #.reshape(self.mxu_context.ifm_tile_shape)
        wgt_tile = wgt_cont.data.view(self.mxu_context.dtype)  #.reshape(self.mxu_context.wgt_tile_shape)
        
        if self.mxu_context.config.ifm_tile_numel != ifm_tile.numel():
            ifm_tile = torch.nn.functional.pad(ifm_tile, (0, self.mxu_context.config.ifm_tile_numel - ifm_tile.numel()), 'constant', 0)
        if self.mxu_context.config.wgt_tile_numel != wgt_tile.numel():
            wgt_tile = torch.nn.functional.pad(wgt_tile, (0, self.mxu_context.config.wgt_tile_numel - wgt_tile.numel()), 'constant', 0)
            
        ifm_tile = ifm_tile.reshape(self.mxu_context.config.ifm_tile_shape)
        wgt_tile = wgt_tile.reshape(self.mxu_context.config.wgt_tile_shape)
        
        if ifm_transposed: ifm_tile = ifm_tile.T
        if wgt_transposed: wgt_tile = wgt_tile.T

        self.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=wgt_tile)
            
        if flush_ofm:
            ofm_tile = self.mxu_context.get_pe_arr_regs()  
                
            if ifm_transposed: 
                ofm_tile = ofm_tile.T
            
            ofm_cont.data = ofm_tile
            
    @core_command_method
    def mxu_tiled_maxpool(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
    ):  
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if psum_cont.data is None:
                psum_tile = torch.zeros(self.mxu_context.config.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
            
            self.mxu_context.load_tile_pe_arr(psum_tile)
        
        ifm_tile = ifm_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
        if ifm_transposed: ifm_tile = ifm_tile.T
        self.mxu_context.execute_maxpool(ifm_tile=ifm_tile)
        
        if flush_ofm:
            ofm_tile = self.mxu_context.get_pe_arr_regs()
            ofm_cont.data = ofm_tile
            
    @core_command_method
    def mxu_tiled_elemwise(
        self,

        op: MXUElementwiseOp,
        src:  DataContainer[torch.Tensor],
        dst:  DataContainer[torch.Tensor],
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        ifm_transposed: bool=False,
    ):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if dst.data is None:
                psum_tile = torch.zeros(self.mxu_context.config.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = dst.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
                    
            self.mxu_context.load_tile_pe_arr(psum_tile)

        ifm_tile = src.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ifm_tile_shape)
        if ifm_transposed: ifm_tile = ifm_tile.T
        
        self.mxu_context.execute_elemwise(ifm_tile=ifm_tile, op=op)
        
        if flush_ofm:
            ofm_tile = self.mxu_context.get_pe_arr_regs()
            dst.data = ofm_tile
            
    @core_command_method
    def mxu_tiled_elemwise_imm(
        self,
        
        op: MXUElementwiseOp,
        imm: int | float,
        dst:  DataContainer[torch.Tensor],
        
        flush_ofm: bool=False,
    ):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        self.mxu_context.execute_elemwise_imm(imm=imm, op=op)
        
        if flush_ofm:
            ofm_tile = self.mxu_context.get_pe_arr_regs()
            dst.data = ofm_tile
        
            
    #############################################################
    # VPU Commands
    #############################################################
        
    @core_command_method
    def vpu_reconfigure(self, vlen: int, vdtype: torch.dtype):
        self.vpu_context.reconfigure_vector_reg_file(vlen=vlen, vdtype=vdtype)
        
    @core_command_method
    def vpu_load_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1, offset: int=0):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        data = data_cont.data.flatten().view(self.vpu_context.vdtype)
        
        for i in range(burst_len):
            st = offset + i * self.vpu_context.vlen
            ed = offset + (i + 1) * self.vpu_context.vlen
            vreg_data = data[st:ed]
            self.vpu_context.set_vector_reg(vreg_idx + i, vreg_data)
        
    @core_command_method
    def vpu_store_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1, offset: int=0):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)

        data: list[torch.Tensor] = []
        
        for i in range(burst_len):
            vreg_data = self.vpu_context.get_vector_reg(vreg_idx + i)
            data.append(vreg_data)
            
        wr_data = torch.cat(data, dim=0).flatten().view(torch.uint8)
        raw_data: torch.Tensor = data_cont.data.reshape(-1).view(torch.uint8).clone()
        raw_data[offset:offset + wr_data.numel()] = wr_data
        
        data_cont.data = raw_data

    @core_command_method
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        for i in range(burst_len):
            vra = vreg_a + i
            vrb = vreg_b + i if vreg_b is not None else None
            vrd = vreg_dest + i if vreg_dest is not None else None
            self.vpu_context.execute_vector_op(opcode, vra, vrb, vrd, inplace=inplace)

class NPUCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'NPUCore'):
        super().__init__()
        
        self.core = core
        
    def _static_inter_core_sync_send_msg(self, dst_core_id: int):
        if not self.core.check_rpc_inbox(self.core.global_context.icnt_core_id):
            return 10  # TODO: Assume that sending a message takes 10 cycles
        return 1
    
    def _static_inter_core_sync_send_rsp_msg(self, dst_core_id: int):
        if not self.core.check_rpc_inbox(self.core.global_context.icnt_core_id):
            return 10  # TODO: Assume that sending a message takes 10 cycles
        return 1
    
    def local_mem_page_read(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        return 1
        
    def local_mem_page_write(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        return row_num
    
    def mxu_load_context(self, psum_cont: DataContainer[torch.Tensor]):
        return self.core.mxu_context.get_preload_pe_arr_cycles()
        
    def mxu_store_context(self, psum_cont: DataContainer[torch.Tensor]):
        return self.core.mxu_context.get_flush_pe_arr_cycles()

    def mxu_tiled_gemm(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        wgt_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        wgt_transposed: bool=False,
        psum_vectored:  bool=False,
    ):
        total_cycles = 0
        
        if preload_psum:
            total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
                
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
                    
        return total_cycles
    
    def mxu_tiled_maxpool(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        psum_vectored:  bool=False,
    ):  
        total_cycles = 0
        
        if preload_psum:
            total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
                
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
                    
        return total_cycles
    
    def mxu_tiled_elemwise(
        self,
        
        op: MXUElementwiseOp,
        src:  DataContainer[torch.Tensor],
        dst:  DataContainer[torch.Tensor],

        preload_psum:   bool=False,
        flush_ofm:      bool=False,
        ifm_transposed: bool=False,
    ):
        total_cycles = 0
        
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
                    
        return total_cycles
    
    def mxu_tiled_elemwise_imm(
        self,
        
        op: MXUElementwiseOp,
        imm: int | float,
        dst:  DataContainer[torch.Tensor],
        
        flush_ofm: bool=False,
    ):
        total_cycles = 0
        
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
                    
        return total_cycles
    
    def vpu_load_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1, offset: int=0):
        return burst_len  # TODO: Assume that loading one vector register takes 1 cycle
        
    def vpu_store_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1, offset: int=0):
        return burst_len  # TODO: Assume that storing one vector register takes 1 cycle
    
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if opcode.is_unary:
            return self.core.vpu_context.unary_op_latency * burst_len
        else:
            return self.core.vpu_context.arith_op_latency * burst_len

