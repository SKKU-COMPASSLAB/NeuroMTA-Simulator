import torch

from neuromta.framework import *

from neuromta.component.context.mem_context import MemContext
from neuromta.component.context.cmap_context import CmapContext
from neuromta.component.context.icnt_context import IcntContext
from neuromta.component.context.vpu_context import VPUConfig, VPUOperator
from neuromta.component.context.mxu_context import MXUConfig, MXUDataflow, MXUElementwiseOp


__all__ = [
    "NPUCore",
] 


class NPUCore(Core):
    def __init__(
        self,
        core_id: int,
        mem_context: MemContext, 
        cmap_context: CmapContext,
        vpu_config: VPUConfig = VPUConfig(),
        mxu_config: MXUConfig = MXUConfig(),
    ):
        super().__init__(
            core_id=core_id,  # coordinate is the core ID (it is guaranteed that each core is assigned with a unique coordinate in the given core grid!)
            cycle_model=NPUCoreCycleModel(core=self),
        )
        
        self.mem_context = mem_context
        self.cmap_context = cmap_context
        
        self.mxu_context = mxu_config.create_context()
        self.vpu_context = vpu_config.create_context()
        
        self.mem_handle = MemoryHandle(
            mem_id=self.core_id.__str__(), 
            base_addr=self.cmap_context.get_base_addr_from_core_id(self.core_id), 
            size=self.cmap_context.config.l1_spm_bank_size
        )
        
        # synchronization variables
        self.ongoing_core_sync_msg: list[int] = []
        
        
    #############################################################
    # Inter-Core Synchronization
    #############################################################
    
    @core_command_method
    def _static_inter_core_sync_send_msg(self, dst_core_id: int):
        pass
    
    def inter_core_sync_send_msg(self, dst_core_id: int):
        if self.check_rpc_inbox(self.cmap_context.icnt_core_id):  # check if it is possible to send NOC transaction request (if not, the )
            noc_write_msg = RPCMessage(self.core_id, self.cmap_context.icnt_core_id, cmd_id="noc_create_data_write_transaction").with_args(src_id=self.core_id, dst_id=dst_core_id, data_size=2)
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
    # Variable Management
    #############################################################
    
    @core_command_method
    def var_allocate(self, ptr: Pointer, initial_value: int=0):
        r = self.mem_handle.allocate_var_ptr(var_size=4, initial_value=initial_value, channel_id=0, dst_ptr=ptr)
        if r is None:
            raise Exception(f"Variable allocation failed in core {self.core_id} (not enough memory)")
    
    @core_command_method
    def var_deallocate(self, ptr: Pointer):
        self.mem_handle.deallocate_ptr(ptr)
        ptr.initialize()
        
    @core_conditional_command_method
    def local_var_compare_and_swap(self, ptr: Pointer, cmp_value: int | Pointer, new_value: int | Pointer):
        if isinstance(cmp_value, Pointer):
            cmp_value = self.mem_handle.get_data_element(cmp_value).content
        if isinstance(new_value, Pointer):
            new_value = self.mem_handle.get_data_element(new_value).content

        var: Variable = self.mem_handle.get_data_element(ptr)
        if var.content == cmp_value:
            var.content = new_value
            return True
        else:
            return False
        
    @core_command_method
    def local_var_atomic_increase(self, ptr: Pointer, value: int | Pointer=1):
        if isinstance(value, Pointer):
            value = self.mem_handle.get_data_element(value).content

        var: Variable = self.mem_handle.get_data_element(ptr)
        var.content += value
        
    @core_conditional_command_method
    def local_var_wait_value(self, ptr: Pointer, value: int | Pointer):
        if isinstance(value, Pointer):
            value = self.mem_handle.get_data_element(value).content

        var: Variable = self.mem_handle.get_data_element(ptr)
        return var.content == value
    
    @jit_prototype
    def var_compare_and_swap(self, ptr: Pointer, cmp_value: int | Pointer, new_value: int | Pointer):
        buffer_owners = self.cmap_context.get_buffer_owner_core_ids(self.core_id, ptr)
        if not len(buffer_owners) == 1:
            raise Exception(f"The variable address {ptr} is not owned by a single core (owners: {buffer_owners}) in core {self.core_id}")
        
        owner_id = buffer_owners[0]
        
        if owner_id != self.core_id:
            var_cas_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_var_compare_and_swap"
            ).with_args(ptr=ptr, cmp_value=cmp_value, new_value=new_value)
            
            self.async_rpc_send_req_msg(var_cas_msg)
            self.async_rpc_wait_rsp_msg(var_cas_msg)
        else:
            self.local_var_compare_and_swap(ptr, cmp_value=cmp_value, new_value=new_value)
    
    @jit_prototype
    def var_atomic_increase(self, ptr: Pointer, value: int | Pointer=1):
        buffer_owners = self.cmap_context.get_buffer_owner_core_ids(self.core_id, ptr)
        if not len(buffer_owners) == 1:
            raise Exception(f"The variable address {ptr} is not owned by a single core (owners: {buffer_owners}) in core {self.core_id}")
        
        owner_id = buffer_owners[0]
        
        if owner_id != self.core_id:
            var_inc_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_var_atomic_increase"
            ).with_args(ptr=ptr, value=value)
            
            self.async_rpc_send_req_msg(var_inc_msg)
            self.async_rpc_wait_rsp_msg(var_inc_msg)
        else:
            self.local_var_atomic_increase(ptr, value=value)
            
    @jit_prototype
    def var_wait_value(self, ptr: Pointer, value: int | Pointer):
        buffer_owners = self.cmap_context.get_buffer_owner_core_ids(self.core_id, ptr)
        if not len(buffer_owners) == 1:
            raise Exception(f"The variable address {ptr} is not owned by a single core (owners: {buffer_owners}) in core {self.core_id}")
        
        owner_id = buffer_owners[0]
        
        if owner_id != self.core_id:
            var_wait_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_var_wait_value"
            ).with_args(ptr=ptr, value=value)
            
            self.async_rpc_send_req_msg(var_wait_msg)
            self.async_rpc_wait_rsp_msg(var_wait_msg)
        else:
            self.local_var_wait_value(ptr, value=value)
        
    #############################################################
    # Buffer Management
    #############################################################
      
    @core_command_method
    def buf_allocate(self, ref: BufferPointer, page_size: int, n_pages: int):
        handle = self.mem_handle.allocate_buffer_ptr(page_size=page_size, n_pages=n_pages, channel_id=0)
        if handle is None:
            raise Exception(f"Buffer allocation failed in core {self.core_id} (not enough memory)")
        ref.initialize(handle=handle)
    
    @core_command_method
    def buf_deallocate(self, ref: BufferPointer):
        handle: BufferHandle = ref.raw_handle
        self.mem_handle.deallocate_ptr(handle)
        ref.initialize()
        
    #############################################################
    # Memory Copy Commands
    #############################################################
    
    @core_command_method
    def mem_container_init(self, container: DataContainer, tensor: torch.Tensor=None, shape: tuple[int,...]=None, dtype: torch.dtype=torch.uint8):
        if isinstance(shape, int):
            shape = (shape,)
        
        if tensor is not None:
            container.data = tensor
        else:
            container.data = torch.zeros(shape, dtype=dtype)

    @core_command_method
    def local_mem_read_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer[torch.Tensor], offset: int=0, size: int=None, copy_layout_width: int=None, copy_layout_pattern: list[tuple[int, int, int, int, int]]=None):
        if isinstance(ptr, BufferPointer):
            handle = ptr.raw_handle

        if size is None:
            size = handle.size - offset
        if container.data is None:
            self.mem_container_init(container, shape=size, dtype=torch.uint8)
        
        cont_data: torch.Tensor = container.data.reshape(-1).view(torch.uint8)
        mem_data: torch.Tensor = self.mem_handle.get_content(handle, shape=(-1,), dtype=torch.uint8)[offset:offset+size].flatten()
        
        if copy_layout_pattern is not None:
            if copy_layout_width is None:
                raise Exception(f"'copy_layout_width' must be specified when 'copy_layout_pattern' is given in core {self.core_id}")
            
            dst_data = cont_data.reshape(-1, copy_layout_width)
            src_data = mem_data.reshape(-1, copy_layout_width)
            
            for dst_idx, src_idx, dst_seg_offset, src_seg_offset, seg_size in copy_layout_pattern:
                dst_data[dst_idx][dst_seg_offset:dst_seg_offset+seg_size] = src_data[src_idx][src_seg_offset:src_seg_offset+seg_size]
                        
            cont_data = dst_data
        else:
            cont_data[:size] = mem_data

        container.data = cont_data
        
    @core_command_method
    def local_mem_write_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer[torch.Tensor], offset: int=0, size: int=None):
        if isinstance(ptr, BufferPointer):
            handle = ptr.raw_handle
        else:
            handle = ptr
            
        if size is None:
            size = handle.size - offset
        if container.data is None:
            self.mem_container_init(container, shape=size, dtype=torch.uint8)
        
        raw_data: torch.Tensor = self.mem_handle.get_content(handle, shape=(-1,), dtype=torch.uint8)
        wr_data:  torch.Tensor = container.data.reshape(-1).view(torch.uint8)
        
        wr_data = wr_data[:size]
    
        raw_data[offset:offset+wr_data.numel()] = wr_data
        self.mem_handle.set_content(handle, raw_data)
    
    @jit_prototype
    def mem_read_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer, offset: int=0, size: int=None, copy_layout_width: int=None, copy_layout_pattern: list[tuple[int, int, int, int, int]]=None):
        buffer_owners = self.cmap_context.get_buffer_owner_core_ids(self.core_id, ptr)
        if not len(buffer_owners) == 1:
            raise Exception(f"The memory read address {ptr} is not owned by a single core (owners: {buffer_owners}) in core {self.core_id}")
        
        owner_id = buffer_owners[0]
        
        if owner_id != self.core_id:
            mem_read_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_mem_read_with_container"
            ).with_args(ptr=ptr, container=container, offset=offset, size=size, copy_layout_width=copy_layout_width, copy_layout_pattern=copy_layout_pattern)
            
            noc_read_msg = RPCMessage(
                self.core_id, 
                self.cmap_context.icnt_core_id, 
                cmd_id="noc_create_data_read_transaction"
            ).with_args(src_id=owner_id, dst_id=self.core_id, data_size=ptr.size)
            
            self.async_rpc_send_req_msg(mem_read_msg)
            self.async_rpc_wait_rsp_msg(mem_read_msg)
            
            self.async_rpc_send_req_msg(noc_read_msg)
            self.async_rpc_wait_rsp_msg(noc_read_msg)
        else:
            self.local_mem_read_with_container(ptr, container, offset=offset, size=size, copy_layout_width=copy_layout_width, copy_layout_pattern=copy_layout_pattern)

    @jit_prototype
    def mem_write_with_container(self, ptr: BufferPointer | Pointer, container: DataContainer, offset: int=0, size: int=None):            
        buffer_owners = self.cmap_context.get_buffer_owner_core_ids(self.core_id, ptr)
        if not len(buffer_owners) == 1:
            raise Exception(f"The memory read address {ptr} is not owned by a single core (owners: {buffer_owners}) in core {self.core_id}")

        owner_id = buffer_owners[0]
        
        if owner_id != self.core_id:
            noc_write_msg = RPCMessage(
                self.core_id, 
                self.cmap_context.icnt_core_id, 
                cmd_id="noc_create_data_write_transaction"
            ).with_args(src_id=self.core_id, dst_id=owner_id, data_size=ptr.size)
            
            mem_write_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=owner_id,
                cmd_id="local_mem_write_with_container"
            ).with_args(ptr=ptr, container=container, offset=offset, size=size)
            
            self.async_rpc_send_req_msg(noc_write_msg)
            self.async_rpc_wait_rsp_msg(noc_write_msg)
            
            self.async_rpc_send_req_msg(mem_write_msg)
            self.async_rpc_wait_rsp_msg(mem_write_msg)
        else:
            self.local_mem_write_with_container(ptr, container, offset=offset, size=size)
        
    @core_command_method
    def mem_concat_containers(self, dst_cont: DataContainer[torch.Tensor], src_conts: list[DataContainer[torch.Tensor]]):
        data_list = [cont.data.reshape(-1).view(torch.uint8) for cont in src_conts]
        dst_cont.data = torch.cat(data_list, dim=0)
        
    @core_command_method
    def local_mem_page_copy(self, dst_ref: BufferPointer, src_ref: BufferPointer, page_offset: int=0):
        dst_handle = dst_ref.raw_handle
        src_handle = src_ref.raw_handle
        
        dst_ptr = dst_handle.page_ptrs[page_offset]
        src_ptr = src_handle.page_ptrs[page_offset]
        
        if self.cmap_context.get_mem_owner_core_id(self.core_id, dst_ptr.addr) != self.core_id:
            raise Exception(f"Invalid destination memory address {dst_ptr.addr} in core {self.core_id}")
        if self.cmap_context.get_mem_owner_core_id(self.core_id, src_ptr.addr) != self.core_id:
            raise Exception(f"Invalid source memory address {src_ptr.addr} in core {self.core_id}")
        
        content = self.mem_handle.get_content(src_ptr)
        self.mem_handle.set_content(dst_ptr, content)

    @jit_prototype
    def mem_page_copy(self, dst_ptr: BufferPointer, src_ptr: BufferPointer, page_offset: int=0):
        container = DataContainer()
        
        dst_page_ptr = dst_ptr.raw_handle.page_ptrs[page_offset]
        dst_page_owner_id = self.cmap_context.get_mem_owner_core_id(self.core_id, dst_page_ptr.addr)
        
        src_page_ptr = src_ptr.raw_handle.page_ptrs[page_offset]
        src_page_owner_id = self.cmap_context.get_mem_owner_core_id(self.core_id, src_page_ptr.addr)
        
        if src_page_owner_id == dst_page_owner_id == self.core_id:
            self.local_mem_page_copy(dst_ptr, src_ptr, page_offset=page_offset)
        else:
            if self.cmap_context.config.check_l1_mem_addr(src_page_ptr.addr):
                src_read_msg = RPCMessage(self.core_id, src_page_owner_id, cmd_id="local_mem_read_with_container").with_args(ptr=src_ptr[page_offset], container=container, offset=0, size=src_ptr.page_size)
            elif self.cmap_context.config.check_main_mem_addr(src_page_ptr.addr):
                src_read_msg = RPCMessage(self.core_id, src_page_owner_id, cmd_id="mem_page_read").with_args(ptr=src_ptr[page_offset], container=container)
            
            if self.cmap_context.config.check_l1_mem_addr(dst_page_ptr.addr):
                dst_write_msg = RPCMessage(self.core_id, dst_page_owner_id, cmd_id="local_mem_write_with_container").with_args(ptr=dst_ptr[page_offset], container=container, offset=0)
            elif self.cmap_context.config.check_main_mem_addr(dst_page_ptr.addr):
                dst_write_msg = RPCMessage(self.core_id, dst_page_owner_id, cmd_id="mem_page_write").with_args(ptr=dst_ptr[page_offset], container=container)
                
            noc_transaction_msgs = []
        
            if self.check_rpc_inbox(self.cmap_context.icnt_core_id):
                if src_page_owner_id != self.core_id:
                    noc_read_msg = RPCMessage(self.core_id, self.cmap_context.icnt_core_id, cmd_id="noc_create_data_read_transaction").with_args(src_id=src_page_owner_id, dst_id=self.core_id, data_size=src_ptr[page_offset].page_size)
                    noc_transaction_msgs.append(noc_read_msg)
                if dst_page_owner_id != self.core_id:
                    noc_write_msg = RPCMessage(self.core_id, self.cmap_context.icnt_core_id, cmd_id="noc_create_data_write_transaction").with_args(src_id=self.core_id, dst_id=dst_page_owner_id, data_size=dst_ptr[page_offset].page_size)
                    noc_transaction_msgs.append(noc_write_msg)
            
            self.async_rpc_send_req_msg(src_read_msg)
            self.async_rpc_wait_rsp_msg(src_read_msg)
            
            for msg in noc_transaction_msgs:
                self.async_rpc_send_req_msg(msg)
                self.async_rpc_wait_rsp_msg(msg)
                
            self.async_rpc_send_req_msg(dst_write_msg)
            self.async_rpc_wait_rsp_msg(dst_write_msg)

    def mem_buffer_copy(self, dst_ref: BufferPointer, src_ref: BufferPointer, n_pages: int):
        for i in range(n_pages):
            with new_parallel_thread(f"PAGE{i}"):
                self.mem_page_copy(dst_ref, src_ref, page_offset=i)

        self.parallel_merge()

    #############################################################
    # MXU Commands
    #############################################################
        
    @core_command_method
    def mxu_reconfigure(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self.mxu_context.reconfigure_dtype(dtype=dtype, acc_dtype=acc_dtype)
    
    @core_command_method
    def mxu_tiled_gemm(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        wgt_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_wgt:   bool=False,
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        wgt_transposed: bool=False,
        psum_vectored:  bool=False,
    ):  
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                if psum_cont.data is None:
                    psum_tile = torch.zeros(self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
                else:
                    psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype)
                    
                    if psum_vectored:
                        psum_tile = psum_tile.flatten().unsqueeze(0)  # (1, N)
                    else:
                        psum_tile = psum_tile.reshape(self.mxu_context.ofm_tile_shape)
                        
                    if ifm_transposed: 
                        psum_tile = psum_tile.T
                    
                self.mxu_context.load_tile_pe_arr(psum_tile)
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"PSUM preload is not supported in WS dataflow")    
        
        if preload_wgt:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                wgt_tile = wgt_cont.data.view(self.mxu_context.dtype).reshape(self.mxu_context.wgt_tile_shape)
                if wgt_transposed: wgt_tile = wgt_tile.T
                self.mxu_context.load_tile_pe_arr(wgt_tile)

        if self.mxu_context.dataflow == MXUDataflow.OS:
            ifm_tile = ifm_cont.data.view(self.mxu_context.dtype)  #.reshape(self.mxu_context.ifm_tile_shape)
            wgt_tile = wgt_cont.data.view(self.mxu_context.dtype)  #.reshape(self.mxu_context.wgt_tile_shape)
            
            if self.mxu_context.ifm_tile_numel != ifm_tile.numel():
                ifm_tile = torch.nn.functional.pad(ifm_tile, (0, self.mxu_context.ifm_tile_numel - ifm_tile.numel()), 'constant', 0)
            if self.mxu_context.wgt_tile_numel != wgt_tile.numel():
                wgt_tile = torch.nn.functional.pad(wgt_tile, (0, self.mxu_context.wgt_tile_numel - wgt_tile.numel()), 'constant', 0)
                
            ifm_tile = ifm_tile.reshape(self.mxu_context.ifm_tile_shape)
            wgt_tile = wgt_tile.reshape(self.mxu_context.wgt_tile_shape)
            
            if ifm_transposed: ifm_tile = ifm_tile.T
            if wgt_transposed: wgt_tile = wgt_tile.T

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=wgt_tile)

        elif self.mxu_context.dataflow == MXUDataflow.WS:
            ifm_tile = ifm_cont.data.view(self.mxu_context.dtype).reshape(self.mxu_context.ifm_tile_shape)
            
            if ifm_transposed: 
                ifm_tile = ifm_tile.T
            
            if psum_cont.data is None:
                psum_tile = torch.zeros(self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype)
                
                if psum_vectored:
                    psum_tile = psum_tile.flatten().unsqueeze(1)  # (N, 1)
                else:
                    psum_tile = psum_tile.reshape(self.mxu_context.ofm_tile_shape)
                    
                if ifm_transposed: 
                    psum_tile = psum_tile.T

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, psum_tile=psum_tile)
            
        if flush_ofm:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                ofm_tile = self.mxu_context.get_pe_arr_regs()   
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                ofm_tile = self.mxu_context.get_acc_regs() 
                
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
            if self.mxu_context.dataflow == MXUDataflow.OS:
                if psum_cont.data is None:
                    psum_tile = torch.zeros(self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
                else:
                    psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ofm_tile_shape)
                
                self.mxu_context.load_tile_pe_arr(psum_tile)
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"PSUM preload is not supported in WS dataflow") 
        
        if self.mxu_context.dataflow == MXUDataflow.OS:
            ifm_tile = ifm_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ofm_tile_shape)
            if ifm_transposed: ifm_tile = ifm_tile.T
            self.mxu_context.execute_maxpool(ifm_tile=ifm_tile)
        elif self.mxu_context.dataflow == MXUDataflow.WS:
            ifm_tile = ifm_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ofm_tile_shape)
            if ifm_transposed: ifm_tile = ifm_tile.T
            
            if psum_cont.data is None:
                psum_tile = torch.zeros(self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ofm_tile_shape)
                    
            self.mxu_context.execute_maxpool(ifm_tile=ifm_tile, psum_tile=psum_tile)
        
        if flush_ofm:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                ofm_tile = self.mxu_context.get_pe_arr_regs()   
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                ofm_tile = self.mxu_context.get_acc_regs()

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
                psum_tile = torch.zeros(self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = dst.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ofm_tile_shape)
                    
            if self.mxu_context.dataflow == MXUDataflow.OS:
                self.mxu_context.load_tile_pe_arr(psum_tile)
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                self.mxu_context.load_tile_acc_regs(psum_tile) 

        ifm_tile = src.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.ifm_tile_shape)
        if ifm_transposed: ifm_tile = ifm_tile.T
        
        self.mxu_context.execute_elemwise(ifm_tile=ifm_tile, op=op)
        
        if flush_ofm:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                ofm_tile = self.mxu_context.get_pe_arr_regs()   
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                ofm_tile = self.mxu_context.get_acc_regs()

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
        
        data = data_cont.data.view(self.vpu_context.vdtype).reshape(-1)
        
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
        if not self.core.check_rpc_inbox(self.core.cmap_context.icnt_core_id):
            return 10  # TODO: Assume that sending a message takes 10 cycles
        return 1
    
    def _static_inter_core_sync_send_rsp_msg(self, dst_core_id: int):
        if not self.core.check_rpc_inbox(self.core.cmap_context.icnt_core_id):
            return 10  # TODO: Assume that sending a message takes 10 cycles
        return 1

    def local_mem_page_copy(self, dst_ref: BufferPointer, src_ref: BufferPointer, page_offset: int=0):
        return self.core.mem_context.l1_config.get_cycles(size=src_ref.page_size)
    
    def mxu_tiled_gemm(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        wgt_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor],
        ofm_cont:  DataContainer[torch.Tensor],
        
        preload_wgt:   bool=False,
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        wgt_transposed: bool=False,
        psum_vectored:  bool=False,
    ):
        total_cycles = 0
        
        if preload_psum:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"PSUM preload is not supported in WS dataflow")

        if preload_wgt:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
                
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_flush_acc_regs_cycles()
                    
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
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"PSUM preload is not supported in WS dataflow")
                
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_flush_acc_regs_cycles()
                    
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
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_flush_acc_regs_cycles()
                    
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

