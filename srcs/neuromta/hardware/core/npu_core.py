import torch

from neuromta.framework import *

from neuromta.hardware.context.mem_context import MemContext
from neuromta.hardware.context.cmap_context import CmapContext
from neuromta.hardware.context.icnt_context import IcntContext
from neuromta.hardware.context.vpu_context import VPUConfig, VPUOperator
from neuromta.hardware.context.mxu_context import MXUConfig, MXUDataflow


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

    #############################################################
    # Variable Management
    #############################################################
    
    @core_command_method
    def var_allocate(self, ptr: Pointer, initial_value: int=0):
        r = self.mem_handle.allocate_var_ptr(var_size=4, initial_value=initial_value, channel_id=0, dst_ptr=ptr)
        if r is None:
            raise Exception(f"[ERROR] Variable allocation failed in core {self.core_id} (not enough memory)")
    
    @core_command_method
    def var_deallocate(self, ptr: Pointer):
        self.mem_handle.deallocate_ptr(ptr)
        ptr.initialize()
        
    @core_conditional_command_method
    def var_compare_and_swap(self, ptr: Pointer, cmp_value: int | Pointer, new_value: int | Pointer):
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
    def var_atomic_increase(self, ptr: Pointer, value: int | Pointer=1):
        if isinstance(value, Pointer):
            value = self.mem_handle.get_data_element(value).content

        var: Variable = self.mem_handle.get_data_element(ptr)
        var.content += value
        
    #############################################################
    # Buffer Management
    #############################################################
      
    @core_command_method
    def buf_allocate(self, ref: BufferPointer, page_size: int, n_pages: int):
        handle = self.mem_handle.allocate_buffer_ptr(page_size=page_size, n_pages=n_pages, is_circular=False, channel_id=0)
        if handle is None:
            raise Exception(f"[ERROR] Buffer allocation failed in core {self.core_id} (not enough memory)")
        ref.initialize(handle=handle)
    
    @core_command_method
    def buf_deallocate(self, ref: BufferPointer):
        handle: BufferHandle = ref.raw_handle
        self.mem_handle.deallocate_ptr(handle)
        ref.initialize()
        
    #############################################################
    # Circular Buffer Management (Intra-Core Control Management)
    #############################################################
    
    @core_command_method
    def cb_allocate(self, ref: BufferPointer, page_size: int, n_pages: int):
        handle = self.mem_handle.allocate_buffer_ptr(page_size=page_size, n_pages=n_pages, is_circular=True, channel_id=0)
        if handle is None:
            raise Exception(f"[ERROR] Circular buffer allocation failed in core {self.core_id} (not enough memory)")
        ref.initialize(handle=handle)
    
    @core_command_method
    def cb_deallocate(self, ref: BufferPointer):
        handle: CircularBufferHandle = ref.raw_handle
        self.mem_handle.deallocate_ptr(handle)
        ref.initialize()
        
    @core_conditional_command_method
    def cb_reserve_back(self, ref: BufferPointer, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        if not handle.check_vacancy(n_pages):
            return False
        handle.allocate_cb_space(n_pages)
        return True
        
    @core_command_method
    def cb_push_back(self, ref: BufferPointer, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        handle.occupy_cb_space(n_pages)
        
    @core_conditional_command_method
    def cb_wait_front(self, ref: BufferPointer, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        return handle.check_occupancy(n_pages)
    
    @core_command_method
    def cb_pop_front(self, ref: BufferPointer, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        handle.deallocate_cb_space(n_pages)
        
    @core_conditional_command_method
    def cb_check_full(self, ref: BufferPointer):
        handle: CircularBufferHandle = ref.raw_handle
        return handle.check_occupancy(handle.n_pages)
    
    @core_conditional_command_method
    def cb_check_empty(self, ref: BufferPointer):
        handle: CircularBufferHandle = ref.raw_handle
        return handle.check_vacancy(handle.n_pages)
        
    #############################################################
    # Memory Copy Commands
    #############################################################
    
    @core_command_method
    def init_container(self, container: DataContainer, shape: tuple[int, ...], dtype: torch.dtype):
        container.data = torch.zeros(shape, dtype=dtype)
    
    @core_command_method
    def mem_read_with_container(self, handle: BufferPointer | Pointer, container: DataContainer, offset: int=0, size: int=None, shape: tuple[int, ...]=(-1,), dtype: torch.dtype=torch.uint8):
        if isinstance(handle, BufferPointer):
            handle = handle.resolve(is_read=True)
        
        if size is None:
            size = handle.size
            
        data: torch.Tensor = self.mem_handle.get_content(handle, shape=(-1,), dtype=torch.uint8)[offset:offset+size]
        container.data = data.view(dtype=dtype).reshape(shape)

    @core_command_method
    def mem_write_with_container(self, handle: BufferPointer | Pointer, container: DataContainer, offset: int=0):
        if isinstance(handle, BufferPointer):
            rd_handle = handle.resolve(is_read=True)
            wr_handle = handle.resolve(is_read=False)
        else:
            rd_handle = handle
            wr_handle = handle
            
        raw_data: torch.Tensor = self.mem_handle.get_content(rd_handle, shape=(-1,), dtype=torch.uint8)
        wr_data:  torch.Tensor = container.data.reshape(-1).view(torch.uint8)
        
        raw_data[offset:offset+wr_data.numel()] = wr_data
        self.mem_handle.set_content(wr_handle, raw_data)
        
    @core_command_method
    def local_memcopy_page(self, dst_ref: BufferPointer, src_ref: BufferPointer, page_offset: int=0):
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)
        
        dst_ptr = dst_handle.page_ptrs[page_offset]
        src_ptr = src_handle.page_ptrs[page_offset]
        
        if self.cmap_context.get_mem_owner_core_id(self.core_id, dst_ptr.addr) != self.core_id:
            raise Exception(f"[ERROR] Invalid destination memory address {dst_ptr.addr} in core {self.core_id}")
        if self.cmap_context.get_mem_owner_core_id(self.core_id, src_ptr.addr) != self.core_id:
            raise Exception(f"[ERROR] Invalid source memory address {src_ptr.addr} in core {self.core_id}")
        
        content = self.mem_handle.get_content(src_ptr)
        self.mem_handle.set_content(dst_ptr, content)

    def mem_page_copy(self, dst_ref: BufferPointer, src_ref: BufferPointer, page_offset: int=0):
        container = DataContainer()
        
        if dst_ref.is_circular:
            dst_owner_id = self.core_id
            dst_mem_type = "L1"
        else:
            dst_ptr = dst_ref.resolve(is_read=False).page_ptrs[page_offset]
            dst_owner_id = self.cmap_context.get_mem_owner_core_id(self.core_id, dst_ptr.addr)
            
            if   self.cmap_context.config.check_l1_mem_addr(dst_ptr.addr):   dst_mem_type = "L1"
            elif self.cmap_context.config.check_main_mem_addr(dst_ptr.addr): dst_mem_type = "MAIN"
            else: raise Exception(f"[ERROR] Invalid destination memory address {dst_ptr.addr} in core {self.core_id}")
            
        if src_ref.is_circular:
            src_owner_id = self.core_id
            src_mem_type = "L1"
        else:
            src_ptr = src_ref.resolve(is_read=True).page_ptrs[page_offset]
            src_owner_id = self.cmap_context.get_mem_owner_core_id(self.core_id, src_ptr.addr)
            
            if   self.cmap_context.config.check_l1_mem_addr(src_ptr.addr):   src_mem_type = "L1"
            elif self.cmap_context.config.check_main_mem_addr(src_ptr.addr): src_mem_type = "MAIN"
            else: raise Exception(f"[ERROR] Invalid source memory address {src_ptr.addr} in core {self.core_id}")
        
        if src_owner_id == dst_owner_id == self.core_id:
            self.local_memcopy_page(dst_ref, src_ref, page_offset=page_offset)
        else:
            if dst_mem_type == "L1":
                dst_write_msg = RPCMessage(self.core_id, dst_owner_id, cmd_id="mem_write_with_container").with_args(handle=dst_ref[page_offset], container=container, offset=0)
            elif dst_mem_type == "MAIN":
                dst_write_msg = RPCMessage(self.core_id, dst_owner_id, cmd_id="mem_page_write").with_args(ptr=dst_ref[page_offset], container=container)
        
            if src_mem_type == "L1":
                src_read_msg = RPCMessage(self.core_id, src_owner_id, cmd_id="mem_read_with_container").with_args(handle=src_ref[page_offset], container=container, offset=0, size=src_ref.size)
            elif src_mem_type == "MAIN":
                src_read_msg = RPCMessage(self.core_id, src_owner_id, cmd_id="mem_page_read").with_args(ptr=src_ref[page_offset], container=container)
                
            noc_transaction_msgs = []
        
            if self.check_rpc_inbox(self.cmap_context.icnt_core_id):  # check if it is possible to send NOC transaction request (if not, the )
                if src_owner_id != self.core_id:
                    noc_read_msg = RPCMessage(self.core_id, self.cmap_context.icnt_core_id, cmd_id="noc_create_data_read_transaction").with_args(src_id=src_owner_id, dst_id=self.core_id, data_size=src_ref[page_offset].page_size)
                    noc_transaction_msgs.append(noc_read_msg)
                if dst_owner_id != self.core_id:
                    noc_write_msg = RPCMessage(self.core_id, self.cmap_context.icnt_core_id, cmd_id="noc_create_data_write_transaction").with_args(src_id=self.core_id, dst_id=dst_owner_id, data_size=dst_ref[page_offset].page_size)
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
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")    
        
        if preload_wgt:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                wgt_tile = wgt_cont.data.view(self.mxu_context.dtype).reshape(self.mxu_context.wgt_tile_shape)
                if wgt_transposed: wgt_tile = wgt_tile.T
                self.mxu_context.load_tile_pe_arr(wgt_tile)

        if self.mxu_context.dataflow == MXUDataflow.OS:
            ifm_tile = ifm_cont.data.view(self.mxu_context.dtype).reshape(self.mxu_context.ifm_tile_shape)
            wgt_tile = wgt_cont.data.view(self.mxu_context.dtype).reshape(self.mxu_context.wgt_tile_shape)
            
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

    #############################################################
    # VPU Commands
    #############################################################
        
    @core_command_method
    def vpu_reconfigure(self, vlen: int, vdtype: torch.dtype):
        self.vpu_context.reconfigure_vector_reg_file(vlen=vlen, vdtype=vdtype)
        
    @core_command_method
    def vpu_load_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        data = data_cont.data.view(self.vpu_context.vdtype).reshape(-1)
        
        for i in range(burst_len):
            st = i * self.vpu_context.vlen
            ed = (i + 1) * self.vpu_context.vlen
            vreg_data = data[st:ed]
            self.vpu_context.set_vector_reg(vreg_idx + i, vreg_data)
        
    @core_command_method
    def vpu_store_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)

        data: list[torch.Tensor] = []
        
        for i in range(burst_len):
            vreg_data = self.vpu_context.get_vector_reg(vreg_idx + i)
            data.append(vreg_data)
            
        data_cont.data = torch.cat(data, dim=0)

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

    def local_memcopy_page(self, dst_ref: BufferPointer, src_ref: BufferPointer, page_offset: int=0):
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
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")

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
    
    def vpu_load_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1):
        return burst_len  # TODO: Assume that loading one vector register takes 1 cycle
        
    def vpu_store_reg(self, data_cont: DataContainer[torch.Tensor], vreg_idx: int, burst_len: int=1):
        return burst_len  # TODO: Assume that storing one vector register takes 1 cycle
    
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if opcode.is_unary:
            return self.core.vpu_context.unary_op_latency * burst_len
        else:
            return self.core.vpu_context.arith_op_latency * burst_len

