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
        self.mem_handle.allocate_var_ptr(var_size=4, initial_value=initial_value, channel_id=0, dst_ptr=ptr)
    
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
        
    #############################################################
    # Circular Buffer Management (Intra-Core Control Management)
    #############################################################

    @core_conditional_command_method
    def cb_reserve_back(self, ref: Reference, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        if not handle.check_vacancy(n_pages):
            return False
        handle.allocate_cb_space(n_pages)
        return True
        
    @core_command_method
    def cb_push_back(self, ref: Reference, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        handle.occupy_cb_space(n_pages)
        
    @core_conditional_command_method
    def cb_wait_front(self, ref: Reference, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        return handle.check_occupancy(n_pages)
    
    @core_command_method
    def cb_pop_front(self, ref: Reference, n_pages: int):
        handle: CircularBufferHandle = ref.raw_handle
        handle.deallocate_cb_space(n_pages)
        
    #############################################################
    # Buffer Management
    #############################################################
        
    @core_command_method
    def local_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")

        if dst_ptr.size != src_ptr.size:
            raise ValueError(f"[ERROR] Page sizes do not match: {dst_ptr.size} != {src_ptr.size}")

        if not self.use_functional_model:
            return  # Terminate the operation if not using functional model

        dst_elem = self.mem_handle.get_data_element(dst_ptr)
        src_elem = self.mem_handle.get_data_element(src_ptr)

        dst_elem.copy_from(src_elem)

    @core_kernel_method
    def local_memcopy_buffer(self, dst_ref: Reference, src_ref: Reference):  
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)

        for dst_ptr, src_ptr in zip(dst_handle.page_ptrs, src_handle.page_ptrs):
            self.local_memcopy_page(dst_ptr, src_ptr)
            
    #############################################################
    # Data Container (NoC Interface)
    #############################################################
    
    @core_command_method
    def mem_page_write(self, ptr: Pointer, container: DataContainer):
        if ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointer.")

        if not isinstance(container, DataContainer):
            raise ValueError("[ERROR] The source container must be a DataContainer instance.")

        if not self.use_functional_model:
            return  # Terminate the operation if not using functional model
        
        page_elem = self.mem_handle.get_data_element(ptr)
        page_elem.content = container.data

    @core_command_method
    def mem_page_read(self, ptr: Pointer, container: DataContainer):
        if ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointer.")

        if not isinstance(container, DataContainer):
            raise ValueError("[ERROR] The target container must be a DataContainer instance.")
        
        if not self.use_functional_model:
            return  # Terminate the operation if not using functional model

        page_elem = self.mem_handle.get_data_element(ptr)
        container.data = page_elem.content.clone()  # Copy the content of the page element to the container
        
    #############################################################
    # Remote Asynchronous Memory Access (without NoC)
    #############################################################
    
    @core_kernel_method
    def async_page_read(self, dst_ptr: Pointer, src_ptr: Pointer):
        src_owner_id = self.cmap_context.get_nxt_mem_core_id(self.core_id, src_ptr.addr)
        dst_owner_id = self.core_id
        
        container = DataContainer()

        mem_reader_msg = RPCMessage(
            src_core_id=dst_owner_id,   # source of the RPC message will be myself
            dst_core_id=src_owner_id,   # destination of the RPC message will be owner of the source pointer,
            cmd_id="mem_page_read",
        ).with_args(
            ptr=src_ptr,
            container=container
        ).with_callbacks(
            # self.mem_page_write(dst_ptr, container)  # load page from container
            method=self.mem_page_write, ptr=dst_ptr, container=container  # load page from container
        )
        
        self.async_rpc_send_req_msg(mem_reader_msg)
            
    @core_kernel_method
    def async_page_write(self, dst_ptr: Pointer, src_ptr: Pointer):
        src_owner_id = self.core_id
        dst_owner_id = self.cmap_context.get_nxt_mem_core_id(self.core_id, dst_ptr.addr)
        
        container = DataContainer()
        
        mem_writer_msg = RPCMessage(
            src_core_id=src_owner_id,   # source of the RPC message will be myself
            dst_core_id=dst_owner_id,   # destination of the RPC message will be owner of the source pointer,
            cmd_id="mem_page_write",
        ).with_args(
            ptr=dst_ptr,
            container=container
        )
        
        self.mem_page_read(src_ptr, container)
        self.async_rpc_send_req_msg(mem_writer_msg)
            
    @core_kernel_method
    def async_buffer_read(self, dst_ref: Reference, src_ref: Reference):
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)  

        for dst_ptr, src_ptr in zip(dst_handle.page_ptrs, src_handle.page_ptrs):
            with new_parallel_thread():
                self.async_page_read(dst_ptr, src_ptr)

    @core_kernel_method
    def async_buffer_write(self, dst_ref: Reference, src_ref: Reference):
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)  

        for dst_ptr, src_ptr in zip(dst_handle.page_ptrs, src_handle.page_ptrs):
            with new_parallel_thread():
                self.async_page_write(dst_ptr, src_ptr)
        
    #############################################################
    # Remote Asynchronous Memory Access (via NoC)
    #############################################################
        
    @core_kernel_method
    def async_noc_page_read(self, dst_ptr: Pointer, src_ptr: Pointer):
        icnt_core_id = self.cmap_context.icnt_core_id
        src_owner_id = self.cmap_context.get_nxt_mem_core_id(self.core_id, src_ptr.addr)
        dst_owner_id = self.core_id
        
        container = DataContainer()

        noc_trans_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=icnt_core_id,
            cmd_id="noc_create_data_read_transaction"
        ).with_args(
            src_id=src_owner_id,
            dst_id=dst_owner_id,
            data_size=dst_ptr.size,
        )
        
        mem_reader_msg = RPCMessage(
            src_core_id=self.core_id,   # source of the RPC message will be myself
            dst_core_id=src_owner_id,   # destination of the RPC message will be owner of the source pointer,
            cmd_id="mem_page_read",
        ).with_args(
            ptr=src_ptr,
            container=container
        ).with_callbacks(
            # self.mem_page_write(dst_ptr, container)  # load page from container
            method=self.mem_page_write, ptr=dst_ptr, container=container  # load page from container
        )
        
        # NOTE: The code below assumes that the memory access and NoC data transfer is done sequentially without any
        # pipelining. I think that this scenario is unrealistic since the real hardware may attempt to pipeline the 
        # data movement all the way through core->router->core.
        # TODO: Check whether the latency model implemented below is accurate.
        
        self.async_rpc_send_req_msg(mem_reader_msg)
        self.async_rpc_send_req_msg(noc_trans_msg)
            
    @core_kernel_method
    def async_noc_page_write(self, dst_ptr: Pointer, src_ptr: Pointer):
        icnt_core_id = self.cmap_context.icnt_core_id
        src_owner_id = self.core_id
        dst_owner_id = self.cmap_context.get_nxt_mem_core_id(self.core_id, dst_ptr.addr)
        
        container = DataContainer()
        
        noc_trans_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=icnt_core_id,
            cmd_id="noc_create_data_write_transaction"
        ).with_args(
            src_id=src_owner_id,
            dst_id=dst_owner_id,
            data_size=dst_ptr.size,
        )
        
        mem_writer_msg = RPCMessage(
            src_core_id=self.core_id,   # source of the RPC message will be myself
            dst_core_id=dst_owner_id,   # destination of the RPC message will be owner of the source pointer,
            cmd_id="mem_page_write",
        ).with_args(
            ptr=dst_ptr,
            container=container
        )
        
        # NOTE: The code below assumes that the memory access and NoC data transfer is done sequentially without any
        # pipelining. I think that this scenario is unrealistic since the real hardware may attempt to pipeline the 
        # data movement all the way through core->router->core.
        # TODO: Check whether the latency model implemented below is accurate.
        
        self.mem_page_read(src_ptr, container)
        
        self.async_rpc_send_req_msg(noc_trans_msg)
        self.async_rpc_send_req_msg(mem_writer_msg)
            
            
    @core_kernel_method
    def async_noc_buffer_read(self, dst_ref: Reference, src_ref: Reference):
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)  

        for dst_ptr, src_ptr in zip(dst_handle.page_ptrs, src_handle.page_ptrs):
            with new_parallel_thread():
                self.async_noc_page_read(dst_ptr, src_ptr)

    @core_kernel_method
    def async_noc_buffer_write(self, dst_ref: Reference, src_ref: Reference):
        dst_handle = dst_ref.resolve(is_read=False)
        src_handle = src_ref.resolve(is_read=True)  

        for dst_ptr, src_ptr in zip(dst_handle.page_ptrs, src_handle.page_ptrs):
            with new_parallel_thread():
                self.async_noc_page_write(dst_ptr, src_ptr)

    #############################################################
    # MXU Commands
    #############################################################
        
    @core_command_method
    def mxu_reconfigure(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self.mxu_context.reconfigure_dtype(dtype=dtype, acc_dtype=acc_dtype)
    
    @core_command_method
    def mxu_tiled_gemm(
        self, 
        ifm_ptr:  Reference,
        wgt_ptr:  Reference,
        psum_ptr: Reference,
        ofm_ptr:  Reference,
        preload_wgt:   bool,
        preload_psum:  bool,
        flush_ofm:     bool,
    ):  
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)

        if preload_psum:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                psum_tile = self.mem_handle.get_content(psum_ptr, shape=self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
                self.mxu_context.load_tile_pe_arr(psum_tile)
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")    
        
        if preload_wgt:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                wgt_tile = self.mem_handle.get_content(wgt_ptr, shape=self.mxu_context.wgt_tile_shape, dtype=self.mxu_context.dtype)
                self.mxu_context.load_tile_pe_arr(wgt_tile)

        if self.mxu_context.dataflow == MXUDataflow.OS:
            ifm_tile = self.mem_handle.get_content(ifm_ptr, shape=self.mxu_context.ifm_tile_shape, dtype=self.mxu_context.dtype)
            wgt_tile = self.mem_handle.get_content(wgt_ptr, shape=self.mxu_context.wgt_tile_shape, dtype=self.mxu_context.dtype)

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=wgt_tile)

        elif self.mxu_context.dataflow == MXUDataflow.WS:
            ifm_tile = self.mem_handle.get_content(ifm_ptr, shape=self.mxu_context.ifm_tile_shape, dtype=self.mxu_context.dtype)
            psum_tile = self.mem_handle.get_content(psum_ptr, shape=self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, psum_tile=psum_tile)
            
        if flush_ofm:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                psum_tile = self.mxu_context.get_pe_arr_regs()   
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                psum_tile = self.mxu_context.get_acc_regs() 
            
            self.mem_handle.set_content(ofm_ptr, psum_tile)

    #############################################################
    # VPU Commands
    #############################################################
        
    @core_command_method
    def vpu_reconfigure(self, vlen: int, vdtype: torch.dtype):
        self.vpu_context.reconfigure_vector_reg_file(vlen=vlen, vdtype=vdtype)
        
    @core_command_method
    def vpu_load_reg(self, ptr: Pointer, ptr_offset: int, vreg_idx: int, burst_len: int=1):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        for i in range(burst_len):
            st = ptr_offset + i * self.vpu_context.vlen
            ed = ptr_offset + (i + 1) * self.vpu_context.vlen
            vreg_data = self.mem_handle.get_content(ptr, shape=(-1,), dtype=self.vpu_context.vdtype)[st:ed]
            self.vpu_context.set_vector_reg(vreg_idx + i, vreg_data)
        
    @core_command_method
    def vpu_store_reg(self, ptr: Pointer, ptr_offset: int, vreg_idx: int, burst_len: int=1):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)

        for i in range(burst_len):
            offset = (ptr_offset + i * self.vpu_context.vlen) * self.vpu_context.vdtype.itemsize
            vreg_data = self.vpu_context.get_vector_reg(vreg_idx + i)
            self.mem_handle.set_content(ptr, vreg_data, offset=offset)

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

    def local_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        return self.core.mem_context.l1_config.get_cycles(size=src_ptr.size)

    @core_command_method
    def mem_page_write(self, ptr: Pointer, container: DataContainer):
        return self.core.mem_context.l1_config.get_cycles(size=ptr.size)

    @core_command_method
    def mem_page_read(self, ptr: Pointer, container: DataContainer):
        return self.core.mem_context.l1_config.get_cycles(size=ptr.size)

    def mxu_tiled_gemm(
        self, 
        ifm_ptr:  Reference,
        wgt_ptr:  Reference,
        psum_ptr: Reference,
        ofm_ptr:  Reference,
        preload_wgt:   bool,
        preload_psum:  bool,
        flush_ofm:     bool,
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
    
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if opcode.is_unary:
            return self.core.vpu_context.unary_op_latency * burst_len
        else:
            return self.core.vpu_context.arith_op_latency * burst_len

