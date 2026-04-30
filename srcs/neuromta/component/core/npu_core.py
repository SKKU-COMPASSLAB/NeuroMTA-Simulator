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
        self.set_mem_handle(mem_handle=self.mem_info.mem_handle)
        
        r, _ = self.icnt_context.core_id_to_coord(self.core_id)
        self._dma_engine_idx = r % self.global_context.n_dma_engine_per_instance  # Assume that each NPU core is connected to one DMA engine in a round-robin manner based on the row coordinate of the core in the NoC topology
    
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
    # Memory Copy Commands
    #############################################################
    
    @jit_prototype
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
            
    @jit_prototype
    def remote_mem_page_read(self, dst_core_id: int, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0, cont_row_zero_pad: int=0):
        src_core_id = self.core_id
        
        self.local_mem_page_read(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset, cont_row_zero_pad)
        
        if dst_core_id != src_core_id:
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
                
    @jit_prototype
    def remote_mem_page_write(self, src_core_id: int, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        dst_core_id = self.core_id
        
        if src_core_id != dst_core_id:
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
        
        self.local_mem_page_write(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset)
    
    @jit_prototype
    def mem_copy(self, dst_ptr: Pointer, src_ptr: Pointer, row_size: int, row_num: int=1, src_row_stride: int=None, dst_row_stride: int=None, dst_row_zero_pad: int=0):
        if not isinstance(dst_ptr, Pointer) or not isinstance(src_ptr, Pointer):
            raise ValueError("dst_ptr and src_ptr must be Pointer instances.")
        if dst_ptr.addr is None or src_ptr.addr is None:
            raise ValueError(f"dst_ptr and src_ptr must have valid addresses before 'mem_copy' method is compiled. {dst_ptr}, {src_ptr}")
        
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
            
        container = DataContainer()
        
        if src_owner_core_id != self.core_id and dst_owner_core_id != self.core_id:
            raise Exception("At least one of the source and destination buffers must belong to the current core for 'mem_copy' method.")

        if src_owner_core_id == self.core_id:
            self.local_mem_page_read(src_ptr, container, row_size, row_num, src_row_stride, row_size, cont_row_zero_pad=dst_row_zero_pad)
        else:
            data_rd_request = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=src_owner_core_id,
                cmd_id="remote_mem_page_read"
            ).with_args(
                dst_core_id=self.core_id,
                ptr=src_ptr,
                container=container,
                row_size=row_size,
                row_num=row_num,
                mem_row_stride=src_row_stride,
                cont_row_stride=row_size,
                cont_row_zero_pad=dst_row_zero_pad
            )
            
            self.async_rpc_send_req_msg(data_rd_request)
            self.async_rpc_wait_rsp_msg(data_rd_request)
        
        if dst_owner_core_id == self.core_id:
            self.local_mem_page_write(dst_ptr, container, row_size, row_num, dst_row_stride, row_size)
        else:
            data_wr_request = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=dst_owner_core_id,
                cmd_id="remote_mem_page_write"
            ).with_args(
                src_core_id=self.core_id,
                ptr=dst_ptr,
                container=container,
                row_size=row_size,
                row_num=row_num,
                mem_row_stride=dst_row_stride,
                cont_row_stride=row_size,
            )
            
            self.async_rpc_send_req_msg(data_wr_request)
            self.async_rpc_wait_rsp_msg(data_wr_request)
        
    @jit_prototype
    def mem_copy_to_fifo(self, ptr: Pointer, fifo_handle: FIFOBufferHandle, entry_id: VariableHandle | int, size: int=None, ref_count: int=1):
        dst_ptr = fifo_handle.get_ptr(entry_id)
        size = size if (size is not None) else fifo_handle.entry_size
        
        self.fifo_wait_until_vacant(fifo_handle, entry_id)
        self.mem_copy(dst_ptr, ptr, row_size=size)
        self.fifo_push(fifo_handle, entry_id, ref_count)
        
    @jit_prototype
    def mem_copy_from_fifo(self, ptr: Pointer, fifo_handle: FIFOBufferHandle, entry_id: VariableHandle | int, size: int=None):
        src_ptr = fifo_handle.get_ptr(entry_id)
        size = size if (size is not None) else fifo_handle.entry_size
        
        self.fifo_wait_until_valid(fifo_handle, entry_id)
        self.mem_copy(ptr, src_ptr, row_size=size)
        self.fifo_pop(fifo_handle, entry_id)
        
    @jit_prototype
    def mem_read(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0, cont_row_zero_pad: int=0):
        if self.check_ptr_belonging(ptr):
            self.local_mem_page_read(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset, cont_row_zero_pad)
        else:
            src_owner_core_id = self.get_buffer_owner(ptr)
            
            data_rd_request = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=src_owner_core_id,
                cmd_id="remote_mem_page_read"
            ).with_args(
                dst_core_id=self.core_id,
                ptr=ptr,
                container=container,
                row_size=row_size,
                row_num=row_num,
                mem_row_stride=mem_row_stride,
                cont_row_stride=cont_row_stride,
                row_pattern=row_pattern,
                cont_row_offset=cont_row_offset,
                cont_row_zero_pad=cont_row_zero_pad
            )
            
            self.async_rpc_send_req_msg(data_rd_request)
            self.async_rpc_wait_rsp_msg(data_rd_request)
            
    @jit_prototype
    def mem_write(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        if self.check_ptr_belonging(ptr):
            self.local_mem_page_write(ptr, container, row_size, row_num, mem_row_stride, cont_row_stride, row_pattern, cont_row_offset)
        else:
            dst_owner_core_id = self.get_buffer_owner(ptr)
            
            data_wr_request = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=dst_owner_core_id,
                cmd_id="remote_mem_page_write"
            ).with_args(
                src_core_id=self.core_id,
                ptr=ptr,
                container=container,
                row_size=row_size,
                row_num=row_num,
                mem_row_stride=mem_row_stride,
                cont_row_stride=cont_row_stride,
                row_pattern=row_pattern,
                cont_row_offset=cont_row_offset
            )
            
            self.async_rpc_send_req_msg(data_wr_request)
            self.async_rpc_wait_rsp_msg(data_wr_request)
            
    @jit_prototype
    def mem_read_from_fifo(self, container: DataContainer[torch.Tensor], fifo_handle: FIFOBufferHandle, entry_id: VariableHandle | int, row_size: int, row_num: int=1, row_pattern: dict[int, int]=None):
        src_ptr = fifo_handle.get_ptr(entry_id)
        row_size = row_size if (row_size is not None) else fifo_handle.entry_size
        
        self.fifo_wait_until_valid(fifo_handle, entry_id)
        self.mem_read(src_ptr, container, row_size=row_size, row_num=row_num, row_pattern=row_pattern)
        self.fifo_pop(fifo_handle, entry_id)
        
    @jit_prototype
    def mem_write_to_fifo(self, container: DataContainer[torch.Tensor], fifo_handle: FIFOBufferHandle, entry_id: VariableHandle | int, row_size: int, row_num: int=1, row_pattern: dict[int, int]=None, ref_count: int=1):
        dst_ptr = fifo_handle.get_ptr(entry_id)
        row_size = row_size if (row_size is not None) else fifo_handle.entry_size
        
        self.fifo_wait_until_vacant(fifo_handle, entry_id)
        self.mem_write(dst_ptr, container, row_size=row_size, row_num=row_num, row_pattern=row_pattern)
        self.fifo_push(fifo_handle, entry_id, ref_count)

    #############################################################
    # MXU Commands
    #############################################################
        
    @core_command_method
    def mxu_reconfigure(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self.mxu_context.reconfigure_dtype(dtype=dtype, acc_dtype=acc_dtype)
        
    @core_command_method
    def mxu_load_context(self, psum_cont: DataContainer[torch.Tensor]):
        psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
        self.mxu_context.set_pe_arr_regs(psum_tile)
        
    @core_command_method
    def mxu_store_context(self, psum_cont: DataContainer[torch.Tensor]):
        psum_tile = self.mxu_context.get_pe_arr_regs()
        psum_cont.data = psum_tile
    
    @core_command_method
    def mxu_tiled_gemm(
        self, 
        
        ifm_cont:  DataContainer[torch.Tensor],
        wgt_cont:  DataContainer[torch.Tensor],
        psum_cont: DataContainer[torch.Tensor] | None,
        ofm_cont:  DataContainer[torch.Tensor] | None,
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        
        ifm_transposed: bool=False,
        wgt_transposed: bool=False,
        psum_vectored:  bool=False,
    ):  
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if psum_cont is None or psum_cont.data is None:
                psum_tile = torch.zeros(self.mxu_context.config.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = psum_cont.data.view(self.mxu_context.acc_dtype)
                
                if psum_vectored:
                    psum_tile = psum_tile.flatten().unsqueeze(0)  # (1, N)
                else:
                    psum_tile = psum_tile.reshape(self.mxu_context.config.ofm_tile_shape)
                    
                if ifm_transposed: 
                    psum_tile = psum_tile.T
                
            self.mxu_context.set_pe_arr_regs(psum_tile)  

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
            if ofm_cont is None:
                raise ValueError("ofm_cont must be provided when flush_ofm is True in mxu_tiled_gemm command.")
            
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
            
            self.mxu_context.set_pe_arr_regs(psum_tile)
        
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
        dst:  DataContainer[torch.Tensor] | None,
        
        preload_psum:  bool=False,
        flush_ofm:     bool=False,
        ifm_transposed: bool=False,
    ):
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if dst is None or dst.data is None:
                psum_tile = torch.zeros(self.mxu_context.config.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)
            else:
                psum_tile = dst.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ofm_tile_shape)
                    
            self.mxu_context.set_pe_arr_regs(psum_tile)

        ifm_tile = src.data.view(self.mxu_context.acc_dtype).reshape(self.mxu_context.config.ifm_tile_shape)
        if ifm_transposed: ifm_tile = ifm_tile.T
        
        self.mxu_context.execute_elemwise(ifm_tile=ifm_tile, op=op)
        
        if flush_ofm:
            if dst is None:
                raise ValueError("dst must be provided when flush_ofm is True in mxu_tiled_elemwise command.")
            
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

