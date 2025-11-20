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
    ):
        super().__init__(
            core_id=core_id, 
            cycle_model=DMACoreCycleModel(core=self)
        )
        
        self.global_context = global_context
        
        self.core_info  = self.global_context.get_core_info(GlobalContextCoreType.DMA, core_id)
        self.mem_info   = self.core_info.owned_mem_info  # Assume that each DMA core owns only one memory
        self.mem_handle = self.mem_info.mem_handle
        
    def check_ptr_belonging(self, ptr: Pointer) -> bool:
        if isinstance(ptr, Pointer):
            addr = ptr.addr
        else:
            addr = ptr
            
        mem_info = self.global_context.get_mem_info_by_address(addr)
        if mem_info.mem_type == GlobalContextMemType.L1:
            return False
        else:
            return self.core_id in mem_info.owner_core_ids
    
    @core_command_method
    def local_mem_page_read(self, ptr: Pointer, size: int, container: DataContainer[torch.Tensor], offset: int=0, mem_row_size: int=None, mem_row_stride: int=None, cont_row_stride: int=None, cont_row_pattern: dict[int, int]=None):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_read' method.")
        
        data = self.mem_handle.get_data(ptr, size=size, dtype=torch.uint8)
        
        if offset != 0:
            data = data[offset:]
            data = torch.nn.functional.pad(data, (0, offset), 'constant', 0)
        
        if mem_row_size is not None:
            if mem_row_stride is None:
                mem_row_stride = mem_row_size
            if cont_row_stride is None:
                cont_row_stride = mem_row_size
                
            if size % mem_row_stride != 0:
                raise ValueError(f"Size {size} is not divisible by row_stride {mem_row_stride} in core {self.core_id} during 'local_mem_page_read' method with row-wise operation.")
            
            data = data.reshape(size // mem_row_stride, mem_row_stride)  # Assume that the data is 2D (rows x row_stride)
            
            if cont_row_pattern is None:
                cont_row_pattern = {i: i for i in range(size // mem_row_stride)}
            
            if not container.is_mem_segment:
                container.data = torch.zeros(cont_row_stride * (size // mem_row_stride), dtype=torch.uint8)
                
            cont_data = container.data.flatten().view(torch.uint8)
            
            for dst_idx, src_idx in cont_row_pattern.items():
                st = offset + (dst_idx * cont_row_stride)
                ed = st + mem_row_size
                cont_data[st:ed] = data[src_idx, :mem_row_size]
                
            container.data = cont_data.reshape(-1, cont_row_stride)
        else:
            container.data = data
        
    @core_command_method
    def local_mem_page_write(self, ptr: Pointer, size: int, container: DataContainer[torch.Tensor], offset: int=0, mem_row_size: int=None, mem_row_stride: int=None, cont_row_stride: int=None, cont_row_pattern: dict[int, int]=None):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_write' method.")
        
        if mem_row_size is not None:
            if mem_row_stride is None:
                mem_row_stride = mem_row_size
            if cont_row_stride is None:
                cont_row_stride = mem_row_size
                
            if size % mem_row_stride != 0:
                raise ValueError(f"Size {size} is not divisible by row_stride {mem_row_stride} in core {self.core_id} during 'local_mem_page_read' method with row-wise operation.")
                
            data = self.mem_handle.get_data(ptr, size=size, dtype=torch.uint8)
            
            if offset != 0:
                data = data[offset:]
                data = torch.nn.functional.pad(data, (0, offset), 'constant', 0)
            
            data = data.reshape(size // mem_row_stride, mem_row_stride)  # Assume that the data is 2D (rows x row_stride)
            
            if cont_row_pattern is None:
                cont_row_pattern = {i: i for i in range(size // mem_row_stride)}
                
            if not container.is_mem_segment:
                raise Exception(f"The container is not a memory segment in core {self.core_id} during 'local_mem_page_write' method with row-wise operation.")
            
            cont_data = container.data.flatten().view(torch.uint8)
            
            for src_idx, dst_idx in cont_row_pattern.items():
                st = offset + (dst_idx * cont_row_stride)
                ed = st + mem_row_size
                data[dst_idx, :mem_row_size] = cont_data[st:ed]
                
            self.mem_handle.set_data(ptr, size=size, data=data)
        else:
            self.mem_handle.set_data(ptr + offset, size=size, data=container.data)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core