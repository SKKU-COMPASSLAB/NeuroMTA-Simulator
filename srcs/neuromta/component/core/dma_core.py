import torch
from neuromta.framework import *

from neuromta.component.context.global_context import *

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
        
    def dump_core_states(self):
        return {
            "mem_handle_state": self.mem_handle.dump_handle_state(),
        }
        
    def load_core_states(self, states: dict):
        self.mem_handle.load_handle_state(states["mem_handle_state"])
        
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
    def local_mem_page_read(self, ptr: Pointer, container: DataContainer[torch.Tensor], row_size: int, row_num: int=1, mem_row_stride: int=None, cont_row_stride: int=None, row_pattern: dict[int, int]=None, cont_row_offset: int=0):
        if not self.check_ptr_belonging(ptr):
            raise Exception(f"Pointer {ptr} does not belong to core {self.core_id} during 'local_mem_page_read' method.")
        
        if mem_row_stride is None:
            mem_row_stride = row_size
        if cont_row_stride is None:
            cont_row_stride = row_size
        if row_pattern is None:
            row_pattern = {i: i for i in range(row_num)}
            
        if not container.is_mem_segment:
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
        
        for d, s in row_pattern.items():
            dst_data = container.data[d, cont_row_offset:cont_row_offset+row_size]
            self.mem_handle.set_data(ptr + (s * mem_row_stride), size=row_size, data=dst_data)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core