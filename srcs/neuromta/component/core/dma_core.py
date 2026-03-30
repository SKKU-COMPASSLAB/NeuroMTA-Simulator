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
        
        self.set_mem_handle(mem_handle=self.mem_info.mem_handle)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core