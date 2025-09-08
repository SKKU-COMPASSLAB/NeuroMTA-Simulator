import torch
from typing import Any, Sequence

from neuromta.framework import *

from neuromta.hardware.context.mem_context import *
from neuromta.hardware.context.icnt_context import *
from neuromta.hardware.context.cmap_context import *
from neuromta.hardware.context.mxu_context import *
from neuromta.hardware.context.vpu_context import *

from neuromta.hardware.core.npu_core import *
from neuromta.hardware.core.dma_core import *
from neuromta.hardware.core.icnt_core import *
from neuromta.hardware.core.main_mem_core import *

from neuromta.hardware.companions.booksim import BookSim2
from neuromta.hardware.companions.dramsim import DRAMSim3


__all__ = [
    "MultiTileAccelerator"
]


class MultiTileAccelerator(Device):
    def __init__(
        self, 
        
        cmap_config: CmapConfig, 
        icnt_config: IcntConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        self.cmap_context = CmapContext(config=cmap_config)
        self.icnt_context = IcntContext(config=icnt_config)
        self.mem_context  = MemContext(config=mem_config)
        
        self.mxu_config = mxu_config
        self.vpu_config = vpu_config
        
        self.npu_core_ids = self.cmap_context.npu_core_ids
        self.dma_core_ids = self.cmap_context.dma_core_ids

        self.npu_core_id_to_idx_mappings = {core_id: idx for idx, core_id in enumerate(self.npu_core_ids)}
        self.dma_core_id_to_idx_mappings = {core_id: idx for idx, core_id in enumerate(self.dma_core_ids)}

        self.npu_cores: list[NPUCore] = [
            NPUCore(core_id=core_id, mem_context=self.mem_context, cmap_context=self.cmap_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for core_id in self.npu_core_ids
        ]
        
        self.dma_cores: list[DMACore] = [
            DMACore(core_id=core_id, mem_context=self.mem_context, cmap_context=self.cmap_context)
            for core_id in self.dma_core_ids
        ]
        
        self.icnt_core = IcntCore(cmap_context=self.cmap_context, icnt_context=self.icnt_context)
        self.main_mem_core = MainMemoryCore(mem_context=self.mem_context, cmap_context=self.cmap_context)
        
        if self.icnt_context.booksim2_enable:
            self.companion_core.register_companion_module(
                self.cmap_context.config.booksim_module_id,
                module=BookSim2(config=self.icnt_context.config.booksim2_config)
            )
        
        if self.mem_context.main_config.dramsim3_enable:
            self.companion_core.register_companion_module(
                self.cmap_context.config.dramsim_module_id,
                module=DRAMSim3(config=self.mem_context.main_config.dramsim3_config)
            )
    
    def get_npu_core(self, core_id: int=None, coord: tuple[int, int]=None, addr: int=None) -> NPUCore:
        if core_id is None and coord is None and addr is None:
            raise Exception(f"[ERROR] Please provide exactly one of core_id, coord, or addr to identify the NPU core.")
            
        if core_id is None:
            if coord is not None:
                core_id = self.icnt_context.coord_to_core_id(coord)
            elif addr is not None:
                addr_space_entry = self.cmap_context.get_addr_space_entry_from_address(addr)
                core_id = addr_space_entry.core_ids[0]  # TODO: only one?

        core_idx = self.npu_core_id_to_idx_mappings[core_id]

        return self.npu_cores[core_idx]
    
    def get_l1_mem_handle(self, core_id: int=None, coord: tuple[int, int]=None, addr: int=None) -> MemoryHandle:
        core = self.get_npu_core(core_id=core_id, coord=coord, addr=addr)
        return core.mem_handle
    
    def get_main_mem_handle(self) -> MemoryHandle:
        return self.main_mem_core.mem_handle

    def create_local_l1_circular_buffer(self, page_size: int, n_pages: int, core_ids: list[int]=None) -> Reference | list[Reference]:
        if core_ids is None:
            core_ids = self.npu_core_ids
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]

        ptrs: list[BufferHandle] = []

        for core_id in core_ids:
            mem_handle = self.get_l1_mem_handle(core_id=core_id)
            ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=True)
            ptrs.append(ptr)

        if len(core_ids) == 1:
            return ptrs[0]
        return ptrs
    
    def create_local_l1_buffer(self, page_size: int, n_pages: int, core_ids: list[int]=None) -> Reference | list[Reference]:
        if core_ids is None:
            core_ids = self.npu_core_ids
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]
            
        ptrs: list[BufferHandle] = []

        for core_id in core_ids:
            mem_handle = self.get_l1_mem_handle(core_id=core_id)
            ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=False)
            ptrs.append(ptr)
        
        if len(core_ids) == 1:
            return ptrs[0]
        return ptrs

    def create_sharded_l1_buffer(self, page_size: int, n_pages: int, core_ids: list[int]=None) -> Reference:
        if core_ids is None:
            core_ids = self.cmap_context.config.get_core_ids(CmapCoreType.NPU)
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]

        mem_handles = [self.get_l1_mem_handle(core_id=core_id) for core_id in core_ids]
        ptr = create_distributed_buffer(mem_handles=mem_handles, page_size=page_size, n_pages=n_pages)

        return ptr

    def create_sharded_main_buffer(self, page_size: int, n_pages: int, channel_id: int | Sequence[int]=None) -> Reference:        
        if channel_id is None:
            channel_id = list(range(self.cmap_context.config.n_main_mem_channels))
        
        mem_handle = self.get_main_mem_handle()
        
        ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=False, channel_id=channel_id)
        return ptr
    
    def set_ptr_content(self, ptr: Reference | Pointer | BufferHandle, content: torch.Tensor):
        if isinstance(ptr, Reference):
            ptr = ptr.resolve(is_read=False)
        
        if isinstance(ptr, Pointer):
            if ptr.ptr_type == PointerType.PAGE:
                self._set_page_var_ptr_content(ptr, content)
            else:
                self._set_page_var_ptr_content(ptr, content)
        elif isinstance(ptr, BufferHandle):
            page_size = ptr.page_size
            n_pages = ptr.n_pages
            
            if content.numel() * content.element_size() != page_size * n_pages:
                raise ValueError(f"[ERROR] Content size {content.numel() * content.element_size()} does not match buffer size {page_size * n_pages}.")
            
            content = content.view(dtype=torch.uint8).reshape((n_pages, page_size))
            
            for page_ptr, page_content in zip(ptr.page_ptrs, content):
                self._set_page_var_ptr_content(page_ptr, page_content)
        else:
            raise Exception(f"[ERROR] Unsupported pointer type: {type(ptr)}. Expected BufferPointer or Pointer.")

    def _set_page_var_ptr_content(self, ptr: Pointer, content: Any):
        if ptr.ptr_type == PointerType.PAGE:
            if not isinstance(content, torch.Tensor):
                raise ValueError(f"[ERROR] Content must be a torch.Tensor for PAGE pointer, got {type(content)}.")
            
            content = content.view(dtype=torch.uint8)
            
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(addr=ptr.addr)
        else:
            raise Exception(f"[ERROR] Unsupported address: {ptr.addr}")

        mem_handle.set_content(ptr, content)

    def get_ptr_content(self, ptr: Reference | Pointer | BufferHandle, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if isinstance(ptr, Reference):
            ptr = ptr.resolve(is_read=True)
            
        if isinstance(ptr, Pointer):
            if ptr.ptr_type == PointerType.PAGE:
                return self._get_page_var_ptr_content(ptr, shape, dtype)
            else:
                return self._get_page_var_ptr_content(ptr)
        elif isinstance(ptr, BufferHandle):
            page_size = ptr.page_size
            n_pages = ptr.n_pages
            
            content = torch.empty((n_pages, page_size), dtype=torch.uint8).contiguous()
            
            for i, page_ptr in enumerate(ptr.page_ptrs):
                content[i, :] = self._get_page_var_ptr_content(page_ptr, shape=(-1,), dtype=torch.uint8)
                
            if dtype is not None:
                content = content.view(dtype=dtype)
            if shape is not None:
                content = content.reshape(shape)

            return content
        else:
            raise Exception(f"[ERROR] Unsupported pointer type: {type(ptr)}. Expected BufferPointer or Pointer.")

    def _get_page_var_ptr_content(self, ptr: Pointer, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(addr=ptr.addr)
        else:
            raise Exception(f"[ERROR] Unsupported address: {ptr.addr}")
        
        content = mem_handle.get_content(ptr, shape=shape, dtype=dtype)

        return content