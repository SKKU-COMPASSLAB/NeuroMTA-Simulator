import pprint
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
    "MCA_DeviceBase",
    "MTA_DeviceBase",
    "MTA_CoreGrid",
]


class MCA_DeviceBase(Device):
    def __init__(
        self, 
        
        cmap_config: CmapConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        self.cmap_context = CmapContext(config=cmap_config)
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
        
        self.main_mem_core = MainMemoryCore(mem_context=self.mem_context, cmap_context=self.cmap_context)
        
        if self.mem_context.main_config.dramsim3_enable:
            self.companion_core.register_companion_module(
                self.cmap_context.config.dramsim_module_id,
                module=DRAMSim3(config=self.mem_context.main_config.dramsim3_config)
            )
    
    def get_npu_core(self, core_id: int=None, addr: int=None) -> NPUCore:
        if core_id is None and addr is None:
            raise Exception(f"Please provide exactly one of core_id, coord, or addr to identify the NPU core.")
            
        if core_id is None:
            if addr is not None:
                addr_space_entry = self.cmap_context.get_addr_space_entry_from_address(addr)
                core_id = addr_space_entry.core_ids[0]  # TODO: only one?

        core_idx = self.npu_core_id_to_idx_mappings[core_id]

        return self.npu_cores[core_idx]

    def get_l1_mem_handle(self, core_id: int=None, addr: int=None) -> MemoryHandle:
        core = self.get_npu_core(core_id=core_id, addr=addr)
        return core.mem_handle
    
    def get_main_mem_handle(self) -> MemoryHandle:
        return self.main_mem_core.mem_handle
    
    def create_local_variable(self, size: int, initial_value: int, core_ids: int | list[int]=None) -> Pointer | list[Pointer]:
        if core_ids is None:
            core_ids = self.npu_core_ids
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]
        
        ptrs: list[Pointer] = []
        
        for core_id in core_ids:
            mem_handle = self.get_l1_mem_handle(core_id=core_id)
            ptr = create_var_ptr(mem_handle=mem_handle, var_size=size, initial_value=initial_value)
            ptrs.append(ptr)
        
        if len(core_ids) == 1:
            return ptrs[0]
        return ptrs
    
    def create_local_l1_buffer(self, page_size: int, n_pages: int, core_ids: list[int]=None) -> BufferPointer | list[BufferPointer]:
        if core_ids is None:
            core_ids = self.npu_core_ids
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]
            
        ptrs: list[BufferHandle] = []

        for core_id in core_ids:
            mem_handle = self.get_l1_mem_handle(core_id=core_id)
            ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages)
            ptrs.append(ptr)
        
        if len(core_ids) == 1:
            return ptrs[0]
        return ptrs

    def create_sharded_main_buffer(self, page_size: int, n_pages: int, channel_id: int | Sequence[int]=None) -> BufferPointer:        
        if channel_id is None:
            channel_id = list(range(self.cmap_context.config.n_main_mem_channels))
        
        mem_handle = self.get_main_mem_handle()
        
        ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, channel_id=channel_id)
        return ptr
    
    def remove_buffer(self, ptr: BufferPointer):
        handle = ptr.raw_handle
        
        for page_ptr in handle.page_ptrs:
            if self.cmap_context.config.check_main_mem_addr(page_ptr.addr):
                mem_handle = self.get_main_mem_handle()
            elif self.cmap_context.config.check_l1_mem_addr(page_ptr.addr):
                mem_handle = self.get_l1_mem_handle(addr=page_ptr.addr)
            else:
                raise Exception(f"Unsupported address: {page_ptr.addr}")
            
            mem_handle.deallocate_ptr(page_ptr)
    
    def set_ptr_content(self, ptr: BufferPointer | Pointer | BufferHandle, content: torch.Tensor):
        if isinstance(ptr, BufferPointer):
            ptr = ptr.raw_handle
        
        if isinstance(ptr, Pointer):
            if ptr.ptr_type == PointerType.PAGE:
                self._set_page_var_ptr_content(ptr, content)
            else:
                self._set_page_var_ptr_content(ptr, content)
        elif isinstance(ptr, BufferHandle):
            page_size = ptr.page_size
            n_pages = ptr.n_pages
            
            if content.numel() * content.element_size() != page_size * n_pages:
                raise ValueError(f"Content size {content.numel() * content.element_size()} does not match buffer size {page_size * n_pages}.")
            
            content = content.view(dtype=torch.uint8).reshape((n_pages, page_size))
            
            for page_ptr, page_content in zip(ptr.page_ptrs, content):
                self._set_page_var_ptr_content(page_ptr, page_content)
        else:
            raise Exception(f"Unsupported pointer type: {type(ptr).__name__}. Expected BufferPointer or Pointer.")

    def _set_page_var_ptr_content(self, ptr: Pointer, content: Any):
        if ptr.ptr_type == PointerType.PAGE:
            if not isinstance(content, torch.Tensor):
                raise ValueError(f"Content must be a torch.Tensor for PAGE pointer, got {type(content)}.")
            
            content = content.flatten().view(dtype=torch.uint8)
            
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(addr=ptr.addr)
        else:
            raise Exception(f"Unsupported address: {ptr.addr}")

        mem_handle.set_content(ptr, content)

    def get_ptr_content(self, ptr: BufferPointer | Pointer | BufferHandle, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if isinstance(ptr, BufferPointer):
            ptr = ptr.raw_handle
            
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
            raise Exception(f"Unsupported pointer type: {type(ptr)}. Expected BufferPointer or Pointer.")

    def _get_page_var_ptr_content(self, ptr: Pointer, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(addr=ptr.addr)
        else:
            raise Exception(f"Unsupported address: {ptr.addr}")
        
        content = mem_handle.get_content(ptr, shape=shape, dtype=dtype)

        return content
    
    def summary(self) -> dict[str, Any]:
        return {
            "device_type": type(self).__name__,
            "npu_cores": len(self.npu_cores),
            "dma_cores": len(self.dma_cores),
            "cmap_config": self.cmap_context.config.summary(),
            "mxu_config": self.mxu_config,
            "vpu_config": self.vpu_config,
            "mem_config": {
                "l1_config": self.mem_context.l1_config.summary(),
                "main_config": self.mem_context.main_config.summary(),
            }
        }
        
    def print_summary(self):
        pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
        pp.pprint(self.summary())


class MTA_CoreGrid(list):
    def __init__(self, offset: tuple[int, int], shape: tuple[int, int], core_ids: list[int]):
        super().__init__(core_ids)
        
        self.offset = offset
        self.shape = shape
        self.core_ids = core_ids
        
    def __getitem__(self, idx: int) -> int:
        if isinstance(idx, tuple):
            return self.core_ids[idx[0] * self.shape[1] + idx[1]]
        return super().__getitem__(idx)


class MTA_DeviceBase(MCA_DeviceBase):
    def __init__(
        self, 
        
        cmap_config: CmapConfig, 
        icnt_config: IcntConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__(cmap_config=cmap_config, mem_config=mem_config, mxu_config=mxu_config, vpu_config=vpu_config)
        
        self.icnt_context = IcntContext(config=icnt_config)
        self.icnt_core = IcntCore(cmap_context=self.cmap_context, icnt_context=self.icnt_context)
        
        if self.icnt_context.booksim2_enable:
            self.companion_core.register_companion_module(
                self.cmap_context.config.booksim_module_id,
                module=BookSim2(config=self.icnt_context.config.booksim2_config)
            )
            
        npu_core_rows, npu_core_cols = [], []
        
        for core_id in self.npu_core_ids:
            coord = self.icnt_context.core_id_to_coord(core_id)
            npu_core_rows.append(coord[0])
            npu_core_cols.append(coord[1])
            
        npu_core_rows = sorted(list(set(npu_core_rows)))
        npu_core_cols = sorted(list(set(npu_core_cols)))
        
        self.npu_core_grid = torch.tensor([[self.icnt_context.coord_to_core_id((r, c)) for c in npu_core_cols]for r in npu_core_rows])
        self.npu_core_grid_enabled = True
        
        for core_id in torch.unique(self.npu_core_grid):
            if core_id not in self.npu_core_ids:
                self.npu_core_grid_enabled = False  # the accelerator does not have a full mesh of NPU cores
                break

    def get_npu_core_grid(self, offset: tuple[int, int], shape: tuple[int, int]) -> MTA_CoreGrid:
        if not self.npu_core_grid_enabled:
            raise Exception("[ERROR] Unable to get npu core grid since the accelerator does not have a full mesh of NPU cores.")

        grid = self.npu_core_grid[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]]
        return MTA_CoreGrid(offset=offset, shape=shape, core_ids=grid.flatten().tolist())

    def create_sharded_l1_buffer(self, page_size: int, n_pages: int, core_ids: list[int]=None, contiguous_n_pages: int=1) -> BufferPointer:
        if core_ids is None:
            core_ids = self.cmap_context.config.get_core_ids(CmapCoreType.NPU)
        if not isinstance(core_ids, Sequence):
            core_ids = [core_ids]

        mem_handles = [self.get_l1_mem_handle(core_id=core_id) for core_id in core_ids]
        ptr = create_distributed_buffer(mem_handles=mem_handles, page_size=page_size, n_pages=n_pages, contiguous_n_pages=contiguous_n_pages)
        
        if ptr is None:
            logger.info(f"Unable to locate distributed buffer with page_size={page_size}, n_pages={n_pages}, contiguous_n_pages={contiguous_n_pages} on cores {core_ids}.")

        return ptr
    
    def summary(self):
        s = super().summary()
        s["icnt_config"] = self.icnt_context.config.summary()
        return s
