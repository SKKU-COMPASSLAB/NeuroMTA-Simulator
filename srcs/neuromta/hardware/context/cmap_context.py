import enum
from typing import Sequence

from neuromta.framework import *
from neuromta.hardware.companions.booksim import BookSim2Config, PYBOOKSIM2_AVAILABLE


__all__ = [
    "CmapCoreType",
    "CmapMemType",
    "CmapConfig",
    "CmapContext",
]


ICNT_CORE_NAME      = "ICNT"
MAIN_MEM_CORE_NAME  = "MAIN_MEM"
BOOKSIM_MODULE_ID   = "BOOKSIM"
DRAMSIM_MODULE_ID   = "DRAMSIM"


class CmapCoreType(enum.Enum):
    NPU     = enum.auto()
    DMA     = enum.auto()
    CACHE   = enum.auto()
    
    @property
    def is_memory_space_owner(self) -> bool:
        return self in (CmapCoreType.NPU, CmapCoreType.DMA)
    

class CmapCoreInfo:
    def __init__(self, core_type: CmapCoreType, base_addr: int=None, addr_space_size: int=None, nxt_level_mem_core_ids: list[int]=None):
        self.core_type = core_type
        self.base_addr = base_addr
        self.addr_space_size = addr_space_size
        self.nxt_level_mem_core_ids = nxt_level_mem_core_ids
        
        if self.nxt_level_mem_core_ids is not None:
            if not isinstance(self.nxt_level_mem_core_ids, Sequence):
                self.nxt_level_mem_core_ids = [self.nxt_level_mem_core_ids]
                
    def __str__(self):
        return f"CmapCoreInfo(core_type={self.core_type}, base_addr={self.base_addr}, addr_space_size={self.addr_space_size}, nxt_level_mem_core_ids={self.nxt_level_mem_core_ids})"


class CmapMemType(enum.Enum):
    L1      = enum.auto()
    MAIN    = enum.auto()
    
    
class AddressSpaceEntry:
    def __init__(self, mem_type: CmapMemType, size: int):
        self.mem_type = mem_type
        self.size = size
        
        self.core_ids: list[int] = []
        
    def add_core_id(self, core_id: int):
        self.core_ids.append(core_id)


class CmapConfig:
    def __init__(
        self,
        
        n_l1_spm_bank: int,
        n_main_mem_channels: int,
        l1_spm_bank_size: int,      # = parse_mem_cap_str("1MB"),
        main_mem_channel_size: int, # = parse_mem_cap_str("1GB"),
    ):
        super().__init__()

        self.n_l1_spm_bank          = n_l1_spm_bank
        self.n_main_mem_channels    = n_main_mem_channels
        self.l1_spm_bank_size       = l1_spm_bank_size
        self.main_mem_channel_size  = main_mem_channel_size
        
        self.icnt_core_id           = ICNT_CORE_NAME
        self.main_mem_core_id       = MAIN_MEM_CORE_NAME
        
        self.booksim_module_id      = BOOKSIM_MODULE_ID
        self.dramsim_module_id      = DRAMSIM_MODULE_ID 
        
        self.core_map: dict[int, CmapCoreInfo] = {}
        self.addr_space: dict[int, AddressSpaceEntry] = {}

        self.main_mem_base_addr = 0x00000000
        self.l1_spm_base_addr = self.main_mem_base_addr + self.n_main_mem_channels * self.main_mem_channel_size
        
        for i in range(self.n_main_mem_channels):
            base_addr = self.main_mem_base_addr + i * self.main_mem_channel_size
            self.addr_space[base_addr] = AddressSpaceEntry(mem_type=CmapMemType.MAIN, size=self.main_mem_channel_size)
        
        for i in range(self.n_l1_spm_bank):
            base_addr = self.l1_spm_base_addr + i * self.l1_spm_bank_size
            self.addr_space[base_addr] = AddressSpaceEntry(mem_type=CmapMemType.L1, size=self.l1_spm_bank_size)
    
    def add_npu_core(self, core_id: int, mem_bank_idx: int, nxt_level_mem_core_ids: list[int]):
        if not isinstance(core_id, int):
            raise TypeError(f"Core ID must be an integer.")
        if core_id in self.core_map.keys():
            raise ValueError(f"Core ID {core_id} is already assigned.")
        
        mem_type=CmapMemType.L1
        base_addr=self.l1_spm_base_addr + mem_bank_idx * self.l1_spm_bank_size
        addr_space_size=self.l1_spm_bank_size
        
        self.core_map[core_id] = CmapCoreInfo(
            core_type=CmapCoreType.NPU,
            base_addr=base_addr,
            addr_space_size=addr_space_size,
            nxt_level_mem_core_ids=nxt_level_mem_core_ids,
        )
        
        if base_addr not in self.addr_space.keys():
            raise Exception(f"[ERROR] Address {base_addr} is not mapped in the address space.")
        
        addr_space_entry = self.addr_space[base_addr]
        
        if addr_space_entry.mem_type != mem_type or addr_space_entry.size != addr_space_size:
            raise ValueError(f"Address {base_addr} is already assigned to a different memory type.")
        
        if len(addr_space_entry.core_ids) >= 1:
            raise Exception(f"[ERROR] Address {base_addr} is already assigned to another core.")

        addr_space_entry.core_ids.append(core_id)
        
    def add_dma_core(self, core_id: int, mem_bank_idx: int):
        if not isinstance(core_id, int):
            raise TypeError(f"Core ID must be an integer.")
        if core_id in self.core_map.keys():
            raise ValueError(f"Core ID {core_id} is already assigned.")
        
        mem_type=CmapMemType.MAIN
        base_addr=self.main_mem_base_addr + mem_bank_idx * self.main_mem_channel_size
        addr_space_size=self.main_mem_channel_size
        
        self.core_map[core_id] = CmapCoreInfo(
            core_type=CmapCoreType.DMA,
            base_addr=base_addr,
            addr_space_size=addr_space_size,
            nxt_level_mem_core_ids=None
        )

        if base_addr not in self.addr_space.keys():
            raise Exception(f"[ERROR] Address {base_addr} is not mapped in the address space.")
        
        addr_space_entry = self.addr_space[base_addr]
        
        if addr_space_entry.mem_type != mem_type or addr_space_entry.size != addr_space_size:
            raise ValueError(f"Address {base_addr} is already assigned to a different memory type.")

        addr_space_entry.core_ids.append(core_id)

    def get_core_ids(self, core_type: CmapCoreType) -> tuple[int]:
        ids = [coord for coord, cinfo in self.core_map.items() if cinfo.core_type == core_type]
        if not ids:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(ids)
        
    def count_core(self, core_type: CmapCoreType) -> int:
        return sum(1 for ctype in self.core_map.values() if ctype == core_type)
    
    def check_main_mem_addr(self, addr: int) -> bool:
        return self.main_mem_base_addr <= addr < self.l1_spm_base_addr
    
    def check_l1_mem_addr(self, addr: int) -> bool:
        return self.l1_spm_base_addr <= addr < (self.l1_spm_base_addr + self.n_l1_spm_bank * self.l1_spm_bank_size)
    
    def get_mem_base_addr(self, addr: int) -> int:
        if self.check_l1_mem_addr(addr):
            base_addr = self.l1_spm_base_addr
            bank_size = self.l1_spm_bank_size
        elif self.check_main_mem_addr(addr):
            base_addr = self.main_mem_base_addr
            bank_size = self.main_mem_channel_size
        else:
            raise ValueError(f"Address {addr} is out of range of main memory and L1 SPM.")
        
        return ((addr - base_addr) // bank_size) * bank_size + base_addr
    

class CmapContext:
    def __init__(
        self,
        
        config: CmapConfig,
    ):
        self._config = config

        self._npu_core_ids: tuple[int] = tuple(sorted(self.config.get_core_ids(CmapCoreType.NPU)))
        self._dma_core_ids: tuple[int] = tuple(sorted(self.config.get_core_ids(CmapCoreType.DMA)))
        
    def get_nxt_mem_core_id(self, src_core_id: int, addr: int) -> int:
        if self.config.check_l1_mem_addr(addr):
            base_addr = self.config.get_mem_base_addr(addr)
            addr_space_entry = self.config.addr_space[base_addr]
            core_id = addr_space_entry.core_ids[0]  # TODO: the numebr of L1 memory owner is always 1 ???
            
            return core_id
        
        src_core_info = self.config.core_map[src_core_id]
        
        if src_core_info.nxt_level_mem_core_ids is None:
            raise ValueError(f"Core ID {src_core_id} does not have a next-level memory.")
        
        for dst_core_id in src_core_info.nxt_level_mem_core_ids:
            dst_core_info = self.config.core_map[dst_core_id]
            
            if dst_core_info.core_type == CmapCoreType.CACHE:
                return dst_core_id
            elif dst_core_info.base_addr <= addr < (dst_core_info.base_addr + dst_core_info.addr_space_size):
                return dst_core_id
            
        raise ValueError(f"Address {addr} is out of range of the next-level memory of core ID {src_core_id}.")

    def get_base_addr_from_core_id(self, core_id: int) -> int:
        return self.config.core_map[core_id].base_addr
            
    def get_addr_space_entry_from_address(self, addr: int) -> AddressSpaceEntry:
        keys = sorted(self.config.addr_space.keys())
        left, right = 0, len(keys) - 1

        while left <= right:
            mid = (left + right) // 2
            base_addr = keys[mid]
            
            addr_space_entry = self.config.addr_space[base_addr]
            size = addr_space_entry.size

            if base_addr <= addr < (base_addr + size):
                return addr_space_entry
            if base_addr > addr:
                right = mid - 1
            else:
                left = mid + 1
                
        return []  # Address does not map to any core
    
    @property
    def config(self) -> CmapConfig:
        return self._config

    @property
    def npu_core_ids(self) -> tuple[int]:
        return self._npu_core_ids

    @property
    def dma_core_ids(self) -> tuple[int]:
        return self._dma_core_ids
    
    @property
    def icnt_core_id(self) -> str:
        return self._config.icnt_core_id
    
    @property
    def main_mem_core_id(self) -> str:
        return self._config.main_mem_core_id
