import enum
import math
from typing import Sequence, Any

from neuromta.framework import *
from neuromta.component.companions.dramsim import DRAMSim3Config


__all__ = [
    "GlobalContextCoreType",
    "GlobalContextCoreInfo",
    "GlobalContextMemType",
    "GlobalContextMemInfo",
    "GlobalContextConfig",
    "GlobalContext",
    "MainMemoryConfig",
]


ICNT_CORE_NAME      = "ICNT"
MAIN_MEM_CORE_NAME  = "MAIN_MEM"
BOOKSIM_MODULE_ID   = "BOOKSIM"
DRAMSIM_MODULE_ID   = "DRAMSIM"


class GlobalContextCoreType(enum.Enum):
    NPU     = enum.auto()  # NPU core
    L1_MEM  = enum.auto()  # L1 memory controller (for future use)
    L2_MEM  = enum.auto()  # L2 memory controller (for future use)
    L3_MEM  = enum.auto()  # L3 memory controller (for future use)
    DMA     = enum.auto()  # DMA engine
    
class GlobalContextCoreInfo:
    def __init__(self, core_type: GlobalContextCoreType, core_id: int):
        self.core_type = core_type
        self.core_id   = core_id
        
        self._owned_mem_info: GlobalContextMemInfo = None
        
    @property
    def owned_mem_info(self) -> 'GlobalContextMemInfo':
        return self._owned_mem_info
    
    @owned_mem_info.setter
    def owned_mem_info(self, mem_info: 'GlobalContextMemInfo'):
        if self._owned_mem_info is not None:
            raise RuntimeError("Owned memory ID is already set.")
        self._owned_mem_info = mem_info
        
    def __eq__(self, value):
        if isinstance(value, GlobalContextCoreInfo):
            return (self.core_type == value.core_type) and (self.core_id == value.core_id)
        return False


class GlobalContextMemType(enum.Enum):
    L1   = enum.auto()  # L1 memory (owned by NPU core)
    L2   = enum.auto()  # L2 memory (shared memory, for future use)
    L3   = enum.auto()  # L3 memory (shared memory, for future use)
    MAIN = enum.auto()  # Main memory (owned and managed by DMA engine)


class GlobalContextMemInfo:
    def __init__(self, mem_type: GlobalContextMemType, mem_id: int, base_addr: int, size: int):
        self.mem_type   = mem_type
        self.mem_id     = mem_id
        self._base_addr  = base_addr
        self._size       = size
        
        self.owner_core_ids: list[int]  = []
        
        self._stack_base_addrs: list[int] = None
        self._stack_sizes:      list[int] = None
        self._stack_write_ptrs: list[int] = None
        
        self._bank_size = min(self._size, MemoryHandle.MAX_BANK_SIZE)
        self._n_banks   = math.ceil(size / self._bank_size)
        
        self._mem_handle: MemoryHandle = MemoryHandle(
            base_addr=self.base_addr,
            bank_size=self._bank_size,
            n_banks=self._n_banks,
        )
        
    def add_owner_core(self, core_info: GlobalContextCoreInfo):
        if core_info.core_id not in self.owner_core_ids:
            self.owner_core_ids.append(core_info.core_id)
            core_info.owned_mem_info = self
        
    def __eq__(self, value):
        if isinstance(value, GlobalContextMemInfo):
            return (self.mem_type == value.mem_type) and (self.mem_id == value.mem_id)
        return False
        
    def create_stack(self, stack_size: int) -> int:
        if not self.is_stack_initialized:
            self._stack_base_addrs = [self._base_addr]
            self._stack_sizes      = [stack_size]
            self._stack_write_ptrs = [0]
        else:
            if sum(self._stack_sizes) + stack_size > self.size:
                raise MemoryError(f"Not enough memory to add stack of size {stack_size}. Available memory size: {self.size - sum(self._stack_sizes)}")
            
            self._stack_base_addrs.append(self._base_addr + sum(self._stack_sizes))
            self._stack_sizes.append(stack_size)
            self._stack_write_ptrs.append(0)
            
        return len(self._stack_sizes) - 1  # return the new stack ID
        
    def allocate_data(self, size: int, stack_id: int=0) -> Pointer:
        if self._stack_base_addrs is None:
            raise RuntimeError("Stacks are not initialized. Call 'create_stack' first.")
        if stack_id < 0 or stack_id >= len(self._stack_sizes):
            raise ValueError(f"Invalid stack_id {stack_id}, must be in range [0, {len(self._stack_sizes)}).")
        if self._stack_write_ptrs[stack_id] + size > self._stack_sizes[stack_id]:
            raise MemoryError(f"Not enough memory in stack {stack_id} to allocate {size} bytes. Available: {self._stack_sizes[stack_id] - self._stack_write_ptrs[stack_id]} bytes.")
        
        addr = self._stack_base_addrs[stack_id] + self._stack_write_ptrs[stack_id]
        self._stack_write_ptrs[stack_id] += size
        return Pointer(addr=addr)
    
    def empty_space(self, stack_id: int=0) -> int:
        if not self.is_stack_initialized:
            raise RuntimeError("Stacks are not initialized. Call 'initialize_stacks' first.")
        if stack_id < 0 or stack_id >= len(self._stack_sizes):
            raise ValueError(f"Invalid stack_id {stack_id}, must be in range [0, {len(self._stack_sizes)}).")
        
        return self._stack_sizes[stack_id] - self._stack_write_ptrs[stack_id]
    
    def remove_stack(self, stack_id: int=-1):
        if not self.is_stack_initialized:
            raise RuntimeError("Stacks are not initialized. Call 'initialize_stacks' first.")
        if stack_id < 0:
            stack_id += len(self._stack_sizes)
        if stack_id < 0 or stack_id >= len(self._stack_sizes):
            raise ValueError(f"Invalid stack_id {stack_id}, must be in range [0, {len(self._stack_sizes)}).")
        
        self._stack_base_addrs[stack_id] = None
        self._stack_write_ptrs[stack_id] = None
        self._stack_sizes[stack_id]      = 0
        
        i = len(self._stack_sizes) - 1
        while i >= 0 and self._stack_sizes[i] == 0:
            del self._stack_base_addrs[i]
            del self._stack_sizes[i]
            del self._stack_write_ptrs[i]
            i -= 1
        
    @property
    def base_addr(self) -> int:
        return self._base_addr
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def mem_handle(self) -> MemoryHandle:
        return self._mem_handle
    
    @property
    def is_stack_initialized(self) -> bool:
        return self._stack_base_addrs is not None
    
    
class MainMemoryConfig:
    def __init__(
        self, 
        
        # Default: HBM2 Configuration
        transfer_speed: int         = 1600, 
        ch_io_width: int            = 1024, 
        burst_len: int              = 256, 
        is_ddr: bool                = True, 
        
        processor_clock_freq: int   = parse_freq_str("1GHz"),
        n_instance: int             = 1,
        channel_size: int           = parse_mem_cap_str("4GB"),
        n_channel_per_instance: int = 1,
        n_cmd_q_per_instance: int   = 8,
        
        # DRAMSim3 Configuation (if needed)
        dramsim3_enable: bool           = False,
        dramsim3_src_config_path: str   = "GDDR6_8Gb_x16.ini",
        dramsim3_dst_config_path: str   = "dramsim3_config.ini",
    ):
        self.transfer_speed         = transfer_speed    # transfer speed per pin (MT/s)
        self.ch_io_width            = ch_io_width       # io channel width (bits)
        self.burst_len              = burst_len         # burst length
        self.is_ddr                 = is_ddr
        self.processor_clock_freq   = processor_clock_freq
        
        self._n_instance             = n_instance
        self._channel_size           = channel_size
        self._n_channel_per_instance = n_channel_per_instance
        self._n_cmd_q_per_instance   = n_cmd_q_per_instance

        self.dramsim3_enable        = dramsim3_enable
        
        if self.dramsim3_enable:
            self.dramsim3_config = DRAMSim3Config(
                src_config_path=dramsim3_src_config_path,
                dst_config_path=dramsim3_dst_config_path,
                processor_clock_freq=processor_clock_freq,
                n_instance=n_instance,
                channel_size=channel_size,
                n_channel_per_instance=n_channel_per_instance,
                n_cmd_q_per_instance=n_cmd_q_per_instance,
            )
        else:
            self.dramsim3_config = None
        
    @property
    def n_instance(self) -> int:
        if self.dramsim3_enable and self.dramsim3_config is not None:
            return self.dramsim3_config.n_instance
        return self._n_instance
        
    @property
    def channel_size(self) -> int:
        if self.dramsim3_enable and self.dramsim3_config is not None:
            return self.dramsim3_config.channel_size * (1024 * 1024)  # MB -> Byte
        return self._channel_size
    
    @property
    def n_channel_per_instance(self) -> int:
        if self.dramsim3_enable and self.dramsim3_config is not None:
            return self.dramsim3_config.n_channel_per_instance
        return self._n_channel_per_instance
    
    @property
    def n_cmd_q_per_instance(self) -> int:
        if self.dramsim3_enable and self.dramsim3_config is not None:
            return self.dramsim3_config.n_cmd_q_per_instance
        return self._n_cmd_q_per_instance
    
    @property
    def n_channels(self) -> int:
        return self.n_instance * self.n_channel_per_instance

    def get_cycles(self, size: int) -> int:
        self.transfer_speed_bytes = (self.transfer_speed * (2 ** 20) * self.ch_io_width * self.ch_num // 8)  # Byte/s
        self.transfer_speed_per_cycles = self.transfer_speed_bytes / self.processor_clock_freq   # Byte/cycle
        
        return math.ceil(size / self.transfer_speed_per_cycles)
    
    def summary(self) -> dict[str, Any]:
        if self.dramsim3_enable:
            return {
                "dramsim3_enable": self.dramsim3_enable,
                "dramsim3_config": self.dramsim3_config.summary(),
            }
        else:
            return {
                "transfer_speed": self.transfer_speed,
                "ch_io_width": self.ch_io_width,
                "ch_num": self.ch_num,
                "burst_len": self.burst_len,
                "is_ddr": self.is_ddr,
                "processor_clock_freq": self.processor_clock_freq,
            }


class GlobalContextConfig:
    def __init__(
        self,
        
        n_npu_core: int,
        n_dma_core: int,
        
        l1_mem_bank_size: int,
        
        main_mem_config: MainMemoryConfig,
        main_mem_min_alloc_size: int = parse_mem_cap_str("32B"),
    ):
        self._n_npu_core = n_npu_core
        self._n_dma_core = n_dma_core
        
        self._l1_mem_bank_size   = l1_mem_bank_size
        
        self._main_mem_config = main_mem_config    
        self._main_mem_min_alloc_size = main_mem_min_alloc_size
        
        self._l1_mem_base_addr   = 0x0000
        self._main_mem_base_addr = self._l1_mem_base_addr + (self._n_npu_core * self._l1_mem_bank_size)
        
        self.icnt_core_id      = ICNT_CORE_NAME
        self.main_mem_core_id  = MAIN_MEM_CORE_NAME
        
        self.booksim_module_id = BOOKSIM_MODULE_ID
        self.dramsim_module_id = DRAMSIM_MODULE_ID 
        
        self._dma_core_ids: list[int] = [i for i in range(self._n_dma_core)]
        self._npu_core_ids: list[int] = [i + self._n_dma_core for i in range(self._n_npu_core)]
        
    @property
    def main_mem_config(self) -> MainMemoryConfig:
        return self._main_mem_config
    
    @property
    def npu_core_ids(self) -> list[int]:
        return self._npu_core_ids
    
    @property
    def dma_core_ids(self) -> list[int]:
        return self._dma_core_ids
    
    def summary(self) -> dict[str, Any]:
        return {
            "n_npu_core": self._n_npu_core,
            "n_dma_core": self._n_dma_core,
            "l1_mem_bank_size": self._l1_mem_bank_size,
            "main_mem_min_alloc_size": self._main_mem_min_alloc_size,
            "main_mem_base_addr": hex(self._main_mem_base_addr),
            "l1_mem_base_addr": hex(self._l1_mem_base_addr),
            "icnt_core_id": self.icnt_core_id,
            "main_mem_core_id": self.main_mem_core_id,
            "booksim_module_id": self.booksim_module_id,
            "dramsim_module_id": self.dramsim_module_id,
            "main_mem_config": self._main_mem_config.summary(),
        }


class GlobalContext:
    def __init__(self, config: GlobalContextConfig):
        self._config = config
        
        self._core_info: dict[Any, GlobalContextCoreInfo] = {}
        self._mem_info:  dict[Any, GlobalContextMemInfo]  = {}
        
        # for main_inst_id in range(self.n_main_mem_instances):
        #     base_addr = self._config._main_mem_base_addr + (main_inst_id * self.main_mem_channel_size * self._config._n_main_mem_cmd_q_per_instance)
        #     size = self.main_mem_channel_size * self._config._n_main_mem_cmd_q_per_instance
            
        #     mem_info = GlobalContextMemInfo(GlobalContextMemType.MAIN, main_inst_id, base_addr, size)
        #     self._mem_info[(GlobalContextMemType.MAIN, main_inst_id)] = mem_info
            
        #     for cmd_q_id in range(self._config._n_main_mem_cmd_q_per_instance):
        #         d = self._config._dma_core_ids[main_inst_id * self._config._n_main_mem_cmd_q_per_instance + cmd_q_id]
        #         core_info = GlobalContextCoreInfo(GlobalContextCoreType.DMA, d)
        #         self._core_info[(GlobalContextCoreType.DMA, d)] = core_info
                
        #         mem_info.add_owner_core(core_info)
        
        for main_inst_id in range(self.n_main_mem_instances):
            inst_base_addr = self._config._main_mem_base_addr + (main_inst_id * self.main_mem_channel_size * self._config.main_mem_config.n_channel_per_instance)
            
            for ch_id in range(self._config.main_mem_config.n_channel_per_instance):
                mem_id = main_inst_id * self._config.main_mem_config.n_channel_per_instance + ch_id
                base_addr = inst_base_addr + (ch_id * self.main_mem_channel_size)
                size = self.main_mem_channel_size
                
                mem_info = GlobalContextMemInfo(GlobalContextMemType.MAIN, mem_id, base_addr, size)
                self._mem_info[(GlobalContextMemType.MAIN, mem_id)] = mem_info
            
            for cmd_q_id in range(self._config.main_mem_config.n_cmd_q_per_instance):
                d = self._config._dma_core_ids[main_inst_id * self._config.main_mem_config.n_cmd_q_per_instance + cmd_q_id]
                core_info = GlobalContextCoreInfo(GlobalContextCoreType.DMA, d)
                self._core_info[(GlobalContextCoreType.DMA, d)] = core_info
                
                ch_id = cmd_q_id % self._config.main_mem_config.n_channel_per_instance
                mem_id = main_inst_id * self._config.main_mem_config.n_channel_per_instance + ch_id
                mem_info = self._mem_info[(GlobalContextMemType.MAIN, mem_id)]
                mem_info.add_owner_core(core_info)
            
        for l1_mem_bank_id in range(self._config._n_npu_core):
            base_addr = self._config._l1_mem_base_addr + (l1_mem_bank_id * self._config._l1_mem_bank_size)
            size = self._config._l1_mem_bank_size
            
            mem_info = GlobalContextMemInfo(GlobalContextMemType.L1, l1_mem_bank_id, base_addr, size)
            self._mem_info[(GlobalContextMemType.L1, l1_mem_bank_id)] = mem_info
            
            n = self._config._npu_core_ids[l1_mem_bank_id]
            core_info = GlobalContextCoreInfo(GlobalContextCoreType.NPU, n)
            self._core_info[(GlobalContextCoreType.NPU, n)] = core_info
            
            mem_info.add_owner_core(core_info)


    def get_core_info(self, core_type: GlobalContextCoreType, core_id: int) -> GlobalContextCoreInfo:
        return self._core_info[(core_type, core_id)]
    
    def get_mem_info(self, mem_type: GlobalContextMemType, mem_id: int) -> GlobalContextMemInfo:
        return self._mem_info[(mem_type, mem_id)]
        
    def get_mem_info_by_address(self, addr: int) -> GlobalContextMemInfo:
        for mem_info in self._mem_info.values():
            if (addr >= mem_info.base_addr) and (addr < mem_info.base_addr + mem_info.size):
                return mem_info
        return None
    
    def get_mem_type_by_address(self, addr: int) -> GlobalContextMemType:
        mem_info = self.get_mem_info_by_address(addr)
        if mem_info is not None:
            return mem_info.mem_type
        return None
    
    def get_main_mem_access_args(self, ptr: Pointer, size: int, is_write: bool) -> dict[str, int] | None:
        if not self.config.main_mem_config.dramsim3_enable:
            return None
        
        addr = ptr.addr - self.main_mem_base_addr
        ch_id = addr // self.main_mem_channel_size
        inst_id = ch_id // self.config.main_mem_config.n_channel_per_instance
        cmd_q_id = ch_id % self.config.main_mem_config.n_channel_per_instance
        addr_offset = addr % (self.main_mem_channel_size * self.config.main_mem_config.n_channel_per_instance)
        
        return {
            "inst_id": inst_id,
            "cmd_q_id": cmd_q_id,
            "addr": addr_offset,
            "size": size,
            "is_write": is_write,
        }
        
    @property
    def n_dma_engine_per_channel(self) -> int:
        return self._config._n_dma_core // self._config.main_mem_config.n_channels
    
    @property
    def config(self) -> GlobalContextConfig:
        return self._config
    
    @property
    def icnt_core_id(self) -> str:
        return self._config.icnt_core_id
    
    @property
    def main_mem_core_id(self) -> str:
        return self._config.main_mem_core_id
    
    @property
    def booksim_module_id(self) -> str:
        return self._config.booksim_module_id
    
    @property
    def dramsim_module_id(self) -> str:
        return self._config.dramsim_module_id
    
    @property
    def npu_core_ids(self) -> list[int]:
        return self._config._npu_core_ids
    
    @property
    def dma_core_ids(self) -> list[int]:
        return self._config._dma_core_ids
    
    @property
    def n_main_mem_cmd_q_per_instance(self) -> int:
        return self._config.main_mem_config.n_cmd_q_per_instance
    
    @property
    def main_mem_base_addr(self) -> int:
        return self._config._main_mem_base_addr
    
    @property
    def main_mem_channel_size(self) -> int:
        return self._config.main_mem_config.channel_size
    
    @property
    def n_main_mem_instances(self) -> int:
        return self._config.main_mem_config.n_instance
    
    @property
    def n_main_mem_cmd_q_per_instance(self) -> int:
        return self._config.main_mem_config.n_cmd_q_per_instance
    
    @property
    def main_mem_size(self) -> int:
        return self.n_main_mem_instances * self._config.main_mem_config.n_channel_per_instance * self._config.main_mem_config.channel_size
    
    
if __name__ == "__main__":
    config = GlobalContextConfig(
        n_npu_core         = 4,
        n_main_mem_instances = 8,
        n_dma_core         = 8,
        l1_mem_bank_size   = parse_mem_cap_str("4MB"),
        main_mem_bank_size = parse_mem_cap_str("32MB"),
    )
    
    global_context = GlobalContext(config)
    
    global_context.create_l1_mem_space(parse_mem_cap_str("2MB"))
    global_context.create_main_mem_space(parse_mem_cap_str("16MB"))
    
    npu_core_ids = global_context.npu_core_ids[:2]
    
    ptr1 = global_context.allocate_l1_local_buffer(parse_mem_cap_str("256KB"), npu_core_ids[0])
    ptr2 = global_context.allocate_l1_local_buffer(parse_mem_cap_str("128KB"), npu_core_ids[1])
    ptr3 = global_context.allocate_l1_sharded_buffer(parse_mem_cap_str("512KB"), npu_core_ids)
    bptr = global_context.allocate_main_buffer(parse_mem_cap_str("4MB"))
    
    print(f"L1 Local Buffer Ptr NPU{npu_core_ids[0]}: {ptr1}")
    print(f"L1 Local Buffer Ptr NPU{npu_core_ids[1]}: {ptr2}")
    print(f"L1 Sharded Buffer Ptr NPU{npu_core_ids[0]} and NPU{npu_core_ids[1]}: {ptr3}")
    print(f"Main Buffer Ptr: {bptr}")
    
    global_context.remove_l1_mem_space()
    global_context.remove_main_mem_space()