import enum
import math
from typing import Sequence, Any

from neuromta.framework import *
from neuromta.component.companions.dramsim import DRAMSim3Config, PYDRAMSIM3_AVAILABLE


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
    def __init__(self, mem_type: GlobalContextMemType, mem_id: int, base_addr: int, size: int, dynamic_space_size: int, scheduled_space_size: int=None):
        self.mem_type   = mem_type
        self.mem_id     = mem_id
        
        self._total_space_size   = size
        self._dynamic_space_size  = dynamic_space_size
        self._scheduled_space_size = scheduled_space_size if scheduled_space_size is not None else (self._total_space_size - dynamic_space_size)

        if self._total_space_size < self._dynamic_space_size + self._scheduled_space_size:
            raise ValueError(f"Total memory size {self._total_space_size} is smaller than the sum of dynamic space size {self._dynamic_space_size} and scheduled space size {self._scheduled_space_size}.")
        
        self.owner_core_ids: list[int]  = []
        
        self._stack_base_addrs: list[int] = None
        self._stack_sizes:      list[int] = None
        self._stack_write_ptrs: list[int] = None
        
        self._bank_size = min(self._total_space_size, MemoryHandle.MAX_BANK_SIZE)
        self._n_banks   = math.ceil(self._total_space_size / self._bank_size)
        
        self._mem_handle: MemoryHandle = MemoryHandle(
            base_addr=base_addr,
            bank_size=self._bank_size,
            n_banks=self._n_banks,
            dynamic_space_size=self._dynamic_space_size,
            static_space_size=self._scheduled_space_size,
        )
        
    def add_owner_core(self, core_info: GlobalContextCoreInfo):
        if core_info.core_id not in self.owner_core_ids:
            self.owner_core_ids.append(core_info.core_id)
            core_info.owned_mem_info = self
            
    def get_owner_id(self, core_id: int) -> int | None:
        if core_id in self.owner_core_ids:
            return self.owner_core_ids.index(core_id)
        return None
        
    def __eq__(self, value):
        if isinstance(value, GlobalContextMemInfo):
            return (self.mem_type == value.mem_type) and (self.mem_id == value.mem_id)
        return False
        
    def create_stack(self, stack_size: int) -> int:
        if not self.is_stack_initialized:
            self._stack_base_addrs = [self._mem_handle.scheduled_space_addr]
            self._stack_sizes      = [stack_size]
            self._stack_write_ptrs = [0]
        else:
            if sum(self._stack_sizes) + stack_size > self.scheduled_space_size:
                raise MemoryError(f"Not enough memory to add stack of size {stack_size}. Available memory size: {self.scheduled_space_size - sum(self._stack_sizes)}")
            
            self._stack_base_addrs.append(self._mem_handle.scheduled_space_addr + sum(self._stack_sizes))
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
    def total_space_size(self) -> int:
        return self._total_space_size
    
    @property
    def dynamic_space_size(self) -> int:
        return self._dynamic_space_size
    
    @property
    def scheduled_space_size(self) -> int:
        return self._scheduled_space_size
    
    @property
    def total_base_addr(self) -> int:
        return self._mem_handle.base_addr
    
    @property
    def dynamic_base_addr(self) -> int:
        return self._mem_handle.dynamic_space_addr
    
    @property
    def scheduled_base_addr(self) -> int:
        return self._mem_handle.scheduled_space_addr
    
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
        
        # DRAMSim3 Configuation (if needed)
        dramsim3_enable: bool           = None,
        dramsim3_src_config_path: str   = "GDDR6_8Gb_x16.ini",
        dramsim3_dst_config_path: str   = "dramsim3_config.ini",
        dramsim3_max_issue_per_cmd_q_per_cycle: int = 1,
        
        # Lightweight Memory Simulation Configuration
        lightweight_read_latency_cycles: int = 0,
        lightweight_write_latency_cycles: int = 0,
        lightweight_channel_bandwidth_bytes_per_cycle: int = 64,
        lightweight_dma_granularity: int = parse_mem_cap_str("256B"),
        lightweight_command_issue_gap_cycles: int = 0,
        lightweight_read_write_turnaround_cycles: int = 0,
        lightweight_request_startup_latency_cycles: int = 0,
        lightweight_burst_size_bytes: int = parse_mem_cap_str("64B"),
        lightweight_n_rank_per_channel: int = 1,
        lightweight_n_bank_group_per_rank: int = 4,
        lightweight_n_bank_per_bank_group: int = 4,
        lightweight_row_size_bytes: int = parse_mem_cap_str("2KB"),
        lightweight_row_hit_latency_cycles: int = 0,
        lightweight_row_miss_penalty_cycles: int = 0,
        lightweight_bank_group_penalty_cycles: int = 0,
        lightweight_dma_max_outstanding_bursts: int = 32,
        lightweight_latency_amortization_bytes: int = parse_mem_cap_str("5KB"),    # magic number!
    ):
        if dramsim3_enable is None:
            dramsim3_enable = PYDRAMSIM3_AVAILABLE
        
        self.transfer_speed         = transfer_speed    # transfer speed per pin (MT/s)
        self.ch_io_width            = ch_io_width       # io channel width (bits)
        self.burst_len              = burst_len         # burst length
        self.is_ddr                 = is_ddr
        self.processor_clock_freq   = processor_clock_freq
        
        self._n_instance             = n_instance
        self._channel_size           = channel_size
        self._n_channel_per_instance = n_channel_per_instance

        self.dramsim3_enable        = dramsim3_enable
        
        if self.dramsim3_enable:
            self.dramsim3_config = DRAMSim3Config(
                src_config_path=dramsim3_src_config_path,
                dst_config_path=dramsim3_dst_config_path,
                processor_clock_freq=processor_clock_freq,
                n_instance=n_instance,
                channel_size=channel_size,
                n_channel_per_instance=n_channel_per_instance,
                max_issue_per_cmd_q_per_cycle=dramsim3_max_issue_per_cmd_q_per_cycle,
            )
        else:
            self.dramsim3_config = None
            
        self.lightweight_read_latency_cycles = lightweight_read_latency_cycles
        self.lightweight_write_latency_cycles = lightweight_write_latency_cycles
        self.lightweight_channel_bandwidth_bytes_per_cycle = lightweight_channel_bandwidth_bytes_per_cycle
        self.lightweight_dma_granularity = lightweight_dma_granularity
        self.lightweight_command_issue_gap_cycles = lightweight_command_issue_gap_cycles
        self.lightweight_read_write_turnaround_cycles = lightweight_read_write_turnaround_cycles
        self.lightweight_request_startup_latency_cycles = lightweight_request_startup_latency_cycles
        self.lightweight_burst_size_bytes = lightweight_burst_size_bytes
        self.lightweight_n_rank_per_channel = lightweight_n_rank_per_channel
        self.lightweight_n_bank_group_per_rank = lightweight_n_bank_group_per_rank
        self.lightweight_n_bank_per_bank_group = lightweight_n_bank_per_bank_group
        self.lightweight_row_size_bytes = lightweight_row_size_bytes
        self.lightweight_row_hit_latency_cycles = lightweight_row_hit_latency_cycles
        self.lightweight_row_miss_penalty_cycles = lightweight_row_miss_penalty_cycles
        self.lightweight_bank_group_penalty_cycles = lightweight_bank_group_penalty_cycles
        self.lightweight_dma_max_outstanding_bursts = lightweight_dma_max_outstanding_bursts
        self.lightweight_latency_amortization_bytes = lightweight_latency_amortization_bytes
        
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
    def n_channels(self) -> int:
        return self.n_instance * self.n_channel_per_instance
    
    @property
    def peak_bandwidth_per_cycle(self) -> float:
        if self.dramsim3_enable:
            return self.dramsim3_config.peak_bandwidth() / self.processor_clock_freq
        else:
            return (self.transfer_speed * self.ch_io_width * self.n_channels) / 8 / self.processor_clock_freq  # Byte/cycle
        
    @property
    def channel_size_per_instance(self) -> int:
        return self.channel_size * self.n_channel_per_instance
    
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
                "burst_len": self.burst_len,
                "is_ddr": self.is_ddr,
                "processor_clock_freq": self.processor_clock_freq,
                "lightweight_read_latency_cycles": self.lightweight_read_latency_cycles,
                "lightweight_write_latency_cycles": self.lightweight_write_latency_cycles,
                "lightweight_channel_bandwidth_bytes_per_cycle": self.lightweight_channel_bandwidth_bytes_per_cycle,
                "lightweight_dma_granularity": self.lightweight_dma_granularity,
                "lightweight_command_issue_gap_cycles": self.lightweight_command_issue_gap_cycles,
                "lightweight_read_write_turnaround_cycles": self.lightweight_read_write_turnaround_cycles,
                "lightweight_request_startup_latency_cycles": self.lightweight_request_startup_latency_cycles,
                "lightweight_burst_size_bytes": self.lightweight_burst_size_bytes,
                "lightweight_n_rank_per_channel": self.lightweight_n_rank_per_channel,
                "lightweight_n_bank_group_per_rank": self.lightweight_n_bank_group_per_rank,
                "lightweight_n_bank_per_bank_group": self.lightweight_n_bank_per_bank_group,
                "lightweight_row_size_bytes": self.lightweight_row_size_bytes,
                "lightweight_row_hit_latency_cycles": self.lightweight_row_hit_latency_cycles,
                "lightweight_row_miss_penalty_cycles": self.lightweight_row_miss_penalty_cycles,
                "lightweight_bank_group_penalty_cycles": self.lightweight_bank_group_penalty_cycles,
                "lightweight_dma_max_outstanding_bursts": self.lightweight_dma_max_outstanding_bursts,
                "lightweight_latency_amortization_bytes": self.lightweight_latency_amortization_bytes,
            }


class MemorySimulator:
    def __init__(self, config: MainMemoryConfig, mem_addr_offset: int):
        self.config = config
        self.mem_addr_offset = mem_addr_offset
        self.mem_addr_end = self.mem_addr_offset + (self.config.n_instance * self.config.n_channel_per_instance * self.config.channel_size)
        self.reset()
        
    def reset(self) -> None:
        self._channel_next_free_cycle = {
            (instance_id, channel_id): 0
            for instance_id in range(self.config.n_instance)
            for channel_id in range(self.config.n_channel_per_instance)
        }
        self._channel_command_next_free_cycle = {
            (instance_id, channel_id): 0
            for instance_id in range(self.config.n_instance)
            for channel_id in range(self.config.n_channel_per_instance)
        }
        self._channel_last_is_write = {
            (instance_id, channel_id): None
            for instance_id in range(self.config.n_instance)
            for channel_id in range(self.config.n_channel_per_instance)
        }
        self._bank_next_free_cycle = {}
        self._bank_group_next_free_cycle = {}
        self._open_row = {}
        
    def _cfg(self, name: str, default: int) -> int:
        return getattr(self.config, name, default)
        
    def check_address_range(self, address: int) -> bool:
        if address < self.mem_addr_offset:
            return False
        if address >= self.mem_addr_end:
            return False
        return True
    
    def get_instance_id_with_address(self, address: int) -> int:
        if not self.check_address_range(address):
            raise ValueError(f"Address {address} is out of range")
        return ((address - self.mem_addr_offset) // self.config.channel_size_per_instance) % self.config.n_instance
    
    def get_memory_mapping(self, address: int) -> dict[str, int]:
        instance_id = self.get_instance_id_with_address(address)
        addr_offset = (address - self.mem_addr_offset) % self.config.channel_size_per_instance
        channel_id = addr_offset // self.config.channel_size
        channel_offset = addr_offset % self.config.channel_size
        burst_size = max(1, self._cfg("lightweight_burst_size_bytes", self.config.lightweight_dma_granularity))
        row_size = max(burst_size, self._cfg("lightweight_row_size_bytes", 2048))
        n_rank = max(1, self._cfg("lightweight_n_rank_per_channel", 1))
        n_bank_group = max(1, self._cfg("lightweight_n_bank_group_per_rank", 1))
        n_bank = max(1, self._cfg("lightweight_n_bank_per_bank_group", 1))
        n_bank_slots = n_rank * n_bank_group * n_bank
        bursts_per_row = max(1, row_size // burst_size)
        burst_index = channel_offset // burst_size
        burst_in_rank_space = burst_index % (n_bank_slots * bursts_per_row)
        bank_slot = burst_in_rank_space % n_bank_slots
        column_burst = burst_in_rank_space // n_bank_slots
        row_id = burst_index // (n_bank_slots * bursts_per_row)
        rank_id = bank_slot // (n_bank_group * n_bank)
        bank_slot_rem = bank_slot % (n_bank_group * n_bank)
        bank_group_id = bank_slot_rem // n_bank
        bank_id = bank_slot_rem % n_bank
        column_offset = (column_burst * burst_size) + (channel_offset % burst_size)
        return {
            "inst_id": instance_id,
            "addr": addr_offset,
            "channel_id": channel_id,
            "channel_offset": channel_offset,
            "rank_id": rank_id,
            "bank_group_id": bank_group_id,
            "bank_id": bank_id,
            "row_id": row_id,
            "column_offset": column_offset,
        }
    
    def _iter_dma_chunks(self, address: int, size: int) -> list[dict[str, int]]:
        if size < 0:
            raise ValueError(f"Invalid size: {size}")
        if size == 0:
            return []
        if not self.check_address_range(address) or not self.check_address_range(address + size - 1):
            raise ValueError(f"Address range [{address}, {address + size}) is out of range")
        
        chunks = []
        remaining = size
        current_addr = address
        granularity = max(1, self.config.lightweight_dma_granularity)
        burst_size = max(1, self._cfg("lightweight_burst_size_bytes", granularity))
        row_size = max(burst_size, self._cfg("lightweight_row_size_bytes", 2048))
        
        while remaining > 0:
            mapping = self.get_memory_mapping(current_addr)
            local_addr = current_addr - self.mem_addr_offset
            granularity_remaining = granularity - (local_addr % granularity)
            burst_remaining = burst_size - (local_addr % burst_size)
            row_remaining = row_size - (mapping["column_offset"] % row_size)
            channel_remaining = self.config.channel_size - mapping["channel_offset"]
            chunk_size = min(remaining, granularity_remaining, burst_remaining, row_remaining, channel_remaining)
            chunks.append({
                "address": current_addr,
                "size": chunk_size,
                **mapping,
            })
            current_addr += chunk_size
            remaining -= chunk_size
        
        return chunks
    
    def _retire_completed_bursts(self, completions: list[int], issue_cycle: int) -> list[int]:
        return [cycle for cycle in completions if cycle > issue_cycle]
    
    def send_request(
        self,
        addr: int,
        size: int,
        is_write: bool,
        current_cycle: int = 0,
    ) -> dict:
        if current_cycle < 0:
            raise ValueError("current_cycle must be non-negative")
        
        chunks = self._iter_dma_chunks(address=addr, size=size)
        raw_base_latency = self.config.lightweight_write_latency_cycles if is_write else self.config.lightweight_read_latency_cycles
        amortization_bytes = max(1, self._cfg("lightweight_latency_amortization_bytes", size if size > 0 else 1))
        latency_scale = min(1.0, size / amortization_bytes) if size > 0 else 0.0
        scale_latency = lambda value: 0 if value <= 0 else max(1, math.ceil(value * latency_scale))
        base_latency = scale_latency(raw_base_latency)
        bandwidth = self.config.lightweight_channel_bandwidth_bytes_per_cycle
        issue_gap = self.config.lightweight_command_issue_gap_cycles
        turnaround = self.config.lightweight_read_write_turnaround_cycles
        request_startup = scale_latency(self._cfg("lightweight_request_startup_latency_cycles", 0))
        row_hit_latency = scale_latency(self._cfg("lightweight_row_hit_latency_cycles", 0))
        row_miss_penalty = scale_latency(self._cfg("lightweight_row_miss_penalty_cycles", 0))
        bank_group_penalty = scale_latency(self._cfg("lightweight_bank_group_penalty_cycles", 0))
        max_outstanding = max(1, self._cfg("lightweight_dma_max_outstanding_bursts", len(chunks) if chunks else 1))
        
        scheduled_chunks = []
        finish_cycle = current_cycle + request_startup
        issue_cycle = current_cycle + request_startup
        outstanding_completions: list[int] = []
        
        for chunk in chunks:
            outstanding_completions = self._retire_completed_bursts(outstanding_completions, issue_cycle)
            if len(outstanding_completions) >= max_outstanding:
                earliest_completion = min(outstanding_completions)
                issue_cycle = max(issue_cycle, earliest_completion)
                outstanding_completions = self._retire_completed_bursts(outstanding_completions, issue_cycle)
            
            channel_key = (chunk["inst_id"], chunk["channel_id"])
            bank_group_key = (*channel_key, chunk["rank_id"], chunk["bank_group_id"])
            bank_key = (*bank_group_key, chunk["bank_id"])
            row_key = bank_key
            
            command_start_cycle = max(issue_cycle, self._channel_command_next_free_cycle[channel_key])
            bank_ready_cycle = self._bank_next_free_cycle.get(bank_key, 0)
            bank_group_ready_cycle = self._bank_group_next_free_cycle.get(bank_group_key, 0)
            row_latency = row_hit_latency if self._open_row.get(row_key) == chunk["row_id"] else row_miss_penalty
            dram_ready_cycle = max(command_start_cycle, bank_ready_cycle, bank_group_ready_cycle) + row_latency
            
            bus_start_cycle = max(dram_ready_cycle, self._channel_next_free_cycle[channel_key])
            last_is_write = self._channel_last_is_write[channel_key]
            if last_is_write is not None and last_is_write != is_write:
                bus_start_cycle += turnaround
            
            transfer_cycles = max(1, math.ceil(chunk["size"] / bandwidth))
            bus_finish_cycle = bus_start_cycle + transfer_cycles
            chunk_finish_cycle = bus_finish_cycle + base_latency
            queue_delay_cycles = max(0, bus_start_cycle - issue_cycle)
            
            self._channel_command_next_free_cycle[channel_key] = command_start_cycle + issue_gap
            self._channel_next_free_cycle[channel_key] = bus_finish_cycle
            self._channel_last_is_write[channel_key] = is_write
            self._bank_next_free_cycle[bank_key] = bus_finish_cycle
            self._bank_group_next_free_cycle[bank_group_key] = bus_start_cycle + bank_group_penalty
            self._open_row[row_key] = chunk["row_id"]
            finish_cycle = max(finish_cycle, chunk_finish_cycle)
            outstanding_completions.append(chunk_finish_cycle)
            
            scheduled_chunk = dict(chunk)
            scheduled_chunk.update({
                "command_start_cycle": command_start_cycle,
                "dram_ready_cycle": dram_ready_cycle,
                "bus_start_cycle": bus_start_cycle,
                "bus_finish_cycle": bus_finish_cycle,
                "finish_cycle": chunk_finish_cycle,
                "transfer_cycles": transfer_cycles,
                "base_latency_cycles": base_latency,
                "row_latency_cycles": row_latency,
                "queue_delay_cycles": queue_delay_cycles,
            })
            scheduled_chunks.append(scheduled_chunk)
            issue_cycle = command_start_cycle + issue_gap
        
        return {
            "current_cycle": current_cycle,
            "finish_cycle": finish_cycle,
            "latency_cycles": finish_cycle - current_cycle,
            "n_chunks": len(scheduled_chunks),
            "chunks": scheduled_chunks,
        }
    
    @property
    def channel_next_free_cycle(self) -> dict[tuple[int, int], int]:
        return dict(self._channel_next_free_cycle)


class GlobalContextConfig:
    def __init__(
        self,
        
        n_npu_core: int,
        n_dma_core: int,
        
        l1_mem_bank_size: int,
        l1_mem_dynamic_space_size_per_bank: int,
        
        main_mem_config: MainMemoryConfig,
        main_mem_min_alloc_size: int = parse_mem_cap_str("32B"),
    ):
        self._n_npu_core = n_npu_core
        self._n_dma_core = n_dma_core
        
        self._l1_mem_bank_size   = l1_mem_bank_size
        self._l1_mem_dynamic_space_size_per_bank = l1_mem_dynamic_space_size_per_bank
        
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
        
        n_dma_cores_per_inst = self._config._n_dma_core // self._config.main_mem_config.n_instance
        if n_dma_cores_per_inst == 0:
            raise ValueError(f"Number of DMA cores ({self._config._n_dma_core}) must be greater than or equal to the number of main memory instances ({self._config.main_mem_config.n_instance}).")
        
        for main_inst_id in range(self.n_main_mem_instances):
            inst_base_addr = self._config._main_mem_base_addr + (main_inst_id * self.main_mem_channel_size * self._config.main_mem_config.n_channel_per_instance)
            
            mem_id = main_inst_id
            base_addr = inst_base_addr
            size = self.main_mem_channel_size * self._config.main_mem_config.n_channel_per_instance
            
            mem_info = GlobalContextMemInfo(GlobalContextMemType.MAIN, mem_id, base_addr, size, dynamic_space_size=0)
            self._mem_info[(GlobalContextMemType.MAIN, mem_id)] = mem_info
            
        for dma_engine_id in range(self._config._n_dma_core):
            main_inst_id = dma_engine_id // n_dma_cores_per_inst
            
            d = self._config._dma_core_ids[dma_engine_id]
            core_info = GlobalContextCoreInfo(GlobalContextCoreType.DMA, d)
            self._core_info[(GlobalContextCoreType.DMA, d)] = core_info
            
            mem_id = main_inst_id
            mem_info = self._mem_info[(GlobalContextMemType.MAIN, mem_id)]
            mem_info.add_owner_core(core_info)
            
        for l1_mem_bank_id in range(self._config._n_npu_core):
            base_addr = self._config._l1_mem_base_addr + (l1_mem_bank_id * self._config._l1_mem_bank_size)
            size = self._config._l1_mem_bank_size
            
            mem_info = GlobalContextMemInfo(GlobalContextMemType.L1, l1_mem_bank_id, base_addr, size, dynamic_space_size=self._config._l1_mem_dynamic_space_size_per_bank)
            self._mem_info[(GlobalContextMemType.L1, l1_mem_bank_id)] = mem_info
            
            n = self._config._npu_core_ids[l1_mem_bank_id]
            core_info = GlobalContextCoreInfo(GlobalContextCoreType.NPU, n)
            self._core_info[(GlobalContextCoreType.NPU, n)] = core_info
            
            mem_info.add_owner_core(core_info)
        
        if self._config.main_mem_config.dramsim3_enable:
            self._main_mem_sim = None
        else:
            self._main_mem_sim = MemorySimulator(
                config=self._config.main_mem_config,
                mem_addr_offset=self._config._main_mem_base_addr,
            )

    def get_core_info(self, core_type: GlobalContextCoreType, core_id: int) -> GlobalContextCoreInfo:
        return self._core_info[(core_type, core_id)]
    
    def get_mem_info(self, mem_type: GlobalContextMemType, mem_id: int) -> GlobalContextMemInfo:
        return self._mem_info[(mem_type, mem_id)]
        
    def get_mem_info_by_address(self, addr: int) -> GlobalContextMemInfo:
        for mem_info in self._mem_info.values():
            if (addr >= mem_info.total_base_addr) and (addr < mem_info.total_base_addr + mem_info.total_space_size):
                return mem_info
        return None
    
    def get_mem_type_by_address(self, addr: int) -> GlobalContextMemType:
        mem_info = self.get_mem_info_by_address(addr)
        if mem_info is not None:
            return mem_info.mem_type
        return None

    def get_main_mem_access_args(self, ptr: Pointer, size: int, is_write: bool) -> dict[str, int] | None:
        addr = ptr.addr - self.main_mem_base_addr
        inst_id = addr // (self.config.main_mem_config.n_channel_per_instance * self.main_mem_channel_size)
        addr_offset = addr % (self.main_mem_channel_size * self.config.main_mem_config.n_channel_per_instance)
        
        return {
            "inst_id": inst_id,
            "addr": addr_offset,
            "size": size,
            "is_write": is_write,
        }
        
    def get_main_mem_strided_access_args(self, ptr: Pointer, row_size: int, row_num: int, stride: int, is_write: bool) -> list[dict[str, int]] | None:
        return [
            self.get_main_mem_access_args(Pointer(ptr.addr + i * stride), row_size, is_write)
            for i in range(row_num)
        ]
        
    @property
    def n_dma_engine_per_instance(self) -> int:
        return self._config._n_dma_core // self._config.main_mem_config.n_instance
    
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
    def main_mem_base_addr(self) -> int:
        return self._config._main_mem_base_addr
    
    @property
    def main_mem_channel_size(self) -> int:
        return self._config.main_mem_config.channel_size
    
    @property
    def n_main_mem_instances(self) -> int:
        return self._config.main_mem_config.n_instance
    
    @property
    def main_mem_size(self) -> int:
        return self.n_main_mem_instances * self._config.main_mem_config.n_channel_per_instance * self._config.main_mem_config.channel_size

    @property
    def main_mem_simulator(self) -> MemorySimulator | None:
        return self._main_mem_sim
    
    @property
    def is_main_mem_simulator_enabled(self) -> bool:
        return self._main_mem_sim is not None
