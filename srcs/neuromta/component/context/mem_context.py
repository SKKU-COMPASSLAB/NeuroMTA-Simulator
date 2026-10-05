import math
from typing import Any

from neuromta.framework.parser_utils import parse_mem_cap_str, parse_freq_str
from neuromta.component.companions.dramsim import DRAMSim3Config, PYDRAMSIM3_AVAILABLE

__all__ = [
    "MemoryContext",
    "MemoryConfig",
    "MemorySimulator",
]


class MemoryConfig:
    def __init__(
        self,

        mem_addr_offset: int         = 0x00000000,

        processor_clock_freq: int   = parse_freq_str("1GHz"),
        n_instance: int=1,
        channel_size: int=parse_mem_cap_str("4GB"),
        n_channel_per_instance: int=1,

        dramsim3_enable: bool = True,
        dramsim3_module_id: str         = "DRAMSIM",
        dramsim3_src_config_path: str   = "GDDR6_8Gb_x16.ini",
        dramsim3_dst_config_path: str   = "dramsim3_config.ini",
        dramsim3_max_issue_per_cmd_q_per_cycle: int = 1,

        lightweight_read_latency_cycles: int = 0,
        lightweight_write_latency_cycles: int = 0,
        lightweight_write_accept_latency_cycles: int = 1,
        lightweight_write_completion_policy: str = "retire",
        lightweight_channel_bandwidth_bytes_per_cycle: int = 64,
        lightweight_dma_granularity: int = parse_mem_cap_str("256B"),
        lightweight_address_mapping: str = "contiguous",
        lightweight_channel_interleave_bytes: int = parse_mem_cap_str("64B"),
        lightweight_address_mapping_scheme: str = "rorabgbachco",
        lightweight_dram_rows: int = 32768,
        lightweight_dram_columns: int = 64,
        lightweight_dram_burst_length: int = 4,
        lightweight_instance_command_issue_gap_cycles: int = 0,
        lightweight_command_issue_gap_cycles: int = 0,
        lightweight_read_write_turnaround_cycles: int = 0,
        lightweight_read_to_write_turnaround_cycles: int | None = None,
        lightweight_write_to_read_turnaround_cycles: int | None = None,
        lightweight_request_startup_latency_cycles: int = 0,
        lightweight_burst_size_bytes: int = parse_mem_cap_str("64B"),
        lightweight_n_rank_per_channel: int = 1,
        lightweight_n_bank_group_per_rank: int = 4,
        lightweight_n_bank_per_bank_group: int = 4,
        lightweight_row_size_bytes: int = parse_mem_cap_str("2KB"),
        lightweight_row_hit_latency_cycles: int = 0,
        lightweight_row_miss_penalty_cycles: int = 0,
        lightweight_row_conflict_penalty_cycles: int | None = None,
        lightweight_bank_group_penalty_cycles: int = 0,
        lightweight_dma_max_outstanding_bursts: int = 32,
        lightweight_channel_max_outstanding_bursts: int = 16,
        lightweight_request_queue_depth: int = 32,
        lightweight_concurrent_request_command_gap_cycles: int = 0,
        lightweight_concurrent_request_command_gap_threshold: int = 0,
        lightweight_concurrent_request_command_gap_limit: int = 1,
        lightweight_latency_amortization_bytes: int = parse_mem_cap_str("5KB"),
        lightweight_enable_latency_amortization: bool = True,

        instance_dma_map: dict[int, int] = None,
    ):
        self.mem_addr_offset = mem_addr_offset

        self.processor_clock_freq = processor_clock_freq
        self.n_instance = n_instance
        self.channel_size = channel_size
        self.n_channel_per_instance = n_channel_per_instance

        self.dramsim3_enable = dramsim3_enable
        self.dramsim3_module_id = dramsim3_module_id
        self.dramsim3_config = None
        if self.dramsim3_enable:
            if not PYDRAMSIM3_AVAILABLE:
                raise RuntimeError("DRAMSim3 is not available. Please ensure that the DRAMSim3 companion module is properly installed and configured.")
            self.dramsim3_config = DRAMSim3Config(
                src_config_path=dramsim3_src_config_path,
                dst_config_path=dramsim3_dst_config_path,
                processor_clock_freq=processor_clock_freq,
                n_instance=n_instance,
                channel_size=channel_size,
                n_channel_per_instance=n_channel_per_instance,
                max_issue_per_cmd_q_per_cycle=dramsim3_max_issue_per_cmd_q_per_cycle,
            )

        if lightweight_channel_bandwidth_bytes_per_cycle <= 0:
            raise ValueError("lightweight_channel_bandwidth_bytes_per_cycle must be positive")
        if lightweight_dma_granularity <= 0:
            raise ValueError("lightweight_dma_granularity must be positive")
        if lightweight_address_mapping not in ["contiguous", "burst_interleaved", "dramsim3"]:
            raise ValueError(f"Unsupported lightweight_address_mapping: {lightweight_address_mapping}")
        if lightweight_channel_interleave_bytes <= 0:
            raise ValueError("lightweight_channel_interleave_bytes must be positive")
        if lightweight_dram_rows <= 0:
            raise ValueError("lightweight_dram_rows must be positive")
        if lightweight_dram_columns <= 0:
            raise ValueError("lightweight_dram_columns must be positive")
        if lightweight_dram_burst_length <= 0:
            raise ValueError("lightweight_dram_burst_length must be positive")
        if lightweight_instance_command_issue_gap_cycles < 0:
            raise ValueError("lightweight_instance_command_issue_gap_cycles must be non-negative")
        if lightweight_command_issue_gap_cycles < 0:
            raise ValueError("lightweight_command_issue_gap_cycles must be non-negative")
        if lightweight_read_write_turnaround_cycles < 0:
            raise ValueError("lightweight_read_write_turnaround_cycles must be non-negative")
        if lightweight_read_to_write_turnaround_cycles is not None and lightweight_read_to_write_turnaround_cycles < 0:
            raise ValueError("lightweight_read_to_write_turnaround_cycles must be non-negative")
        if lightweight_write_to_read_turnaround_cycles is not None and lightweight_write_to_read_turnaround_cycles < 0:
            raise ValueError("lightweight_write_to_read_turnaround_cycles must be non-negative")
        if lightweight_request_startup_latency_cycles < 0:
            raise ValueError("lightweight_request_startup_latency_cycles must be non-negative")
        if lightweight_write_accept_latency_cycles < 0:
            raise ValueError("lightweight_write_accept_latency_cycles must be non-negative")
        if lightweight_write_completion_policy not in ["retire", "accept"]:
            raise ValueError(f"Unsupported lightweight_write_completion_policy: {lightweight_write_completion_policy}")
        if lightweight_burst_size_bytes <= 0:
            raise ValueError("lightweight_burst_size_bytes must be positive")
        if lightweight_n_rank_per_channel <= 0:
            raise ValueError("lightweight_n_rank_per_channel must be positive")
        if lightweight_n_bank_group_per_rank <= 0:
            raise ValueError("lightweight_n_bank_group_per_rank must be positive")
        if lightweight_n_bank_per_bank_group <= 0:
            raise ValueError("lightweight_n_bank_per_bank_group must be positive")
        if lightweight_row_size_bytes <= 0:
            raise ValueError("lightweight_row_size_bytes must be positive")
        if lightweight_row_hit_latency_cycles < 0:
            raise ValueError("lightweight_row_hit_latency_cycles must be non-negative")
        if lightweight_row_miss_penalty_cycles < 0:
            raise ValueError("lightweight_row_miss_penalty_cycles must be non-negative")
        if lightweight_row_conflict_penalty_cycles is not None and lightweight_row_conflict_penalty_cycles < 0:
            raise ValueError("lightweight_row_conflict_penalty_cycles must be non-negative")
        if lightweight_bank_group_penalty_cycles < 0:
            raise ValueError("lightweight_bank_group_penalty_cycles must be non-negative")
        if lightweight_dma_max_outstanding_bursts <= 0:
            raise ValueError("lightweight_dma_max_outstanding_bursts must be positive")
        if lightweight_channel_max_outstanding_bursts <= 0:
            raise ValueError("lightweight_channel_max_outstanding_bursts must be positive")
        if lightweight_request_queue_depth <= 0:
            raise ValueError("lightweight_request_queue_depth must be positive")
        if lightweight_concurrent_request_command_gap_cycles < 0:
            raise ValueError("lightweight_concurrent_request_command_gap_cycles must be non-negative")
        if lightweight_concurrent_request_command_gap_threshold < 0:
            raise ValueError("lightweight_concurrent_request_command_gap_threshold must be non-negative")
        if lightweight_concurrent_request_command_gap_limit < 0:
            raise ValueError("lightweight_concurrent_request_command_gap_limit must be non-negative")
        if lightweight_latency_amortization_bytes <= 0:
            raise ValueError("lightweight_latency_amortization_bytes must be positive")

        self.lightweight_read_latency_cycles = lightweight_read_latency_cycles
        self.lightweight_write_latency_cycles = lightweight_write_latency_cycles
        self.lightweight_write_accept_latency_cycles = lightweight_write_accept_latency_cycles
        self.lightweight_write_completion_policy = lightweight_write_completion_policy
        self.lightweight_channel_bandwidth_bytes_per_cycle = lightweight_channel_bandwidth_bytes_per_cycle
        self.lightweight_dma_granularity = lightweight_dma_granularity
        self.lightweight_address_mapping = lightweight_address_mapping
        self.lightweight_channel_interleave_bytes = lightweight_channel_interleave_bytes
        self.lightweight_address_mapping_scheme = lightweight_address_mapping_scheme
        self.lightweight_dram_rows = lightweight_dram_rows
        self.lightweight_dram_columns = lightweight_dram_columns
        self.lightweight_dram_burst_length = lightweight_dram_burst_length
        self.lightweight_instance_command_issue_gap_cycles = lightweight_instance_command_issue_gap_cycles
        self.lightweight_command_issue_gap_cycles = lightweight_command_issue_gap_cycles
        self.lightweight_read_write_turnaround_cycles = lightweight_read_write_turnaround_cycles
        self.lightweight_read_to_write_turnaround_cycles = (
            lightweight_read_write_turnaround_cycles
            if lightweight_read_to_write_turnaround_cycles is None
            else lightweight_read_to_write_turnaround_cycles
        )
        self.lightweight_write_to_read_turnaround_cycles = (
            lightweight_read_write_turnaround_cycles
            if lightweight_write_to_read_turnaround_cycles is None
            else lightweight_write_to_read_turnaround_cycles
        )
        self.lightweight_request_startup_latency_cycles = lightweight_request_startup_latency_cycles
        self.lightweight_burst_size_bytes = lightweight_burst_size_bytes
        self.lightweight_n_rank_per_channel = lightweight_n_rank_per_channel
        self.lightweight_n_bank_group_per_rank = lightweight_n_bank_group_per_rank
        self.lightweight_n_bank_per_bank_group = lightweight_n_bank_per_bank_group
        self.lightweight_row_size_bytes = lightweight_row_size_bytes
        self.lightweight_row_hit_latency_cycles = lightweight_row_hit_latency_cycles
        self.lightweight_row_miss_penalty_cycles = lightweight_row_miss_penalty_cycles
        self.lightweight_row_conflict_penalty_cycles = (
            lightweight_row_miss_penalty_cycles
            if lightweight_row_conflict_penalty_cycles is None
            else lightweight_row_conflict_penalty_cycles
        )
        self.lightweight_bank_group_penalty_cycles = lightweight_bank_group_penalty_cycles
        self.lightweight_dma_max_outstanding_bursts = lightweight_dma_max_outstanding_bursts
        self.lightweight_channel_max_outstanding_bursts = lightweight_channel_max_outstanding_bursts
        self.lightweight_request_queue_depth = lightweight_request_queue_depth
        self.lightweight_concurrent_request_command_gap_cycles = lightweight_concurrent_request_command_gap_cycles
        self.lightweight_concurrent_request_command_gap_threshold = lightweight_concurrent_request_command_gap_threshold
        self.lightweight_concurrent_request_command_gap_limit = lightweight_concurrent_request_command_gap_limit
        self.lightweight_latency_amortization_bytes = lightweight_latency_amortization_bytes
        self.lightweight_enable_latency_amortization = lightweight_enable_latency_amortization

        self._instance_dma_map: dict[int, int] = instance_dma_map
        if self._instance_dma_map is None:
            self._instance_dma_map = {instance_id: None for instance_id in range(self.n_instance)}
        else:
            _map_keys = set(self._instance_dma_map.keys())
            _instance_ids = set(range(self.n_instance))
            if _map_keys != _instance_ids:
                raise ValueError(f"Invalid instance_dma_map keys. Expected keys: {_instance_ids}, but got: {_map_keys}")

    def set_dma_id_to_instance(self, instance_id: int, dma_id: int):
        if instance_id not in self._instance_dma_map:
            raise ValueError(f"Invalid instance_id: {instance_id}")
        if self._instance_dma_map[instance_id] is not None:
            raise ValueError(f"Instance {instance_id} is already assigned to DMA {self._instance_dma_map[instance_id]}")
        self._instance_dma_map[instance_id] = dma_id

        return self

    @property
    def channel_size_per_instance(self) -> int:
        return self.channel_size * self.n_channel_per_instance

    @property
    def mem_addr_end(self) -> int:
        return self.mem_addr_offset + self.n_instance * self.channel_size_per_instance

    @property
    def peak_bandwidth_per_cycle(self) -> float:
        if self.dramsim3_enable:
            return self.dramsim3_config.peak_bandwidth() / self.processor_clock_freq
        return self.n_instance * self.n_channel_per_instance * self.lightweight_channel_bandwidth_bytes_per_cycle
    
    
class MemorySimulator:
    def __init__(self, config: MemoryConfig, mem_addr_offset: int = 0):
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
        self._instance_command_next_free_cycle = {
            instance_id: 0
            for instance_id in range(self.config.n_instance)
        }
        self._instance_request_completions = {
            instance_id: []
            for instance_id in range(self.config.n_instance)
        }
        self._instance_burst_completions = {
            instance_id: []
            for instance_id in range(self.config.n_instance)
        }
        self._channel_burst_completions = {
            (instance_id, channel_id): []
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
        self._initialize_address_mapping()
        
    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    def _initialize_address_mapping(self) -> None:
        scheme = self._cfg("lightweight_address_mapping_scheme", "rorabgbachco")
        fields = [scheme[index:index + 2] for index in range(0, len(scheme), 2)]
        burst_length = max(1, self._cfg("lightweight_dram_burst_length", 1))
        columns = max(burst_length, self._cfg("lightweight_dram_columns", burst_length))
        field_counts = {
            "ch": self.config.n_channel_per_instance,
            "ra": self._cfg("lightweight_n_rank_per_channel", 1),
            "bg": self._cfg("lightweight_n_bank_group_per_rank", 1),
            "ba": self._cfg("lightweight_n_bank_per_bank_group", 1),
            "ro": self._cfg("lightweight_dram_rows", 1),
            "co": max(1, columns // burst_length),
        }
        if len(fields) != 6 or set(fields) != set(field_counts):
            raise ValueError(f"Invalid lightweight_address_mapping_scheme: {scheme}")
        field_widths = {}
        for field, count in field_counts.items():
            if count & (count - 1):
                raise ValueError(f"Address mapping field {field} count must be a power of two: {count}")
            field_widths[field] = count.bit_length() - 1
        position = 0
        self._address_field_positions = {}
        for field in reversed(fields):
            width = field_widths[field]
            self._address_field_positions[field] = (position, (1 << width) - 1)
            position += width
        burst_size = max(1, self._cfg("lightweight_burst_size_bytes", self.config.lightweight_dma_granularity))
        if burst_size & (burst_size - 1):
            raise ValueError(f"lightweight_burst_size_bytes must be a power of two: {burst_size}")
        self._address_shift_bits = burst_size.bit_length() - 1

    def _extract_address_field(self, shifted_address: int, field: str) -> int:
        position, mask = self._address_field_positions[field]
        return (shifted_address >> position) & mask
        
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
        address_mapping = self._cfg("lightweight_address_mapping", "contiguous")
        if address_mapping == "dramsim3":
            shifted_address = addr_offset >> self._address_shift_bits
            channel_id = self._extract_address_field(shifted_address, "ch")
            rank_id = self._extract_address_field(shifted_address, "ra")
            bank_group_id = self._extract_address_field(shifted_address, "bg")
            bank_id = self._extract_address_field(shifted_address, "ba")
            row_id = self._extract_address_field(shifted_address, "ro")
            column_id = self._extract_address_field(shifted_address, "co")
            burst_size = max(1, self._cfg("lightweight_burst_size_bytes", self.config.lightweight_dma_granularity))
            channel_offset = addr_offset % self.config.channel_size
            column_offset = column_id * burst_size + (addr_offset % burst_size)
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
        if address_mapping == "burst_interleaved":
            interleave_bytes = max(1, self._cfg("lightweight_channel_interleave_bytes", 64))
            stripe_index = addr_offset // interleave_bytes
            stripe_offset = addr_offset % interleave_bytes
            channel_id = stripe_index % self.config.n_channel_per_instance
            channel_offset = (stripe_index // self.config.n_channel_per_instance) * interleave_bytes + stripe_offset
        else:
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
        enable_amortization = self._cfg("lightweight_enable_latency_amortization", True)
        amortization_bytes = max(1, self._cfg("lightweight_latency_amortization_bytes", size if size > 0 else 1))
        latency_scale = min(1.0, size / amortization_bytes) if enable_amortization and size > 0 else 1.0
        scale_latency = lambda value: 0 if value <= 0 else max(1, math.ceil(value * latency_scale))
        base_latency = scale_latency(raw_base_latency)
        write_accept_latency = self._cfg("lightweight_write_accept_latency_cycles", 1)
        write_completion_policy = self._cfg("lightweight_write_completion_policy", "retire")
        bandwidth = self.config.lightweight_channel_bandwidth_bytes_per_cycle
        instance_issue_gap = self._cfg("lightweight_instance_command_issue_gap_cycles", 0)
        issue_gap = self.config.lightweight_command_issue_gap_cycles
        read_to_write_turnaround = self._cfg("lightweight_read_to_write_turnaround_cycles", self.config.lightweight_read_write_turnaround_cycles)
        write_to_read_turnaround = self._cfg("lightweight_write_to_read_turnaround_cycles", self.config.lightweight_read_write_turnaround_cycles)
        request_startup = scale_latency(self._cfg("lightweight_request_startup_latency_cycles", 0))
        row_hit_latency = scale_latency(self._cfg("lightweight_row_hit_latency_cycles", 0))
        row_miss_penalty = scale_latency(self._cfg("lightweight_row_miss_penalty_cycles", 0))
        row_conflict_penalty = scale_latency(self._cfg("lightweight_row_conflict_penalty_cycles", row_miss_penalty))
        bank_group_penalty = scale_latency(self._cfg("lightweight_bank_group_penalty_cycles", 0))
        max_outstanding = max(1, self._cfg("lightweight_dma_max_outstanding_bursts", len(chunks) if chunks else 1))
        channel_max_outstanding = max(1, self._cfg("lightweight_channel_max_outstanding_bursts", max_outstanding))
        request_queue_depth = max(1, self._cfg("lightweight_request_queue_depth", 32))
        concurrent_gap = self._cfg("lightweight_concurrent_request_command_gap_cycles", 0)
        concurrent_gap_threshold = self._cfg("lightweight_concurrent_request_command_gap_threshold", 0)
        concurrent_gap_limit = self._cfg("lightweight_concurrent_request_command_gap_limit", 1)

        instance_id = chunks[0]["inst_id"] if chunks else self.get_instance_id_with_address(addr)
        request_completions = self._retire_completed_bursts(self._instance_request_completions[instance_id], current_cycle)
        request_admit_cycle = current_cycle
        if len(request_completions) >= request_queue_depth:
            request_admit_cycle = min(request_completions)
            request_completions = self._retire_completed_bursts(request_completions, request_admit_cycle)
        concurrent_requests = len(request_completions)
        saturated_requests = max(0, concurrent_requests - concurrent_gap_threshold)
        contention_gap = min(saturated_requests, concurrent_gap_limit) * concurrent_gap
        effective_instance_issue_gap = instance_issue_gap + contention_gap
        self._instance_request_completions[instance_id] = request_completions
        
        scheduled_chunks = []
        finish_cycle = request_admit_cycle + request_startup
        retire_finish_cycle = finish_cycle
        accept_finish_cycle = finish_cycle
        first_data_cycle = None
        issue_cycle = request_admit_cycle + request_startup
        
        for chunk in chunks:
            instance_completions = self._retire_completed_bursts(self._instance_burst_completions[chunk["inst_id"]], issue_cycle)
            channel_key = (chunk["inst_id"], chunk["channel_id"])
            channel_completions = self._retire_completed_bursts(self._channel_burst_completions[channel_key], issue_cycle)
            if len(instance_completions) >= max_outstanding or len(channel_completions) >= channel_max_outstanding:
                earliest_completion = min(
                    min(instance_completions) if len(instance_completions) >= max_outstanding else math.inf,
                    min(channel_completions) if len(channel_completions) >= channel_max_outstanding else math.inf,
                )
                issue_cycle = max(issue_cycle, earliest_completion)
                instance_completions = self._retire_completed_bursts(instance_completions, issue_cycle)
                channel_completions = self._retire_completed_bursts(channel_completions, issue_cycle)
            self._instance_burst_completions[chunk["inst_id"]] = instance_completions
            self._channel_burst_completions[channel_key] = channel_completions
            
            bank_group_key = (*channel_key, chunk["rank_id"], chunk["bank_group_id"])
            bank_key = (*bank_group_key, chunk["bank_id"])
            row_key = bank_key
            
            command_start_cycle = max(
                issue_cycle,
                self._instance_command_next_free_cycle[chunk["inst_id"]],
                self._channel_command_next_free_cycle[channel_key],
            )
            bank_ready_cycle = self._bank_next_free_cycle.get(bank_key, 0)
            bank_group_ready_cycle = self._bank_group_next_free_cycle.get(bank_group_key, 0)
            open_row = self._open_row.get(row_key)
            if open_row == chunk["row_id"]:
                row_latency = row_hit_latency
            elif open_row is None:
                row_latency = row_miss_penalty
            else:
                row_latency = row_conflict_penalty
            dram_ready_cycle = max(command_start_cycle, bank_ready_cycle, bank_group_ready_cycle) + row_latency
            
            bus_start_cycle = max(dram_ready_cycle, self._channel_next_free_cycle[channel_key])
            last_is_write = self._channel_last_is_write[channel_key]
            if last_is_write is not None and last_is_write != is_write:
                bus_start_cycle += write_to_read_turnaround if last_is_write else read_to_write_turnaround
            
            transfer_cycles = max(1, math.ceil(chunk["size"] / bandwidth))
            bus_finish_cycle = bus_start_cycle + transfer_cycles
            chunk_finish_cycle = bus_finish_cycle + base_latency
            chunk_accept_cycle = command_start_cycle + write_accept_latency if is_write else chunk_finish_cycle
            queue_delay_cycles = max(0, bus_start_cycle - issue_cycle)
            
            self._instance_command_next_free_cycle[chunk["inst_id"]] = command_start_cycle + effective_instance_issue_gap
            self._channel_command_next_free_cycle[channel_key] = command_start_cycle + issue_gap
            self._channel_next_free_cycle[channel_key] = bus_finish_cycle
            self._channel_last_is_write[channel_key] = is_write
            self._bank_next_free_cycle[bank_key] = bus_finish_cycle
            self._bank_group_next_free_cycle[bank_group_key] = bus_start_cycle + bank_group_penalty
            self._open_row[row_key] = chunk["row_id"]
            retire_finish_cycle = max(retire_finish_cycle, chunk_finish_cycle)
            accept_finish_cycle = max(accept_finish_cycle, chunk_accept_cycle)
            finish_cycle = accept_finish_cycle if is_write and write_completion_policy == "accept" else retire_finish_cycle
            first_data_cycle = chunk_finish_cycle if first_data_cycle is None else min(first_data_cycle, chunk_finish_cycle)
            outstanding_completion_cycle = (
                chunk_accept_cycle
                if is_write and write_completion_policy == "accept"
                else chunk_finish_cycle
            )
            instance_completions.append(outstanding_completion_cycle)
            channel_completions.append(outstanding_completion_cycle)
            
            scheduled_chunk = dict(chunk)
            scheduled_chunk.update({
                "command_start_cycle": command_start_cycle,
                "dram_ready_cycle": dram_ready_cycle,
                "bus_start_cycle": bus_start_cycle,
                "bus_finish_cycle": bus_finish_cycle,
                "finish_cycle": chunk_finish_cycle,
                "accept_cycle": chunk_accept_cycle,
                "transfer_cycles": transfer_cycles,
                "base_latency_cycles": base_latency,
                "row_latency_cycles": row_latency,
                "queue_delay_cycles": queue_delay_cycles,
            })
            scheduled_chunks.append(scheduled_chunk)
            issue_cycle = command_start_cycle + effective_instance_issue_gap

        self._instance_request_completions[instance_id].append(retire_finish_cycle)
        
        return {
            "current_cycle": current_cycle,
            "finish_cycle": finish_cycle,
            "latency_cycles": finish_cycle - current_cycle,
            "accept_finish_cycle": accept_finish_cycle,
            "retire_finish_cycle": retire_finish_cycle,
            "first_data_cycle": current_cycle if first_data_cycle is None else first_data_cycle,
            "request_admit_cycle": request_admit_cycle,
            "concurrent_requests": concurrent_requests,
            "contention_gap_cycles": contention_gap,
            "n_chunks": len(scheduled_chunks),
            "chunks": scheduled_chunks,
        }
    
    @property
    def channel_next_free_cycle(self) -> dict[tuple[int, int], int]:
        return dict(self._channel_next_free_cycle)
    

class MemoryContext:
    def __init__(
        self,
        config: MemoryConfig,
    ):
        self._config = config

        if self._config.dramsim3_enable:
            self._simulator = None
        else:  # use lightweight memory simulator instead of DRAMSim3
            self._simulator = MemorySimulator(config, mem_addr_offset=self.config.mem_addr_offset)   # MemoryConfig is perpectly compatible with MemorySimulator (no need to create a MemoryConfig in neuromta.component)

    def get_dma_id_with_instance_id(self, instance_id: int) -> int:
        if instance_id not in self._config._instance_dma_map:
            raise ValueError(f"Invalid instance_id: {instance_id}")
        return self._config._instance_dma_map[instance_id]

    def get_instance_id_with_dma_id(self, dma_id: int) -> int:
        for instance_id, mapped_dma_id in self._config._instance_dma_map.items():
            if mapped_dma_id == dma_id:
                return instance_id
        raise ValueError(f"DMA ID {dma_id} is not mapped to any instance")

    def get_instance_id_with_address(self, address: int) -> int:
        if not self.check_address_range(address):
            raise ValueError(f"Address {address} is out of range")
        return ((address - self.mem_addr_offset) // self.channel_size_per_instance) % self.n_instance

    def get_memory_mapping(self, address: int) -> dict[str, int]:
        instance_id = self.get_instance_id_with_address(address)
        addr_offset = (address - self.mem_addr_offset) % self.channel_size_per_instance
        channel_id = addr_offset // self.channel_size
        channel_offset = addr_offset % self.channel_size
        return {
            "inst_id": instance_id,
            "addr": addr_offset,
            "channel_id": channel_id,
            "channel_offset": channel_offset,
        }

    def get_dma_id_with_address(self, address: int) -> int:
        instance_id = self.get_instance_id_with_address(address)
        return self.get_dma_id_with_instance_id(instance_id)

    def get_mem_access_args(self, address: int, size: int, is_write: bool) -> dict:
        mapping = self.get_memory_mapping(address)

        return {
            "inst_id": mapping["inst_id"],
            "addr": mapping["addr"],
            "size": size,
            "is_write": is_write,
        }

    def get_address_with_instance_id(self, instance_id: int, addr_offset: int) -> int:
        if instance_id < 0 or instance_id >= self.n_instance:
            raise ValueError(f"Invalid instance_id: {instance_id}")
        if addr_offset < 0 or addr_offset >= self.channel_size_per_instance:
            raise ValueError(f"Invalid addr_offset: {addr_offset}. Must be in range [0, {self.channel_size_per_instance})")

        return (instance_id * self.channel_size_per_instance) + addr_offset

    def check_address_range(self, address: int) -> bool:
        if address < self.mem_addr_offset:
            return False
        if address >= (self.mem_addr_offset + self.n_instance * self.channel_size_per_instance):
            return False
        return True

    def get_instance_addr(self, instance_id: int, offset: int) -> int:
        if instance_id < 0 or instance_id >= self.n_instance:
            raise ValueError(f"Invalid instance_id: {instance_id}")
        if offset < 0 or offset >= self.channel_size_per_instance:
            raise ValueError(f"Invalid offset: {offset}. Must be in range [0, {self.channel_size_per_instance})")

        return self.mem_addr_offset + (instance_id * self.channel_size_per_instance) + offset

    @property
    def mem_addr_offset(self) -> int:
        return self._config.mem_addr_offset

    @property
    def mem_addr_end(self) -> int:
        return self._config.mem_addr_offset + self.n_instance * self.channel_size_per_instance

    @property
    def n_channels(self) -> int:
        return self._config.n_instance

    @property
    def channel_size(self) -> int:
        return self._config.channel_size

    @property
    def channel_size_per_instance(self) -> int:
        return self._config.channel_size_per_instance

    @property
    def n_channel_per_instance(self) -> int:
        return self._config.n_channel_per_instance

    @property
    def n_instance(self) -> int:
        return self._config.n_instance

    @property
    def config(self) -> MemoryConfig:
        return self._config

    @property
    def is_simulator_available(self) -> bool:
        return self._simulator is not None

    @property
    def simulator(self) -> MemorySimulator:
        if self._simulator is None:
            raise RuntimeError("Memory simulator is not available. Please use the DRAMSim3 companion module is properly installed and configured.")
        return self._simulator

    @property
    def peak_bandwidth_per_cycle(self) -> float:
        return self._config.peak_bandwidth_per_cycle
