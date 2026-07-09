import math
import random
from typing import Any

from neuromta.framework import *
from neuromta.component.companions.booksim import BookSim2Config, pybooksim2, PYBOOKSIM2_AVAILABLE


__all__ = [
    "IcntConfig",
    "IcntContext",
    "IcntSimulator",
    "ICNT_CHANNEL_MODE_UNIDIRECTIONAL",
    "ICNT_CHANNEL_MODE_BIDIRECTIONAL_SHARED",
]


ICNT_CHANNEL_MODE_UNIDIRECTIONAL = "unidirectional"
ICNT_CHANNEL_MODE_BIDIRECTIONAL_SHARED = "bidirectional_shared"


def _opposite_direction(direction: str) -> str:
    if direction == "N":
        return "S"
    if direction == "S":
        return "N"
    if direction == "E":
        return "W"
    if direction == "W":
        return "E"
    raise ValueError(f"Invalid direction: {direction}")


class IcntConfig:
    def __init__(
        self,
        
        processor_clock_freq: int,
        shape: tuple[int, int], 
        flit_size: int          = parse_mem_cap_str("32B"),
        max_payload_size: int   = 256,
        subnets: int            = 1,
        booksim2_enable: bool   = None,
        booksim2_kwargs: dict[str, Any] = None,
        
        lightweight_router_latency_cycles: int = 1,
        lightweight_link_latency_cycles: int = 1,
        lightweight_flits_per_cycle_per_channel: int = 1,
        lightweight_injection_flits_per_cycle: int = 1,
        lightweight_egress_flits_per_cycle: int = 1,
        lightweight_channel_mode: str = ICNT_CHANNEL_MODE_UNIDIRECTIONAL,
        lightweight_router_allocation_cycles: int = 1,
        lightweight_packet_startup_cycles: int = 1,
        lightweight_min_packet_cycles: int = 1,
        lightweight_payload_issue_gap_cycles: int = 1,
    ):  
        if booksim2_enable is None:
            booksim2_enable = PYBOOKSIM2_AVAILABLE
        
        x_dim = shape[0]
        y_dim = shape[1]
        
        booksim2_config = BookSim2Config(
            processor_clock_freq=processor_clock_freq,
            flit_size=flit_size,
            subnets=subnets,
            x=x_dim,
            y=y_dim,
            xr=1,   # no concentration by default
            yr=1,   # no concentration by default
        )
        
        if booksim2_enable:
            if booksim2_kwargs is not None:
                for field, value in booksim2_kwargs.items():
                    booksim2_config.update_field(field, value)
            
        self.processor_clock_freq = processor_clock_freq
        self.shape = shape
        self.flit_size = flit_size
        self.max_payload_size = max_payload_size
        self.subnets = subnets
        
        self.booksim2_config = booksim2_config
        self.booksim2_enable = booksim2_enable
        
        self.lightweight_router_latency_cycles = lightweight_router_latency_cycles
        self.lightweight_link_latency_cycles = lightweight_link_latency_cycles
        self.lightweight_flits_per_cycle_per_channel = lightweight_flits_per_cycle_per_channel
        self.lightweight_injection_flits_per_cycle = lightweight_injection_flits_per_cycle
        self.lightweight_egress_flits_per_cycle = lightweight_egress_flits_per_cycle
        self.lightweight_channel_mode = lightweight_channel_mode
        self.lightweight_router_allocation_cycles = lightweight_router_allocation_cycles
        self.lightweight_packet_startup_cycles = lightweight_packet_startup_cycles
        self.lightweight_min_packet_cycles = lightweight_min_packet_cycles
        self.lightweight_payload_issue_gap_cycles = lightweight_payload_issue_gap_cycles
        
        self._core_map: dict[tuple[int, int], int] = {}
    
    def update_core_map(self, coord: tuple[int, int], core_id: int):
        self._core_map[coord] = core_id
    
    @property
    def peak_bisection_bandwidth(self) -> float:
        if self.booksim2_enable:
            return self.booksim2_config.peak_bisection_bandwidth()
        
        bisection_channels = min(self.shape) * 2
        channel_bandwidth = (
            self.flit_size
            * self.lightweight_flits_per_cycle_per_channel
            * self.processor_clock_freq
        )
        return channel_bandwidth * self.subnets * bisection_channels
    
    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "flit_size": self.flit_size,
            "booksim2_enable": self.booksim2_enable,
            "booksim2_config": self.booksim2_config.summary() if self.booksim2_enable else None,
        }
        

class IcntSimulator:
    def __init__(self, config: IcntConfig):
        self.config = config
        self.reset()
        
    def reset(self) -> None:
        self._resource_next_free_cycle: dict[tuple, int] = {}
        
    def node_id_to_coord(self, node_id: int) -> tuple[int, int]:
        n_nodes = self.config.shape[0] * self.config.shape[1]
        if node_id < 0 or node_id >= n_nodes:
            raise ValueError(f"Invalid node_id: {node_id}")
        return (node_id // self.config.shape[1], node_id % self.config.shape[1])
    
    def coord_to_node_id(self, coord: tuple[int, int]) -> int:
        row, col = coord
        if row < 0 or row >= self.config.shape[0]:
            raise ValueError(f"Invalid row coordinate: {row}")
        if col < 0 or col >= self.config.shape[1]:
            raise ValueError(f"Invalid column coordinate: {col}")
        return row * self.config.shape[1] + col
    
    def core_id_to_coord(self, core_id: Any) -> tuple[int, int]:
        for coord, cid in self.config._core_map.items():
            if cid == core_id:
                return coord
        raise ValueError(f"Core ID {core_id} not found in core map.")
    
    def core_id_to_node_id(self, core_id: Any) -> int:
        return self.coord_to_node_id(self.core_id_to_coord(core_id))
    
    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
    
    def get_xy_route(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> list[dict[str, Any]]:
        route = []
        cur_row, cur_col = src_coord
        dst_row, dst_col = dst_coord
        
        while cur_col != dst_col:
            next_col = cur_col + (1 if dst_col > cur_col else -1)
            direction = "E" if next_col > cur_col else "W"
            next_coord = (cur_row, next_col)
            route.append({"src_coord": (cur_row, cur_col), "dst_coord": next_coord, "direction": direction})
            cur_col = next_col
        
        while cur_row != dst_row:
            next_row = cur_row + (1 if dst_row > cur_row else -1)
            direction = "S" if next_row > cur_row else "N"
            next_coord = (next_row, cur_col)
            route.append({"src_coord": (cur_row, cur_col), "dst_coord": next_coord, "direction": direction})
            cur_row = next_row
        
        return route
    
    def _physical_channel_key(self, subnet: int, src_coord: tuple[int, int], dst_coord: tuple[int, int], direction: str) -> tuple:
        if self.config.lightweight_channel_mode == ICNT_CHANNEL_MODE_BIDIRECTIONAL_SHARED:
            endpoint_a, endpoint_b = sorted([src_coord, dst_coord])
            return ("physical_channel", subnet, endpoint_a, endpoint_b)
        return ("physical_channel", subnet, src_coord, dst_coord, direction)
    
    def get_resource_sequence(self, src_id: int, dst_id: int, subnet: int) -> list[dict[str, Any]]:
        src_coord = self.node_id_to_coord(src_id)
        dst_coord = self.node_id_to_coord(dst_id)
        route = self.get_xy_route(src_coord, dst_coord)
        resources = [{
            "kind": "injection_port",
            "key": ("injection_port", subnet, src_coord),
            "coord": src_coord,
            "latency_after": 0,
        }]
        
        for hop in route:
            hop_src = hop["src_coord"]
            hop_dst = hop["dst_coord"]
            direction = hop["direction"]
            resources.extend([
                {
                    "kind": "router_output_port",
                    "key": ("router_output_port", subnet, hop_src, direction),
                    "coord": hop_src,
                    "direction": direction,
                    "latency_after": self.config.lightweight_router_latency_cycles,
                },
                {
                    "kind": "physical_channel",
                    "key": self._physical_channel_key(subnet, hop_src, hop_dst, direction),
                    "src_coord": hop_src,
                    "dst_coord": hop_dst,
                    "direction": direction,
                    "latency_after": self.config.lightweight_link_latency_cycles,
                },
                {
                    "kind": "router_input_port",
                    "key": ("router_input_port", subnet, hop_dst, _opposite_direction(direction)),
                    "coord": hop_dst,
                    "direction": _opposite_direction(direction),
                    "latency_after": 0,
                },
            ])
        
        resources.append({
            "kind": "ejection_port",
            "key": ("ejection_port", subnet, dst_coord),
            "coord": dst_coord,
            "latency_after": 0,
        })
        return resources
    
    def _reserve_resource(self, key: tuple, ready_cycle: int, hold_cycles: int) -> dict[str, int]:
        next_free_cycle = self._resource_next_free_cycle.get(key, 0)
        start_cycle = max(ready_cycle, next_free_cycle)
        finish_cycle = start_cycle + hold_cycles
        self._resource_next_free_cycle[key] = finish_cycle
        return {
            "start_cycle": start_cycle,
            "finish_cycle": finish_cycle,
            "queue_delay_cycles": max(0, start_cycle - ready_cycle),
        }
    
    def _send_payload(
        self,
        src_id: int,
        dst_id: int,
        subnet: int,
        n_flits: int,
        is_write: bool,
        is_response: bool,
        current_cycle: int,
        payload_index: int,
    ) -> dict:
        src_coord = self.node_id_to_coord(src_id)
        dst_coord = self.node_id_to_coord(dst_id)
        route = self.get_xy_route(src_coord, dst_coord)
        serialization_cycles = max(1, math.ceil(n_flits / self.config.lightweight_flits_per_cycle_per_channel))
        injection_cycles = max(1, math.ceil(n_flits / self.config.lightweight_injection_flits_per_cycle))
        egress_cycles = max(1, math.ceil(n_flits / self.config.lightweight_egress_flits_per_cycle))
        router_alloc_cycles = max(1, getattr(self.config, "lightweight_router_allocation_cycles", 1))
        packet_startup_cycles = max(0, getattr(self.config, "lightweight_packet_startup_cycles", 1))
        min_packet_cycles = max(1, getattr(self.config, "lightweight_min_packet_cycles", 1))
        
        scheduled_resources = []
        resource_index = 0
        
        injection_resource = {
            "kind": "injection_port",
            "key": ("injection_port", subnet, src_coord),
            "coord": src_coord,
        }
        injection = self._reserve_resource(
            injection_resource["key"],
            ready_cycle=current_cycle + packet_startup_cycles,
            hold_cycles=injection_cycles,
        )
        scheduled = dict(injection_resource)
        scheduled.update({
            "resource_index": resource_index,
            "hold_cycles": injection_cycles,
            **injection,
        })
        scheduled_resources.append(scheduled)
        resource_index += 1
        
        head_ready_cycle = injection["start_cycle"]
        tail_ready_cycle = injection["finish_cycle"]
        
        for hop in route:
            hop_src = hop["src_coord"]
            hop_dst = hop["dst_coord"]
            direction = hop["direction"]
            router_resource = {
                "kind": "router_output_port",
                "key": ("router_output_port", subnet, hop_src, direction),
                "coord": hop_src,
                "direction": direction,
            }
            router = self._reserve_resource(
                router_resource["key"],
                ready_cycle=head_ready_cycle,
                hold_cycles=router_alloc_cycles,
            )
            scheduled = dict(router_resource)
            scheduled.update({
                "resource_index": resource_index,
                "hold_cycles": router_alloc_cycles,
                **router,
            })
            scheduled_resources.append(scheduled)
            resource_index += 1
            
            channel_resource = {
                "kind": "physical_channel",
                "key": self._physical_channel_key(subnet, hop_src, hop_dst, direction),
                "src_coord": hop_src,
                "dst_coord": hop_dst,
                "direction": direction,
            }
            channel = self._reserve_resource(
                channel_resource["key"],
                ready_cycle=router["finish_cycle"] + self.config.lightweight_router_latency_cycles,
                hold_cycles=serialization_cycles,
            )
            scheduled = dict(channel_resource)
            scheduled.update({
                "resource_index": resource_index,
                "hold_cycles": serialization_cycles,
                **channel,
            })
            scheduled_resources.append(scheduled)
            resource_index += 1
            
            head_ready_cycle = channel["start_cycle"] + self.config.lightweight_link_latency_cycles
            tail_ready_cycle = channel["finish_cycle"] + self.config.lightweight_link_latency_cycles
        
        ejection_resource = {
            "kind": "ejection_port",
            "key": ("ejection_port", subnet, dst_coord),
            "coord": dst_coord,
        }
        ejection = self._reserve_resource(
            ejection_resource["key"],
            ready_cycle=tail_ready_cycle,
            hold_cycles=egress_cycles,
        )
        scheduled = dict(ejection_resource)
        scheduled.update({
            "resource_index": resource_index,
            "hold_cycles": egress_cycles,
            **ejection,
        })
        scheduled_resources.append(scheduled)
        
        finish_cycle = max(ejection["finish_cycle"], current_cycle + min_packet_cycles)
        return {
            "payload_index": payload_index,
            "current_cycle": current_cycle,
            "finish_cycle": finish_cycle,
            "latency_cycles": finish_cycle - current_cycle,
            "src_id": src_id,
            "dst_id": dst_id,
            "src_coord": src_coord,
            "dst_coord": dst_coord,
            "subnet": subnet,
            "n_flits": n_flits,
            "is_write": is_write,
            "is_response": is_response,
            "hop_count": self.compute_hop_cnt(src_coord, dst_coord),
            "serialization_cycles": serialization_cycles,
            "injection_cycles": injection_cycles,
            "egress_cycles": egress_cycles,
            "resources": scheduled_resources,
        }
    
    def send_request(
        self,
        src_core_id: Any,
        dst_core_id: Any,
        data_size: int,
        is_write: bool = False,
        is_response: bool | None = None,
        current_cycle: int = 0,
    ) -> dict:
        if current_cycle < 0:
            raise ValueError("current_cycle must be non-negative")
        if data_size < 0:
            raise ValueError(f"Invalid data_size: {data_size}")
        if is_response is None:
            is_response = not is_write
        
        src_id = self.core_id_to_node_id(src_core_id)
        dst_id = self.core_id_to_node_id(dst_core_id)
        n_flits = math.ceil(data_size / self.config.flit_size)
        if n_flits == 0:
            return {
                "current_cycle": current_cycle,
                "finish_cycle": current_cycle,
                "latency_cycles": 0,
                "src_core_id": src_core_id,
                "dst_core_id": dst_core_id,
                "src_id": src_id,
                "dst_id": dst_id,
                "data_size": data_size,
                "n_flits": 0,
                "n_payloads": 0,
                "is_write": is_write,
                "is_response": is_response,
                "payloads": [],
            }
        
        n_payloads = math.ceil(n_flits / self.config.max_payload_size)
        payloads = []
        payload_issue_gap = max(0, getattr(self.config, "lightweight_payload_issue_gap_cycles", 1))
        for payload_index in range(n_payloads):
            payload_flits = min(self.config.max_payload_size, n_flits - payload_index * self.config.max_payload_size)
            subnet = (src_id + dst_id + payload_index) % self.config.subnets
            payloads.append(self._send_payload(
                src_id=src_id,
                dst_id=dst_id,
                subnet=subnet,
                n_flits=payload_flits,
                is_write=is_write,
                is_response=is_response,
                current_cycle=current_cycle + payload_index * payload_issue_gap,
                payload_index=payload_index,
            ))
        
        finish_cycle = max(payload["finish_cycle"] for payload in payloads)
        return {
            "current_cycle": current_cycle,
            "finish_cycle": finish_cycle,
            "latency_cycles": finish_cycle - current_cycle,
            "src_core_id": src_core_id,
            "dst_core_id": dst_core_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "src_coord": self.node_id_to_coord(src_id),
            "dst_coord": self.node_id_to_coord(dst_id),
            "data_size": data_size,
            "n_flits": n_flits,
            "n_payloads": n_payloads,
            "is_write": is_write,
            "is_response": is_response,
            "payloads": payloads,
        }
    
    @property
    def resource_next_free_cycle(self) -> dict[tuple, int]:
        return dict(self._resource_next_free_cycle)


class IcntContext:
    def __init__(self, config: IcntConfig,):
        self._config = config
        
        if self._config.booksim2_enable:
            self._icnt_sim = None
        else:
            self._icnt_sim = IcntSimulator(config)
    
    def coord_to_core_id(self, coord: tuple[int, int]) -> Any:
        return self.core_map[coord]
    
    def core_id_to_coord(self, core_id: Any) -> tuple[int, int]:
        for coord, cid in self.core_map.items():
            if cid == core_id:
                return coord
        raise ValueError(f"Core ID {core_id} not found in core map.")

    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])

    def get_data_packet_latency(self, src_id: int, dst_id: int, data_size: int) -> int:
        src_coord = self.core_id_to_coord(src_id)
        dst_coord = self.core_id_to_coord(dst_id)
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (data_size // self.config.flit_size) + 1
    
    def get_icnt_data_transfer_args(self, src_id: int, dst_id: int, data_size: int, is_write: bool) -> list[dict[str, int]]:
        n_flits = math.ceil(data_size / self.config.flit_size)
        n_payloads = math.ceil(n_flits / self.config.max_payload_size)
        payload_size = min(n_flits, self.config.max_payload_size)
        
        return [{
            "src_id": src_id,
            "dst_id": dst_id,
            "subnet": (src_id + dst_id + i) % self.config.booksim2_config._subnets,
            "n_flits": min(payload_size, n_flits - i * payload_size),
            "is_write": is_write,
            "is_response": not is_write,
        } for i in range(n_payloads)]
        
    @property
    def flit_size(self) -> int:
        return self.config.flit_size
    
    @property
    def max_payload_size(self) -> int:
        return self.config.max_payload_size
    
    @property
    def booksim2_enable(self) -> bool:
        return self.config.booksim2_enable

    @property
    def config(self) -> IcntConfig:
        return self._config
    
    @property
    def core_map(self) -> dict[tuple[int, int], int]:
        return self._config._core_map

    @property
    def icnt_simulator(self) -> IcntSimulator:
        if self._icnt_sim is None:
            raise RuntimeError("ICNT simulator is not initialized. Ensure that booksim2_enable is set to False.")
        return self._icnt_sim
    
    @property
    def is_icnt_simulator_enabled(self) -> bool:
        return self._icnt_sim is not None
