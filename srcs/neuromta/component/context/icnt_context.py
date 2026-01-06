import math
import random
from typing import Any

from neuromta.framework import *
from neuromta.component.companions.booksim import BookSim2Config, pybooksim2


__all__ = [
    "IcntConfig",
    "IcntContext",
]


class IcntConfig:
    def __init__(
        self,
        
        shape: tuple[int, int], 
        flit_size: int          = parse_mem_cap_str("32B"),
        max_payload_size: int   = 256,
        subnets: int            = 1,
        booksim2_enable: bool   = False,
        booksim2_kwargs: dict[str, Any] = None,
    ):  
        if booksim2_enable:
            x_dim = shape[0]
            y_dim = shape[1]
            
            booksim2_config = BookSim2Config(
                flit_size=flit_size,
                subnets=subnets,
                x=x_dim,
                y=y_dim,
                xr=1,   # no concentration by default
                yr=1,   # no concentration by default
            )
            
            if booksim2_kwargs is not None:
                for field, value in booksim2_kwargs.items():
                    booksim2_config.update_field(field, value)
        else:
            booksim2_config = None
            
        self.shape = shape
        self.flit_size = flit_size
        self.max_payload_size = max_payload_size
        
        self.booksim2_config = booksim2_config
        self.booksim2_enable = booksim2_enable
        
        self._core_map: dict[tuple[int, int], int] = {}
    
    def update_core_map(self, coord: tuple[int, int], core_id: int):
        self._core_map[coord] = core_id
    
    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "flit_size": self.flit_size,
            "booksim2_enable": self.booksim2_enable,
            "booksim2_config": self.booksim2_config.summary() if self.booksim2_enable else None,
        }


class IcntContext:
    def __init__(self, config: IcntConfig,):
        self._config = config
    
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
        # subnet = (src_id + dst_id) % self.config.booksim2_config._subnets
        # subnet = random.randint(0, self.config.booksim2_config._subnets - 1) if self.config.booksim2_enable else 0
        n_flits = math.ceil(data_size / self.config.flit_size)
        n_payloads = math.ceil(n_flits / self.config.max_payload_size)
        payload_size = min(n_flits, self.config.max_payload_size)
        
        return [{
            "src_id": src_id,
            "dst_id": dst_id,
            "subnet": (src_id + dst_id + i) % self.config.booksim2_config._subnets if self.config.booksim2_enable else 0, #subnet,
            "n_flits": min(payload_size, n_flits - i * payload_size),
            "is_write": is_write,
            "is_response": not is_write,  # TODO: remove this feature! (data packets are response to the read and request to the write)
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
