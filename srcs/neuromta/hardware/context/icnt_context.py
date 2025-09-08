from typing import Any

from neuromta.framework import *
from neuromta.hardware.companions.booksim import BookSim2Config


__all__ = [
    "IcntConfig",
    "IcntContext",
]


class IcntConfig(dict):
    def __init__(
        self,
        
        shape: tuple[int, int], 
        flit_size: int                  = parse_mem_cap_str("16B"),
        control_packet_size: int        = parse_mem_cap_str("32B"),
        booksim2_enable: bool           = False,
        booksim2_config: BookSim2Config = None,
    ):
        super().__init__()
        
        if booksim2_enable and booksim2_config is None:
            x_dim = shape[0]
            y_dim = shape[1]
            
            booksim2_config = BookSim2Config(
                subnets=1,  # TODO: is number of subnets always 1?
                x=x_dim,
                y=y_dim,
                xr=1,   # no concentration by default
                yr=1,   # no concentration by default
            )
            
        self.shape = shape
        self.flit_size = flit_size
        self.control_packet_size = control_packet_size
        
        self.booksim2_config = booksim2_config
        self.booksim2_enable = booksim2_enable
        
    def coord_to_core_id(self, coord: tuple[int, int]) -> int:
        return coord[0] * self.shape[1] + coord[1]
    
    def core_id_to_coord(self, core_id: Any) -> tuple[int, int]:
        if isinstance(core_id, str):
            core_id = int(core_id)
        col = core_id % self.shape[1]
        row = core_id // self.shape[1]
        return (row, col)


class IcntContext:
    def __init__(self, config: IcntConfig,):
        self._config = config
        
    def coord_to_core_id(self, coord: tuple[int, int]) -> int:
        return self.config.coord_to_core_id(coord=coord)
    
    def core_id_to_coord(self, core_id: Any) -> tuple[int, int]:
        return self.config.core_id_to_coord(core_id=core_id)

    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
    
    def get_control_packet_latency(self, src_id: int, dst_id: int) -> int:
        src_coord = self.core_id_to_coord(src_id)
        dst_coord = self.core_id_to_coord(dst_id)
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (self.config.control_packet_size // self.config.flit_size)

    def get_data_packet_latency(self, src_id: int, dst_id: int, data_size: int) -> int:
        src_coord = self.core_id_to_coord(src_id)
        dst_coord = self.core_id_to_coord(dst_id)
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (data_size // self.config.flit_size) + 1
    
    @property
    def flit_size(self) -> int:
        return self.config.flit_size
    
    @property
    def control_packet_size(self) -> int:
        return self.config.control_packet_size
    
    @property
    def booksim2_enable(self) -> bool:
        return self.config.booksim2_enable

    @property
    def config(self) -> IcntConfig:
        return self._config
