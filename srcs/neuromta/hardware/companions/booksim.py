from typing import Any, Callable
from ctypes import c_void_p
from neuromta.framework import *

try:
    import pybooksim2
    PYBOOKSIM2_AVAILABLE = True
except ImportError as e:
    PYBOOKSIM2_AVAILABLE = False
 
    
__all__ = [
    "PYBOOKSIM2_AVAILABLE",
    "BookSim2",
    "BookSim2Config",
]


class BookSim2Config:
    def __init__(self, subnets: int, x: int, y: int, xr: int, yr: int):
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")
        
        self._subnets: int = subnets
        self._x: int = x
        self._y: int = y
        self._xr: int = xr
        self._yr: int = yr

        self._config: c_void_p = pybooksim2.create_config_torus_2d(subnets, x, y, xr, yr)

    def create_icnt(self) -> c_void_p:
        return pybooksim2.create_icnt(config=self._config)
    
    def summary(self) -> dict[str, Any]:
        return {
            "subnets": self._subnets,
            "x": self._x,
            "y": self._y,
            "xr": self._xr,
            "yr": self._yr,
        }


class BookSim2(CompanionModule):
    def __init__(self, config: BookSim2Config):
        super().__init__()
        
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")

        self._icnt = config.create_icnt()
        self.config = config

    def update_cycle_time(self, cycle_time):
        pybooksim2.icnt_cycle_step(icnt=self._icnt, cycles=cycle_time)

    def create_command(self, src_id: int, dst_id: int, subnet: int, n_flits: int, is_write: bool, is_response: bool) -> Any:
        return pybooksim2.create_icnt_cmd_data_packet(src_id, dst_id, subnet, n_flits, is_write, is_response)

    def dispatch_command(self, cmd, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        return pybooksim2.icnt_dispatch_cmd(icnt=self._icnt, cmd=cmd, dispatch_callback=dispatch_callback, execute_callback=execute_callback)

    def check_command_executed(self, cmd) -> bool:
        return pybooksim2.check_icnt_cmd_received(cmd=cmd)
