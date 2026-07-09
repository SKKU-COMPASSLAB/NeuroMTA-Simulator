import os
from typing import Any, Callable
from ctypes import c_void_p
from neuromta.framework import *

try:
    import pybooksim2
    PYBOOKSIM2_AVAILABLE = True
except ImportError as e:
    PYBOOKSIM2_AVAILABLE = False
    
PYBOOKSIM2_AVAILABLE = os.environ.get("NEUROMTA_DISABLE_BOOKSIM", "0") != "1" and PYBOOKSIM2_AVAILABLE

logger.debug(f"pybooksim2 availability: {PYBOOKSIM2_AVAILABLE}")
if not PYBOOKSIM2_AVAILABLE:
    logger.debug(f"pybooksim2 is not available. Using a simplified BookSim2 model instead. To enable pybooksim2, please install it via 'pip install externals/pybooksim2'")
 
    
__all__ = [
    "PYBOOKSIM2_AVAILABLE",
    "BookSim2",
    "BookSim2Config",
]

class BookSim2Config:
    def __init__(self, processor_clock_freq: float, flit_size: int, subnets: int, x: int, y: int, xr: int, yr: int):
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")
        
        self.processor_clock_freq: float = processor_clock_freq
        self._flit_size: int = flit_size
        self._subnets: int = subnets
        self._x: int = x
        self._y: int = y
        self._xr: int = xr
        self._yr: int = yr

        self._config: c_void_p = pybooksim2.create_config_torus_2d(subnets, x, y, xr, yr)
        
    def peak_bandwidth_per_router(self) -> float:
        return self._flit_size * self._subnets * self.processor_clock_freq
    
    def peak_bisection_bandwidth(self) -> float:
        return self.peak_bandwidth_per_router() * min(self._x, self._y) * 2

    def create_icnt(self) -> c_void_p:
        return pybooksim2.create_icnt(config=self._config)
    
    def update_field(self, field: str, value: Any):
        pybooksim2.update_config(self._config, field, value)
    
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

    def create_command(self, src_id: int, dst_id: int, subnet: int, n_flits: int, is_write: bool, is_response: bool) -> CompanionCommandSignature:
        capsule = pybooksim2.create_icnt_cmd_data_packet(src_id, dst_id, subnet, n_flits, is_write, is_response)
        cmd = CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs={
            "src_id": src_id,
            "dst_id": dst_id,
            "subnet": subnet,
            "n_flits": n_flits,
            "is_write": is_write,
            "is_response": is_response,
        })
        return cmd

    def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        return pybooksim2.icnt_dispatch_cmd(icnt=self._icnt, cmd=cmd.capsule, dispatch_callback=dispatch_callback, execute_callback=execute_callback)

    def get_stats(self) -> dict[int, dict[str, float | int]]:
        return pybooksim2.get_icnt_router_stats(icnt=self._icnt)
