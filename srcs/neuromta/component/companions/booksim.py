import os
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
    def __init__(self, flit_size: int, subnets: int, x: int, y: int, xr: int, yr: int):
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")
        
        self._flit_size: int = flit_size
        self._subnets: int = subnets
        self._x: int = x
        self._y: int = y
        self._xr: int = xr
        self._yr: int = yr

        self._config: c_void_p = pybooksim2.create_config_torus_2d(subnets, x, y, xr, yr)
        
    def peak_bandwidth_per_router(self) -> float:
        return self._flit_size * self._subnets

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
        
        self._bw_profile_hook_id = None
        self._bw_profile_resolution = 1  # in cycles
        self._bw_tx_profiles: dict[int, list[int]] = {}
        self._bw_rx_profiles: dict[int, list[int]] = {}

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
    
    def enable_bandwidth_profiling(self, resolution: int = 1):
        self._bw_profile_resolution = resolution
        self._bw_profile_hook_id = self.companion_core.register_companion_module_hook(self.module_id, self._bandwidth_profile_hook)
        
    def disable_bandwidth_profiling(self):
        self.companion_core.unregister_companion_module_hook(self.module_id, self._bw_profile_hook_id)
        
    def _bandwidth_profile_hook(self, cmd: CompanionCommandSignature):
        src_id      = cmd.kwargs["src_id"]
        dst_id      = cmd.kwargs["dst_id"]
        n_flits     = cmd.kwargs["n_flits"]
        issue_time  = cmd.issue_time
        commit_time = cmd.commit_time
        
        i_r = issue_time // self._bw_profile_resolution
        c_r = commit_time // self._bw_profile_resolution
        bw  = n_flits * self.config._flit_size / ((commit_time - issue_time) if (commit_time - issue_time) > 0 else 1)  # in bytes per cycle
        
        if src_id not in self._bw_tx_profiles.keys():
            self._bw_tx_profiles[src_id] = []
        if dst_id not in self._bw_rx_profiles.keys():
            self._bw_rx_profiles[dst_id] = []
            
        tx_profile = self._bw_tx_profiles[src_id]
        rx_profile = self._bw_rx_profiles[dst_id]
        
        while len(tx_profile) <= c_r:
            tx_profile.append(0)
        while len(rx_profile) <= c_r:
            rx_profile.append(0)
            
        for t in range(i_r, c_r + 1):
            tx_profile[t] += bw
            rx_profile[t] += bw
            
    def dump_bandwidth_profiles(self) -> dict[str, dict[int, list[int]]]:
        return {
            "tx": self._bw_tx_profiles,
            "rx": self._bw_rx_profiles,
        }
    
    def save_bandwidth_profiles_as_file(self, dirname: str):
        profiles = self.dump_bandwidth_profiles()
        filename_fmt = "{node_id}_{traffic_type}.csv"
        
        os.makedirs(dirname, exist_ok=True)
        
        for node_id, profile in profiles["tx"].items():
            profile_path = os.path.join(dirname, filename_fmt.format(node_id=node_id, traffic_type="tx"))
            with open(profile_path, "wt") as file:
                file.write("timestamp,bandwidth[bytes/cycle]\n")
                for i, bw in enumerate(profile):
                    file.write(f"{i*self._bw_profile_resolution},{bw}\n")

            logger.info(f"BookSim2 bandwidth profile saved at \"{profile_path}\"")

        for node_id, profile in profiles["rx"].items():
            profile_path = os.path.join(dirname, filename_fmt.format(node_id=node_id, traffic_type="rx"))
            with open(profile_path, "wt") as file:
                file.write("timestamp,bandwidth[bytes/cycle]\n")
                for i, bw in enumerate(profile):
                    file.write(f"{i*self._bw_profile_resolution},{bw}\n")

            logger.info(f"BookSim2 bandwidth profile saved at \"{profile_path}\"")
