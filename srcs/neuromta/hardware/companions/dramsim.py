import os
import math
import configparser
from typing import Any, Callable

from neuromta.framework import *

try:
    import pydramsim3
    PYDRAMSIM3_AVAILABLE = True
except ImportError as e:
    PYDRAMSIM3_AVAILABLE = False
    

__all__ = [
    "PYDRAMSIM3_AVAILABLE",
    "DRAMSim3",
    "DRAMSim3Config"
    "create_new_dramsim_config_file",
]


def create_new_dramsim_config_file(
    src_config_path: str, 
    new_config_path: str,
    
    # channel_size: int,
    # n_channel: int,
    system_params: dict[str, int] = None,
    dram_structure_params: dict[str, int] = None,
):
    if not os.path.isfile(src_config_path):
        src_config_path = pydramsim3.PYDRAMSIM_MSYS_CONFIG_PATH(src_config_path)
    if not os.path.isfile(src_config_path):
        raise FileNotFoundError(f"DRAMSim3 config file '{src_config_path}' not found.")

    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)

    src_config = configparser.ConfigParser()
    src_config.read(src_config_path)
    
    # src_config["system"]["channel_size"] = str(channel_size)
    # src_config["system"]["channels"] = str(n_channel)
    
    if system_params is not None:
        for key, value in system_params.items():
            src_config["system"][key] = str(value)
    
    if dram_structure_params is not None:
        for key, value in dram_structure_params.items():
            src_config["dram_structure"][key] = str(value)

    with open(new_config_path, "w") as new_file:
        src_config.write(new_file)


class DRAMSim3Config:
    def __init__(
        self, 
        config_path: str,  #="GDDR5_8Gb_x32", 
        processor_clock_freq: int,  #=parse_freq_str("1GHz"),
        cmd_queue_num: int,
    ):  
        if not os.path.isfile(config_path):
            config_path = pydramsim3.PYDRAMSIM_MSYS_CONFIG_PATH(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"DRAMSim3 config file '{config_path}' not found.")

        self.config_path = config_path
        self.processor_clock_freq = processor_clock_freq
        self.cmd_queue_num = cmd_queue_num

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": os.path.abspath(self.config_path),
            "processor_clock_freq": self.processor_clock_freq,
        }


class DRAMSim3(CompanionModule):
    def __init__(self, config: DRAMSim3Config):
        super().__init__()
        
        self.config = config
        
        self._msys = pydramsim3.create_msys(
            config_file=self.config.config_path,
            output_dir=pydramsim3.PYDRAMSIM_DEFAULT_OUT_DIR,
            cmd_queue_num=self.config.cmd_queue_num,
        )
        
        self._mem_clock_time = pydramsim3.msys_get_tck(self._msys)
        self._ref_clock_time = 1 / (self.config.processor_clock_freq * (1e-9))
        self._rem_clock_sync_time = 0
        
        self._bw_profile_hook_id = None
        self._bw_profile_resolution = 1  # in cycles
        self._bw_rd_profiles: dict[int, list[int]] = {}
        self._bw_wr_profiles: dict[int, list[int]] = {}

    def update_cycle_time(self, cycle_time):
        self._rem_clock_sync_time += cycle_time * self._ref_clock_time
        
        mem_cycles = math.floor(self._rem_clock_sync_time / self._mem_clock_time)
        self._rem_clock_sync_time -= mem_cycles * self._mem_clock_time
        
        pydramsim3.msys_cycle_step(msys=self._msys, cycles=mem_cycles)

    def create_command(self, cmd_q_id: int, addr: int, size: int, is_write: bool) -> CompanionCommandSignature:
        capsule = pydramsim3.create_msys_cmd(cmd_q_id=cmd_q_id, addr=addr, size=size, is_write=is_write)
        cmd = CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs={
            "cmd_q_id": cmd_q_id,
            "addr": addr,
            "size": size,
            "is_write": is_write,
        })
        return cmd

    def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        return pydramsim3.msys_dispatch_cmd(msys=self._msys, cmd=cmd.capsule, dispatch_callback=dispatch_callback, execute_callback=execute_callback)
    
    def enable_bandwidth_profiling(self, resolution: int = 1):
        self._bw_profile_resolution = resolution
        self._bw_profile_hook_id = self.companion_core.register_companion_module_hook(self.module_id, self._bandwidth_profile_hook)
        
    def disable_bandwidth_profiling(self):
        self.companion_core.unregister_companion_module_hook(self.module_id, self._bw_profile_hook_id)
        
    def _bandwidth_profile_hook(self, cmd: CompanionCommandSignature):
        cmd_q_id   = cmd.kwargs["cmd_q_id"]
        size       = cmd.kwargs["size"]
        is_write   = cmd.kwargs["is_write"]
        issue_time  = cmd.issue_time
        commit_time = cmd.commit_time
        
        i_r = issue_time // self._bw_profile_resolution
        c_r = commit_time // self._bw_profile_resolution
        bw  = size / ((commit_time - issue_time) if (commit_time - issue_time) > 0 else 1)  # in bytes per cycle
        
        if cmd_q_id not in self._bw_wr_profiles.keys():
            self._bw_wr_profiles[cmd_q_id] = []
        if cmd_q_id not in self._bw_rd_profiles.keys():
            self._bw_rd_profiles[cmd_q_id] = []

        profile = self._bw_rd_profiles[cmd_q_id] if not is_write else self._bw_wr_profiles[cmd_q_id]

        while len(profile) <= c_r:
            profile.append(0)
            
        for t in range(i_r, c_r + 1):
            profile[t] += bw

    def dump_bandwidth_profiles(self) -> dict[str, dict[int, list[int]]]:
        return {
            "read": self._bw_rd_profiles,
            "write": self._bw_wr_profiles,
        }
    
    def save_bandwidth_profiles_as_file(self, dirname: str):
        profiles = self.dump_bandwidth_profiles()
        filename_fmt = "{cmd_q_id}_{traffic_type}.csv"

        os.makedirs(dirname, exist_ok=True)

        for cmd_q_id, profile in profiles["read"].items():
            profile_path = os.path.join(dirname, filename_fmt.format(cmd_q_id=cmd_q_id, traffic_type="read"))
            with open(profile_path, "wt") as file:
                file.write("timestamp,bandwidth[bytes/cycle]\n")
                for i, bw in enumerate(profile):
                    file.write(f"{i * self._bw_profile_resolution},{bw}\n")
                
            logger.info(f"DRAMSim3 bandwidth profile saved at \"{profile_path}\"")

        for cmd_q_id, profile in profiles["write"].items():
            profile_path = os.path.join(dirname, filename_fmt.format(cmd_q_id=cmd_q_id, traffic_type="write"))
            with open(profile_path, "wt") as file:
                file.write("timestamp,bandwidth[bytes/cycle]\n")
                for i, bw in enumerate(profile):
                    file.write(f"{i * self._bw_profile_resolution},{bw}\n")

            logger.info(f"DRAMSim3 bandwidth profile saved at \"{profile_path}\"")
