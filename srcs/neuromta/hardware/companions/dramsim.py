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
        raise FileNotFoundError(f"[ERROR] DRAMSim3 config file '{src_config_path}' not found.")

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
            raise FileNotFoundError(f"[ERROR] DRAMSim3 config file '{config_path}' not found.")

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

    def update_cycle_time(self, cycle_time):
        self._rem_clock_sync_time += cycle_time * self._ref_clock_time
        
        mem_cycles = math.floor(self._rem_clock_sync_time / self._mem_clock_time)
        self._rem_clock_sync_time -= mem_cycles * self._mem_clock_time
        
        pydramsim3.msys_cycle_step(msys=self._msys, cycles=mem_cycles)

    def create_command(self, cmd_q_id: int, addr: int, size: int, is_write: bool) -> Any:
        return pydramsim3.create_msys_cmd(cmd_q_id=cmd_q_id, addr=addr, size=size, is_write=is_write)

    def dispatch_command(self, cmd, dispatch_callback: Callable, execute_callback: Callable) -> bool:
        return pydramsim3.msys_dispatch_cmd(msys=self._msys, cmd=cmd, dispatch_callback=dispatch_callback, execute_callback=execute_callback)

    def check_command_executed(self, cmd) -> bool:
        return pydramsim3.check_msys_cmd_executed(cmd=cmd)
