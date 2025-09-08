import math
from neuromta.framework import *
from neuromta.hardware.companions.dramsim import DRAMSim3Config


__all__ = [
    "MainMemoryConfig",
    "L1MemoryConfig",
    "MemConfig",
    "MemContext",
]

class MainMemoryConfig:
    def __init__(
        self, 
        
        # Default: HBM2 Configuration
        transfer_speed: int         = 1600, 
        ch_io_width: int            = 1024, 
        ch_num: int                 = 1, 
        burst_len: int              = 256, 
        is_ddr: bool                = True, 
        processor_clock_freq: int   = parse_freq_str("1GHz"),
        
        # DRAMSim3 Configuation (if needed)
        dramsim3_enable: bool           = False,
        dramsim3_config: DRAMSim3Config = None,
    ):
        self.transfer_speed         = transfer_speed    # transfer speed per pin (MT/s)
        self.ch_io_width            = ch_io_width       # io channel width (bits)
        self.ch_num                 = ch_num            # number of channels
        self.burst_len              = burst_len         # burst length
        self.is_ddr                 = is_ddr
        self.processor_clock_freq   = processor_clock_freq

        self.dramsim3_config        = dramsim3_config
        self.dramsim3_enable        = dramsim3_enable

    def get_cycles(self, size: int) -> int:
        self.transfer_speed_bytes = (self.transfer_speed * (2 ** 20) * self.ch_io_width * self.ch_num // 8)  # Byte/s
        self.transfer_speed_per_cycles = self.transfer_speed_bytes / self.processor_clock_freq   # Byte/cycle
        
        return math.ceil(size / self.transfer_speed_per_cycles)
    
    
class L1MemoryConfig:
    def __init__(
        self,
        
        access_gran: int = parse_mem_cap_str("32B"),
    ):
        self.access_gran = access_gran
        
    def get_cycles(self, size: int) -> int:
        return math.ceil(size / self.access_gran)


class MemConfig:
    def __init__(
        self,
        l1_config: L1MemoryConfig = L1MemoryConfig(),
        main_config: MainMemoryConfig = MainMemoryConfig(),
    ):
        super().__init__()
        
        self.l1_config = l1_config
        self.main_config = main_config

class MemContext:
    def __init__(self, config: MemConfig):
        self._config = config
    
    @property
    def l1_config(self) -> L1MemoryConfig:
        return self.config.l1_config

    @property
    def main_config(self) -> MainMemoryConfig:
        return self.config.main_config
    
    @property
    def config(self) -> MemConfig:
        return self._config