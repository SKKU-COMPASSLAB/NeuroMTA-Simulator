import os
import math
import torch

from neuromta.framework import *
from neuromta.component import *

from neuromta.component.companions.booksim import PYBOOKSIM2_AVAILABLE
from neuromta.component.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config, create_new_dramsim_config_file


__all__ = [
    "GoogleTPUConfig",
    "GoogleTPUDevice",
]


GOOGLE_TPU_IP_ROOT = os.path.abspath(os.path.dirname(__file__))
GOOGLE_TPU_IP_CACHE_DIR = os.path.join(GOOGLE_TPU_IP_ROOT, ".cache")
GOOGLE_TPU_IP_DRAMSIM_CONFIG_FMT = os.path.join(GOOGLE_TPU_IP_CACHE_DIR, "dramsim_{config_name}.ini").format



class GoogleTPUConfig(dict):
    def __init__(
        self,
        
        processor_clock_freq: int,
        global_config: GlobalContextConfig,
        icnt_config: IcntConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["processor_clock_freq"] = processor_clock_freq
        self["global_config"] = global_config
        self["icnt_config"] = icnt_config
        self["mxu_config"] = mxu_config
        self["vpu_config"] = vpu_config
        
    @classmethod
    def V4(cls) -> 'GoogleTPUConfig':
        config_name = "v4"
        
        processor_clock_freq    = parse_freq_str("1GHz")
        main_mem_channel_size   = parse_mem_cap_str("2GB")
        l1_mem_bank_size        = parse_mem_cap_str("48MB")
        n_main_mem_channels     = 16
        
        n_dma_core = n_main_mem_channels
        n_npu_core = 2
        icnt_shape = (2, max(n_dma_core, n_npu_core))
        
        if PYDRAMSIM3_AVAILABLE:
            dramsim3_config_path    = GOOGLE_TPU_IP_DRAMSIM_CONFIG_FMT(config_name=config_name)
            dramsim3_channel_size   = main_mem_channel_size // (1024 * 1024)    # GB -> MB
            
            create_new_dramsim_config_file(
                src_config_path="HBM2_8Gb_x128.ini",
                new_config_path=dramsim3_config_path,
                system_params={
                    "channel_size": dramsim3_channel_size,
                    "channels": n_main_mem_channels,
                },
            )
            
            dramsim3_config = DRAMSim3Config(
                config_path=dramsim3_config_path,
                processor_clock_freq=processor_clock_freq,
                cmd_queue_num=n_main_mem_channels,
            )
        else:
            dramsim3_config = None

        main_mem_config = MainMemoryConfig(
            # STATIC MEMORY CONFIG (used if pydramsim is not available)
            transfer_speed=7000,      # MT/s (DDR6 typical speed)
            ch_io_width=32,           # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=32,             # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=processor_clock_freq,
            
            # DRAMSIM CONFIG
            dramsim3_enable=PYDRAMSIM3_AVAILABLE,
            dramsim3_config=dramsim3_config,
        )
        
        icnt_config = IcntConfig(                   # INTERCONNECT CONFIG
            shape=icnt_shape,                       # - 12x16 torus
            subnets=2,                              # - 2 subnets (full-duplex)
            flit_size=parse_mem_cap_str("32B"),     # - 32B flit size (the unit of flow control)
            max_payload_size=256,                   # - 256 in flits in maximum as a payload = 8KB
            booksim2_enable=PYBOOKSIM2_AVAILABLE,   # - theoretical bandwidth per direction: 32B * 2 * 1GHz = 64GB/s
        )
        
        global_config = GlobalContextConfig(
            n_npu_core=n_npu_core,
            n_main_mem_channel=n_main_mem_channels,
            n_dma_core=n_dma_core,
            l1_mem_bank_size=l1_mem_bank_size,
            main_mem_bank_size=main_mem_channel_size,
            main_mem_config=main_mem_config,
        )
        
        for i, n in enumerate(global_config.npu_core_ids):
            icnt_config.update_core_map(coord=(0, i), core_id=n)
        for i, n in enumerate(global_config.dma_core_ids):
            icnt_config.update_core_map(coord=(1, i), core_id=n)
        
        mxu_config = MXUConfig(
            pe_arr_height=128,
            pe_arr_width=128,
            seq_len=128,  # TODO: sequence length is 128? 256? for now decide for simple tiling ...
            dtype=torch.float32,
            acc_dtype=torch.float32,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("512B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            processor_clock_freq=processor_clock_freq,
            global_config=global_config,
            icnt_config=icnt_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )

class GoogleTPUDevice(MCA_DeviceBase):
    def __init__(self, processor_clock_freq, global_config, icnt_config, mxu_config, vpu_config):
        super().__init__(global_config, icnt_config, mxu_config, vpu_config)
        
        self.processor_clock_freq = processor_clock_freq