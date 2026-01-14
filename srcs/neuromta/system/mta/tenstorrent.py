import os
import math
import torch

from neuromta.framework import *
from neuromta.component import *

from neuromta.component.companions.booksim import PYBOOKSIM2_AVAILABLE, BookSim2Config
from neuromta.component.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config

try:
    from pydramsim3 import create_new_dramsim_config_file
except ImportError:
    create_new_dramsim_config_file = None


__all__ = [
    "TenstorrentConfig",
    "TenstorrentDevice",
]


TENSTORRENT_IP_ROOT = os.path.abspath(os.path.dirname(__file__))
TENSTORRENT_IP_CACHE_DIR = os.path.join(TENSTORRENT_IP_ROOT, ".cache")
TENSTORRENT_IP_DRAMSIM_CONFIG_FMT = os.path.join(TENSTORRENT_IP_CACHE_DIR, "dramsim_{config_name}.ini").format


class TenstorrentConfig(dict):
    def __init__(
        self,
        
        processor_clock_freq: int,
        icnt_config: IcntConfig, 
        global_config: GlobalContextConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["processor_clock_freq"] = processor_clock_freq
        self["icnt_config"] = icnt_config
        self["global_config"] = global_config
        self["mxu_config"] = mxu_config
        self["vpu_config"] = vpu_config
        
    @classmethod
    def BLACKHOLE(cls) -> 'TenstorrentConfig':
        config_name = "blackhole"
        
        processor_clock_freq    = parse_freq_str("1GHz")
        main_mem_channel_size   = parse_mem_cap_str("4GB")
        l1_mem_bank_size        = parse_mem_cap_str("1.5MB")

        icnt_shape = (12, 16)
        n_npu_core = 12 * 14
        n_dma_core = 12 * 2
        
        n_main_mem_cmd_q_per_instance = 3
        n_main_mem_instances = math.ceil(n_dma_core / n_main_mem_cmd_q_per_instance)

        if PYDRAMSIM3_AVAILABLE:
            dramsim3_config_path    = TENSTORRENT_IP_DRAMSIM_CONFIG_FMT(config_name=config_name)
            dramsim3_channel_size   = main_mem_channel_size // (1024 * 1024)    # GB -> MB
            
            create_new_dramsim_config_file(
                src_config_path="GDDR6_8Gb_x16.ini",
                new_config_path=dramsim3_config_path,
                system_params={
                    "channel_size": dramsim3_channel_size,
                    "channels": n_main_mem_cmd_q_per_instance,  # original config is for 3 channels
                    # "address_mapping": "rorababgchco",
                },
                dram_structure_params={
                    "bankgroups": 1  # TODO: more authentic way of doing this..?
                }
            )
            
            dramsim3_config = DRAMSim3Config(
                config_path=dramsim3_config_path,
                processor_clock_freq=processor_clock_freq,
                n_instance=n_main_mem_instances,
                n_cmd_q_per_instance=n_main_mem_cmd_q_per_instance,
            )
        else:
            dramsim3_config = None

        main_mem_config = MainMemoryConfig(
            # STATIC MEMORY CONFIG (used if pydramsim is not available)
            transfer_speed=7000,         # MT/s (DDR6 typical speed)
            ch_io_width=32,              # bits (DDR6 typical channel width)
            ch_num=n_main_mem_instances, # channels (example for DDR6)
            burst_len=32,                # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=processor_clock_freq,
            
            # DRAMSIM CONFIG
            dramsim3_enable=PYDRAMSIM3_AVAILABLE,
            dramsim3_config=dramsim3_config,
        )
        
        icnt_config = IcntConfig(                   # INTERCONNECT CONFIG
            shape=icnt_shape,                       # - 12x16 torus
            subnets=5,                              # - 5 subnets
            flit_size=parse_mem_cap_str("64B"),     # - 64B flit size (the unit of flow control)
            max_payload_size=4,                     # - 4 in flits in maximum as a payload = 256B
            booksim2_enable=PYBOOKSIM2_AVAILABLE,   # - theoretical bandwidth per direction: 64B * 5 * 1GHz = 320GB/s
            booksim2_kwargs={
                "in_ports": 32,
                "out_ports": 32,
                "input_speedup": 32,
                "output_speedup": 32,
            }
        )
        
        global_config = GlobalContextConfig(
            n_npu_core=n_npu_core,
            n_dma_core=n_dma_core,
            
            n_main_mem_instances=n_main_mem_instances,
            n_main_mem_cmd_q_per_instance=n_main_mem_cmd_q_per_instance,
            
            l1_mem_bank_size=l1_mem_bank_size,
            main_mem_bank_size=main_mem_channel_size,
            main_mem_config=main_mem_config,
        )

        dma_core_to_coord_map: dict[int, list[tuple[int, int]]] = {}
        dma_cmap_col = [0, 8]
        npu_core_to_coord_map: dict[int, tuple[int, int]] = {}
        
        for main_mem_inst_idx in range(n_main_mem_instances):
            c = dma_cmap_col[main_mem_inst_idx % 2]
            for main_mem_cmd_q_idx in range(n_main_mem_cmd_q_per_instance):
                r = main_mem_inst_idx // 2 * n_main_mem_cmd_q_per_instance + main_mem_cmd_q_idx
                d = global_config.dma_core_ids[main_mem_inst_idx * n_main_mem_cmd_q_per_instance + main_mem_cmd_q_idx]
                dma_core_to_coord_map[d] = (r, c)
        
        cnt = 0
        for r in range(icnt_shape[0]):
            for c in range(icnt_shape[1]):
                if c in dma_cmap_col:
                    continue
                
                n = global_config.npu_core_ids[cnt]
                npu_core_to_coord_map[n] = (r, c)
                cnt += 1
                
        for d, coord in dma_core_to_coord_map.items():
            icnt_config.update_core_map(coord, d)
        for n, coord in npu_core_to_coord_map.items():
            icnt_config.update_core_map(coord, n)
        
        mxu_config = MXUConfig(
            peak_op_per_cycle=4096,  # from the tenstorrent blackhole specsheet (refer to the official github repo)
            preload_cycle=1,         # assume pipelining (additional 1 cycle for tail latency)
            flush_cycle=1,           # assume pipelining (additional 1 cycle for tail latency)
            
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            op_latency_per_byte=0.5,  # the peak performance assumes bfloat16 (2 bytes per operation)
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
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


class TenstorrentDevice(MTA_DeviceBase):
    def __init__(self, processor_clock_freq, global_config, icnt_config, mxu_config, vpu_config):
        super().__init__(global_config, icnt_config, mxu_config, vpu_config)
        
        self.processor_clock_freq = processor_clock_freq
