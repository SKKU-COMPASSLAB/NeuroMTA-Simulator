import os
import math
import torch

from neuromta.framework import *
from neuromta.component import *

from neuromta.component.companions.booksim import PYBOOKSIM2_AVAILABLE, BookSim2Config
from neuromta.component.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config, create_new_dramsim_config_file


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
        cmap_config: CmapConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["processor_clock_freq"] = processor_clock_freq
        self["icnt_config"] = icnt_config
        self["cmap_config"] = cmap_config
        self["mem_config"] = mem_config
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
        n_dma_core_per_channel = 3
        n_main_mem_channels = math.ceil(n_dma_core / n_dma_core_per_channel)
        
        # Interconnect Configuration
        #   - 12x16 torus network
        #   - 32B channel width (2 flits per cycle)
        #   - 2 subnets (randomized duplex network)
        #   - peak bandwidth per router: 2 * 32B * 1GHz * 2 = 128GB/s
        icnt_ch_width = parse_mem_cap_str("32B") * 5  # 2 flits per cycle
        icnt_subnet_num = 2  # randomized duplex network
        
        icnt_config = IcntConfig(
            shape=icnt_shape,
            subnets=icnt_subnet_num,
            flit_size=icnt_ch_width,
            booksim2_enable=PYBOOKSIM2_AVAILABLE,
        )
        
        cmap_config = CmapConfig(
            n_l1_spm_bank=n_npu_core,
            n_main_mem_channels=n_main_mem_channels,
            l1_spm_bank_size=l1_mem_bank_size,
            main_mem_channel_size=main_mem_channel_size,
        )
        
        dma_core_group: dict[int, list[int]] = {}

        for row in range(12):
            inter_ch_idx = row // n_dma_core_per_channel
            intra_ch_idx = row % n_dma_core_per_channel
            
            dma_core_ch_col_0_id = icnt_config.coord_to_core_id((row, 0))
            dma_core_ch_col_1_id = icnt_config.coord_to_core_id((row, 8))
            
            cmap_config.add_dma_core(core_id=dma_core_ch_col_0_id, mem_bank_idx=inter_ch_idx * 2)
            cmap_config.add_dma_core(core_id=dma_core_ch_col_1_id, mem_bank_idx=inter_ch_idx * 2 + 1)
            
            dma_core_group[intra_ch_idx] = dma_core_group.get(intra_ch_idx, []) + [dma_core_ch_col_0_id, dma_core_ch_col_1_id]
        
        for row in range(12):
            intra_ch_idx = row % n_dma_core_per_channel
            
            for i in range(7):
                cmap_config.add_npu_core(core_id=icnt_config.coord_to_core_id((row, 1 + i)), mem_bank_idx=(row * 14) + i,     nxt_level_mem_core_ids=dma_core_group[intra_ch_idx])
                cmap_config.add_npu_core(core_id=icnt_config.coord_to_core_id((row, 9 + i)), mem_bank_idx=(row * 14) + i + 7, nxt_level_mem_core_ids=dma_core_group[intra_ch_idx])


        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )

        if PYDRAMSIM3_AVAILABLE:
            dramsim3_config_path    = TENSTORRENT_IP_DRAMSIM_CONFIG_FMT(config_name=config_name)
            dramsim3_channel_size   = main_mem_channel_size // (1024 * 1024)    # GB -> MB
            
            create_new_dramsim_config_file(
                src_config_path="GDDR6_8Gb_x16.ini",
                new_config_path=dramsim3_config_path,
                system_params={
                    "channel_size": dramsim3_channel_size,
                    "channels": n_main_mem_channels,
                },
                dram_structure_params={
                    "bankgroups": 2  # TODO: more authentic way of doing this..?
                }
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
            transfer_speed=7000,        # MT/s (DDR6 typical speed)
            ch_io_width=32,             # bits (DDR6 typical channel width)
            ch_num=n_main_mem_channels, # channels (example for DDR6)
            burst_len=32,               # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=processor_clock_freq,
            
            # DRAMSIM CONFIG
            dramsim3_enable=PYDRAMSIM3_AVAILABLE,
            dramsim3_config=dramsim3_config,
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
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
            cmap_config=cmap_config,
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )


class TenstorrentDevice(MTA_DeviceBase):
    def __init__(self, processor_clock_freq, cmap_config, icnt_config, mem_config, mxu_config, vpu_config):
        super().__init__(cmap_config, icnt_config, mem_config, mxu_config, vpu_config)
        
        self.processor_clock_freq = processor_clock_freq
