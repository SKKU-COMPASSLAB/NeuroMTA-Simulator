import os
import math
import torch

from neuromta.framework import *
from neuromta.component import *

from neuromta.component.companions.booksim import PYBOOKSIM2_AVAILABLE
from neuromta.component.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config

try:
    from pydramsim3 import create_new_dramsim_config_file
except ImportError:
    create_new_dramsim_config_file = None

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
    def V4(
        cls,
        processor_clock_freq: int = parse_freq_str("1GHz"),
        main_mem_channel_size: int = parse_mem_cap_str("2GB"),
        l1_mem_bank_size: int = parse_mem_cap_str("48MB"),
        l1_mem_dynamic_space_size_per_bank: int = 0,
    ) -> 'GoogleTPUConfig':
        config_name = "v4"
        
        n_dma_core = 8
        n_npu_core = 8
        icnt_shape = (2, max(n_dma_core, n_npu_core))
        
        n_main_mem_instances = 1
        n_main_mem_channel_per_instance = 8
        n_main_mem_cmd_q_per_instance = 8

        main_mem_config = MainMemoryConfig(
            # STATIC MEMORY CONFIG
            processor_clock_freq=processor_clock_freq,
            n_instance=n_main_mem_instances,
            channel_size=main_mem_channel_size,
            n_channel_per_instance=n_main_mem_channel_per_instance,
            n_cmd_q_per_instance=n_main_mem_cmd_q_per_instance,
            
            # DRAMSIM CONFIG
            dramsim3_enable=PYDRAMSIM3_AVAILABLE,
            dramsim3_src_config_path="HBM2_8Gb_x128.ini",
            dramsim3_dst_config_path=GOOGLE_TPU_IP_DRAMSIM_CONFIG_FMT(config_name=config_name),
        )
        
        icnt_config = IcntConfig(                       # INTERCONNECT CONFIG
            processor_clock_freq=processor_clock_freq,  # - 1.35GHz (from the tenstorrent blackhole specsheet, refer to the official github repo)
            shape=icnt_shape,                           # - 12x16 torus
            subnets=2,                                  # - 2 independent NoCs (one per router)
            flit_size=parse_mem_cap_str("64B"),         # - 64B flit size (flow-control unit)
            max_payload_size=32,                        # - max payload = 32 flits (= 2048B)
            booksim2_enable=PYBOOKSIM2_AVAILABLE,
            booksim2_kwargs={
                "routing_delay": 1,
                "vc_alloc_delay": 1,
                "sw_alloc_delay": 1,
                "st_prepare_delay": 0,
                "st_final_delay": 1,
                
                "input_speedup": 1,
                "output_speedup": 1,
                "internal_speedup": 1.0,
                
                "num_vcs": 16,
                "vc_buf_size": 8,
            }
        )
        
        global_config = GlobalContextConfig(
            n_npu_core=n_npu_core,
            n_dma_core=n_dma_core,
            
            l1_mem_bank_size=l1_mem_bank_size,
            l1_mem_dynamic_space_size_per_bank=l1_mem_dynamic_space_size_per_bank,
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