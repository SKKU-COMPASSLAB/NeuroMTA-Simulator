import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *
from neuromta.system.software.tenstorrent import *


TMP_DIR = os.path.join(os.curdir, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)


if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    orig_shape = (32, 4, 32, 4)
    flat_shape = (orig_shape[0], orig_shape[1] * orig_shape[2] * orig_shape[3])
    
    ifm  = torch.randint(low=1, high=64, size=orig_shape, dtype=torch.int16)
    ofm  = torch.zeros(flat_shape, dtype=torch.int16)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    spad_ld_pp_space    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group)
    spad_st_pp_space    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group)
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(4, 4), blocked_mapping=False).allocate().update(ifm)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(4, 4), blocked_mapping=False).allocate()  # TODO: x dimension shard size should always be the same for input and output buffers!
    
    operator = MCA_OP_FLATTEN(
        device, core_group, spad_ld_pp_space, spad_st_pp_space, 
        ifm_b, ofm_b, 
        auto_dispatch=True,
        mapping_strategy=MCA_OperatorMapper.OUTPUT_STATIONARY
    )
    
    tmp_ouput_path = os.path.join(TMP_DIR, "pipelined_mapping.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(operator.summary(), f, indent=4)
        logger.info(f"Pipelined mapping summary saved to '{tmp_ouput_path}'.")
        
    with MonitoringWindow() as monitor:
        for core_id in core_group.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    simulated = ofm_b.restore()
    reference = torch.flatten(ifm, start_dim=1)
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    # print(f"mismatch positions:\n{simulated != reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"total number of zeros in simulated output: {(simulated == 0).sum().item()}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
