import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


TMP_DIR = os.path.join(os.curdir, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group1 = device.get_npu_core_group((0, 0), (2, 2))
    core_group2 = device.get_npu_core_group((0, 2), (2, 2))
    
    M, N, K = 256, 256, 256
    dtype = torch.int32
    acc_dtype = torch.int32
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = True  # Enable broadcast optimization to reduce memory and NoC traffic
    
    ifm  = torch.randint(low=0, high=128, size=(M, K), dtype=dtype)
    wgt  = torch.randint(low=0, high=128, size=(N, K), dtype=dtype)
    bias = torch.randint(low=0, high=256, size=(N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    main_data_mem_space  = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    l1_data_mem_space1   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group1)
    spad_ld_pp_space1    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group1)
    spad_st_pp_space1    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group1)
    
    ifm_b1  = MCA_TensorBuffer(mem_space=l1_data_mem_space1,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).allocate().update(ifm)
    wgt_b1  = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False          ).allocate().update(wgt)
    bias_b1 = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32), blocked_mapping=False          ).allocate().update(bias)
    ofm_b1  = MCA_TensorBuffer(mem_space=l1_data_mem_space1,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).allocate()
    
    operator1 = MCA_OP_LINEAR(
        device, core_group1, spad_ld_pp_space1, spad_st_pp_space1, 
        ifm_b1, wgt_b1, bias_b1, ofm_b1, 
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=False, 
        mapping_strategy=MCA_OperatorMapper.OUTPUT_STATIONARY
    )
    
    l1_data_mem_space2   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group2)
    spad_ld_pp_space2    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group2)
    spad_st_pp_space2    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group2)
    
    wgt_b2  = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False          ).allocate().update(wgt)
    bias_b2 = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32), blocked_mapping=False          ).allocate().update(bias)
    ofm_b2  = MCA_TensorBuffer(mem_space=l1_data_mem_space2,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).allocate()
    
    operator2 = MCA_OP_LINEAR(
        device, core_group2, spad_ld_pp_space2, spad_st_pp_space2, 
        ofm_b1, wgt_b2, bias_b2, ofm_b2, 
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=False, 
        mapping_strategy=MCA_OperatorMapper.OUTPUT_STATIONARY
    )
    
    operator1.pipeline(
        dst_op=operator2,
        src_buf_name="ofm",
        dst_buf_name="ifm",
    ).dispatch()
        
    with MonitoringWindow() as monitor:
        for core_id in core_group1.core_ids + core_group2.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * M * N * K
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b2.restore()
    reference = torch.nn.functional.linear(ifm.to(acc_dtype), wgt.to(acc_dtype), bias=bias)
    reference = torch.nn.functional.linear(reference, wgt.to(acc_dtype), bias=bias)
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
