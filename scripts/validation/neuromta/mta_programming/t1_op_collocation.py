import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *


TMP_DIR = os.path.join(os.curdir, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)


if __name__ == "__main__":
    # torch.set_printoptions(linewidth=1024, threshold=10000)
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group1 = device.get_npu_core_group((0, 0), (8, 6))
    core_group2 = device.get_npu_core_group((0, 6), (8, 6))
    
    N, H, W, C = 1, 32, 32, 32
    FH, FW, K = 3, 3, 32
    STRIDE, PADDING, DILATION = (1, 1), (1, 1), (1, 1)
    OH = (H + 2 * PADDING[0] - DILATION[0] * (FH - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (FW - 1) - 1) // STRIDE[1] + 1
    
    dtype = torch.int16
    acc_dtype = torch.int16
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = True  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    ifm  = torch.randint(low=0, high=64, size=(N, H, W, C), dtype=dtype)
    wgt  = torch.randint(low=0, high=64, size=(FH, FW, K, C), dtype=dtype)
    bias = torch.randint(low=0, high=64, size=(K,), dtype=acc_dtype)
    ofm  = torch.zeros((N, OH, OW, K), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space1   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group1)
    main_data_mem_space  = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    spad_ld_pp_space1    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group1)
    spad_st_pp_space1    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group1)
    
    ifm_b1  = MCA_TensorBuffer(mem_space=l1_data_mem_space1,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate().update(ifm)
    wgt_b1  = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate().update(wgt)
    bias_b1 = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=bias.shape, dtype=bias.dtype, shard_shape=(1 , 32), blocked_mapping=False).allocate().update(bias)
    ofm_b1  = MCA_TensorBuffer(mem_space=l1_data_mem_space1,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate()
    
    MCA_OP_CONV2D(
        device, core_group1, spad_ld_pp_space1, spad_st_pp_space1, 
        ifm_b1, wgt_b1, bias_b1, ofm_b1, 
        stride=STRIDE, padding=PADDING, dilation=DILATION,
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=True, 
        mapping_strategy=MCA_OperatorMapper.CONTIGUOUS
    )
    
    l1_data_mem_space2   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group2)
    spad_ld_pp_space2    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group2)
    spad_st_pp_space2    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group2)
    
    ifm_b2  = MCA_TensorBuffer(mem_space=l1_data_mem_space2,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate().update(ifm)
    wgt_b2  = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate().update(wgt)
    bias_b2 = MCA_TensorBuffer(mem_space=main_data_mem_space,  shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32), blocked_mapping=False).allocate().update(bias)
    ofm_b2  = MCA_TensorBuffer(mem_space=l1_data_mem_space2,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=False).allocate()
    
    MCA_OP_CONV2D(
        device, core_group2, spad_ld_pp_space2, spad_st_pp_space2, 
        ifm_b2, wgt_b2, bias_b2, ofm_b2, 
        stride=STRIDE, padding=PADDING, dilation=DILATION,
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=True, 
        mapping_strategy=MCA_OperatorMapper.CONTIGUOUS
    )
        
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
    
    total_ops = 2 * OH * OW * K * C * FH * FW
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated1 = ofm_b1.restore()
    simulated2 = ofm_b2.restore()
    reference = torch.nn.functional.conv2d(
        input=ifm.permute(0, 3, 1, 2).to(acc_dtype).contiguous(), 
        weight=wgt.permute(2, 3, 0, 1).to(acc_dtype).contiguous(), 
        bias=bias.to(acc_dtype), 
        stride=STRIDE, 
        padding=PADDING, 
        dilation=DILATION
    ).permute(0, 2, 3, 1)
    
    # print(f"simulated 1:\n{simulated1}")
    # print(f"simulated 2:\n{simulated2}")
    # print(f"reference:\n{reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated1 != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation 1 {'PASSED' if torch.equal(simulated1, reference) else 'FAILED'}")
    num_mismatches = (simulated2 != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation 2 {'PASSED' if torch.equal(simulated2, reference) else 'FAILED'}")