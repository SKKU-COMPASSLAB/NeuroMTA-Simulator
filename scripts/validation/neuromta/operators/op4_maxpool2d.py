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
    # torch.set_printoptions(linewidth=1024, threshold=10000)
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group((0, 0), (8, 8))
    
    N, H, W, C = 1, 60, 60, 64
    WINDOW = (2, 2)
    STRIDE, PADDING, DILATION = (2, 2), (0, 0), (1, 1)
    OH = (H + 2 * PADDING[0] - DILATION[0] * (WINDOW[0] - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (WINDOW[1] - 1) - 1) // STRIDE[1] + 1
    
    Ws = 32
    Cs = 32
    
    dtype = torch.int16
    acc_dtype = torch.int16
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = True  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    ifm  = torch.randint(low=0, high=64, size=(N, H, W, C), dtype=dtype)
    ofm  = torch.zeros((N, OH, OW, C), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    spad_ld_pp_space    = device.create_l1_mem_space(parse_mem_cap_str("480KB"), core_group=core_group)
    spad_st_pp_space    = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group)
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(W, Cs), blocked_mapping=False).allocate().update(ifm)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(OW, Cs), blocked_mapping=False).allocate()
    
    operator = MCA_OP_MAXPOOL2D(
        device, core_group, spad_ld_pp_space, spad_st_pp_space, 
        ifm_b, ofm_b, 
        window=WINDOW, stride=STRIDE, padding=PADDING, dilation=DILATION,
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=True, 
        mapping_strategy=MCA_OperatorMapper.CONTIGUOUS,
        use_collective_tile_load=False,
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
    
    total_ops = 2 * OH * OW * C * WINDOW[0] * WINDOW[1]
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.nn.functional.max_pool2d(
        input=ifm.permute(0, 3, 1, 2).to(acc_dtype).contiguous(), 
        kernel_size=WINDOW,
        stride=STRIDE, 
        padding=PADDING, 
        dilation=DILATION
    ).permute(0, 2, 3, 1)
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
