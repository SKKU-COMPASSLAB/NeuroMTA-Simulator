import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    M, N, K = 512, 512, 512
    Ms, Ns, Ks = 4, 4, 4
    dtype = torch.int32
    acc_dtype = torch.int32
    blocked_mapping = False  # Enable blocked mapping for better data locality
    broadcast_optimize = True  # Enable broadcast optimization to reduce memory and NoC traffic
    
    ifm  = torch.randint(low=0, high=128, size=(M, K), dtype=dtype)
    wgt  = torch.randint(low=0, high=128, size=(N, K), dtype=dtype)
    bias = torch.randint(low=0, high=256, size=(N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    ifm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group)
    param_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    ofm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group)
    spad_ld_pp_space = device.create_l1_mem_space(parse_mem_cap_str("256KB"), core_group=core_group)
    spad_st_pp_space = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group)
    
    ifm_b  = MCA_TensorBuffer(mem_space=ifm_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks), blocked_mapping=blocked_mapping).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks), blocked_mapping=blocked_mapping).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns), blocked_mapping=blocked_mapping).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=ofm_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=blocked_mapping).allocate()
    
    operator = MCA_OP_LINEAR(
        device, core_group, spad_ld_pp_space, spad_st_pp_space, 
        ifm_b, wgt_b, bias_b, ofm_b, 
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=True, 
        mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN
    )
    
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping.json")
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
    
    total_ops = 2 * M * N * K
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    if not torch.equal(simulated, reference):
        mismatch_report = os.path.join(os.curdir, ".tmp", "mismatch_report.txt")
        with open(mismatch_report, "w") as f:
            content = []
            for i in range(M):
                for j in range(N):
                    sim_val = simulated[i, j].item()
                    ref_val = reference[i, j].item()
                    if sim_val != ref_val:
                        content.append(f"Mismatch at position ({i}, {j}): simulated={sim_val}, reference={ref_val}\n")
            f.writelines(content)
        logger.error(f"Mismatch report saved to '{mismatch_report}'.")