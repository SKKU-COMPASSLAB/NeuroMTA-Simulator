import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


if __name__ == "__main__":
    torch.set_printoptions(linewidth=4096)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group1 = device.get_npu_core_group((0, 0), (4, 4))
    core_group2 = device.get_npu_core_group((0, 4), (4, 4))
    
    M, N, K = 512, 512, 512
    Ms, Ns, Ks = 4, 4, 4
    Mt, Nt, Kt = 32, 32, 32
    dtype     = torch.int32
    acc_dtype = torch.int32
    
    if N != K:
        raise ValueError("This demo only supports square weight matrices (N == K).")
    
    ifm  = torch.randint(0, 32, (M, K), dtype=dtype).reshape(M, K)
    wgt  = torch.randint(0, 32, (N, K), dtype=dtype).reshape(N, K)
    bias = torch.randint(0, 32, (N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    param_mem_space  = device.create_main_mem_space(parse_mem_cap_str("2GB"))
    
    ifm_mem_space1    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group1)
    ofm_mem_space1    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group1)
    spad_ld_pp_space1 = device.create_l1_mem_space(parse_mem_cap_str("192KB"), core_group=core_group1)
    spad_st_pp_space1 = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group1)
    
    ifm_mem_space2    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group2)
    ofm_mem_space2    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group2)
    spad_ld_pp_space2 = device.create_l1_mem_space(parse_mem_cap_str("192KB"), core_group=core_group2)
    spad_st_pp_space2 = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group2)
    
    
    ifm1_b  = MCA_TensorBuffer(mem_space=ifm_mem_space1,  shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks), blocked_mapping=True).allocate().update(ifm)
    wgt1_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks), blocked_mapping=True).allocate().update(wgt)
    bias1_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns), blocked_mapping=True).allocate().update(bias)
    
    ifm2_b  = MCA_TensorBuffer(mem_space=ofm_mem_space1,  shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=True).allocate()
    wgt2_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,  dtype=acc_dtype,  shard_grid=(Ns, Ks), blocked_mapping=True).allocate().update(wgt.to(acc_dtype))
    bias2_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns), blocked_mapping=True).allocate().update(bias)
    
    ofm_b   = MCA_TensorBuffer(mem_space=ofm_mem_space2,  shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=True).allocate()
    
    operator1 = MCA_OP_LINEAR(
        device=device,
        core_group=core_group1,
        spad_ld_mem_space=spad_ld_pp_space1,
        spad_st_mem_space=spad_st_pp_space1,
        ifm=ifm1_b,
        wgt=wgt1_b,
        bias=bias1_b,
        ofm=ifm2_b,
    )
    
    operator2 = MCA_OP_LINEAR(
        device=device,
        core_group=core_group2,
        spad_ld_mem_space=spad_ld_pp_space2,
        spad_st_mem_space=spad_st_pp_space2,
        ifm=ifm2_b,
        wgt=wgt2_b,
        bias=bias2_b,
        ofm=ofm_b,
    )
    
    operator1.pipeline(
        dst_op=operator2,
        src_buf_name="ofm",
        dst_buf_name="ifm"
    )
    
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping_operator1.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(operator1.summary(), f, indent=4)
        logger.info(f"Mapping summary for operator 1 saved to '{tmp_ouput_path}'.")
        
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping_operator2.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(operator2.summary(), f, indent=4)
        logger.info(f"Mapping summary for operator 2 saved to '{tmp_ouput_path}'.")
    
    operator1.dispatch()
    # operator2.dispatch()  # dispatched automatically by operator1.dispatch()
        
    with MonitoringWindow() as monitor:
        for core_id in core_group1.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"GROUP 1 {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
            
        for core_id in core_group2.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"GROUP 2 {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()

    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * M * N * K * 2  # 2 MACs per multiply-add + 2 pipelined ops
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    
    reference_ofm1 = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    reference = torch.matmul(reference_ofm1, wgt.t().to(acc_dtype)) + bias
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
