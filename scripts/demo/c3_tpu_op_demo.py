import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mca.google_tpu import *


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = GoogleTPUConfig.V4()
    device = GoogleTPUDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group(0, 1)
    
    M, N, K = 512, 512, 512
    Ms, Ns, Ks = 2, 2, 2
    Mt, Nt, Kt = 128, 128, 128
    dtype = torch.int8
    acc_dtype = torch.int32
    
    ifm  = torch.arange(M * K, dtype=dtype).reshape(M, K)
    wgt  = torch.arange(N * K, dtype=dtype).reshape(N, K)
    bias = torch.arange(N, dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    ifm_tile_size  = Mt * Kt * dtype.itemsize
    wgt_tile_size  = Nt * Kt * dtype.itemsize
    bias_tile_size = Nt * acc_dtype.itemsize
    ofm_tile_size  = Mt * Nt * acc_dtype.itemsize
    
    ifm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("4MB"), core_group=core_group)
    param_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    ofm_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("4MB"), core_group=core_group)
    spad_pp_space   = device.create_l1_mem_space(parse_mem_cap_str("4MB"), core_group=core_group)
    
    ifm_b  = MCA_TensorBuffer(mem_space=ifm_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks)).tiling(tile_shape=(Mt, Kt)).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks)).tiling(tile_shape=(Nt, Kt)).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns)).tiling(tile_shape=(1,  Nt)).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=ofm_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns)).tiling(tile_shape=(Mt, Nt)).allocate()
    
    operator = MCA_LINEAR(core_group, spad_pp_space, ifm_b, wgt_b, bias_b, ofm_b).dispatch(device)
        
    with MonitoringWindow() as monitor:
        for core_id in core_group.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"NPUCore {core_id:<3d}", ncols=60)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
        
    tmp_ouput_path = os.path.join(os.curdir, ".tmp", "pipelined_mapping.json")
    with open(tmp_ouput_path, "w") as f:
        json.dump(operator.summary(), f, indent=4)
        logger.info(f"Pipelined mapping summary saved to '{tmp_ouput_path}'.")
    
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