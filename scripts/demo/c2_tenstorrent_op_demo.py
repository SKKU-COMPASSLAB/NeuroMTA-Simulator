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
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    M, N, K = 1024, 1024, 1024
    Ms, Ns, Ks = 32, 32, 32
    dtype = torch.bfloat16
    acc_dtype = torch.bfloat16
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = True  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    ifm  = torch.randint(low=0, high=128, size=(M, K), dtype=dtype)
    wgt  = torch.randint(low=0, high=128, size=(N, K), dtype=dtype)
    bias = torch.randint(low=0, high=256, size=(N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    Mt = M // Ms
    Nt = N // Ns
    Kt = K // Ks
    
    if sim_mode == "all_l1":
        _ifm_size_per_core  = math.ceil(Ms / core_group.shape[0]) * math.ceil(Ks / core_group.shape[1]) * (Mt * Kt * dtype.itemsize)
        _ofm_size_per_core  = math.ceil(Ms / core_group.shape[0]) * math.ceil(Ns / core_group.shape[1]) * (Mt * Nt * acc_dtype.itemsize)
        _wgt_size_per_core  = math.ceil(Ns * Ks / len(core_group)) * (Nt * Kt * dtype.itemsize)
        _bias_size_per_core = math.ceil(Ns / len(core_group)) * (Nt * acc_dtype.itemsize)
        
        _l1_total_per_core     = parse_mem_cap_str("1MB")  # total L1 memory size in Tenstorrent Tensix Core is 1.5MB
        _l1_data_size_per_core = math.ceil((_ifm_size_per_core + _wgt_size_per_core + _bias_size_per_core + _ofm_size_per_core))
        _spad_size_per_core    = _l1_total_per_core - _l1_data_size_per_core  # remaining L1 SPAD size per core
        _spad_st_size_per_core = max(2*32*32*acc_dtype.itemsize, math.floor(_spad_size_per_core * 0.15))
        _spad_ld_size_per_core = _spad_size_per_core - _spad_st_size_per_core
        
        logger.info(f"benchmark L1 memory map: Data: {_l1_data_size_per_core / 1024:.2f} KB, SPAD Load: {_spad_ld_size_per_core / 1024:.2f} KB, SPAD Store: {_spad_st_size_per_core / 1024:.2f} KB")
        
        l1_data_mem_space = device.create_l1_mem_space(_l1_data_size_per_core, core_group=core_group)
        spad_ld_pp_space  = device.create_l1_mem_space(_spad_ld_size_per_core, core_group=core_group)
        spad_st_pp_space  = device.create_l1_mem_space(_spad_st_size_per_core, core_group=core_group)
        
        ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space, shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks), blocked_mapping=blocked_mapping).allocate().update(ifm)
        wgt_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks), blocked_mapping=blocked_mapping).allocate().update(wgt)
        bias_b = MCA_TensorBuffer(mem_space=l1_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns), blocked_mapping=blocked_mapping).allocate().update(bias)
        ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=blocked_mapping).allocate()
    elif sim_mode == "partial_l1":
        _ifm_size_per_core  = math.ceil(Ms / core_group.shape[0]) * math.ceil(Ks / core_group.shape[1]) * (Mt * Kt * dtype.itemsize)
        _ofm_size_per_core  = math.ceil(Ms / core_group.shape[0]) * math.ceil(Ns / core_group.shape[1]) * (Mt * Nt * acc_dtype.itemsize)
        
        _l1_total_per_core     = parse_mem_cap_str("1MB")  # total L1 memory size in Tenstorrent Tensix Core is 1.5MB
        _l1_data_size_per_core = math.ceil((_ifm_size_per_core + _ofm_size_per_core))
        _spad_size_per_core    = _l1_total_per_core - _l1_data_size_per_core  # remaining L1 SPAD size per core
        _spad_st_size_per_core = max(2*32*32*acc_dtype.itemsize, math.floor(_spad_size_per_core * 0.15))
        _spad_ld_size_per_core = _spad_size_per_core - _spad_st_size_per_core
        
        logger.info(f"benchmark L1 memory map: Data: {_l1_data_size_per_core / 1024:.2f} KB, SPAD Load: {_spad_ld_size_per_core / 1024:.2f} KB, SPAD Store: {_spad_st_size_per_core / 1024:.2f} KB")
        
        l1_data_mem_space   = device.create_l1_mem_space(_l1_data_size_per_core, core_group=core_group)
        main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
        spad_ld_pp_space    = device.create_l1_mem_space(_spad_ld_size_per_core, core_group=core_group)
        spad_st_pp_space    = device.create_l1_mem_space(_spad_st_size_per_core, core_group=core_group)
        
        ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_grid=(Ms, Ks), blocked_mapping=blocked_mapping).allocate().update(ifm)
        wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_grid=(Ns, Ks), blocked_mapping=False          ).allocate().update(wgt)
        bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_grid=(1,  Ns), blocked_mapping=False          ).allocate().update(bias)
        ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_grid=(Ms, Ns), blocked_mapping=blocked_mapping).allocate()
    else:
        raise ValueError(f"Unsupported simulation mode '{sim_mode}'")
    
    operator = MCA_OP_LINEAR(
        device, core_group, spad_ld_pp_space, spad_st_pp_space, 
        ifm_b, wgt_b, bias_b, ofm_b, 
        broadcast_optimize=broadcast_optimize, 
        auto_dispatch=True, 
        mapping_strategy=MCA_OperatorMapper.CONTIGUOUS
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
    
    total_ops = 2 * M * N * K
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    if not torch.equal(simulated, reference):
        mismatch_report = os.path.join(TMP_DIR, "mismatch_report.txt")
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
    
    print(l1_data_mem_space.owner_ids)
    for inst_id, cnt in sorted(device.global_context._history.items(), key=lambda x: x[0]):
        print(f"DRAMSim3 Instance {inst_id} handled {cnt} commands.")