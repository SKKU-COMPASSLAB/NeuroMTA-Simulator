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
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    M, N, K = 256, 256, 1024
    dtype = torch.int16
    acc_dtype = torch.int16
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
    
    # l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group)
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=device.get_npu_core_group()).override(core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    set_global_mca_op_option(
        spad_ld_mem_space_size=parse_mem_cap_str("256KB"), 
        spad_st_mem_space_size=parse_mem_cap_str("256KB"),
    )
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).tiling((32, 32))#.allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False          ).tiling((32, 32))#.allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32), blocked_mapping=False          ).tiling((1,  32))#.allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).tiling((32, 32))#.allocate()
    
    operator = MCA_OP_LINEAR(
        ifm_b, wgt_b, bias_b, ofm_b, 
    )
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator)
    
    global_recipe=MCA_OperatorGraphCompiler.GlobalRecipe(
        device=device,
        global_core_group=core_group,
        core_group_shape=(4, 4),
        op_recipes={
            operator.op_type: MCA_OperatorGraphCompiler.OperatorRecipe(
                spatial_reuse_target_buf_idx=1,
                use_broadcast_optimize=broadcast_optimize,
            )
        },
    )
    
    compiled_ops = compiler.compile(global_recipe, target_ops="ALL")
    
    ifm_b.allocate().update(ifm)
    wgt_b.allocate().update(wgt)
    bias_b.allocate().update(bias)
    ofm_b.allocate()
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()
    
    for op_id, compiled_op in compiled_ops.items():
        compiled_op.dispatch(device, slot_id="MAIN")
        
        tmp_output_path = os.path.join(TMP_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(compiled_op.summary(), f, indent=4)
            logger.info(f"Pipelined mapping summary saved to '{tmp_output_path}'.")
        
    with MonitoringWindow() as monitor:
        for core_id in core_group.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"{core_id:<3d}", ncols=40)
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
