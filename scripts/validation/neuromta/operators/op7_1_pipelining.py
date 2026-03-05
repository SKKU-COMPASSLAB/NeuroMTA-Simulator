import os
import json
import time
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *
from neuromta.system.software.tenstorrent import *


FILEROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
LOGDIR = os.path.join(FILEROOT, ".logs")
SUMMARY_DIR = os.path.join(LOGDIR, FILENAME)

os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_group = device.get_npu_core_group((0, 0), (8, 8))
    sub_core_groups = core_group.split(shape=(4, 4))
    
    M, N, K = 512, 512, 512
    dtype = torch.int16
    acc_dtype = torch.int16
    broadcast_optimize = False  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    ifm  = torch.randint(low=0, high=128, size=(M, K), dtype=dtype)
    wgt  = torch.randint(low=0, high=128, size=(N, K), dtype=dtype)
    bias = torch.randint(low=0, high=256, size=(N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm1_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(ifm)
    wgt1_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(wgt)
    bias1_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(bias)
    ofm1_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32)).tiling((32, 32))  # does not allocate intermediate buffer to check pipelining effect 
    wgt2_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(wgt)
    bias2_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(bias)
    ofm2_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32)).tiling((32, 32))  # does not allocate intermediate buffer to check pipelining effect
    wgt3_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(wgt)
    bias3_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(bias)
    ofm3_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate()
    
    operator1 = MCA_OP_LINEAR(ifm1_b, wgt1_b, bias1_b, ofm1_b).initialize_core_group(sub_core_groups[0])
    operator2 = MCA_OP_LINEAR(ofm1_b, wgt2_b, bias2_b, ofm2_b).initialize_core_group(sub_core_groups[1])
    operator3 = MCA_OP_LINEAR(ofm2_b, wgt3_b, bias3_b, ofm3_b).initialize_core_group(sub_core_groups[2])
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator1)
    compiler.add_op(operator2)
    compiler.add_op(operator3)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=parse_mem_cap_str("512KB")
    )
    
    compiled_ops = compiler.compile(global_recipe)
    
    for op_id, compiled_op in compiled_ops.items():
        compiled_op.dispatch(device, slot_id="MAIN")
        
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(compiled_op.summary(), f, indent=4)
            logger.info(f"Pipelined mapping summary saved to '{tmp_output_path}'.")
        
    with MonitoringWindow() as monitor:
        for core_group in [sub_core_groups[0], sub_core_groups[1], sub_core_groups[2]]:
            for core_id in core_group.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar_idx = monitor.add_core_pbar(desc=f"{core_id:<3d}", ncols=40)
                monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * M * N * K * 2
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    # simulated1 = ofm1_b.restore()
    # simulated2 = ofm2_b.restore()
    simulated3 = ofm3_b.restore()
    reference1 = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    reference2 = torch.matmul(reference1.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    reference3 = torch.matmul(reference2.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
    
    # print(f"simulated1:\n{simulated1}")
    # print(f"simulated2:\n{simulated2}")
    print(f"simulated3:\n{simulated3}")
    # print(f"reference1:\n{reference1}")
    # print(f"reference2:\n{reference2}")
    print(f"reference3:\n{reference3}")
    
    # print(f"simulation1 {'PASSED' if torch.equal(simulated1, reference1) else 'FAILED'}")
    # print(f"simulation2 {'PASSED' if torch.equal(simulated2, reference2) else 'FAILED'}")
    print(f"simulation3 {'PASSED' if torch.equal(simulated3, reference3) else 'FAILED'}")
