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
    
    core_group = device.get_npu_core_group((0, 0), (12, 14))
    core_group_shape = (2, 2)
    sub_core_groups = core_group.split(shape=core_group_shape)
    
    dtype = torch.int16
    acc_dtype = torch.int16
    broadcast_optimize = False  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    BS = 32
    
    x = torch.randint(low=-64, high=64, size=(BS, 784), dtype=dtype)
    linear1_wgt = torch.randint(low=-64, high=64, size=(256, 784), dtype=dtype)
    linear1_bias = torch.randint(low=-128, high=128, size=(256,), dtype=acc_dtype)
    linear1_ofm = torch.nn.functional.linear(x, linear1_wgt, linear1_bias).to(acc_dtype)
    relu1_ofm = torch.nn.functional.relu(linear1_ofm)
    linear2_wgt = torch.randint(low=-64, high=64, size=(128, 256), dtype=dtype)
    linear2_bias = torch.randint(low=-128, high=128, size=(128,), dtype=acc_dtype)
    linear2_ofm = torch.nn.functional.linear(relu1_ofm, linear2_wgt, linear2_bias).to(acc_dtype)
    relu2_ofm = torch.nn.functional.relu(linear2_ofm)
    linear3_wgt = torch.randint(low=-64, high=64, size=(10, 128), dtype=dtype)
    linear3_bias = torch.randint(low=-128, high=128, size=(10,), dtype=acc_dtype)
    linear3_ofm = torch.nn.functional.linear(relu2_ofm, linear3_wgt, linear3_bias).to(acc_dtype)
    
    logger.info(f"Micro-benchmark Configuration")
    logger.info(f"  x shape: {x.shape}, dtype: {x.dtype}")
    logger.info(f"  linear1_wgt shape: {linear1_wgt.shape}, dtype: {linear1_wgt.dtype}")
    logger.info(f"  linear1_bias shape: {linear1_bias.shape}, dtype: {linear1_bias.dtype}")
    logger.info(f"  linear1_ofm shape: {linear1_ofm.shape}, dtype: {linear1_ofm.dtype}")
    logger.info(f"  relu1_ofm shape: {relu1_ofm.shape}, dtype: {relu1_ofm.dtype}")
    logger.info(f"  linear2_wgt shape: {linear2_wgt.shape}, dtype: {linear2_wgt.dtype}")
    logger.info(f"  linear2_bias shape: {linear2_bias.shape}, dtype: {linear2_bias.dtype}")
    logger.info(f"  linear2_ofm shape: {linear2_ofm.shape}, dtype: {linear2_ofm.dtype}")
    logger.info(f"  relu2_ofm shape: {relu2_ofm.shape}, dtype: {relu2_ofm.dtype}")
    logger.info(f"  linear3_wgt shape: {linear3_wgt.shape}, dtype: {linear3_wgt.dtype}")
    logger.info(f"  linear3_bias shape: {linear3_bias.shape}, dtype: {linear3_bias.dtype}")
    logger.info(f"  linear3_ofm shape: {linear3_ofm.shape}, dtype: {linear3_ofm.dtype}")
    
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    x_b            = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=x.shape,            dtype=x.dtype,            shard_shape=(BS, 56)).tiling((32, 32)).allocate().update(x)
    linear1_wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear1_wgt.shape,  dtype=linear1_wgt.dtype,  shard_shape=(32, 56)).tiling((32, 32)).allocate().update(linear1_wgt)
    linear1_bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear1_bias.shape, dtype=linear1_bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(linear1_bias.unsqueeze(0))
    linear1_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear1_ofm.shape,  dtype=linear1_ofm.dtype,  shard_shape=(BS, 32)).tiling((32, 32))
    relu1_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=relu1_ofm.shape,    dtype=relu1_ofm.dtype,    shard_shape=(BS, 32)).tiling((32, 32))
    linear2_wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_wgt.shape,  dtype=linear2_wgt.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(linear2_wgt)
    linear2_bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_bias.shape, dtype=linear2_bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(linear2_bias.unsqueeze(0))
    linear2_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_ofm.shape,  dtype=linear2_ofm.dtype,  shard_shape=(BS, 32)).tiling((32, 32))
    relu2_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=relu2_ofm.shape,    dtype=relu2_ofm.dtype,    shard_shape=(BS, 32)).tiling((32, 32))
    linear3_wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_wgt.shape,  dtype=linear3_wgt.dtype,  shard_shape=(10, 32)).tiling((32, 32)).allocate().update(linear3_wgt)
    linear3_bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_bias.shape, dtype=linear3_bias.dtype, shard_shape=(1,  10)).tiling((1,  32)).allocate().update(linear3_bias.unsqueeze(0))
    linear3_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_ofm.shape,  dtype=linear3_ofm.dtype,  shard_shape=(BS, 10)).tiling((32, 32)).allocate()
    
    linear1_core_group = sub_core_groups[0]
    relu1_core_group   = sub_core_groups[1]
    linear2_core_group = sub_core_groups[2]
    relu2_core_group   = sub_core_groups[3]
    linear3_core_group = sub_core_groups[4]
    
    operator1 = MCA_OP_LINEAR(x_b, linear1_wgt_b, linear1_bias_b, linear1_ofm_b).initialize_core_group(linear1_core_group)
    operator2 = MCA_OP_RELU(linear1_ofm_b, relu1_ofm_b).initialize_core_group(relu1_core_group)
    operator3 = MCA_OP_LINEAR(relu1_ofm_b, linear2_wgt_b, linear2_bias_b, linear2_ofm_b).initialize_core_group(linear2_core_group)
    operator4 = MCA_OP_RELU(linear2_ofm_b, relu2_ofm_b).initialize_core_group(relu2_core_group)
    operator5 = MCA_OP_LINEAR(relu2_ofm_b, linear3_wgt_b, linear3_bias_b, linear3_ofm_b).initialize_core_group(linear3_core_group)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator1)
    compiler.add_op(operator2)
    compiler.add_op(operator3)
    compiler.add_op(operator4)
    compiler.add_op(operator5)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=parse_mem_cap_str("128KB")
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
        
    with MonitoringWindow() as monitor:
        core_group_with_names: dict[str, MCA_CoreGroup] = {
            "L1": linear1_core_group,
            "R1": relu1_core_group,
            "L2": linear2_core_group,
            "R2": relu2_core_group,
            "L3": linear3_core_group,
        }
        
        for core_group_name, core_group in core_group_with_names.items():
            for core_id in core_group.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar_idx = monitor.add_core_pbar(desc=f"{core_group_name} {core_id:<3d}", ncols=40)
                monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    linear3_simulated = linear3_ofm_b.restore()
    
    print(f"simulation3 {'PASSED' if torch.equal(linear3_simulated, linear3_ofm) else 'FAILED'}")
