import os
import json
import time
import torch
import argparse

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
    parser = argparse.ArgumentParser(description="Validate OP6 AvgPool2D operator on Tenstorrent hardware.")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
    parser.add_argument('--debug-command', action="store_true", help="Whether to enable command-level debugging", dest="debug_command")
    parser.add_argument('--report-mismatch', action="store_true", help="Whether to generate mismatch report when validation fails", dest="report_mismatch")
    parser.add_argument('--bcast-queue-depth', type=int, default=16, help="The depth of the broadcast queue", dest="bcast_queue_depth")
    parser.add_argument('--pipeline-gran', type=int, default=8, help="The number of micro-operations per pipeline stage", dest="pipeline_gran")
    parser.add_argument('--max-timestamp', type=int, default=-1, help="Maximum timestamp to run the simulation", dest="max_timestamp")
    args = parser.parse_args()

    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.debug_command)
    
    core_groups = device.get_npu_core_group((0, 0), (8, 8)).split(shape=(1, 1))
    
    dtype = torch.int16
    acc_dtype = torch.int16
    # # broadcast_optimize = not args.no_bcast  # Enable broadcast optimization to reduce memory and NoC traffic
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
    linear1_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear1_ofm.shape,  dtype=linear1_ofm.dtype,  shard_shape=(BS, 32)).tiling((32, 32)).allocate()
    relu1_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=relu1_ofm.shape,    dtype=relu1_ofm.dtype,    shard_shape=(BS, 32)).tiling((32, 32)).allocate()
    linear2_wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_wgt.shape,  dtype=linear2_wgt.dtype,  shard_shape=(32, 32)).tiling((32, 32)).allocate().update(linear2_wgt)
    linear2_bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_bias.shape, dtype=linear2_bias.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(linear2_bias.unsqueeze(0))
    linear2_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear2_ofm.shape,  dtype=linear2_ofm.dtype,  shard_shape=(BS, 32)).tiling((32, 32)).allocate()
    relu2_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=relu2_ofm.shape,    dtype=relu2_ofm.dtype,    shard_shape=(BS, 32)).tiling((32, 32)).allocate()
    linear3_wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_wgt.shape,  dtype=linear3_wgt.dtype,  shard_shape=(10, 32)).tiling((32, 32)).allocate().update(linear3_wgt)
    linear3_bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_bias.shape, dtype=linear3_bias.dtype, shard_shape=(1,  10)).tiling((1,  32)).allocate().update(linear3_bias.unsqueeze(0))
    linear3_ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=linear3_ofm.shape,  dtype=linear3_ofm.dtype,  shard_shape=(BS, 10)).tiling((32, 32)).allocate()
    
    operator1 = MCA_OP_LINEAR(x_b, linear1_wgt_b, linear1_bias_b, linear1_ofm_b)
    operator2 = MCA_OP_RELU(linear1_ofm_b, relu1_ofm_b)
    operator3 = MCA_OP_LINEAR(relu1_ofm_b, linear2_wgt_b, linear2_bias_b, linear2_ofm_b)
    operator4 = MCA_OP_RELU(linear2_ofm_b, relu2_ofm_b)
    operator5 = MCA_OP_LINEAR(relu2_ofm_b, linear3_wgt_b, linear3_bias_b, linear3_ofm_b)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator1)
    compiler.add_op(operator2)
    compiler.add_op(operator3)
    compiler.add_op(operator4)
    compiler.add_op(operator5)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        core_groups=core_groups,
        spad_space_size_per_core=parse_mem_cap_str("64KB"),
        pipeline_granularity=args.pipeline_gran,
        broadcast_optimize_queue_depth=args.bcast_queue_depth,
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
    
    if args.monitor:
        with MonitoringWindow(device, core_groups) as monitor:
            st = time.time()
            device.run_kernels(max_timestamp=args.max_timestamp)
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels(max_timestamp=args.max_timestamp)
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    linear1_simulated = linear1_ofm_b.restore()
    relu1_simulated   = relu1_ofm_b.restore()
    linear2_simulated = linear2_ofm_b.restore()
    relu2_simulated   = relu2_ofm_b.restore()
    linear3_simulated = linear3_ofm_b.restore()
    
    print(f"simulation1 {'PASSED' if torch.equal(linear1_simulated, linear1_ofm) else 'FAILED'}")
    print(f"simulation2 {'PASSED' if torch.equal(relu1_simulated,   relu1_ofm  ) else 'FAILED'}")
    print(f"simulation3 {'PASSED' if torch.equal(linear2_simulated, linear2_ofm) else 'FAILED'}")
    print(f"simulation4 {'PASSED' if torch.equal(relu2_simulated,   relu2_ofm  ) else 'FAILED'}")
    print(f"simulation5 {'PASSED' if torch.equal(linear3_simulated, linear3_ofm) else 'FAILED'}")
