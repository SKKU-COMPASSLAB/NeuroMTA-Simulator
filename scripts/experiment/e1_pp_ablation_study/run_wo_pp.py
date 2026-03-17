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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CNN micro-benchmark with pipeline parallelism (PP).")
    parser.add_argument("-o", "--output-dir", type=str, default=SUMMARY_DIR, help="Directory to save the compiled operator summaries and simulation results.", dest="output_dir")
    parser.add_argument("--l1-buf-size", type=int, default=parse_mem_cap_str("512KB"), help="Size of the on-chip buffer for each core", dest="l1_buf_size")
    parser.add_argument("--l1-interm", action="store_true", help="Whether to store the intermediate results in L1 buffer to enable faster operator chaining.", dest="l1_interm")
    args = parser.parse_args()
    return args

def get_benchmark_name(args):
    return f"L1BUF{args.l1_buf_size}_IFM_{'L1' if args.l1_interm else 'DRAM'}"


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.INFO)
    
    args = parse_args()
    output_dir = args.output_dir
    l1_buf_size = args.l1_buf_size
    l1_interm = args.l1_interm
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=False)
    
    core_group = device.get_npu_core_group((0, 0), (8, 8))
    core_group_shape = (2, 2)
    sub_core_groups = core_group.split(shape=core_group_shape)
    
    dtype = torch.int16
    acc_dtype = torch.int16
    broadcast_optimize = False  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    x = torch.randint(low=-64, high=64, size=(1, 3, 224, 224), dtype=dtype)
    conv_wgt = torch.randint(low=-64, high=64, size=(96, 3, 11, 11), dtype=dtype)
    conv_bias = torch.randint(low=-128, high=128, size=(96,), dtype=acc_dtype)
    conv_ofm = torch.nn.functional.conv2d(x, conv_wgt, conv_bias, stride=4, padding=2).to(acc_dtype)
    relu_ofm = torch.nn.functional.relu(conv_ofm)
    maxpool_ofm = torch.nn.functional.max_pool2d(relu_ofm, kernel_size=3, stride=2).to(acc_dtype)
    
    logger.info(f"Micro-benchmark Configuration")
    logger.info(f"  x shape: {x.shape}, dtype: {x.dtype}")
    logger.info(f"  conv_wgt shape: {conv_wgt.shape}, dtype: {conv_wgt.dtype}")
    logger.info(f"  conv_bias shape: {conv_bias.shape}, dtype: {conv_bias.dtype}")
    logger.info(f"  conv_ofm shape: {conv_ofm.shape}, dtype: {conv_ofm.dtype}")
    logger.info(f"  relu_ofm shape: {relu_ofm.shape}, dtype: {relu_ofm.dtype}")
    logger.info(f"  maxpool_ofm shape: {maxpool_ofm.shape}, dtype: {maxpool_ofm.dtype}")
    
    nmta_x_shape = x.permute(0, 2, 3, 1).shape  # NCHW -> NHWC
    nmta_conv_wgt_shape = conv_wgt.permute(2, 3, 0, 1).shape  # OIHW -> OHWI
    nmta_conv_bias_shape = conv_bias.unsqueeze(0).shape  # (96,) -> (1, 96)
    nmta_conv_ofm_shape = conv_ofm.permute(0, 2, 3, 1).shape  # NCHW -> NHWC
    nmta_relu_ofm_shape = relu_ofm.permute(0, 2, 3, 1).shape  # NCHW -> NHWC
    nmta_maxpool_ofm_shape = maxpool_ofm.permute(0, 2, 3, 1).shape  # NCHW -> NHWC
    
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    l1_data_mem_space = device.create_l1_mem_space(parse_mem_cap_str("1.5MB") - l1_buf_size, core_group)
    
    x_b           = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_x_shape,           dtype=x.dtype,           shard_shape=(56, 3 )).tiling((32, 32)).allocate().update(x.permute(0, 2, 3, 1))  # NCHW -> NHWC
    conv_wgt_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_wgt_shape,    dtype=conv_wgt.dtype,    shard_shape=(32, 3 )).tiling((32, 32)).allocate().update(conv_wgt.permute(2, 3, 0, 1))  # OIHW -> OHWI
    conv_bias_b   = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_bias_shape,   dtype=conv_bias.dtype,   shard_shape=(1,  32)).tiling((1,  32)).allocate().update(conv_bias.unsqueeze(0))
    try:
        _interm_mem_space = l1_data_mem_space if l1_interm else main_data_mem_space
        conv_ofm_b    = MCA_TensorBuffer(mem_space=_interm_mem_space, shape=nmta_conv_ofm_shape,    dtype=conv_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32)).allocate()
        relu_ofm_b    = MCA_TensorBuffer(mem_space=_interm_mem_space, shape=nmta_relu_ofm_shape,    dtype=relu_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32)).allocate()
    except Exception as e:
        args.l1_interm = False
        conv_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_ofm_shape,    dtype=conv_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32)).allocate()
        relu_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_relu_ofm_shape,    dtype=relu_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32)).allocate()
    maxpool_ofm_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_maxpool_ofm_shape, dtype=maxpool_ofm.dtype, shard_shape=(27, 32)).tiling((32, 32)).allocate()
    
    conv_core_group = core_group
    relu_core_group = core_group
    maxpool_core_group = core_group
    
    operator1 = MCA_OP_CONV2D(x_b, conv_wgt_b, conv_bias_b, conv_ofm_b, stride=(4, 4), padding=(2, 2)).initialize_core_group(conv_core_group)
    operator2 = MCA_OP_RELU(conv_ofm_b, relu_ofm_b).initialize_core_group(relu_core_group)
    operator3 = MCA_OP_MAXPOOL2D(relu_ofm_b, maxpool_ofm_b, window=(3, 3), stride=(2, 2)).initialize_core_group(maxpool_core_group)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator1)
    compiler.add_op(operator2)
    compiler.add_op(operator3)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=l1_buf_size
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    summary_output_dir = os.path.join(output_dir, get_benchmark_name(args))
    os.makedirs(summary_output_dir, exist_ok=True)
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")

    st = time.time()
    device.run_kernels()
    ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    summary_result_path = os.path.join(summary_output_dir, f"simulation_summary.json")
    with open(summary_result_path, "w") as f:
        json.dump({
            "simulation_time_ms": (ed - st)*1000,
            "l1_buf_size": l1_buf_size,
            "use_l1_data_space": l1_interm,
            "timestamp": device.timestamp,
        }, f, indent=4)
        logger.info(f"Simulation summary saved to '{summary_result_path}'.")
