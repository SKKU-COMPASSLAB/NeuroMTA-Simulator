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
    
    core_group = device.get_npu_core_group((0, 0), (8, 8))
    core_group_shape = (2, 2)
    sub_core_groups = core_group.split(shape=core_group_shape)
    
    dtype = torch.int16
    acc_dtype = torch.int16
    # # broadcast_optimize = not args.no_bcast  # Enable broadcast optimization to reduce memory and NoC traffic
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
    
    x_b           = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_x_shape,           dtype=x.dtype,           shard_shape=(56, 3 )).tiling((32, 32)).allocate().update(x.permute(0, 2, 3, 1))  # NCHW -> NHWC
    conv_wgt_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_wgt_shape,    dtype=conv_wgt.dtype,    shard_shape=(32, 3 )).tiling((32, 32)).allocate().update(conv_wgt.permute(2, 3, 0, 1))  # OIHW -> OHWI
    conv_bias_b   = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_bias_shape,   dtype=conv_bias.dtype,   shard_shape=(1,  32)).tiling((1,  32)).allocate().update(conv_bias.unsqueeze(0))
    conv_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_conv_ofm_shape,    dtype=conv_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32))
    relu_ofm_b    = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_relu_ofm_shape,    dtype=relu_ofm.dtype,    shard_shape=(55, 32)).tiling((32, 32))
    maxpool_ofm_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=nmta_maxpool_ofm_shape, dtype=maxpool_ofm.dtype, shard_shape=(27, 32)).tiling((32, 32)).allocate()
    
    conv_core_group = MCA_CoreGroup.merge_core_groups(sub_core_groups[0:14])
    relu_core_group = sub_core_groups[14]
    maxpool_core_group = sub_core_groups[15]
    
    operator1 = MCA_OP_CONV2D(x_b, conv_wgt_b, conv_bias_b, conv_ofm_b, stride=(4, 4), padding=(2, 2)).initialize_core_group(conv_core_group)
    operator2 = MCA_OP_RELU(conv_ofm_b, relu_ofm_b).initialize_core_group(relu_core_group)
    operator3 = MCA_OP_MAXPOOL2D(relu_ofm_b, maxpool_ofm_b, window=(3, 3), stride=(2, 2)).initialize_core_group(maxpool_core_group)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator1)
    compiler.add_op(operator2)
    compiler.add_op(operator3)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=parse_mem_cap_str("512KB"),
        pipeline_granularity=args.pipeline_gran,
        broadcast_optimize_queue_depth=args.bcast_queue_depth,
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
            
    # profilers = {
    #     op.op_id: ExecutionTimeProfiler(device, op.core_group, ["MEM", "EXE"])
    #     for op in [operator1, operator2, operator3]
    # }
        
    # with MonitoringWindow() as monitor:
    #     core_group_with_names = {
    #         "CV": conv_core_group,
    #         "RE": relu_core_group,
    #         "MP": maxpool_core_group,
    #     }
        
    #     for core_group_name, core_group in core_group_with_names.items():
    #         for core_id in core_group.core_ids:
    #             core = device.get_npu_core(core_id=core_id)
    #             pbar_idx = monitor.add_core_pbar(desc=f"{core_group_name} {core_id:<3d}", ncols=40)
    #             monitor.pbar_handles[pbar_idx].bind_core(core)
    if args.monitor:
        with MonitoringWindow(device, [conv_core_group, relu_core_group, maxpool_core_group]) as monitor:
            st = time.time()
            device.run_kernels(max_timestamp=args.max_timestamp)
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels(max_timestamp=args.max_timestamp)
        ed = time.time()
    
    # for op_id, profiler in profilers.items():
    #     profiler_report_path = os.path.join(SUMMARY_DIR, f"execution_time_profile_{op_id}.json")
    #     with open(profiler_report_path, "w") as f:
    #         json.dump(profiler.summary(), f, indent=4)
    #         logger.info(f"Execution time profile saved to '{profiler_report_path}'.")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    maxpool_simulated = maxpool_ofm_b.restore().permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    print(f"simulation3 {'PASSED' if torch.equal(maxpool_simulated, maxpool_ofm) else 'FAILED'}")
    
    # if not torch.equal(conv_simulated, conv_ofm):
    #     mismatch_report = os.path.join(SUMMARY_DIR, "conv_mismatch_report.txt")
    #     with open(mismatch_report, "w") as f:
    #         content = []
    #         s = conv_simulated.flatten()
    #         r = conv_ofm.flatten()
    #         for i in range(s.shape[0]):
    #             sim_val = s[i].item()
    #             ref_val = r[i].item()
    #             if sim_val != ref_val:
    #                 content.append(f"Mismatch at position ({i}): simulated={sim_val}, reference={ref_val}\n")
    #         f.writelines(content)
    #     logger.error(f"Mismatch report saved to '{mismatch_report}'.")
    #     logger.error(f"Total mismatches: {len(content)}/{s.numel()}")
        
    # if not torch.equal(relu_simulated, relu_ofm):
    #     mismatch_report = os.path.join(SUMMARY_DIR, "relu_mismatch_report.txt")
    #     with open(mismatch_report, "w") as f:
    #         content = []
    #         s = relu_simulated.flatten()
    #         r = relu_ofm.flatten()
    #         for i in range(s.shape[0]):
    #             sim_val = s[i].item()
    #             ref_val = r[i].item()
    #             if sim_val != ref_val:
    #                 content.append(f"Mismatch at position ({i}): simulated={sim_val}, reference={ref_val}\n")
    #         f.writelines(content)
    #     logger.error(f"Mismatch report saved to '{mismatch_report}'.")
    #     logger.error(f"Total mismatches: {len(content)}/{s.numel()}")
        
    if not torch.equal(maxpool_simulated, maxpool_ofm):
        mismatch_report = os.path.join(SUMMARY_DIR, "maxpool_mismatch_report.txt")
        with open(mismatch_report, "w") as f:
            content = []
            s = maxpool_simulated.flatten()
            r = maxpool_ofm.flatten()
            for i in range(s.shape[0]):
                sim_val = s[i].item()
                ref_val = r[i].item()
                if sim_val != ref_val:
                    content.append(f"Mismatch at position ({i}): simulated={sim_val}, reference={ref_val}\n")
            f.writelines(content)
        logger.error(f"Mismatch report saved to '{mismatch_report}'.")
        logger.error(f"Total mismatches: {len(content)}/{s.numel()}")
