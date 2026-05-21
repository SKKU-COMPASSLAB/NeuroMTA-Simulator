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
    parser = argparse.ArgumentParser(description="Validate OP1 Linear operator on Tenstorrent hardware.")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
    parser.add_argument('--debug-command', action="store_true", help="Whether to enable command-level debugging", dest="debug_command")
    parser.add_argument('--report-mismatch', action="store_true", help="Whether to generate mismatch report when validation fails", dest="report_mismatch")
    parser.add_argument('--max-timestamp', type=int, default=-1, help="Maximum timestamp to run the simulation", dest="max_timestamp")
    parser.add_argument('--save-profile', action="store_true", help="Whether to save profiler data to files", dest="save_profile")
    parser.add_argument('--save-compile-summary', action="store_true", help="Whether to save compilation summary to files", dest="save_compile_summary")
    args = parser.parse_args()

    torch.set_printoptions(profile="full", linewidth=2048)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    torch.manual_seed(0)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.debug_command)
    
    core_group = device.get_npu_core_group((0, 0), (12, 14))
    
    # N, H, W, C = 1, 13, 13, 256
    # FH, FW, K = 3, 3, 256
    # STRIDE, PADDING, DILATION = (1, 1), (1, 1), (1, 1)
    # N, H, W, C = 1, 14, 14, 256
    # FH, FW, K = 3, 3, 512
    # STRIDE, PADDING, DILATION = (1, 1), (1, 1), (1, 1)
    # N, H, W, C = 1, 56, 56, 64
    # FH, FW, K = 3, 3, 64
    # STRIDE, PADDING, DILATION = (1, 1), (1, 1), (1, 1)
    N, H, W, C = 1, 224, 224, 3
    FH, FW, K = 11, 11, 96
    STRIDE, PADDING, DILATION = (4, 4), (2, 2), (1, 1)
    # N, H, W, C = 1, 224, 224, 3
    # FH, FW, K = 7, 7, 64
    # STRIDE, PADDING, DILATION = (2, 2), (3, 3), (1, 1)
    OH = (H + 2 * PADDING[0] - DILATION[0] * (FH - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (FW - 1) - 1) // STRIDE[1] + 1
    
    dtype = torch.int16
    acc_dtype = torch.int16
    
    ifm  = torch.ones((N, H, W, C), dtype=dtype)
    wgt  = torch.ones((FH, FW, K, C), dtype=dtype)
    bias = torch.ones((K,), dtype=acc_dtype) * 2
    ofm  = torch.zeros((N, OH, OW, K), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ifm.shape,  dtype=ifm.dtype).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype).allocate()
    
    operator = MCA_OP_CONV2D(
        ifm_b, wgt_b, bias_b, ofm_b, 
        stride=STRIDE, padding=PADDING, dilation=DILATION,
    )
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        core_groups=[core_group],
        spad_space_size_per_core=parse_mem_cap_str("1MB"),
        broadcast_optimize_queue_depth=4,
        broadcast_optimize_max_ref_cnt=16,
        context_buffer_slot_num=8,
        ld_ex_buffer_slot_num=8,
        ex_st_buffer_slot_num=4,
        concurrent_load_num=2,
        temporal_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.ALL,       # ifm temporal reuse
        spatial_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_MAIN,   # weight broadcast
        # temporal_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_L1,       # ifm temporal reuse
        # spatial_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_MAIN,   # weight broadcast
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
    
    if args.save_profile:
        profilers = [
            DRAMBandwidthProfiler(device, record_type="BOTH"),
            InterconnectBandwidthProfiler(device),
            ThreadUtilizationProfiler(device, core_group, slot_id="LD"),
            ThreadUtilizationProfiler(device, core_group, slot_id="EX"),
            ThreadUtilizationProfiler(device, core_group, slot_id="ST"),
        ]
        
        profiler_saver = ProfilerFileSaverHub(output_dir=os.path.join(SUMMARY_DIR, "profiles"))
        profiler_saver.add_profilers(*profilers)
    
    if args.monitor:
        with MonitoringWindow(device, core_group) as monitor:
            st = time.time()
            device.run_kernels(max_timestamp=args.max_timestamp)
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels(max_timestamp=args.max_timestamp)
        ed = time.time()
    
    if args.save_profile:
        profiler_saver.close()
    
    for profiler, saver_metadata in zip(profilers, profiler_saver.metadata):
        logger.info(f"Profile {profiler.metric_id} saved with {len(saver_metadata['profiler_ids'])} files")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * OH * OW * K * C * FH * FW
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.nn.functional.conv2d(
        input=ifm.permute(0, 3, 1, 2).to(acc_dtype).contiguous(), 
        weight=wgt.permute(2, 3, 0, 1).to(acc_dtype).contiguous(), 
        bias=bias.to(acc_dtype), 
        stride=STRIDE, 
        padding=PADDING, 
        dilation=DILATION
    ).permute(0, 2, 3, 1)
    
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    if args.report_mismatch:
        if not torch.equal(simulated, reference):
            mismatch_report = os.path.join(SUMMARY_DIR, "conv_mismatch_report.txt")
            with open(mismatch_report, "w") as f:
                content = []
                for n in range(N):
                    for oh in range(OH):
                        for ow in range(OW):
                            for k in range(K):
                                sim_val = simulated[n, oh, ow, k].item()
                                ref_val = reference[n, oh, ow, k].item()
                                if sim_val != ref_val:
                                    content.append(f"Mismatch at position ({n}, {oh}, {ow}, {k}): simulated={sim_val}, reference={ref_val}, difference={abs(sim_val - ref_val)}\n")
                f.writelines(content)
            logger.error(f"Mismatch report saved to '{mismatch_report}'.")
            logger.error(f"Total mismatches: {len(content)}/{simulated.numel()}")
