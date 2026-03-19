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
    parser = argparse.ArgumentParser(description="Validate OP3 Conv2D operator on Tenstorrent hardware.")
    parser.add_argument('--no-bcast', action="store_true", help="Whether not to use broadcast", dest="no_bcast")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
    parser.add_argument('--debug-command', action="store_true", help="Whether to enable command-level debugging", dest="debug_command")
    parser.add_argument('--report-mismatch', action="store_true", help="Whether to generate mismatch report when validation fails", dest="report_mismatch")
    args = parser.parse_args()

    torch.set_printoptions(profile="full", linewidth=2048)
    # torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG if args.debug_command else LogLevel.INFO)
    torch.manual_seed(0)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.debug_command)
    
    core_group = MCA_CoreGroup.merge_core_groups(device.get_npu_core_group((0, 0), (12, 12)).split(shape=(4, 4)))
    
    N, H, W, C = 1, 224, 224, 3
    FH, FW, K = 11, 11, 96
    STRIDE, PADDING, DILATION = (4, 4), (2, 2), (1, 1)
    OH = (H + 2 * PADDING[0] - DILATION[0] * (FH - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (FW - 1) - 1) // STRIDE[1] + 1
    
    Wt = 56
    OWt = 55
    Ct = 10
    Kt = 32
    
    if (W % Wt != 0): Wt = W
    if (OW % OWt != 0): OWt = OW
    if C % Ct != 0: Ct = C
    if K % Kt != 0: Kt = K
    
    dtype = torch.int16
    acc_dtype = torch.int16
    broadcast_optimize = not args.no_bcast  # Enable broadcast optimization to reduce memory and NoC traffic
    
    ifm  = torch.randint(low=0, high=64, size=(N, H, W, C), dtype=dtype)
    wgt  = torch.randint(low=0, high=64, size=(FH, FW, K, C), dtype=dtype)
    bias = torch.randint(low=0, high=64, size=(K,), dtype=acc_dtype)
    ofm  = torch.zeros((N, OH, OW, K), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(Wt,  Ct)).tiling((32, 32)).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(Kt,  Ct)).tiling((32, 32)).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,   Kt)).tiling((1,  32)).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(OWt, Kt)).tiling((32, 32)).allocate()
    
    operator = MCA_OP_CONV2D(
        ifm_b, wgt_b, bias_b, ofm_b, 
        stride=STRIDE, padding=PADDING, dilation=DILATION,
    ).initialize_core_group(core_group)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=parse_mem_cap_str("128KB"),
        broadcast_optimize=broadcast_optimize,
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
    
    profilers = [
        DRAMBandwidthProfiler(device, record_type="BOTH"),
        InterconnectBandwidthProfiler(device),
        ThreadUtilizationProfiler(device, core_group, slot_id="LD"),
        ThreadUtilizationProfiler(device, core_group, slot_id="EX"),
        ThreadUtilizationProfiler(device, core_group, slot_id="ST"),
    ]
    
    if args.monitor:
        with MonitoringWindow(device, core_group, profilers) as monitor:
            st = time.time()
            device.run_kernels()
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
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
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    if args.report_mismatch:
        if not torch.equal(simulated, reference):
            mismatch_report = os.path.join(SUMMARY_DIR, "conv_mismatch_report.txt")
            with open(mismatch_report, "w") as f:
                content = []
                s = simulated.flatten()
                r = reference.flatten()
                for i in range(s.shape[0]):
                    sim_val = s[i].item()
                    ref_val = r[i].item()
                    if sim_val != ref_val:
                        content.append(f"Mismatch at position ({i}): simulated={sim_val}, reference={ref_val}\n")
                f.writelines(content)
            logger.error(f"Mismatch report saved to '{mismatch_report}'.")
            logger.error(f"Total mismatches: {len(content)}/{s.numel()}")
