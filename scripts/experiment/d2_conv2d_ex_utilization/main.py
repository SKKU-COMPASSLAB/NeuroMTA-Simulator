import os
import json
import time
import torch
import argparse

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import *


FILEROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
LOGDIR = os.path.join(FILEROOT, ".logs")
SUMMARY_DIR = os.path.join(LOGDIR, FILENAME)

os.makedirs(LOGDIR, exist_ok=True)
# os.makedirs(SUMMARY_DIR, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test simple mapping of a Conv2d operator on Tenstorrent hardware.")
    parser.add_argument("--batch", default=1, type=int, help="Batch size", dest="batch")
    parser.add_argument("--in-channels", default=3, type=int, help="Input channels", dest="in_channels")
    parser.add_argument("--out-channels", default=64, type=int, help="Output channels", dest="out_channels")
    parser.add_argument("--input-height", default=224, type=int, help="Input height", dest="input_height")
    parser.add_argument("--input-width", default=224, type=int, help="Input width", dest="input_width")
    parser.add_argument("--kernel-height", default=11, type=int, help="Kernel height", dest="kernel_height")
    parser.add_argument("--kernel-width", default=11, type=int, help="Kernel width", dest="kernel_width")
    parser.add_argument("--stride-height", default=4, type=int, help="Stride height", dest="stride_height")
    parser.add_argument("--stride-width", default=4, type=int, help="Stride width", dest="stride_width")
    parser.add_argument("--padding-height", default=2, type=int, help="Padding height", dest="padding_height")
    parser.add_argument("--padding-width", default=2, type=int, help="Padding width", dest="padding_width")
    parser.add_argument("--dilation-height", default=1, type=int, help="Dilation height", dest="dilation_height")
    parser.add_argument("--dilation-width", default=1, type=int, help="Dilation width", dest="dilation_width")
    parser.add_argument("--groups", default=1, type=int, help="Conv2d groups", dest="groups")
    parser.add_argument('--l1-buf-size', default=parse_mem_cap_str("128KB"), type=int, help="L1 buffer size per core", dest="l1_buf_size")
    parser.add_argument('--use-l1-cache', action="store_true", help="Whether to load input tensors from L1 buffer (instead of main memory)", dest="use_l1_cache")
    parser.add_argument('--use-bcast', action="store_true", help="Whether to use broadcast", dest="use_bcast")
    parser.add_argument('-o', '--output', default=SUMMARY_DIR, type=str, help="Directory to save the mapping summary and profiler report.", dest="output_dir")
    args = parser.parse_args()
    
    SUMMARY_DIR = args.output_dir
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    N = args.batch
    C = args.in_channels
    K = args.out_channels
    H = args.input_height
    W = args.input_width
    FH = args.kernel_height
    FW = args.kernel_width
    STRIDE = (args.stride_height, args.stride_width)
    PADDING = (args.padding_height, args.padding_width)
    DILATION = (args.dilation_height, args.dilation_width)
    GROUPS = args.groups

    if C % GROUPS != 0:
        raise ValueError(f"in_channels ({C}) must be divisible by groups ({GROUPS})")
    if K % GROUPS != 0:
        raise ValueError(f"out_channels ({K}) must be divisible by groups ({GROUPS})")

    OH = (H + 2 * PADDING[0] - DILATION[0] * (FH - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (FW - 1) - 1) // STRIDE[1] + 1

    dtype = torch.int16
    acc_dtype = torch.int16

    # NeuroMTA Conv2d uses NHWC input and OHWI weight layout.
    x = torch.randint(-32, 32, (N, H, W, C), dtype=dtype)
    w = torch.randint(-32, 32, (FH, FW, K, C // GROUPS), dtype=dtype)
    b = torch.randint(-32, 32, (K,), dtype=acc_dtype)
    y = torch.nn.functional.conv2d(
        input=x.permute(0, 3, 1, 2).to(acc_dtype).contiguous(),
        weight=w.permute(2, 3, 0, 1).to(acc_dtype).contiguous(),
        bias=b.to(acc_dtype),
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        groups=GROUPS,
    ).permute(0, 2, 3, 1)
    
    # Create a Tenstorrent device
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    device.initialize()
    
    device.set_command_debug_verbosity(True)
    logger.set_print_options(log_level="DEBUG")
    
    def kernel_debug_hook(core: Core, kernel: Kernel):
        logger.debug(f"core id: {core.core_id}, kernel id: {kernel.kernel_id}, issue_time: {kernel.issue_time}, commit_time: {kernel.commit_time}")
    
    core_group = device.get_npu_core_group((0, 0), (12, 12))
    profiler = ExecutionTimeProfiler(device, core_group, ["LD", "EX", "ST"])
    
    # Create memory space and buffers
    main_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    l1_mem_space = device.create_l1_mem_space(parse_mem_cap_str("1.5MB") - args.l1_buf_size, core_group)
    
    wt = 56 if (W % 56 == 0) else W
    owt = 55 if (OW % 55 == 0) else OW
    ct = C // GROUPS
    kt = 32 if (K % 32 == 0) else K

    bufs = [
        MCA_TensorBuffer(
            l1_mem_space if args.use_l1_cache else main_mem_space,
            x.shape,
            x.dtype,
            shard_shape=(wt, ct),
        ).tiling((32, 32)).allocate().update(x),
        MCA_TensorBuffer(
            l1_mem_space if args.use_l1_cache else main_mem_space,
            w.shape,
            w.dtype,
            shard_shape=(kt, ct),
        ).tiling((32, 32)).allocate().update(w),
        MCA_TensorBuffer(
            l1_mem_space if args.use_l1_cache else main_mem_space,
            b.shape,
            b.dtype,
            shard_shape=(1, kt),
        ).tiling((1, 32)).allocate().update(b),
        MCA_TensorBuffer(
            l1_mem_space if args.use_l1_cache else main_mem_space,
            y.shape,
            y.dtype,
            shard_shape=(owt, kt),
        ).tiling((32, 32)).allocate(),
    ]
    
    # Create operator signature
    op = MCA_OP_CONV2D(
        *bufs,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        groups=GROUPS,
    ).initialize_core_group(core_group)
    
    # Compile operator
    compiler_recipe = MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=args.l1_buf_size,
        broadcast_optimize=args.use_bcast,
    )
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(op)
    compiled_ops = compiler.compile(compiler_recipe).dispatch()
    
    # Execute operator
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
    
    with MonitoringWindow() as monitor:
        for core_id in core_group.core_ids:
            core = device.get_npu_core(core_id=core_id)
            pbar_idx = monitor.add_core_pbar(desc=f"{core_id:<3d}", ncols=40)
            monitor.pbar_handles[pbar_idx].bind_core(core)
        
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    for core_id, core_summary in profiler.summary().items():
        for slot_id, slot_summary in core_summary.items():
            logger.info(f"Core {core_id} Slot {slot_id}: Active Time = {slot_summary['active_time_cycles']} cycles out of {slot_summary['final_commit_cycles']} cycles total ({slot_summary['active_utilization']*100:.2f}% active)")
    
    profiler_report_path = os.path.join(SUMMARY_DIR, "execution_time_profile.json")
    with open(profiler_report_path, "w") as f:
        json.dump(profiler.summary(), f, indent=4)
        logger.info(f"Execution time profile saved to '{profiler_report_path}'.")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    simulated = bufs[-1].restore()
    reference = y
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    # profiler.print_report()
    
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