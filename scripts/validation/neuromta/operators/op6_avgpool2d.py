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
    parser.add_argument('--no-bcast', action="store_true", help="Whether not to use broadcast", dest="no_bcast")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
    parser.add_argument('--report-mismatch', action="store_true", help="Whether to generate mismatch report when validation fails", dest="report_mismatch")
    args = parser.parse_args()

    # torch.set_printoptions(linewidth=1024, threshold=10000)
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG if args.monitor else LogLevel.INFO)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.monitor)
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    N, H, W, C = 1, 60, 60, 64
    WINDOW = (2, 2)
    STRIDE, PADDING, DILATION = WINDOW, (0, 0), (1, 1)
    OH = (H + 2 * PADDING[0] - DILATION[0] * (WINDOW[0] - 1) - 1) // STRIDE[0] + 1
    OW = (W + 2 * PADDING[1] - DILATION[1] * (WINDOW[1] - 1) - 1) // STRIDE[1] + 1
    
    Ws = 32
    Cs = 32
    
    dtype = torch.bfloat16
    acc_dtype = torch.bfloat16
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = not args.no_bcast  # Enable broadcast optimization to reduce memory and NoC traffic
    sim_mode = "partial_l1"
    
    ifm  = torch.randint(low=0, high=64, size=(N, H, W, C), dtype=dtype)
    ofm  = torch.zeros((N, OH, OW, C), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(W,  Cs)).tiling((32, 32)).allocate().update(ifm)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(OW, Cs)).tiling((32, 32)).allocate()
    
    operator = MCA_OP_AVGPOOL2D(
        ifm_b, ofm_b, 
        window=WINDOW, stride=STRIDE, padding=PADDING, dilation=DILATION,
        use_collective_tile_load=False,
    ).initialize_core_group(core_group)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        spad_space_size_per_core=parse_mem_cap_str("512KB"),
        broadcast_optimize=broadcast_optimize,
    )
    
    compiled_ops = compiler.compile(global_recipe).dispatch()
    
    for op_id, summary in compiled_ops.summary().items():
        tmp_output_path = os.path.join(SUMMARY_DIR, f"op_summary_{op_id}.json")
        with open(tmp_output_path, "w") as f:
            json.dump(summary, f, indent=4)
            logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
        
    if args.monitor:
        with MonitoringWindow() as monitor:
            for core_id in core_group.core_ids:
                core = device.get_npu_core(core_id=core_id)
                pbar_idx = monitor.add_core_pbar(desc=f"{core_id:<3d}", ncols=40)
                monitor.pbar_handles[pbar_idx].bind_core(core)

            st = time.time()
            device.run_kernels()
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels()
        ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * OH * OW * C * WINDOW[0] * WINDOW[1]
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    reference = torch.nn.functional.avg_pool2d(
        input=ifm.permute(0, 3, 1, 2).to(acc_dtype).contiguous(), 
        kernel_size=WINDOW,
        stride=STRIDE, 
        padding=PADDING, 
        # dilation=DILATION  # torch.nn.functional.avg_pool2d does not support dilation
    ).permute(0, 2, 3, 1)
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    total_elements = ofm.numel()
    num_mismatches = (simulated != reference).sum().item()
    print(f"total elements: {total_elements}, mismatches: {num_mismatches}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")

    if args.report_mismatch:
        if not torch.equal(simulated, reference):
            mismatch_report = os.path.join(SUMMARY_DIR, "avgpool_mismatch_report.txt")
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
