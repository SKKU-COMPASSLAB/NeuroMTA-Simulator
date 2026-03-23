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
    parser = argparse.ArgumentParser(description="Test simple mapping of a linear operator on Tenstorrent hardware.")
    parser.add_argument('-m', default=512, type=int, help="M dimension size", dest="m")
    parser.add_argument('-n', default=512, type=int, help="N dimension size", dest="n")
    parser.add_argument('-k', default=256, type=int, help="K dimension size", dest="k")
    parser.add_argument('--l1-buf-size', default=parse_mem_cap_str("128KB"), type=int, help="L1 buffer size per core", dest="l1_buf_size")
    parser.add_argument('--use-l1-cache', action="store_true", help="Whether to load input tensors from L1 buffer (instead of main memory)", dest="use_l1_cache")
    parser.add_argument('--use-bcast', action="store_true", help="Whether to use broadcast", dest="use_bcast")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation.", dest="monitor")
    parser.add_argument('-o', '--output', default=SUMMARY_DIR, type=str, help="Directory to save the mapping summary and profiler report.", dest="output_dir")
    args = parser.parse_args()
    
    SUMMARY_DIR = args.output_dir
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    M, N, K = args.m, args.n, args.k
    
    x = torch.randint(-32, 32, (M, K), dtype=torch.int16)
    w = torch.randint(-32, 32, (N, K), dtype=torch.int16)
    b = torch.randint(-32, 32, (N,),   dtype=torch.int16)
    y = torch.nn.functional.linear(x, w, bias=b)
    
    # Create a Tenstorrent device
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    device.initialize()
    
    device.set_command_debug_verbosity(False)
    logger.set_print_options(log_level="DEBUG")
    
    def kernel_debug_hook(core: Core, kernel: Kernel):
        logger.debug(f"core id: {core.core_id}, kernel id: {kernel.kernel_id}, issue_time: {kernel.issue_time}, commit_time: {kernel.commit_time}")
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    # Create memory space and buffers
    main_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    l1_mem_space = device.create_l1_mem_space(parse_mem_cap_str("1.5MB") - args.l1_buf_size, core_group)
    
    bufs = [
        MCA_TensorBuffer(l1_mem_space if args.use_l1_cache else main_mem_space, x.shape, x.dtype, shard_shape=(32, 32)).tiling((32, 32)).allocate().update(x),
        MCA_TensorBuffer(l1_mem_space if args.use_l1_cache else main_mem_space, w.shape, w.dtype, shard_shape=(32, 32)).tiling((32, 32)).allocate().update(w),
        MCA_TensorBuffer(l1_mem_space if args.use_l1_cache else main_mem_space, b.shape, b.dtype, shard_shape=(1,  32)).tiling((1,  32)).allocate().update(b),
        MCA_TensorBuffer(l1_mem_space if args.use_l1_cache else main_mem_space, y.shape, y.dtype, shard_shape=(32, 32)).tiling((32, 32)).allocate()
    ]
    
    # Create operator signature
    op = MCA_OP_LINEAR(*bufs).initialize_core_group(core_group)
    
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
        with MonitoringWindow(device, core_group, profilers) as monitor:
            st = time.time()
            device.run_kernels()
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels()
        ed = time.time()
        
    profiler_saver.close()
    
    for profiler, saver_metadata in zip(profilers, profiler_saver.metadata):
        logger.info(f"Profile {profiler.metric_id} saved with {len(saver_metadata['profiler_ids'])} files")
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    simulated = bufs[-1].restore()
    reference = y
    
    # print(f"simulated:\n{simulated}")
    # print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    # profiler.print_report()