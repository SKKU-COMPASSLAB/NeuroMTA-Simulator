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
    parser.add_argument('--report-icnt-stats', action="store_true", help="Whether to report interconnect statistics after simulation", dest="report_icnt_stats")
    parser.add_argument('--report-dram-stats', action="store_true", help="Whether to report DRAM statistics after simulation", dest="report_dram_stats")
    parser.add_argument('--bcast-queue-depth', type=int, default=16, help="The depth of the broadcast queue", dest="bcast_queue_depth")
    parser.add_argument('--pipeline-gran', type=int, default=32, help="The number of micro-operations per pipeline stage", dest="pipeline_gran")
    parser.add_argument('--spad-size', type=str, default="512KB", help="The size of the scratchpad memory per core (e.g., '256KB')", dest="spad_size")
    parser.add_argument('--max-timestamp', type=int, default=-1, help="Maximum timestamp to run the simulation", dest="max_timestamp")
    args = parser.parse_args()
    
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.debug_command)
    
    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    M, N, K = 1, 32000, 4096
    dtype = torch.int16
    acc_dtype = torch.int16
    
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("30GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=(M, K), dtype=torch.bfloat16, shard_shape=(1,  32)).tiling((32, 32)).allocate()
    wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=(N, K), dtype=torch.bfloat16, shard_shape=(32, 32)).tiling((32, 32)).allocate()
    bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=(1, N), dtype=torch.bfloat16, shard_shape=(1,  32)).tiling((1,  32)).allocate()
    ofm_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=(M, N), dtype=torch.bfloat16, shard_shape=(1,  32)).tiling((32, 32)).allocate()
    
    operator = MCA_OP_LINEAR(ifm_b, wgt_b, bias_b, ofm_b)
    
    compiler = MCA_OperatorGraphCompiler()
    compiler.add_op(operator)
    
    global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
        device=device,
        core_groups=[core_group],
        spad_space_size_per_core=parse_mem_cap_str(args.spad_size),
        pipeline_granularity=args.pipeline_gran,
        broadcast_optimize_queue_depth=args.bcast_queue_depth,
    )
    
    compiled_ops = compiler.compile(global_recipe)
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()
    
    compiled_ops.dispatch()
    
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
            device.run_kernels(max_timestamp=args.max_timestamp)
            ed = time.time()
    else:
        st = time.time()
        device.run_kernels(max_timestamp=args.max_timestamp)
        ed = time.time()
        
    profiler_saver.close()
    
    for profiler, saver_metadata in zip(profilers, profiler_saver.metadata):
        logger.info(f"Profile {profiler.metric_id} saved with {len(saver_metadata['profiler_ids'])} files")
        
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    total_ops = 2 * M * N * K
    throughput = (total_ops / device.timestamp)
    print(f"overall throughput: {throughput:.2f} OP/cycle")
    
    simulated = ofm_b.restore()
    
    print(f"simulated:\n{simulated}")
        
    if args.report_icnt_stats:
        import pprint
        from neuromta.component.context.global_context import BOOKSIM_MODULE_ID
        from neuromta.component.companions.booksim import BookSim2
        
        booksim2_module: BookSim2 = device.companion_core._companion_modules[BOOKSIM_MODULE_ID]
        pprint.pprint(booksim2_module.get_stats(), indent=4)

    if args.report_dram_stats:
        import pprint
        from neuromta.component.context.global_context import DRAMSIM_MODULE_ID
        from neuromta.component.companions.dramsim import DRAMSim3
        
        dram_module: DRAMSim3 = device.companion_core._companion_modules[DRAMSIM_MODULE_ID]
        pprint.pprint(dram_module.get_stats(), indent=4)