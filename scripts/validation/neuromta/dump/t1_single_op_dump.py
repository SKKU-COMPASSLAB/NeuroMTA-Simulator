# import faulthandler
# import gc

# faulthandler.enable()

import os
import json
import time
import torch
import argparse
import random
import numpy as np

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *
from neuromta.system.software.tenstorrent import *


FILEROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
LOGDIR = os.path.join(FILEROOT, ".logs")
SUMMARY_DIR = os.path.join(LOGDIR, FILENAME)
DUMP_DIR = os.path.join(SUMMARY_DIR, "dumps")

os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN deterministic config (performance issues may arise)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate OP1 Linear operator on Tenstorrent hardware.")
    parser.add_argument('--no-bcast', action="store_true", help="Whether not to use broadcast", dest="no_bcast")
    parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
    parser.add_argument('--debug-command', action="store_true", help="Whether to enable command-level debugging", dest="debug_command")
    parser.add_argument('--create-dump', action="store_true", help="Whether to create dump file for the operator execution", dest="create_dump")
    args = parser.parse_args()
    
    seed_everything(seed=42)
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG if args.debug_command else LogLevel.INFO)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=args.debug_command)

    core_group = device.get_npu_core_group((0, 0), (4, 4))
    
    dump_saver = DumpSaver(device)
    
    M, N, K = 256, 256, 256
    dtype = torch.int16
    acc_dtype = torch.int16
    blocked_mapping = True  # Enable blocked mapping for better data locality
    broadcast_optimize = not args.no_bcast  # Enable broadcast optimization to reduce memory and NoC traffic
    
    ifm  = torch.randint(low=0, high=128, size=(M, K), dtype=dtype)
    wgt  = torch.randint(low=0, high=128, size=(N, K), dtype=dtype)
    bias = torch.randint(low=0, high=256, size=(N,), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    bias_size = bias.numel() * bias.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_data_mem_space   = device.create_l1_mem_space(parse_mem_cap_str("1MB"), core_group=device.get_npu_core_group()).override(core_group)
    main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).tiling((32, 32)).allocate().update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(32, 32), blocked_mapping=False          ).tiling((32, 32)).allocate().update(wgt)
    bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=bias.shape, dtype=bias.dtype, shard_shape=(1,  32), blocked_mapping=False          ).tiling((1,  32)).allocate().update(bias)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(32, 32), blocked_mapping=blocked_mapping).tiling((32, 32)).allocate()
    
    if args.create_dump:
        operator = MCA_OP_LINEAR(
            ifm_b, wgt_b, bias_b, ofm_b, 
        ).initialize_core_group(core_group)
        
        compiler = MCA_OperatorGraphCompiler()
        compiler.add_op(operator)
        
        global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
            device=device,
            spad_space_size_per_core=parse_mem_cap_str("512KB"),
            broadcast_optimize=broadcast_optimize,
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
                
        dump_saver.dump_to_file(DUMP_DIR, f"{FILENAME}_dump")
    else:
        dump_saver.load_from_file(DUMP_DIR, f"{FILENAME}_dump")
        
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
        
        simulated = ofm_b.restore()
        reference = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias
        
        print(f"simulated:\n{simulated}")
        print(f"reference:\n{reference}")
        print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
        
        # print("--- 메인 로직 종료. 강제 GC 시작 ---")
        # # 2. 종료(Exit) 단계로 가기 전에 런타임에서 강제로 GC를 돌려 범인 색출
        # gc.set_debug(gc.DEBUG_LEAK) # 메모리 릭 및 해제 로그 출력
        # gc.collect() 
        # print("--- 강제 GC 완료 (여기까지 오면 GC 문제가 아님) ---")
    