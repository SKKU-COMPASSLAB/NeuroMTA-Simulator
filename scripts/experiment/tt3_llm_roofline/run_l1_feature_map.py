import os
import json
import argparse
import multiprocessing as mp
from neuromta.component.implementation import operator
import torch
import math

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *
from neuromta.system.software.tenstorrent import *


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser(description="Tenstorrent Device Benchmark Suite")
parser.add_argument("-o", "--output", type=str, default=f"{FILE_NAME}.csv", help="Output file to save benchmark results")
parser.add_argument("-n", "--n-workers", type=int, default=mp.cpu_count(), help="Number of parallel worker processes")
parser.add_argument('--monitor', action="store_true", help="Whether to show real-time monitoring window during simulation", dest="monitor")
parser.add_argument('--skip-execution', action="store_true", help="Whether to skip kernel execution and only perform compilation and profiling setup", dest="skip_execution")
args = parser.parse_args()

OUTPUT_DIR = os.path.join(ROOT_DIR, ".logs")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, FILE_NAME)
output_path = os.path.join(OUTPUT_DIR, args.output)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def _find_smallest_divisor_above(num: int, threshold: int) -> int:
    for i in range(threshold, num + 1):
        if num % i == 0:
            return i
    return num

def _get_shard_shape_from_tensor_shape(tensor_shape: tuple[int]) -> tuple[int]:
    hh = tensor_shape[-2]
    ww = tensor_shape[-1]
    
    return _find_smallest_divisor_above(hh, 32), _find_smallest_divisor_above(ww, 32)


class Benchmark:
    def __init__(
        self, 
        signature: str,
        M:  int, N:  int, K:  int,
        dtype:     torch.dtype,
        acc_dtype: torch.dtype,
    ):
        self.signature: str = signature
        self.M: int = M
        self.N: int = N
        self.K: int = K
        
        self.dtype:     torch.dtype = dtype
        self.acc_dtype: torch.dtype = acc_dtype
        
        self._timestamp:    int = 0
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        self._total_ops:    int = (self.M * self.N * self.K) * 2 + (self.M * self.N)  # MACs + Bias Add
        
    @property
    def ifm_shape(self) -> tuple[int]:
        return (self.M, self.K)
    
    @property
    def wgt_shape(self) -> tuple[int]:
        return (self.N, self.K)
    @property
    def bias_shape(self) -> tuple[int]:
        return (1, self.N)
    
    @property
    def ofm_shape(self) -> tuple[int]:
        return (self.M, self.N)
    
    @property
    def ifm_shard_shape(self) -> tuple[int]:
        return _get_shard_shape_from_tensor_shape(self.ifm_shape)
    
    @property
    def wgt_shard_shape(self) -> tuple[int]:
        return _get_shard_shape_from_tensor_shape(self.wgt_shape)
    
    @property
    def bias_shard_shape(self) -> tuple[int]:
        return _get_shard_shape_from_tensor_shape(self.bias_shape)
    
    @property
    def ofm_shard_shape(self) -> tuple[int]:
        return _get_shard_shape_from_tensor_shape(self.ofm_shape)
    
    @property
    def ifm_total_size(self) -> int:
        return self.M * self.K * self.dtype.itemsize
    
    @property
    def wgt_total_size(self) -> int:
        return self.N * self.K * self.dtype.itemsize
    
    @property
    def bias_total_size(self) -> int:
        return self.N * self.dtype.itemsize
    
    @property
    def ofm_total_size(self) -> int:
        return self.M * self.N * self.acc_dtype.itemsize
        
    def run(self, device: TenstorrentDevice, core_group: MTA_CoreGrid):
        _l1_total_per_core     = parse_mem_cap_str("1.4MB")  # total L1 memory size in Tenstorrent Tensix Core is 1.5MB
        _spad_size_per_core    = parse_mem_cap_str("512KB")
        _l1_data_size_per_core = _l1_total_per_core - _spad_size_per_core
        
        logger.info(f"benchmark memory map per core {self.signature}: Data: {_l1_data_size_per_core / 1024:.2f} KB, SPAD: {_spad_size_per_core / 1024:.2f} KB")
        
        try:
            l1_data_mem_space = device.create_l1_mem_space(_l1_data_size_per_core, core_group=core_group)
            main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("32GB"))
            
            ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=self.ifm_shape,  dtype=self.dtype,     shard_shape=self.ifm_shard_shape).tiling((32, 32)).allocate()
            wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=self.wgt_shape,  dtype=self.dtype,     shard_shape=self.wgt_shard_shape).tiling((32, 32)).allocate()
            bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=self.bias_shape, dtype=self.dtype,     shard_shape=self.bias_shard_shape).tiling((1,  32)).allocate()
            ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=self.ofm_shape,  dtype=self.acc_dtype, shard_shape=self.ofm_shard_shape).tiling((32, 32)).allocate()
            
            self._l1_traffic:   int = 0
            self._main_traffic: int = 0
            
            for b in [ifm_b, wgt_b, bias_b, ofm_b]:
                if b.mem_space.mem_type == GlobalContextMemType.L1:
                    self._l1_traffic += b.total_size
                else:
                    self._main_traffic += b.total_size
                    
            op = MCA_OP_LINEAR( 
                ifm_b, wgt_b, bias_b, ofm_b, 
            )
            
            compiler = MCA_OperatorGraphCompiler()
            compiler.add_op(op)
            
            global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
                device=device,
                core_groups=[core_group],
                spad_space_size_per_core=_spad_size_per_core,
                pipeline_granularity=32,
                broadcast_optimize_queue_depth=16,
            )
            
            compiled_ops = compiler.compile(global_recipe)
        
            device.remove_all_l1_mem_space()
            device.remove_all_main_mem_space()
            
            compiled_ops.dispatch()
            
            benchmark_summary_dir = os.path.join(SUMMARY_DIR, self.signature)
            compilation_summary_dir = os.path.join(benchmark_summary_dir, "summaries")
            profiler_summary_dir = os.path.join(benchmark_summary_dir, "profiles")
            os.makedirs(benchmark_summary_dir, exist_ok=True)
            os.makedirs(compilation_summary_dir, exist_ok=True)
            os.makedirs(profiler_summary_dir, exist_ok=True)
            
            for op_id, summary in compiled_ops.summary().items():
                tmp_output_path = os.path.join(compilation_summary_dir, f"op_summary_{op_id}.json")
                with open(tmp_output_path, "w") as f:
                    json.dump(summary, f, indent=4)
                    logger.info(f"Mapping summary saved to '{tmp_output_path}'.")
                
            if args.skip_execution:
                import pandas as pd
                with open(output_path, "r") as f:
                    df = pd.read_csv(f)
                    existing_timestamp = df[df["Benchmark"] == self.signature]["Timestamp (cycles)"].values
                    if len(existing_timestamp) > 0:
                        self._timestamp = int(existing_timestamp[0])
                    else:
                        logger.warning(f"No existing timestamp found for benchmark {self.signature} in backup logs. Setting timestamp to 0.")
                        self._timestamp = 0             
            else:
                profilers = [
                    DRAMBandwidthProfiler(device, record_type="BOTH"),
                    InterconnectBandwidthProfiler(device),
                    ThreadUtilizationProfiler(device, core_group, slot_id="LD"),
                    ThreadUtilizationProfiler(device, core_group, slot_id="EX"),
                    ThreadUtilizationProfiler(device, core_group, slot_id="ST"),
                ]
                
                profiler_saver = ProfilerFileSaverHub(output_dir=profiler_summary_dir)
                profiler_saver.add_profilers(*profilers)
            
                if args.monitor:
                    with MonitoringWindow(device, core_group, profilers, sim_name=self.signature) as monitor:
                        device.run_kernels()
                else:
                    device.run_kernels()
                    
                self._timestamp = device.timestamp
                
                profiler_saver.close()
                device.reset_simulation()
                
        except Exception as e:
            logger.error(f"Error during benchmark {self.signature}: {e}")
            return False
        
        return True  # Indicate successful run without L1 memory overflow
        
    @property
    def timestamp(self) -> int:
        return self._timestamp
    
    @property
    def total_ops(self) -> int:
        return self._total_ops
    
    @property
    def l1_traffic(self) -> int:
        return self._l1_traffic
    
    @property
    def main_traffic(self) -> int:
        return self._main_traffic
    
    
class BenchmarkProcess(mp.Process):
    def __init__(self, benchmark: Benchmark, device_config: TenstorrentConfig, core_group_offset: tuple[int, int], core_group_shape: tuple[int, int], return_dict: dict, worker_sem):
        super().__init__()
        self.benchmark = benchmark
        self.device_config = device_config
        self.core_group_offset = core_group_offset
        self.core_group_shape = core_group_shape
        self.return_dict = return_dict
        self.worker_sem = worker_sem
        
    def run(self):
        self.worker_sem.acquire()
        logger.info(f"process started for {self.benchmark.signature}")

        device = TenstorrentDevice(**self.device_config)
        device.initialize()
        device.set_command_debug_verbosity(verbose=False)
        
        core_group = device.get_npu_core_group(self.core_group_offset, self.core_group_shape)
        
        flag = self.benchmark.run(device, core_group)
        if not flag:
            return
        
        self.return_dict[self.benchmark.signature] = {
            "timestamp":    self.benchmark.timestamp,
            "total_ops":    self.benchmark.total_ops,
            "l1_traffic":   self.benchmark.l1_traffic,
            "main_traffic": self.benchmark.main_traffic,
        }
        
        self.worker_sem.release()
        logger.info(f"process finished for {self.benchmark.signature}")


benchmarks = [
    Benchmark("qkv_proj (prefill)",  M=2048, N=4096,  K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("up_proj (prefill)",   M=2048, N=11008, K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("down_proj (prefill)", M=2048, N=4096,  K=11008, dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("lm_head (prefill)",   M=2048, N=32000, K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    
    Benchmark("qkv_proj (decode)",   M=1,    N=4096,  K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("up_proj (decode)",    M=1,    N=11008, K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("down_proj (decode)",  M=1,    N=4096,  K=11008, dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
    Benchmark("lm_head (decode)",    M=1,    N=32000, K=4096,  dtype=torch.bfloat16, acc_dtype=torch.bfloat16),
]

if __name__ == "__main__":
    try:
        import os
        import sys
        
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        
        import visualize
    except ImportError as e:
        logger.error("Error importing visualize module:", e)
        visualize = None
    
    manager = mp.Manager()
    return_dict = manager.dict()
    config = TenstorrentConfig.BLACKHOLE()
    
    n_workers = min(args.n_workers, len(benchmarks))
    worker_sem = mp.Semaphore(n_workers)
    
    processes: list[BenchmarkProcess] = []
    for benchmark in benchmarks:
        p = BenchmarkProcess(benchmark, config, (0, 0), (12, 14), return_dict, worker_sem)
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    
    with open(output_path, "w") as f:
        f.write("Benchmark,Timestamp (cycles),Total OPs,L1 Memory Traffic (Bytes),Main Memory Traffic (Bytes),Performance (OPs/cycle),Arithmetic Intensity (OPs/Byte),L1 Bandwidth (Byte/cycle),Main Bandwidth (Byte/cycle),Total Bandwidth (Byte/cycle)\n")
        for benchmark in benchmarks:
            if benchmark.signature not in return_dict:
                logger.error(f"Missing results for benchmark {benchmark.signature}")
                continue
            
            result = return_dict[benchmark.signature]
            
            timestamp    = result["timestamp"]
            total_ops    = result["total_ops"]
            l1_traffic   = result["l1_traffic"]
            main_traffic = result["main_traffic"]
            
            ops_per_cycle   = total_ops / timestamp
            l1_bandwidth    = l1_traffic / timestamp
            main_bandwidth  = main_traffic / timestamp
            total_bandwidth = l1_bandwidth + main_bandwidth
            arith_intensity = (total_ops / (main_traffic + l1_traffic)) if (main_traffic + l1_traffic) != 0 else 0
            
            f.write(f"{benchmark.signature},{timestamp},{total_ops},{l1_traffic},{main_traffic},{ops_per_cycle:.2f},{arith_intensity:.2f},{l1_bandwidth:.2f},{main_bandwidth:.2f},{total_bandwidth:.2f}\n")
    
    print(f"Benchmark results saved to '{output_path}'.")
    
    if visualize is not None:
        global_context_config: GlobalContextConfig = config["global_config"]
        icnt_config: IcntConfig = config["icnt_config"]
        mxu_config: MXUConfig = config["mxu_config"]
        dramsim_config = global_context_config.main_mem_config.dramsim3_config
        booksim_config = icnt_config.booksim2_config
        img_path = os.path.join(OUTPUT_DIR, f"{FILE_NAME}.png")
        
        mem_peak_bw = dramsim_config.peak_bandwidth() / 1e9  # in GB/s
        noc_bisection_bw = booksim_config.peak_bandwidth_per_router() * config["processor_clock_freq"] / 1e9
        
        print(f"=== DRAMSim3 Configuration ===")
        print(f"peak bandwidth: {mem_peak_bw:.2f} GB/s")
        print(f"number of channels per instance: {dramsim_config.n_cmd_q_per_instance}")
        print(f"number of instances: {dramsim_config.n_instance}")
        
        print(f"=== BookSim2 Configuration ===")
        print(f"bisection bandwidth: {noc_bisection_bw:.2f} GB/s")
        print(f"number of subnets: {booksim_config._subnets}")
        print(f"flit size: {booksim_config._flit_size} Bytes")
        
        visualize.draw(
            peak_perf = 8 * 8 * mxu_config.peak_op_per_cycle,  # 4x4 PE array
            peak_mem_bw = mem_peak_bw,
            peak_noc_bw = noc_bisection_bw,
            src_path=output_path,
            img_path=img_path,
            img_title=f"{type(config).__name__} Roofline Analysis - {FILE_NAME}"
        )
        
        print(f"Roofline visualization saved to '{img_path}'.")