import os
import abc
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


class Benchmark(abc.ABC):
    @property
    def signature(self) -> str:
        return f"UnknownBenchmark({id(self):016x})"
    
    @abc.abstractmethod
    def run(self, device: TenstorrentDevice) -> bool:
        pass
    
    
class Conv2dBenchmark(Benchmark):
    def __init__(
        self,
        fifo_buffer_slot_num: int,
    ):
        self.fifo_buffer_slot_num = fifo_buffer_slot_num
        self._cache_buffer_size = 0
        
        self.core_group_offset: tuple[int, int] = (0, 0)
        self.core_group_shape:  tuple[int, int] = (8, 8)
        
        self.N = 1
        self.C = 128
        self.H = 28
        self.W = 28
        self.K = 256
        self.FH = 3
        self.FW = 3
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        
        self.OH = math.floor((self.H + 2 * self.padding[0] - self.dilation[0] * (self.FH - 1) - 1) / self.stride[0] + 1)
        self.OW = math.floor((self.W + 2 * self.padding[1] - self.dilation[1] * (self.FW - 1) - 1) / self.stride[1] + 1)
        
        self.dtype:     torch.dtype = torch.int16
        self.acc_dtype: torch.dtype = torch.int16
        
        self._timestamp:    int = 0
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        self._total_ops:    int = self.N * self.OH * self.OW * self.K * (self.C * self.FH * self.FW * 2) + self.N * self.OH * self.OW * self.K
        
    @property
    def ifm_shape(self) -> tuple[int]:
        return (self.N, self.H, self.W, self.C)
    
    @property
    def wgt_shape(self) -> tuple[int]:
        return (self.FH, self.FW, self.K, self.C)
    
    @property
    def bias_shape(self) -> tuple[int]:
        return (1, self.K)
    
    @property
    def ofm_shape(self) -> tuple[int]:
        return (self.N, self.OH, self.OW, self.K)
    
    @property
    def ifm_total_size(self) -> int:
        return self.N * self.H * self.W * self.C * self.dtype.itemsize
    
    @property
    def wgt_total_size(self) -> int:
        return self.FH * self.FW * self.K * self.C * self.dtype.itemsize
    
    @property
    def bias_total_size(self) -> int:
        return self.K * self.dtype.itemsize
    
    @property
    def ofm_total_size(self) -> int:
        return self.N * self.OH * self.OW * self.K * self.acc_dtype.itemsize
        
    def run(self, device: TenstorrentDevice):
        try:
            core_group = device.get_npu_core_group(self.core_group_offset, self.core_group_shape)
        except:
            logger.error(f"Failed to get core group with offset {self.core_group_offset} and shape {self.core_group_shape}. Check if the configuration is valid for the device.")
            return False
        
        _l1_total_per_core     = parse_mem_cap_str("1.5MB")  # total L1 memory size in Tenstorrent Tensix Core is 1.5MB
        _spad_size_per_core    = parse_mem_cap_str("512KB")
        _context_buffer_slot_num = 8
        _l1_data_size_per_core = _l1_total_per_core - _spad_size_per_core
        
        logger.info(f"benchmark memory map per core {self.signature}: Data: {_l1_data_size_per_core / 1024:.2f} KB, SPAD: {_spad_size_per_core / 1024:.2f} KB")
        
        try:
            l1_data_mem_space = device.create_l1_mem_space(_l1_data_size_per_core, core_group=core_group)
            main_data_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
            
            ifm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=self.ifm_shape,  dtype=self.dtype).allocate()
            wgt_b  = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=self.wgt_shape,  dtype=self.dtype).allocate()
            bias_b = MCA_TensorBuffer(mem_space=main_data_mem_space, shape=self.bias_shape, dtype=self.dtype).allocate()
            ofm_b  = MCA_TensorBuffer(mem_space=l1_data_mem_space,   shape=self.ofm_shape,  dtype=self.acc_dtype).allocate()
            
            self._l1_traffic:   int = 0
            self._main_traffic: int = 0
            
            for b in [ifm_b, wgt_b, bias_b, ofm_b]:
                if b.mem_space.mem_type == GlobalContextMemType.L1:
                    self._l1_traffic += b.total_size
                else:
                    self._main_traffic += b.total_size
                    
            op = MCA_OP_CONV2D( 
                ifm_b, wgt_b, bias_b, ofm_b,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation, 
            )
            
            compiler = MCA_OperatorGraphCompiler()
            compiler.add_op(op)
            
            global_recipe=MCA_OperatorGraphCompiler.CompileRecipe(
                device=device,
                core_groups=[core_group],
                spad_space_size_per_core=_spad_size_per_core,
                fifo_buffer_slot_num=self.fifo_buffer_slot_num,
                context_buffer_slot_num=_context_buffer_slot_num,
                temporal_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL,
                spatial_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN,
            )
            
            compiled_ops = compiler.compile(global_recipe)
            self._cache_buffer_size = compiled_ops._env.op_meta[op.op_id].cache_buffer_size
        
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
                if args.monitor:
                    with MonitoringWindow(device, core_group, sim_name=self.signature) as monitor:
                        device.run_kernels()
                else:
                    device.run_kernels()
                    
                self._timestamp = device.timestamp
                device.reset_simulation()
                
        except Exception as e:
            logger.error(f"Error during benchmark {self.signature}: {e}")
            return False
        
        return True  # Indicate successful run without L1 memory overflow
    
    @property
    def n_cores(self) -> int:
        return self.core_group_shape[0] * self.core_group_shape[1]
        
    @property
    def timestamp(self) -> int:
        return self._timestamp
    
    @property
    def ld_ex_buffer_size(self) -> int:
        return self.fifo_buffer_slot_num * 2  # Assuming each slot is 2KB
    
    @property
    def ex_st_buffer_size(self) -> int:
        return self.fifo_buffer_slot_num * 2  # Assuming each slot is 2KB
    
    @property
    def bcast_buffer_size(self) -> int:
        return self.fifo_buffer_slot_num * 2  # Report-only estimate; actual FIFO sizing is owned by CompileRecipe.
    
    @property
    def cache_buffer_size(self) -> int:
        return self._cache_buffer_size // 1024
    
    @property
    def signature(self) -> str:
        ifm_shape_signature = "x".join(map(str, self.ifm_shape))
        wgt_shape_signature = "x".join(map(str, self.wgt_shape))
        stride_signature = "x".join(map(str, self.stride)) if self.stride[0] != self.stride[1] else str(self.stride[0])
        padding_signature = "x".join(map(str, self.padding)) if self.padding[0] != self.padding[1] else str(self.padding[0])
        dilation_signature = "x".join(map(str, self.dilation)) if self.dilation[0] != self.dilation[1] else str(self.dilation[0])
        return f"CV_C{self.n_cores}_{ifm_shape_signature}_{wgt_shape_signature}_{stride_signature}_{padding_signature}_{dilation_signature}_FIFO{self.fifo_buffer_slot_num}"
    
    
class BenchmarkProcess(mp.Process):
    def __init__(self, benchmark: Conv2dBenchmark, device_config: TenstorrentConfig, return_dict: dict, worker_sem):
        super().__init__()
        self.benchmark = benchmark
        self.device_config = device_config
        self.return_dict = return_dict
        self.worker_sem = worker_sem
        
    def run(self):
        self.worker_sem.acquire()
        try:
            logger.info(f"process started for {self.benchmark.signature}")

            device = TenstorrentDevice(**self.device_config)
            device.initialize()
            device.set_command_debug_verbosity(verbose=False)
            
            flag = self.benchmark.run(device)
            if not flag:
                return
            
            self.return_dict[self.benchmark.signature] = {
                "timestamp":    self.benchmark.timestamp,
                "n_cores":      self.benchmark.n_cores,
                "fifo_buffer_slot_num": self.benchmark.fifo_buffer_slot_num,
                "cache_buffer_size": self.benchmark.cache_buffer_size,
            }
            
            logger.info(f"process finished for {self.benchmark.signature}")
        finally:
            self.worker_sem.release()

fifo_buffer_slot_nums = [1, 2, 4, 8, 16, 24, 32, 40]

benchmarks = [
    Conv2dBenchmark(
        fifo_buffer_slot_num=fifo_buffer_slot_num,
    )
    for fifo_buffer_slot_num in fifo_buffer_slot_nums
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
        p = BenchmarkProcess(benchmark, config, return_dict, worker_sem)
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    
    with open(output_path, "w") as f:
        f.write("Benchmark,Number of Cores,Timestamp (cycles),FIFO Buffer Depth,Cache Buffer Size (KB)\n")
        for benchmark in benchmarks:
            if benchmark.signature not in return_dict:
                logger.error(f"Missing results for benchmark {benchmark.signature}")
                continue
            
            result = return_dict[benchmark.signature]
            
            timestamp    = result["timestamp"]
            n_cores      = result["n_cores"]
            fifo_buffer_slot_num = result["fifo_buffer_slot_num"]
            cache_buffer_size = result["cache_buffer_size"]
            
            f.write(f"{benchmark.signature},{n_cores},{timestamp},{fifo_buffer_slot_num},{cache_buffer_size}\n")
    
    print(f"Benchmark results saved to '{output_path}'.")
    
    if visualize is not None:
        img_path = os.path.join(OUTPUT_DIR, f"exp3_2_conv2d_dse_l1_mem.png")
        
        visualize.draw(
            src_path=output_path,
            img_path=img_path,
        )
        
        print(f"Roofline visualization saved to '{img_path}'.")
