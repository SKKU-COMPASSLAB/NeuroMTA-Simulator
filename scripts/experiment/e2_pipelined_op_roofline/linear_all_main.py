import os
import argparse
import threading
import multiprocessing as mp
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


class Benchmark:
    def __init__(
        self, 
        M:  int, N:  int, K:  int,
        Ms: int, Ns: int, Ks: int,
        dtype:     torch.dtype,
        acc_dtype: torch.dtype,
        mapping_strategy: str = MCA_OperatorMapper.OUTPUT_STATIONARY
    ):
        if N != K:
            raise ValueError("This benchmark only supports square weight matrices (N == K).")
        
        self.M: int = M
        self.N: int = N
        self.K: int = K
        
        self.Ms: int = Ms
        self.Ns: int = Ns
        self.Ks: int = Ks
        
        self.dtype:     torch.dtype = dtype
        self.acc_dtype: torch.dtype = acc_dtype
        
        self.mapping_strategy: str = mapping_strategy
        
        self._timestamp:    int = 0
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        self._total_ops:    int = ((self.M * self.N * self.K) * 2 + (self.M * self.N)) * 2  # MACs + Bias Add
        
    def run(self, device: TenstorrentDevice, core_group1: MTA_CoreGrid, core_group2: MTA_CoreGrid):
        if (self.M // 32) % self.Ms != 0:
            self.Ms = 1
        if (self.N // 32) % self.Ns != 0:
            self.Ns = 1
        if (self.K // 32) % self.Ks != 0:
            self.Ks = 1
            
        ifm  = torch.randint(low=0, high=128, size=(self.M, self.K), dtype=self.dtype)
        wgt  = torch.randint(low=0, high=128, size=(self.N, self.K), dtype=self.dtype)
        bias = torch.randint(low=0, high=256, size=(self.N,), dtype=self.dtype)
        
        param_mem_space  = device.create_main_mem_space(parse_mem_cap_str("1GB"))
        
        ifm_mem_space1    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group1)
        ofm_mem_space1    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group1)
        spad_ld_pp_space1 = device.create_l1_mem_space(parse_mem_cap_str("256KB"), core_group=core_group1)
        spad_st_pp_space1 = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group1)
        
        ifm_mem_space2    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group2)
        ofm_mem_space2    = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group=core_group2)
        spad_ld_pp_space2 = device.create_l1_mem_space(parse_mem_cap_str("256KB"), core_group=core_group2)
        spad_st_pp_space2 = device.create_l1_mem_space(parse_mem_cap_str("32KB"), core_group=core_group2)
        
        ifm1_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=ifm.shape,         dtype=ifm.dtype,       shard_grid=(self.Ms, self.Ks)).allocate().update(ifm)
        wgt1_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,         dtype=wgt.dtype,       shard_grid=(self.Ns, self.Ks)).allocate().update(wgt)
        bias1_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape,        dtype=bias.dtype,      shard_grid=(1,  self.Ns),    ).allocate().update(bias)
        
        ifm2_b  = MCA_TensorBuffer(mem_space=ofm_mem_space1,  shape=(self.M, self.N),  dtype=self.acc_dtype,  shard_grid=(self.Ms, self.Ns)).allocate()
        wgt2_b  = MCA_TensorBuffer(mem_space=param_mem_space, shape=wgt.shape,         dtype=self.acc_dtype,  shard_grid=(self.Ns, self.Ks)).allocate().update(wgt.to(self.acc_dtype))
        bias2_b = MCA_TensorBuffer(mem_space=param_mem_space, shape=bias.shape,        dtype=bias.dtype,      shard_grid=(1,  self.Ns),    ).allocate().update(bias)
        ofm_b   = MCA_TensorBuffer(mem_space=ofm_mem_space2,  shape=(self.M, self.N),  dtype=self.acc_dtype,  shard_grid=(self.Ms, self.Ns)).allocate()
        
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        
        for b in [ifm1_b, wgt1_b, bias1_b, ifm2_b, wgt2_b, bias2_b, ofm_b]:
            if b.mem_space.mem_type == GlobalContextMemType.L1:
                self._l1_traffic += b.total_size
            else:
                self._main_traffic += b.total_size
            
        
        operator1 =  MCA_OP_LINEAR(
            device, core_group1, 
            spad_ld_pp_space1, spad_st_pp_space1, 
            ifm1_b, wgt1_b, bias1_b, ifm2_b, 
            broadcast_optimize=True, 
            mapping_strategy=self.mapping_strategy,
        )
           
        operator2 = MCA_OP_LINEAR(
            device, core_group2, 
            spad_ld_pp_space2, spad_st_pp_space2, 
            ifm2_b, wgt2_b, bias2_b, ofm_b, 
            broadcast_optimize=True, 
            mapping_strategy=self.mapping_strategy,
        )
        
        operator1.pipeline(
            dst_op=operator2,
            src_buf_name="ofm",
            dst_buf_name="ifm",
        )
        
        operator1.dispatch()
        
        device.run_kernels()

        self._timestamp = device.timestamp
        
        device.reset_simulation()
        
        ifm_mem_space1.remove()
        ofm_mem_space1.remove()
        spad_ld_pp_space1.remove()
        spad_st_pp_space1.remove()
        
        ifm_mem_space2.remove()
        ofm_mem_space2.remove()
        spad_ld_pp_space2.remove()
        spad_st_pp_space2.remove()
        
        param_mem_space.remove()
        
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
    
    @property
    def signature(self) -> str:
        ms = self.mapping_strategy.lower()
        if ms == MCA_OperatorMapper.OUTPUT_STATIONARY:
            ms_str = "os"
        elif ms == MCA_OperatorMapper.ROUND_ROBIN:
            ms_str = "rr"
        else:
            ms_str = "unk"
        return f"{self.M}x{self.N}x{self.K}_{self.Ms}x{self.Ns}x{self.Ks}_{str(self.dtype).split('.')[-1]}_{str(self.acc_dtype).split('.')[-1]}_{ms_str}"
    
    
class BenchmarkProcess(mp.Process):
    def __init__(
        self, 
        
        benchmark: Benchmark, 
        device_config: TenstorrentConfig, 
        
        core_group_offset1: tuple[int, int], 
        core_group_shape1:  tuple[int, int],
        
        core_group_offset2: tuple[int, int], 
        core_group_shape2:  tuple[int, int], 
        
        return_dict: dict
    ):
        super().__init__()
        self.benchmark = benchmark
        self.device_config = device_config
        
        self.core_group_offset1 = core_group_offset1
        self.core_group_shape1  = core_group_shape1
        
        self.core_group_offset2 = core_group_offset2
        self.core_group_shape2  = core_group_shape2
        
        self.return_dict = return_dict
        
    def run(self):
        device = TenstorrentDevice(**self.device_config)
        device.initialize()
        device.set_command_debug_verbosity(verbose=False)
        
        core_group1 = device.get_npu_core_group(
            offset=self.core_group_offset1,
            shape=self.core_group_shape1
        )
        
        core_group2 = device.get_npu_core_group(
            offset=self.core_group_offset2,
            shape=self.core_group_shape2
        )
        
        self.benchmark.run(device, core_group1, core_group2)
        
        self.return_dict[self.benchmark.signature] = {
            "timestamp":    self.benchmark.timestamp,
            "total_ops":    self.benchmark.total_ops,
            "l1_traffic":   self.benchmark.l1_traffic,
            "main_traffic": self.benchmark.main_traffic,
        }


benchmarks = [
    # Benchmarks: Square Matrices with Varying Sizes
    Benchmark(M=1024, N=1024, K=1024, Ms=8, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=512 , N=512,  K=512,  Ms=8, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=256 , N=256,  K=256,  Ms=4, Ns=4, Ks=4, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=128 , N=128,  K=128,  Ms=4, Ns=4, Ks=4, dtype=torch.int32, acc_dtype=torch.int32),
    
    # Benchmarks: Rectangular Matrices with Skewed Dimensions (Arithmetic Intensity Variation)
    Benchmark(M=512,  N=1024, K=1024, Ms=8, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=256,  N=1024, K=1024, Ms=8, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=128,  N=1024, K=1024, Ms=4, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=64,   N=1024, K=1024, Ms=2, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32),
    Benchmark(M=32,   N=1024, K=1024, Ms=1, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32, mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN),
    Benchmark(M=8,    N=1024, K=1024, Ms=1, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32, mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN),
    Benchmark(M=4,    N=1024, K=1024, Ms=1, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32, mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN),
    Benchmark(M=2,    N=1024, K=1024, Ms=1, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32, mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN),
    Benchmark(M=1,    N=1024, K=1024, Ms=1, Ns=8, Ks=8, dtype=torch.int32, acc_dtype=torch.int32, mapping_strategy=MCA_OperatorMapper.ROUND_ROBIN),
]

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
    
    parser = argparse.ArgumentParser(description="Tenstorrent Device Benchmark Suite")
    parser.add_argument("-o", "--output", type=str, default=f"{FILE_NAME}.csv", help="Output file to save benchmark results")
    args = parser.parse_args()

    output_dir = os.path.join(ROOT_DIR, ".logs")
    output_path = os.path.join(output_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    manager = mp.Manager()
    return_dict = manager.dict()
    config = TenstorrentConfig.BLACKHOLE()
    
    processes: list[BenchmarkProcess] = []
    for benchmark in benchmarks:
        p = BenchmarkProcess(benchmark, config, (0, 0), (4, 4), (0, 4), (4, 4), return_dict)
        p.start()
        processes.append(p)
        print(f"Started benchmark process for: {benchmark.signature}")
        
    def _monitor_and_notify(proc: BenchmarkProcess):
        proc.join()
        print(f"Benchmark process finished for: {proc.benchmark.signature}")

    monitor_threads: list[threading.Thread] = []
    for proc in processes:
        t = threading.Thread(target=_monitor_and_notify, args=(proc,), daemon=True)
        t.start()
        monitor_threads.append(t)

    for t in monitor_threads:
        t.join()
    
    with open(output_path, "w") as f:
        f.write("Benchmark,Timestamp (cycles),Total OPs,L1 Memory Traffic (Bytes),Main Memory Traffic (Bytes),Performance (OPs/cycle),Arithmetic Intensity (OPs/Byte),L1 Bandwidth (Byte/cycle),Main Bandwidth (Byte/cycle),Total Bandwidth (Byte/cycle)\n")
        for benchmark in benchmarks:
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