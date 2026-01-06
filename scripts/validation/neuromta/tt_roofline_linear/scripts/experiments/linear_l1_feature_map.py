import argparse
import os

import torch
import ttnn
from tracy import Profiler, signpost
from loguru import logger


warmup_iters = 5
test_iters = 1


class Benchmark:
    def __init__(
        self, 
        M:  int, N:  int, K:  int,
    ):
        self.M: int = M
        self.N: int = N
        self.K: int = K
        
        self._timestamp:    int = 0
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        self._total_ops:    int = (self.M * self.N * self.K) * 2 + (self.M * self.N)  # MACs + Bias Add
        
    def run(self, device):
        global warmup_iters
        global test_iters
        
        # tensor creation examples
        logger.info("\n--- TT-NN Tensor Creation with Tiles (1024x1024) ---")
        
        ifm  = torch.rand((self.M, self.K), dtype=torch.bfloat16)
        wgt  = torch.rand((self.K, self.N), dtype=torch.bfloat16)
        bias = torch.rand((1,      self.N), dtype=torch.bfloat16)
        
        tt_ifm  = ttnn.from_torch(ifm,  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_wgt  = ttnn.from_torch(wgt,  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        for i in range(warmup_iters + test_iters):
            is_warmup = i < warmup_iters
            test_name = f"WARMUP {i+1}" if is_warmup else f"{self.signature} {i - warmup_iters + 1}"
            
            if not is_warmup:
                signpost(self.signature, f"TEST {i - warmup_iters + 1}")
            matmul_result = ttnn.linear(tt_ifm, tt_wgt, bias=tt_bias, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=4, x=4))
            if not is_warmup:
                signpost(self.signature, "END")
                
        self._l1_traffic:   int = 0
        self._main_traffic: int = 0
        
        self._l1_traffic   += ifm.numel()  * ifm.element_size()
        self._main_traffic += wgt.numel()  * wgt.element_size()
        self._main_traffic += bias.numel() * bias.element_size()
        self._l1_traffic   += self.M * self.N * torch.bfloat16.itemsize  # output write-back to L1
    
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
        return f"{self.M}x{self.N}x{self.K}"


benchmarks = [
    # Benchmarks: Square Matrices with Varying Sizes
    Benchmark(M=512 , N=512,  K=512),
    Benchmark(M=256 , N=256,  K=256),
    Benchmark(M=128 , N=128,  K=128),
    
    # Benchmarks: Rectangular Matrices with Skewed Dimensions (Arithmetic Intensity Variation)
    Benchmark(M=512,  N=1024, K=1024),
    Benchmark(M=256,  N=1024, K=1024),
    Benchmark(M=128,  N=1024, K=1024),
    Benchmark(M=64,   N=1024, K=1024),
    Benchmark(M=32,   N=1024, K=1024),
    Benchmark(M=8,    N=1024, K=1024),
    Benchmark(M=4,    N=1024, K=1024),
    Benchmark(M=2,    N=1024, K=1024),
    Benchmark(M=1,    N=1024, K=1024),
]


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

    parser = argparse.ArgumentParser(description="Tenstorrent Device Benchmark Suite")
    parser.add_argument("-o", "--output", type=str, default=f"{FILE_NAME}.csv", help="Output file to save benchmark results")
    parser.add_argument("-n", "--n-workers", type=int, default=1, help="Number of parallel worker processes")
    args = parser.parse_args()

    output_dir = os.path.join(ROOT_DIR, ".logs")
    output_path = os.path.join(output_dir, args.output)
    trace_path = os.path.join(output_dir, f"{os.path.splitext(args.output)[0]}.tracy")
    os.makedirs(output_dir, exist_ok=True)
    
    return_dict: dict[str, dict[str, int]] = {}
    
    profiler = Profiler()
    profiler.enable()
    
    # Open the Device
    device = ttnn.open_device(device_id=0)
    
    for b in benchmarks:
        logger.info(f"\n--- Running Benchmark: Linear Operator with Dimensions ({b.signature}) ---")
        b.run(device)
        
        return_dict[b.signature] = {
            "timestamp":    0,  # placeholder for cycle count
            "total_ops":    b.total_ops,
            "l1_traffic":   b.l1_traffic,
            "main_traffic": b.main_traffic,
        }
        
    # close the device
    ttnn.close_device(device)
    
    profiler.disable()
        
    with open(output_path, "w") as f:
        f.write("Benchmark,Time (us),Total OPs,L1 Memory Traffic (Bytes),Main Memory Traffic (Bytes),Performance (OPs/cycle),Arithmetic Intensity (OPs/Byte),L1 Bandwidth (Byte/cycle),Main Bandwidth (Byte/cycle),Total Bandwidth (Byte/cycle)\n")
        for benchmark in benchmarks:
            result = return_dict[benchmark.signature]
            
            total_ops    = result["total_ops"]
            l1_traffic   = result["l1_traffic"]
            main_traffic = result["main_traffic"]
            
            arith_intensity = (total_ops / (main_traffic + l1_traffic)) if (main_traffic + l1_traffic) != 0 else 0
            
            f.write(f"{benchmark.signature},0,{total_ops},{l1_traffic},{main_traffic},0,{arith_intensity:.2f},0,0,0\n")
    
    print(f"Benchmark results saved to '{output_path}'.")
    
    
