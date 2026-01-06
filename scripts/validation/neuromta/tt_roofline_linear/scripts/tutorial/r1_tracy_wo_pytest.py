from tracy import Profiler, signpost

import torch
import numpy as np
import ttnn
from loguru import logger

def test_main():
    profiler = Profiler()
    profiler.enable()
    
    # Open the Device
    device = ttnn.open_device(device_id=0)
    
    # tensor creation examples
    logger.info("\n--- TT-NN Tensor Creation with Tiles (1024x1024) ---")
    
    ifm = torch.rand((1024, 1024), dtype=torch.float32)
    wgt = torch.rand((1024, 1024), dtype=torch.float32)
    bias = torch.rand((1, 1024), dtype=torch.float32)
    
    tt_ifm  = ttnn.from_torch(ifm,  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_wgt  = ttnn.from_torch(wgt,  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    
    warmup_iters = 5
    test_iters = 2
    
    for i in range(warmup_iters + test_iters):
        is_warmup = i < warmup_iters
        test_name = f"WARMUP {i+1}" if is_warmup else f"TEST {i - warmup_iters + 1}"
        
        signpost("START", test_name)
        # matmul_result = ttnn.matmul(tt_ifm, tt_wgt, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=4, x=4)) + tt_bias
        matmul_result = ttnn.linear(tt_ifm, tt_wgt, bias=tt_bias, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=4, x=4))
        signpost("END", test_name)
    
    logger.info(f"Matrix Multiplication:\n{matmul_result}")
    
    # close the device
    ttnn.close_device(device)
    
    profiler.disable()

if __name__ == "__main__":
    test_main()