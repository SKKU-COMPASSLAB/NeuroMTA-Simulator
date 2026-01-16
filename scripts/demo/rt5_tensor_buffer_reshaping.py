import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group((0, 0), (2, 2))
    
    l1_mem_space   = device.create_l1_mem_space(size_per_bank=parse_mem_cap_str("1MB"), core_group=core_grid)
    main_mem_space = device.create_main_mem_space(size_per_channel=parse_mem_cap_str("1GB"))
                
    N, M, K = 2, 14, 14
    
    original_tensor = torch.arange(N * M * K, dtype=torch.int32).reshape(N, M, K)
    tensor_buffer = MCA_TensorBuffer(
        mem_space=l1_mem_space,
        shape=(N, M, K), 
        dtype=torch.int32, 
        shard_shape=(7, 7),
        blocked_mapping=True,
    ).tiling(tile_shape=(8, 8))
    
    tensor_buffer.allocate()
    tensor_buffer.update(original_tensor)
    
    reshaped_buffer = tensor_buffer.reshape(N * K, M)
    reshaped_tensor = reshaped_buffer.restore()
    
    core = device.get_npu_core(core_id=core_grid[0])
    spm_ptr = l1_mem_space.allocate(core.core_id, size=parse_mem_cap_str("64KB"))
    
    device.run_kernels()
    
    print("Original Tensor:")
    print(original_tensor)
    
    print("Reshaped Tensor:")
    print(reshaped_tensor)
    
    print("test passed:", torch.equal(original_tensor.reshape(N * K, M), reshaped_tensor))