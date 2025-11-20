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
    
    M, K = 128, 128
    
    original_tensor = torch.arange(M * K, dtype=torch.int32).reshape(M, K)
    tensor_buffer = MCA_TensorBuffer(
        mem_space=l1_mem_space,
        
        shape=(M, K), 
        dtype=torch.int32, 

        shard_grid=(2, 2),
        
        blocked_mapping=True,
    ).tiling(tile_shape=(32, 32))
    
    tensor_buffer.allocate()
    tensor_buffer.update(original_tensor)
    restored_tensor = tensor_buffer.restore()
    
    for h in range(tensor_buffer._n_y_shards):
        for w in range(tensor_buffer._n_x_shards):
            ptr = tensor_buffer.get_shard_ptr(h, w)
            shard = device.mem_get_data(ptr, size=128 * 128 // 4 * 4, dtype=torch.int32)
            print(f"shard ({h}, {w}):\n{shard.reshape(M // 2, K // 2)}")
    
    print(f"original tensor:\n{original_tensor}")
    print(f"restored tensor:\n{restored_tensor}")
    print(f"test {'passed' if torch.equal(original_tensor, restored_tensor) else 'failed'}")