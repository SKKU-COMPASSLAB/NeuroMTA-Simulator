import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def main(core: NPUCore, t: MCA_TensorBuffer, spm_ptr: int, y_s: int, x_s: int, y_t: int, x_t: int):
    container = DataContainer(shape=t.tile_shape, dtype=t.dtype)
    
    src_ptr, src_size, src_row_size, src_row_stride, dst_row_stride = t.get_tile_ptr_read_args(y_s, x_s, y_t, x_t)
    
    core.local_mem_copy(spm_ptr, src_ptr, size=src_size, src_row_size=src_row_size, src_row_stride=src_row_stride, dst_row_stride=dst_row_stride)
    core.local_mem_page_read(spm_ptr, t.tile_size, container, mem_row_size=t.tile_shape[1] * t.dtype.itemsize)
    
    core.debug_core_with_ambiguous_func(
        lambda container: logger.info(f"CORE: {core.core_id} tile ({y_s}, {x_s}, {y_t}, {x_t}):\n{container.data.flatten().view(t.dtype).reshape(t.tile_shape)}"),
        container
    )


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
            
    M, K = 32, 32
    
    original_tensor = torch.arange(M * K, dtype=torch.int32).reshape(M, K)
    tensor_buffer = MCA_TensorBuffer(
        mem_space=l1_mem_space,
        
        shape=(M, K), 
        dtype=torch.int32, 
        
        shard_grid=(2, 2),
        
        blocked_mapping=True,
    ).tiling(tile_shape=(8, 8))
    
    tensor_buffer.allocate()
    tensor_buffer.update(original_tensor)
    restored_tensor = tensor_buffer.restore()
    
    shard_y, shard_x = tensor_buffer.shard_grid
    tile_y, tile_x = tensor_buffer.tile_grid_per_shard
    
    for y_s in range(shard_y):
        for x_s in range(shard_x):
            for y_t in range(tile_y):
                for x_t in range(tile_x):
                    core_id = core_grid[y_s, x_s]
                    core = device.get_npu_core(core_id=core_id)
                    spm_ptr = l1_mem_space.allocate(core.core_id, size=parse_mem_cap_str("16KB"))
                    
                    kernel = main(core, tensor_buffer, spm_ptr, y_s, x_s, y_t, x_t)
                    kernel.dispatch(f"MAIN")
    
    device.run_kernels()