import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *


@jit_prototype
def main(core: NPUCore, tensor_buffer: MCA_TensorBuffer, copied_buffer: MCA_TensorBuffer, spm_ptr: int):
    shard_y, shard_x = tensor_buffer.shard_grid
    tile_y, tile_x = tensor_buffer.tile_grid_per_shard
    
    for y_s in range(shard_y):
        for x_s in range(shard_x):
            for y_t in range(tile_y):
                for x_t in range(tile_x):
                    container = DataContainer(shape=tensor_buffer.tile_shape, dtype=tensor_buffer.dtype)
                    
                    src_ptr, row_size, row_num, src_row_stride, dst_row_stride = tensor_buffer.get_tile_ptr_read_args(y_s, x_s, y_t, x_t)
                    
                    core.local_mem_copy(spm_ptr, src_ptr, row_size, row_num, src_row_stride, dst_row_stride, nowait=False)
                    core.local_mem_page_read(spm_ptr, container, 8*8*4)
                    
                    core.debug_core_with_ambiguous_func(
                        lambda container: logger.info(f"tile ({y_s}, {x_s}, {y_t}, {x_t}):\n{container.data.flatten().view(torch.int32).reshape(8, 8)}"),
                        container
                    )
                    
                    dst_ptr, row_size, row_num, src_row_stride, dst_row_stride = copied_buffer.get_tile_ptr_write_args(y_s, x_s, y_t, x_t)
                    core.local_mem_page_write(spm_ptr, container, 8*8*4)
                    core.local_mem_copy(dst_ptr, spm_ptr, row_size, row_num, src_row_stride, dst_row_stride)


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
    
    copied_buffer = MCA_TensorBuffer(
        mem_space=l1_mem_space,
        shape=(N, M, K), 
        dtype=torch.int32, 
        shard_shape=(7, 7),
        blocked_mapping=True,
    ).tiling(tile_shape=(8, 8))
    
    tensor_buffer.allocate()
    copied_buffer.allocate()
    
    tensor_buffer.update(original_tensor)
    
    core = device.get_npu_core(core_id=core_grid[0])
    spm_ptr = l1_mem_space.allocate(core.core_id, size=parse_mem_cap_str("64KB"))
    
    kernel = main(core, tensor_buffer, copied_buffer, spm_ptr)
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    restored_tensor = copied_buffer.restore()
    
    print("Original Tensor:")
    print(original_tensor)
    
    print("Restored Tensor:")
    print(restored_tensor)
    
    print("test passed:", torch.equal(original_tensor, restored_tensor))