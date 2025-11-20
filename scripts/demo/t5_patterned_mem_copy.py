import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def main(
    core: NPUCore, 
    
    dst_ptr: Pointer, 
    src_ptr: Pointer, 
    
    size: int, 
    
    src_row_size: int=None, 
    src_row_stride: int=None, 
    dst_row_stride: int=None, 
    dst_row_pattern: dict[int, int]=None
):
    core.local_mem_copy(
        dst_ptr=dst_ptr, 
        src_ptr=src_ptr, 
        size=size,
        src_row_size=src_row_size, 
        src_row_stride=src_row_stride, 
        dst_row_stride=dst_row_stride, 
        dst_row_pattern=dst_row_pattern
    )


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group(offset=(0, 0), shape=(1, 1))
    core_id = core_grid[0, 0]
    core = device.get_npu_core(core_id=core_id)
    
    l1_mem_space   = device.create_l1_mem_space(size_per_bank=parse_mem_cap_str("1MB"), core_group=core_grid)
    
    M, K = 8, 8
    N = 16
    dtype = torch.int32
    src = torch.arange(M*K, dtype=dtype).reshape(M, K)
    # dst = torch.zeros(N, K, dtype=dtype)
    
    src_size = src.numel() * src.element_size()
    dst_size = N * K * dtype.itemsize
    
    src_ptr = l1_mem_space.allocate(core_id=core_id, size=src_size)
    dst_ptr = l1_mem_space.allocate(core_id=core_id, size=dst_size)
    
    device.mem_set_data(src_ptr, src_size, src)
    
    src_row_size = K * src.element_size()
    src_row_stride = K * src.element_size()
    dst_row_stride = K * dtype.itemsize
    dst_row_pattern = {
        0: 0,
        4: 1,
        8: 2,
        10: 3,
    }
    
    kernel = main(
        core, dst_ptr, src_ptr, 
        size=src_size, 
        src_row_size=src_row_size, 
        src_row_stride=src_row_stride, 
        dst_row_stride=dst_row_stride, 
        dst_row_pattern=dst_row_pattern
    )
    
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    simulated = device.mem_get_data(dst_ptr, dst_size, dtype=torch.int32).reshape(N, K)
    reference = torch.zeros(N, K, dtype=dtype)
    for dst_row_idx, src_row_idx in dst_row_pattern.items():
        reference[dst_row_idx, :] = src[src_row_idx, :]
        
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()