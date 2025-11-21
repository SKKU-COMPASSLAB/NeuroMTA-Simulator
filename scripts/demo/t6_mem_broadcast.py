import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def main(
    core: NPUCore, 
    
    dst_ptrs: list[Pointer], 
    src_ptr: Pointer, 
    
    size: int, 
    
    src_row_size: int=None, 
    src_row_stride: int=None, 
    dst_row_stride: int=None, 
    dst_row_pattern: dict[int, int]=None
):
    core.local_mem_broadcast(
        dst_ptrs=dst_ptrs, 
        src_ptr=src_ptr, 
        size=size,
        src_row_size=src_row_size, 
        src_row_stride=src_row_stride, 
        dst_row_stride=dst_row_stride, 
        dst_row_pattern=dst_row_pattern,
        nowait=False
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
    N = 8
    dtype = torch.int32
    src = torch.arange(M*K, dtype=dtype).reshape(M, K)
    
    src_size = src.numel() * src.element_size()
    dst_size = N * K * dtype.itemsize
    
    src_ptr = l1_mem_space.allocate(core_id=core_id, size=src_size)
    dst_ptr0 = l1_mem_space.allocate(core_id=core_id, size=dst_size)
    dst_ptr1 = l1_mem_space.allocate(core_id=core_id, size=dst_size)
    dst_ptrs = [dst_ptr0, dst_ptr1]
    
    device.mem_set_data(src_ptr, src_size, src)
    
    src_row_size = K * src.element_size()
    src_row_stride = K * src.element_size()
    dst_row_stride = K * dtype.itemsize
    
    kernel = main(
        core, dst_ptrs, src_ptr, 
        size=src_size, 
        src_row_size=src_row_size, 
        src_row_stride=src_row_stride, 
        dst_row_stride=dst_row_stride, 
        dst_row_pattern=None
    )
    
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    simulated0 = device.mem_get_data(dst_ptr0, dst_size, dtype=torch.int32).reshape(N, K)
    simulated1 = device.mem_get_data(dst_ptr1, dst_size, dtype=torch.int32).reshape(N, K)
    reference = src
        
    print(f"simulated0:\n{simulated0}")
    print(f"simulated1:\n{simulated1}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated0, reference) and torch.equal(simulated1, reference) else 'FAILED'}")
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()