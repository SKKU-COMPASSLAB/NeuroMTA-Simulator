import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def main(core: NPUCore, main_in_ptr: Pointer, main_out_ptr: Pointer, l1_ptr: Pointer, size: int):
    core.local_mem_copy(l1_ptr,       main_in_ptr, size)
    core.local_mem_copy(main_out_ptr, l1_ptr,      size)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group(offset=(0, 0), shape=(2, 2))
    
    l1_mem_space   = device.create_l1_mem_space(size_per_bank=parse_mem_cap_str("1MB"), core_group=core_grid)
    main_mem_space = device.create_main_mem_space(size_per_channel=parse_mem_cap_str("1GB"))
    
    core = device.get_npu_core(core_id=core_grid[0, 0])
    size = parse_mem_cap_str("4KB")
    
    main_in_ptr  = main_mem_space.allocate(channel_id=0, size=size)
    main_out_ptr = main_mem_space.allocate(channel_id=0, size=size)
    l1_ptr       = l1_mem_space.allocate(core.core_id, size=size)
    
    data = torch.arange(0, size // 4, dtype=torch.int32)
    device.mem_set_data(main_in_ptr, size, data)
    
    kernel = main(core, main_in_ptr, main_out_ptr, l1_ptr, size)
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    result = device.mem_get_data(main_out_ptr, size, dtype=torch.int32)
    
    print(f"reference: {data}")
    print(f"simulated: {result}")
    print(f"simulation {'PASSED' if torch.equal(data, result) else 'FAILED'}")
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()