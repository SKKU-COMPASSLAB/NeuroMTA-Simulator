import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *


@jit_prototype
def main(core: NPUCore, main_in_ptrs: list[Pointer], main_out_ptrs: list[Pointer], l1_ptrs: list[Pointer], size: int, burst_count: int = 4):
    for i in range(burst_count):
        with new_parallel_thread():
            core.local_mem_copy(l1_ptrs[i], main_in_ptrs[i], size, nowait=True)  # does not wait for the NoC and DRAM transactions
            
    core.parallel_merge()
    core.async_rpc_wait_all()  # wait for all the NoC and DRAM transactions to complete
    core.debug_core_with_ambiguous_func(lambda x: logger.info(f"[TIMESTAMP={x.timestamp}] SYNC POINT"), core)
    
    for i in range(burst_count):
        with new_parallel_thread():
            core.local_mem_copy(main_out_ptrs[i], l1_ptrs[i], size, nowait=True)  # does not wait for the NoC and DRAM transactions
    
    core.parallel_merge()
    core.async_rpc_wait_all()  # wait for all the NoC and DRAM transactions to complete
    core.debug_core_with_ambiguous_func(lambda x: logger.info(f"[TIMESTAMP={x.timestamp}] SYNC POINT"), core)


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
    
    
    main_in_ptrs: list[Pointer] = []
    main_out_ptrs: list[Pointer] = []
    l1_ptrs: list[Pointer] = []
    burst_count = 4
    
    for _ in range(burst_count):
        main_in_ptrs.append(main_mem_space.allocate(channel_id=0, size=size))
        main_out_ptrs.append(main_mem_space.allocate(channel_id=0, size=size))
        l1_ptrs.append(l1_mem_space.allocate(core.core_id, size=size))
    
    data = torch.arange(0, size // 4, dtype=torch.int32)
    for main_in_ptr in main_in_ptrs:
        device.mem_set_data(main_in_ptr, size, data)
    
    kernel = main(core, main_in_ptrs, main_out_ptrs, l1_ptrs, size, burst_count)
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    for i in range(burst_count):
        result = device.mem_get_data(main_out_ptrs[i], size, dtype=torch.int32)
        
        print(f"BURST {i}:")
        print(f"  - reference: {data}")
        print(f"  - simulated: {result}")
        print(f"  - simulation {'PASSED' if torch.equal(data, result) else 'FAILED'}")
        
    print(f"simulation terminated in {device.timestamp} cycles")
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()