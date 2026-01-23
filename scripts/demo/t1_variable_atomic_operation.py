import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *


@jit_prototype
def core0_main(core: NPUCore, var1: VariableHandle, var2: VariableHandle):
    for i in range(5):
        core.var_atomic_wait(var1, 0)
        core.var_atomic_increase(var2, 1)
        core.var_atomic_compare_and_swap(var1, cmp_value=0, new_value=1)
        
@jit_prototype
def core1_main(core: NPUCore, var1: VariableHandle, var2: VariableHandle):
    for i in range(5):
        core.var_atomic_wait(var1, 1)
        core.var_atomic_increase(var2, 10)
        core.var_atomic_compare_and_swap(var1, cmp_value=1, new_value=0)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=True)
    
    core_grid = device.get_npu_core_group(offset=(0, 0), shape=(2, 2))
    
    l1_mem_space = device.create_l1_mem_space(size_per_bank=parse_mem_cap_str("1MB"), core_group=core_grid)
    main_mem_space = device.create_main_mem_space(size_per_channel=parse_mem_cap_str("4GB"))
    
    core0 = device.get_npu_core(core_id=core_grid[0, 0])
    core1 = device.get_npu_core(core_id=core_grid[0, 1])
    
    var1 = VariableHandle(initial_value=0)
    var2 = VariableHandle(initial_value=0)
    
    kernel0 = core0_main(core0, var1, var2)
    kernel1 = core1_main(core1, var1, var2)
    
    kernel0.dispatch("MAIN")
    kernel1.dispatch("MAIN")
    
    device.run_kernels()

    result = var2._value
    
    print(f"Final result: {result}")  # Expected: 55