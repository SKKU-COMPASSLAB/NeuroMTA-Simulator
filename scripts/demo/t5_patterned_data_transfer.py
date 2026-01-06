import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def kernel_container_init(
    core: NPUCore,
    container: DataContainer[torch.Tensor],
    shape: tuple[int, ...],
    dtype: torch.dtype,
):
    core.local_data_container_init(container, shape, dtype)
    core.debug_core_with_ambiguous_func(lambda: logger.info(f"Container initialized with shape {shape} and dtype {dtype}"))
    

@jit_prototype
def kernel_mem_init(
    core: NPUCore,
    ptr: Pointer,
    size: int,
    data: torch.Tensor,
):
    core.local_mem_init(ptr, size, data)
    core.debug_core_with_ambiguous_func(lambda: logger.info(f"Pointer {ptr} initialized with data:\n{data.reshape(-1, 8)}"))


@jit_prototype
def kernel_mem_copy(
    core: NPUCore, 
    ptr: Pointer,
    container: DataContainer[torch.Tensor], 
    row_size: int, 
    row_num: int=1, 
    mem_row_stride: int=None, 
    cont_row_stride: int=None, 
    row_pattern: dict[int, int]=None, 
    cont_row_offset: int=0
):
    core.local_mem_page_read(
        ptr, container,
        row_size=row_size,
        row_num=row_num,
        mem_row_stride=mem_row_stride,
        cont_row_stride=cont_row_stride,
        row_pattern=row_pattern,
        cont_row_offset=cont_row_offset,
    )
    
    core.debug_core_with_ambiguous_func(lambda: logger.info(f"Data read into container:\n{container.data.reshape(-1, 8)}"))


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
    
    ptr1 = l1_mem_space.allocate(core.core_id, size=256)
    ptr2 = l1_mem_space.allocate(core.core_id, size=256)
    
    container_shape = (8, 32)
    container_dtype = torch.int8
    container = DataContainer()
    
    kernel1 = kernel_container_init(
        core,
        container,
        shape=container_shape,
        dtype=container_dtype,
    ).dispatch("MAIN")
    
    kernel2 = kernel_mem_init(
        core,
        ptr1,
        size=256,
        data=torch.arange(0, 256, dtype=container_dtype),
    ).dispatch("MAIN")
    
    kernel3 = kernel_mem_init(
        core,
        ptr2,
        size=256,
        data=torch.arange(256, 512, dtype=container_dtype),
    ).dispatch("MAIN")
    
    kernel4 = kernel_mem_copy(
        core,
        ptr1,
        container,
        row_size=8,
        row_num=8,
    ).dispatch("MAIN")
    
    kernel5 = kernel_mem_copy(
        core,
        ptr2,
        container,
        row_size=8,
        cont_row_stride=16,
        row_pattern={i+4: i for i in range(8)},
    ).dispatch("MAIN")
    
    device.run_kernels()
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()