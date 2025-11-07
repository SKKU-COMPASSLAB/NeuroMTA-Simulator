import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def core_0_read(core: NPUCore, n_pages: int, local_l1_load_sem: Pointer, remote_main: BufferPointer, local_l1: BufferPointer):
    for i in range(n_pages):
        core.mem_buffer_copy(local_l1[i], remote_main[i], 1)
        core.var_atomic_increase(local_l1_load_sem, 1)
        
@jit_prototype
def core_1_read(core: NPUCore, n_pages: int, remote_l1_load_sem: Pointer, local_l1_load_sem: Pointer, remote_l1: BufferPointer, local_l1: BufferPointer):
    for i in range(n_pages):
        core.var_wait_value(remote_l1_load_sem, i+1)
        core.mem_buffer_copy(local_l1[i], remote_l1[i], 1)
        core.var_atomic_increase(local_l1_load_sem, 1)
        
@jit_prototype
def core_1_compute(core: NPUCore, n_pages: int, local_l1_in_sem: Pointer, local_l1_out_sem: Pointer, local_l1_in: BufferPointer, load_l1_out: BufferPointer):
    container = DataContainer()
    
    core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
    
    for i in range(n_pages):
        core.var_wait_value(local_l1_in_sem, i+1)

        core.mem_read_with_container(local_l1_in[i], container)
        core.mxu_tiled_elemwise(MXUElementwiseOp.ADD, container, container, preload_psum=True, flush_ofm=True)
        core.mem_write_with_container(load_l1_out[i], container)

        core.var_atomic_increase(local_l1_out_sem, 1)
        
@jit_prototype
def core_1_write(core: NPUCore, n_pages: int, local_l1_sem: Pointer, local_l1: BufferPointer, remote_main: BufferPointer):
    for i in range(n_pages):
        core.var_wait_value(local_l1_sem, i+1)
        
        core.mem_buffer_copy(remote_main[i], local_l1[i], 1)


if __name__ == "__main__":
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(True)
    
    n_matrix = 4
    matrix_dim = 32
    matrix_dtype = torch.int32
    
    page_size = matrix_dim * matrix_dim * matrix_dtype.itemsize
    n_pages = n_matrix
    
    core_ids = device.npu_core_ids[:2]
    
    core0_l1_in_sem = device.create_local_variable(1, 0, core_ids=[core_ids[0]])
    core1_l1_in_sem = device.create_local_variable(1, 0, core_ids=[core_ids[1]])
    core1_l1_out_sem = device.create_local_variable(1, 0, core_ids=[core_ids[1]])
    
    core0_l1_in = device.create_local_l1_buffer(page_size=page_size, n_pages=n_pages, core_ids=[core_ids[0]])
    core1_l1_in = device.create_local_l1_buffer(page_size=page_size, n_pages=n_pages, core_ids=[core_ids[1]])
    core1_l1_out = device.create_local_l1_buffer(page_size=page_size, n_pages=n_pages, core_ids=[core_ids[1]])
    
    main_in  = device.create_sharded_main_buffer(page_size=page_size, n_pages=n_pages)
    main_out = device.create_sharded_main_buffer(page_size=page_size, n_pages=n_pages)
    
    matrix = torch.arange(n_matrix * matrix_dim * matrix_dim, dtype=matrix_dtype).reshape(n_matrix, matrix_dim, matrix_dim)
    device.set_ptr_content(main_in, matrix)
    
    npu_core_0 = device.get_npu_core(core_ids[0])
    npu_core_1 = device.get_npu_core(core_ids[1])
    
    kernel1 = core_0_read   (npu_core_0, n_pages, core0_l1_in_sem,  main_in,            core0_l1_in)
    kernel2 = core_1_read   (npu_core_1, n_pages, core0_l1_in_sem,  core1_l1_in_sem,    core0_l1_in, core1_l1_in)
    kernel3 = core_1_compute(npu_core_1, n_pages, core1_l1_in_sem,  core1_l1_out_sem,   core1_l1_in, core1_l1_out)
    kernel4 = core_1_write  (npu_core_1, n_pages, core1_l1_out_sem, core1_l1_out,       main_out)
    
    kernel1.dispatch(slot_id="dma_load")
    kernel2.dispatch(slot_id="dma_load")
    kernel3.dispatch(slot_id="compute")
    kernel4.dispatch(slot_id="dma_store")

    device.run_kernels()
    
    result = device.get_ptr_content(main_out, shape=(n_matrix, -1), dtype=matrix_dtype)
    
    print(result)
