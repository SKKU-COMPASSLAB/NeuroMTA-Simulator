import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.mta.tenstorrent import *


@jit_prototype
def dma_load_kernel(core: NPUCore, n_pages: int, load_var: Pointer, main_in: BufferPointer, l1_in: BufferPointer):
    for i in range(n_pages):
        core.mem_buffer_copy(l1_in[i], main_in[i], 1)
        core.var_atomic_increase(load_var, 1)
        
@jit_prototype
def compute_kernel(core: NPUCore, n_pages: int, load_var: Pointer, store_var: Pointer, l1_in: BufferPointer, l1_out: BufferPointer):
    container = DataContainer()
    
    core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
    
    for i in range(n_pages):
        core.var_wait_value(load_var, i+1)

        core.mem_read_with_container(l1_in[i], container)
        core.mxu_tiled_elemwise(MXUElementwiseOp.ADD, container, container, preload_psum=True, flush_ofm=True)
        core.mem_write_with_container(l1_out[i], container)

        core.var_atomic_increase(store_var, 1)
        
@jit_prototype
def dma_store_kernel(core: NPUCore, n_pages: int, store_var: Pointer, l1_out: BufferPointer, main_out: BufferPointer):
    for i in range(n_pages):
        core.var_wait_value(store_var, i+1)
        
        core.mem_buffer_copy(main_out[i], l1_out[i], 1)


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
    
    core_id = device.npu_core_ids[0]
    
    load_var    = device.create_local_variable(4, 0, core_ids=[core_id])
    store_var   = device.create_local_variable(4, 0, core_ids=[core_id])
    
    main_in  = device.create_sharded_main_buffer(page_size=page_size, n_pages=n_pages)
    main_out = device.create_sharded_main_buffer(page_size=page_size, n_pages=n_pages)
    
    l1_in  = device.create_local_l1_buffer(page_size=page_size, n_pages=n_pages, core_ids=[core_id])
    l1_out = device.create_local_l1_buffer(page_size=page_size, n_pages=n_pages, core_ids=[core_id])
    
    matrix = torch.arange(n_matrix * matrix_dim * matrix_dim, dtype=matrix_dtype).reshape(n_matrix, matrix_dim, matrix_dim)
    
    device.set_ptr_content(main_in, matrix)
    
    npu_core = device.get_npu_core(core_id)

    kernel1 = dma_load_kernel(npu_core, n_pages, load_var, main_in, l1_in)
    kernel2 = compute_kernel(npu_core, n_pages, load_var, store_var, l1_in, l1_out)
    kernel3 = dma_store_kernel(npu_core, n_pages, store_var, l1_out, main_out)
    
    npu_core.dispatch_main_kernel("dma_load", kernel1)
    npu_core.dispatch_main_kernel("compute", kernel2)
    npu_core.dispatch_main_kernel("dma_store", kernel3)
    
    device.run_kernels()
    
    result = device.get_ptr_content(main_out, shape=(n_matrix, -1), dtype=matrix_dtype)
    
    print(result)
