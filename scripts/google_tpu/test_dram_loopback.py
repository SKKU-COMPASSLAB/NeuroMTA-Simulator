import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import GoogleTPUConfig, GoogleTPUDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".logs", FILENAME)


@core_kernel_method
def kernel_main(core: NPUCore, main_in_ptr: Reference, l1_ptr: Reference, main_out_ptr: Reference, n_pages: int):
    for i in range(n_pages):
        core.async_buffer_read(l1_ptr[0], main_in_ptr[i])
        core.async_buffer_write(main_out_ptr[i], l1_ptr[0])


if __name__ == "__main__":
    config = GoogleTPUConfig.V4()
    
    device = GoogleTPUDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    page_size = 32 * 32 * 4
    n_pages = 4
    dtype = torch.int32
    
    npu_core = device.npu_cores[0]
    
    main_in_ptr     = device.create_sharded_main_buffer(page_size, n_pages)
    main_out_ptr    = device.create_sharded_main_buffer(page_size, n_pages)
    l1_ptr          = device.create_local_l1_buffer(page_size, 1, core_ids=npu_core.core_id)

    for i in range(n_pages):
        content = torch.zeros(page_size // dtype.itemsize, dtype=dtype).fill_(i+1)
        device.set_ptr_content(main_in_ptr[i], content)
    
    kernel = kernel_main(npu_core, main_in_ptr, l1_ptr, main_out_ptr, n_pages=n_pages)
    npu_core.dispatch_main_kernel("compute", kernel=kernel)
    
    st = time.time()
    device.run_kernels(verbose=True, max_steps=-1, save_trace=True, save_trace_dir=TRACE_DIR)
    ed = time.time()
    
    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    print(f"\n=== INPUT  BUFFER ===\n{device.get_ptr_content(main_in_ptr[0], shape=(-1,), dtype=dtype)}")
    print(f"\n=== OUTPUT BUFFER ===\n{device.get_ptr_content(main_out_ptr[0], shape=(-1,), dtype=dtype)}")
