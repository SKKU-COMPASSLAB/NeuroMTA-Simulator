import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import GoogleTPUConfig, GoogleTPUDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".logs", FILENAME)


if __name__ == "__main__":
    config = GoogleTPUConfig.V4()
    
    device = GoogleTPUDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 128
    N = 128
    K = 128
    dtype = torch.int32
    acc_dtype = torch.int32
    
    ifm  = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt  = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum = torch.ones((M, N), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size = ifm.numel() * ifm.element_size()
    wgt_size = wgt.numel() * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size = ofm.numel() * ofm.element_size()
    
    ifm_ptr  = device.create_local_l1_buffer(page_size=ifm_size,  n_pages=1, core_ids=device.npu_core_ids[0])
    wgt_ptr  = device.create_local_l1_buffer(page_size=wgt_size,  n_pages=1, core_ids=device.npu_core_ids[0])
    psum_ptr = device.create_local_l1_buffer(page_size=psum_size, n_pages=1, core_ids=device.npu_core_ids[0])
    ofm_ptr  = device.create_local_l1_buffer(page_size=ofm_size,  n_pages=1, core_ids=device.npu_core_ids[0])

    device.set_ptr_content(ifm_ptr, ifm)
    device.set_ptr_content(wgt_ptr, wgt)
    device.set_ptr_content(psum_ptr, psum)
    device.set_ptr_content(ofm_ptr, ofm)

    @core_kernel_method
    def gemm_kernel(
        core: NPUCore,    
        
        ifm_ptr:  Reference,
        wgt_ptr:  Reference,
        psum_ptr: Reference,
        ofm_ptr:  Reference
    ):  
        core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
        core.mxu_tiled_gemm(
            ifm_ptr=ifm_ptr,
            wgt_ptr=wgt_ptr,
            psum_ptr=psum_ptr,
            ofm_ptr=ofm_ptr,
            preload_wgt=True,
            preload_psum=False,
            flush_ofm=True
        )
        
        core.vpu_reconfigure(vlen=128, vdtype=dtype)
        core.vpu_load_reg(ifm_ptr, 0, 0, 4)
        core.vpu_load_reg(wgt_ptr, 0, 4, 4)
        core.vpu_execute(VPUOperator.ADD, 0, 4, 8, inplace=False, burst_len=4)
        core.vpu_store_reg(ofm_ptr, 0, 8, 4)
        
    kernel = gemm_kernel(
        device.npu_cores[0],
        ifm_ptr=ifm_ptr,
        wgt_ptr=wgt_ptr,
        psum_ptr=psum_ptr,
        ofm_ptr=ofm_ptr
    )
    device.npu_cores[0].dispatch_main_kernel("compute", kernel=kernel)

    st = time.time()
    device.run_kernels(verbose=True, max_steps=-1, save_trace=True, save_trace_dir=TRACE_DIR)
    ed = time.time()
    
    print(f"\nkernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    
    reference = torch.matmul(ifm, wgt) + psum
    reference[0:4, :] = ifm[0:4, :] + wgt[0:4, :]  # Simulate the effect of the VPU operation
    simulated = device.get_ptr_content(ofm_ptr, shape=(M, N), dtype=acc_dtype)

    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nsimulation terminated with valid result: {torch.allclose(reference, simulated)}")