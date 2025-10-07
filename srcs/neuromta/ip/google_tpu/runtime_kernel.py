import torch

from neuromta.framework import *
from neuromta.hardware import *


__all__ = [
    "TPU_RT_KERNEL_TILED_LINEAR_BURST_NKM",
]


@MCA_RT_KERNEL
def  TPU_RT_KERNEL_TILED_LINEAR_BURST_NKM(
    core: NPUCore,
        
    buf_ifm:  BufferPointer, 
    buf_wgt:  BufferPointer, 
    buf_bias: BufferPointer | None,
    buf_ofm:  BufferPointer,
    
    # Parameters
    n_tile_num: int, k_tile_num: int, m_tile_num: int, 
    dtype: torch.dtype, acc_dtype: torch.dtype,
):
    containers = [DataContainer() for _ in range(4)]

    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)

    for n_it in range(n_tile_num):
        for k_it in range(k_tile_num):
            for m_it in range(m_tile_num):
                if k_it == 0:
                    if buf_bias is not None:
                        core.mem_read_with_container(buf_bias[n_it], containers[2])
                    else:
                        core.mem_container_init(containers[2], shape=(1, 128), dtype=acc_dtype)
                else:
                    core.mem_read_with_container(buf_ofm[n_it * m_tile_num + m_it], containers[2])
                
                core.mem_read_with_container(buf_ifm[k_it * m_tile_num + m_it], containers[0])
                core.mem_read_with_container(buf_wgt[k_it * n_tile_num + n_it], containers[1])
                
                core.mxu_tiled_gemm(
                    *containers, 
                    preload_wgt=(m_it == 0),
                    preload_psum=False,
                    flush_ofm=True,
                    ifm_transposed=True,
                    psum_vectored=(k_it == 0),
                )
                
                core.mem_write_with_container(buf_ofm[n_it * m_tile_num + m_it], containers[3])
