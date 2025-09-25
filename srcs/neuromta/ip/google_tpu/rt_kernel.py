import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *


class TPU_RT_KERNEL_TILED_LINEAR_BURST_NKM(MCA_RuntimeKernel):
    def __init__(
        self,
        
        core: NPUCore,
        
        buf_ifm:  BufferPointer, 
        buf_wgt:  BufferPointer, 
        buf_bias: BufferPointer | None,
        buf_ofm:  BufferPointer,
        
        # Parameters
        n_tile_num: int, k_tile_num: int, m_tile_num: int, 
        dtype: torch.dtype, acc_dtype: torch.dtype,
    ):
        super().__init__(core=core)
        
        self.buf_ifm  = buf_ifm
        self.buf_wgt  = buf_wgt
        self.buf_bias = buf_bias
        self.buf_ofm  = buf_ofm
        
        self.n_tile_num = n_tile_num
        self.k_tile_num = k_tile_num
        self.m_tile_num = m_tile_num
        
        self.dtype = dtype
        self.acc_dtype = acc_dtype
    
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        containers = [DataContainer() for _ in range(4)]
        
        self.core.mxu_reconfigure(dtype=self.dtype, acc_dtype=self.acc_dtype)
        
        for n_it in range(self.n_tile_num):
            for k_it in range(self.k_tile_num):
                for m_it in range(self.m_tile_num):
                    if k_it == 0:
                        if self.buf_bias is not None:
                            self.core.mem_read_with_container(self.buf_bias[n_it], containers[2])
                        else:
                            self.core.init_container(containers[2], shape=(1, 128), dtype=self.acc_dtype)
                    else:
                        self.core.mem_read_with_container(self.buf_ofm[n_it * self.m_tile_num + m_it], containers[2])
                    
                    self.core.mem_read_with_container(self.buf_ifm[k_it * self.m_tile_num + m_it], containers[0])
                    self.core.mem_read_with_container(self.buf_wgt[k_it * self.n_tile_num + n_it], containers[1])
                    
                    self.core.mxu_tiled_gemm(
                        *containers, 
                        preload_wgt=(m_it == 0),
                        preload_psum=False,
                        flush_ofm=True,
                        ifm_transposed=True,
                        psum_vectored=(k_it == 0),
                    )
                    
                    self.core.mem_write_with_container(self.buf_ofm[n_it * self.m_tile_num + m_it], containers[3])
