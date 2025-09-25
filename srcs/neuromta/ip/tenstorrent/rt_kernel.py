import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "TT_RT_KERNEL_TILED_LINEAR_BURST_K",
]


class TT_RT_KERNEL_TILED_LINEAR_BURST_K(MCA_RuntimeKernel):
    def __init__(
        self,
        
        core: NPUCore,
        
        buf_ifm:  BufferPointer, 
        buf_wgt:  BufferPointer, 
        buf_bias: BufferPointer | None,
        buf_ofm:  BufferPointer,
        
        # Intra-core communication interfaces
        cb_ifm:  BufferPointer, 
        cb_wgt:  BufferPointer, 
        cb_bias: BufferPointer | None, 
        cb_ofm:  BufferPointer,
        
        # Parameters
        k_tile_num: int, load_burst_len: int, dtype: torch.dtype, acc_dtype: torch.dtype,
    ):
        super().__init__(core=core)
        
        self.buf_ifm  = buf_ifm
        self.buf_wgt  = buf_wgt
        self.buf_bias = buf_bias
        self.buf_ofm  = buf_ofm
        
        self.cb_ifm = cb_ifm
        self.cb_wgt = cb_wgt
        self.cb_psum = cb_bias
        self.cb_ofm = cb_ofm
        
        self.k_tile_num = k_tile_num
        self.load_burst_len = load_burst_len
        
        self.dtype = dtype
        self.acc_dtype = acc_dtype
    
    def _pipeline_load_step(self, pipe_in: BufferPointer, cb: BufferPointer, st: int, n_pages: int):
        ed = st + n_pages
        
        if pipe_in.is_circular:
            self.core.cb_wait_front(pipe_in, n_pages)
            pipe_st, pipe_ed = 0, n_pages
        else:
            pipe_st, pipe_ed = st, ed
        
        self.core.cb_reserve_back(cb, n_pages)
        self.core.mem_buffer_copy(cb[0:n_pages], pipe_in[pipe_st:pipe_ed], n_pages)
        self.core.cb_push_back(cb, n_pages)
        
        if pipe_in.is_circular:
            self.core.cb_pop_front(pipe_in, n_pages)

    @MCA_RT_KERNEL_THREAD
    def RD_THREAD(self):
        if self.buf_bias is not None:
            with new_parallel_thread("LD_BIAS"): self._pipeline_load_step(self.buf_bias, self.cb_psum, 0, 1)
        
        for st in range(0, self.k_tile_num, self.load_burst_len):
            ed = min(st + self.load_burst_len, self.k_tile_num)
            n_pages = ed - st

            with new_parallel_thread("LD_IFM"): self._pipeline_load_step(self.buf_ifm, self.cb_ifm, st, n_pages)
            with new_parallel_thread("LD_WGT"): self._pipeline_load_step(self.buf_wgt, self.cb_wgt, st, n_pages)

            self.core.parallel_merge()
    
    @MCA_RT_KERNEL_THREAD
    def EX_THREAD(self):
        containers = [DataContainer() for _ in range(4)]
        
        self.core.mxu_reconfigure(dtype=self.dtype, acc_dtype=self.acc_dtype)

        self.core.cb_wait_front(self.cb_psum, 1)
        self.core.cb_reserve_back(self.cb_ofm, 1)
        
        self.core.mem_read_with_container(self.cb_psum[0], containers[2])
        
        for k_it in range(self.k_tile_num):
            preload_psum = True if k_it == 0 else False
            flush_ofm    = True if (k_it == (self.k_tile_num - 1)) else False
            
            self.core.cb_wait_front(self.cb_ifm, 1)
            self.core.cb_wait_front(self.cb_wgt, 1)
            
            self.core.mem_read_with_container(self.cb_ifm[0], containers[0])
            self.core.mem_read_with_container(self.cb_wgt[0], containers[1])
            
            self.core.mxu_tiled_gemm(
                *containers,
                preload_psum=preload_psum,
                flush_ofm=flush_ofm,
                wgt_transposed=True,    # TODO: assume that the weight matrix is transposed (for contiguous memory access)
                psum_vectored=True,     # TODO: assume taht the psum is a bias vector, not the partial sum matrix
            )
            
            self.core.cb_pop_front(self.cb_ifm, 1)
            self.core.cb_pop_front(self.cb_wgt, 1)
        
        self.core.mem_write_with_container(self.cb_ofm[0], containers[3])
        
        self.core.cb_pop_front(self.cb_psum, 1)
        self.core.cb_push_back(self.cb_ofm, 1)
    
    @MCA_RT_KERNEL_THREAD
    def WR_THREAD(self):
        self.core.cb_wait_front(self.cb_ofm, 1)
        self.core.mem_buffer_copy(self.buf_ofm[0], self.cb_ofm[0], n_pages=1)
        self.core.cb_pop_front(self.cb_ofm, 1)