import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "TT_RT_KERNEL_TILED_LINEAR_BURST_K",
]


@MCA_RT_KERNEL
def TT_RT_KERNEL_TILED_LINEAR_BURST_K(
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
    k_tile_num: int, load_burst_len: int, 
    dtype: torch.dtype, acc_dtype: torch.dtype,
):
    
    def _pipeline_load_step(pipe_in: BufferPointer, cb: BufferPointer, st: int, n_pages: int):
        ed = st + n_pages
        
        if pipe_in.is_circular:
            core.cb_wait_front(pipe_in, n_pages)
            pipe_st, pipe_ed = 0, n_pages
        else:
            pipe_st, pipe_ed = st, ed
        
        core.cb_reserve_back(cb, n_pages)
        core.mem_buffer_copy(cb[0:n_pages], pipe_in[pipe_st:pipe_ed], n_pages)
        core.cb_push_back(cb, n_pages)
        
        if pipe_in.is_circular:
            core.cb_pop_front(pipe_in, n_pages)

    def RD_THREAD(
        buf_ifm:  BufferPointer, 
        buf_wgt:  BufferPointer, 
        buf_bias: BufferPointer | None,
        
        cb_ifm:  BufferPointer, 
        cb_wgt:  BufferPointer, 
        cb_bias: BufferPointer | None, 
        
        k_tile_num: int, load_burst_len: int, 
    ):
        if buf_bias is not None:
            with new_parallel_thread("LD_BIAS"): _pipeline_load_step(buf_bias, cb_bias, 0, 1)
        
        for st in range(0, k_tile_num, load_burst_len):
            ed = min(st + load_burst_len, k_tile_num)
            n_pages = ed - st

            with new_parallel_thread("LD_IFM"): _pipeline_load_step(buf_ifm, cb_ifm, st, n_pages)
            with new_parallel_thread("LD_WGT"): _pipeline_load_step(buf_wgt, cb_wgt, st, n_pages)

            core.parallel_merge()
    
    def EX_THREAD(
        cb_ifm:  BufferPointer, 
        cb_wgt:  BufferPointer, 
        cb_bias: BufferPointer | None, 
        cb_ofm:  BufferPointer,
        
        k_tile_num: int,
        
        dtype: torch.dtype, 
        acc_dtype: torch.dtype,
    ):
        containers = [DataContainer() for _ in range(4)]
        
        core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)

        core.cb_wait_front(cb_bias, 1)
        core.cb_reserve_back(cb_ofm, 1)
        
        core.mem_read_with_container(cb_bias[0], containers[2])
        
        for k_it in range(k_tile_num):
            preload_psum = True if k_it == 0 else False
            flush_ofm    = True if (k_it == (k_tile_num - 1)) else False
            
            core.cb_wait_front(cb_ifm, 1)
            core.cb_wait_front(cb_wgt, 1)
            
            core.mem_read_with_container(cb_ifm[0], containers[0])
            core.mem_read_with_container(cb_wgt[0], containers[1])
            
            core.mxu_tiled_gemm(
                *containers,
                preload_psum=preload_psum,
                flush_ofm=flush_ofm,
                wgt_transposed=True,    # TODO: assume that the weight matrix is transposed (for contiguous memory access)
                psum_vectored=True,     # TODO: assume taht the psum is a bias vector, not the partial sum matrix
            )
            
            core.cb_pop_front(cb_ifm, 1)
            core.cb_pop_front(cb_wgt, 1)
        
        core.mem_write_with_container(cb_ofm[0], containers[3])
        
        core.cb_pop_front(cb_bias, 1)
        core.cb_push_back(cb_ofm, 1)
    
    def WR_THREAD(
        buf_ofm:  BufferPointer,
        cb_ofm:  BufferPointer,
    ):
        core.cb_wait_front(cb_ofm, 1)
        core.mem_buffer_copy(buf_ofm[0], cb_ofm[0], n_pages=1)
        core.cb_pop_front(cb_ofm, 1)
        
        
    with new_parallel_thread("RD_THREAD"): RD_THREAD(buf_ifm, buf_wgt, buf_bias, cb_ifm, cb_wgt, cb_bias, k_tile_num, load_burst_len)
    with new_parallel_thread("EX_THREAD"): EX_THREAD(cb_ifm, cb_wgt, cb_bias, cb_ofm, k_tile_num, dtype, acc_dtype)
    with new_parallel_thread("WR_THREAD"): WR_THREAD(buf_ofm, cb_ofm)
    
    core.parallel_merge()