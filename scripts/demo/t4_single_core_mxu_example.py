import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import *


@jit_prototype
def main(
    core: NPUCore, 
    
    ifm_ptr:  Pointer,  # input features    -> shape: (M, K)
    wgt_ptr:  Pointer,  # weight parameters -> shape: (N, K) (transposed)
    bias_ptr: Pointer,  # bias parameters   -> shape: (N,)
    ofm_ptr:  Pointer,  # output features   -> shape: (M, N)
    
    M: int,  # number of IFM and OFM rows 
    N: int,  # number of WGT and OFM columns
    K: int,  # number of IFM columns and WGT rows
    
    dtype:      torch.dtype=torch.int32,
    acc_dtype:  torch.dtype=torch.int32,
):
    core.mxu_reconfigure(dtype=dtype, acc_dtype=acc_dtype)
    
    m_tile = math.ceil(M / 32)
    n_tile = math.ceil(N / 32)
    k_tile = math.ceil(K / 32)
    
    i_row_tile_size = 32 * dtype.itemsize
    o_row_tile_size = 32 * acc_dtype.itemsize
    
    i_row_stride = K * dtype.itemsize
    o_row_stride = N * acc_dtype.itemsize
    
    containers: list[DataContainer[torch.Tensor]] = [DataContainer() for _ in range(4)]
    
    for m_t in range(m_tile):
        for n_t in range(n_tile):
            bias_tile_ptr = bias_ptr + n_t * o_row_tile_size
            ofm_tile_ptr  = ofm_ptr + (m_t * 32 * o_row_stride) + (n_t * o_row_tile_size)
            
            core.local_mem_page_read(bias_tile_ptr, o_row_tile_size, containers[2])
            
            for k_t in range(k_tile):
                ifm_tile_ptr = ifm_ptr + (m_t * 32 * i_row_stride) + (k_t * i_row_tile_size)
                wgt_tile_ptr = wgt_ptr + (n_t * 32 * i_row_stride) + (k_t * i_row_tile_size)
                
                core.local_mem_page_read(ifm_tile_ptr, i_row_stride * 32, containers[0], mem_row_size=i_row_tile_size, mem_row_stride=i_row_stride)
                core.local_mem_page_read(wgt_tile_ptr, i_row_stride * 32, containers[1], mem_row_size=i_row_tile_size, mem_row_stride=i_row_stride)  # same access pattern with IFM
                
                preload_psum = (k_t == 0)
                flush_ofm    = (k_t == (k_tile - 1))
                
                core.mxu_tiled_gemm(
                    *containers,

                    preload_psum=preload_psum,
                    flush_ofm=flush_ofm,

                    wgt_transposed=True,    # TODO: assume that the weight matrix is transposed (for contiguous memory access)
                    psum_vectored=True,     # TODO: assume taht the psum is a bias vector, not the partial sum matrix
                )
            
            core.local_mem_page_write(ofm_tile_ptr, o_row_stride * 32, containers[3], mem_row_size=o_row_tile_size, mem_row_stride=o_row_stride)


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
    
    M, N, K = 128, 128, 128  # Small sizes for demo
    dtype = torch.int32
    acc_dtype = torch.int32
    
    ifm = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt = torch.arange(0, N * K, dtype=dtype).reshape(N, K)
    bias = torch.arange(0, N, dtype=acc_dtype)
    
    ifm_ptr  = l1_mem_space.allocate(core.core_id, size=ifm.numel() * dtype.itemsize)
    wgt_ptr  = l1_mem_space.allocate(core.core_id, size=wgt.numel() * dtype.itemsize)
    bias_ptr = l1_mem_space.allocate(core.core_id, size=bias.numel() * acc_dtype.itemsize)
    ofm_ptr  = l1_mem_space.allocate(core.core_id, size=M * N * acc_dtype.itemsize)
    
    device.mem_set_data(ifm_ptr,  ifm.numel()  * dtype.itemsize,     ifm)
    device.mem_set_data(wgt_ptr,  wgt.numel()  * dtype.itemsize,     wgt)
    device.mem_set_data(bias_ptr, bias.numel() * acc_dtype.itemsize, bias)
    
    kernel = main(
        core,
        ifm_ptr, wgt_ptr, bias_ptr, ofm_ptr,
        M, N, K,
        dtype=dtype,
        acc_dtype=acc_dtype,
    )
    
    kernel.dispatch("MAIN")
    device.run_kernels()
    
    simulated = device.mem_get_data(ofm_ptr, M * N * acc_dtype.itemsize, dtype=acc_dtype).reshape(M, N)
    reference = torch.matmul(ifm.to(acc_dtype), wgt.t().to(acc_dtype)) + bias.reshape(1, -1)
    
    print(f"simulated:\n{simulated}")
    print(f"reference:\n{reference}")
    print(f"simulation {'PASSED' if torch.equal(simulated, reference) else 'FAILED'}")
    
    device.remove_all_l1_mem_space()
    device.remove_all_main_mem_space()