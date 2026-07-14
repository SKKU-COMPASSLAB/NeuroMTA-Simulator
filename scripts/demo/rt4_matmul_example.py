import math
import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import *


def check_valid(container: DataContainer[torch.Tensor], actual_data: torch.Tensor, coord: tuple[int, ...], mem_params: tuple[int, ...]):
    shape = actual_data.shape
    dtype = actual_data.dtype
    
    sim_data = container.data.flatten().view(dtype).reshape(shape)
    
    if torch.equal(sim_data, actual_data):
        logger.info(f"Data validation PASSED for coordinate {coord} - mem {mem_params}.")
    else:
        logger.error(f"Data validation FAILED for coordinate {coord} - mem {mem_params}!")
        logger.debug(f"Expected data:\n{actual_data}")
        logger.debug(f"Simulated data:\n{sim_data}")


@jit_prototype
def main(core: NPUCore, ifm: MCA_TensorBuffer, wgt: MCA_TensorBuffer, ofm: MCA_TensorBuffer, ifm_raw: torch.Tensor, wgt_raw: torch.Tensor, ifm_l1_ptr: Pointer, wgt_l1_ptr: Pointer, ofm_l1_ptr: Pointer):
    M, K = ifm.shape
    N, _ = wgt.shape
    
    Ms, Ks = ifm.shard_grid
    Ns, _ = wgt.shard_grid
    
    Mt, Kt = ifm.tile_grid_per_shard
    Nt, _ = wgt.tile_grid_per_shard
    
    containers = [DataContainer() for _ in range(4)]
    
    core.mxu_reconfigure(dtype=ifm.dtype, acc_dtype=ofm.dtype)
    
    ifm_raw_sharded = ifm_raw.reshape(Ms, M//Ms, Ks, K//Ks).permute(0, 2, 1, 3).reshape(Ms, Ks, M//Ms, K//Ks)
    wgt_raw_sharded = wgt_raw.reshape(Ns, N//Ns, Ks, K//Ks).permute(0, 2, 1, 3).reshape(Ns, Ks, N//Ns, K//Ks)
    
    ifm_mt_pad = math.ceil((M//Ms) / 32) * 32 - (M//Ms)
    ifm_kt_pad = math.ceil((K//Ks) / 32) * 32 - (K//Ks)
    wgt_nt_pad = math.ceil((N//Ns) / 32) * 32 - (N//Ns)
    wgt_kt_pad = math.ceil((K//Ks) / 32) * 32 - (K//Ks)
    
    ifm_raw_sharded_padded = torch.nn.functional.pad(ifm_raw_sharded, (0, ifm_kt_pad, 0, ifm_mt_pad), mode='constant', value=0)
    wgt_raw_sharded_padded = torch.nn.functional.pad(wgt_raw_sharded, (0, wgt_kt_pad, 0, wgt_nt_pad), mode='constant', value=0)
    
    for m_s in range(Ms):
        for m_t in range(Mt):
            for n_s in range(Ns):
                for n_t in range(Nt):
                    for k_s in range(Ks):
                        for k_t in range(Kt):
                            for c in containers:
                                core.local_data_container_init(c, shape=(32, 32), dtype=torch.int32)
                            
                            core.local_mem_init(ifm_l1_ptr, size=ifm.tile_size)
                            core.local_mem_init(wgt_l1_ptr, size=wgt.tile_size)
                            
                            ifm_src_ptr, ifm_row_size, ifm_row_num, ifm_src_row_stride, ifm_dst_row_stride = ifm.get_tile_ptr_read_args(m_s, k_s, m_t, k_t)
                            wgt_src_ptr, wgt_row_size, wgt_row_num, wgt_src_row_stride, wgt_dst_row_stride = wgt.get_tile_ptr_read_args(n_s, k_s, n_t, k_t)
                            
                            core.local_mem_copy(ifm_l1_ptr, ifm_src_ptr, row_size=ifm_row_size, row_num=ifm_row_num, src_row_stride=ifm_src_row_stride, dst_row_stride=ifm_dst_row_stride, nowait=False)
                            core.local_mem_copy(wgt_l1_ptr, wgt_src_ptr, row_size=wgt_row_size, row_num=wgt_row_num, src_row_stride=wgt_src_row_stride, dst_row_stride=wgt_dst_row_stride, nowait=False)
                            
                            core.local_mem_page_read(ifm_l1_ptr, containers[0], row_size=32*ifm.dtype.itemsize, row_num=32)
                            core.local_mem_page_read(wgt_l1_ptr, containers[1], row_size=32*wgt.dtype.itemsize, row_num=32)
                            
                            ifm_raw_tile = ifm_raw_sharded_padded[m_s, k_s, m_t*32:(m_t+1)*32, k_t*32:(k_t+1)*32]
                            wgt_raw_tile = wgt_raw_sharded_padded[n_s, k_s, n_t*32:(n_t+1)*32, k_t*32:(k_t+1)*32]
                            
                            core.debug_core_with_ambiguous_func(check_valid, containers[0], ifm_raw_tile, (m_s, k_s, m_t, k_t), (ifm_l1_ptr.addr, ifm_src_ptr.addr, ifm_row_size, ifm_row_num, ifm_src_row_stride, ifm_dst_row_stride))
                            core.debug_core_with_ambiguous_func(check_valid, containers[1], wgt_raw_tile, (n_s, k_s, n_t, k_t), (wgt_l1_ptr.addr, wgt_src_ptr.addr, wgt_row_size, wgt_row_num, wgt_src_row_stride, wgt_dst_row_stride))
                            
                            flush_ofm = (k_s == Ks-1 and k_t == Kt-1)
                            
                            core.mxu_tiled_gemm(
                                *containers,
                                preload_psum=False,
                                flush_ofm=flush_ofm,
                                wgt_transposed=True,
                            )
                            
                            if flush_ofm:
                                core.local_mem_page_write(ofm_l1_ptr, containers[3], row_size=32*ofm.dtype.itemsize, row_num=32)
                                
                                ofm_dst_ptr, ofm_row_size, ofm_row_num, ofm_src_row_stride, ofm_dst_row_stride = ofm.get_tile_ptr_write_args(m_s, n_s, m_t, n_t)
                                core.local_mem_copy(ofm_dst_ptr, ofm_l1_ptr, row_size=ofm_row_size, row_num=ofm_row_num, src_row_stride=ofm_src_row_stride, dst_row_stride=ofm_dst_row_stride, nowait=False)
                                

if __name__ == "__main__":
    torch.set_printoptions(linewidth=1024)
    logger.set_print_options(log_level=LogLevel.DEBUG)
    
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize()
    device.set_command_debug_verbosity(verbose=False)
    
    core_group = device.get_npu_core_group((0, 0), (2, 2))
    core_id = core_group[0, 0]
    core = device.get_npu_core(core_id=core_id)
            
    M, N, K = 100, 100, 100
    dtype = torch.int32
    acc_dtype = torch.int32
    
    ifm = torch.randint(low=0, high=10, size=(M, K), dtype=dtype)
    wgt = torch.randint(low=0, high=10, size=(N, K), dtype=dtype)
    ofm = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size  = ifm.numel() * ifm.dtype.itemsize
    wgt_size  = wgt.numel() * wgt.dtype.itemsize
    ofm_size  = ofm.numel() * ofm.dtype.itemsize
    
    l1_mem_space   = device.create_l1_mem_space(size_per_bank=parse_mem_cap_str("1MB"), core_group=core_group)
    main_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
    
    ifm_b  = MCA_TensorBuffer(mem_space=l1_mem_space,   shape=ifm.shape,  dtype=ifm.dtype,  shard_shape=(50, 50), blocked_mapping=True).allocate().tiling((32, 32)).update(ifm)
    wgt_b  = MCA_TensorBuffer(mem_space=main_mem_space, shape=wgt.shape,  dtype=wgt.dtype,  shard_shape=(50, 50)                      ).allocate().tiling((32, 32)).update(wgt)
    ofm_b  = MCA_TensorBuffer(mem_space=l1_mem_space,   shape=ofm.shape,  dtype=ofm.dtype,  shard_shape=(50, 50), blocked_mapping=True).allocate().tiling((32, 32))
    
    ifm_l1_ptr = l1_mem_space.allocate(core_id, parse_mem_cap_str("4KB"))
    wgt_l1_ptr = l1_mem_space.allocate(core_id, parse_mem_cap_str("4KB"))
    ofm_l1_ptr = l1_mem_space.allocate(core_id, parse_mem_cap_str("4KB"))

    kernel = main(core, ifm_b, wgt_b, ofm_b, ifm, wgt, ifm_l1_ptr, wgt_l1_ptr, ofm_l1_ptr)
    kernel.dispatch("MAIN")
    
    device.run_kernels()
    
    reference = torch.matmul(ifm.to(dtype=acc_dtype), wgt.t().to(dtype=acc_dtype))
    simulated = ofm_b.restore()
    
    print("reference:")
    print(reference)
    print("\nsimulated:")
    print(simulated)
    print(f"\nsimuation {'PASSED' if torch.equal(reference, simulated) else 'FAILED'}")