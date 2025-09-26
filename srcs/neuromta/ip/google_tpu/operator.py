import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import *
from neuromta.ip.google_tpu.rt_kernel import *
from neuromta.ip.common.rt_kernel import *


@MCA_RT_OPERATOR
def TPU_RT_DMA_LOAD(device: GoogleTPUDevice, core_id: int, main_buf: MCA_TensorBuffer, l1_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    l1_buf = MCA_TensorBuffer(shape=main_buf.tensor_shape, dtype=main_buf.tensor_dtype, layout=l1_layout, device=device, core_ids=[core_id,])
    
    core = device.get_npu_core(core_id=core_id)
    with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
        core.mem_buffer_copy(l1_buf.reference, main_buf.reference, n_pages=main_buf.n_pages)
    return l1_buf
     
@MCA_RT_OPERATOR
def TPU_RT_DMA_STORE(device: GoogleTPUDevice, core_id: int, l1_buf: MCA_TensorBuffer, main_layout: MCA_TensorMemoryLayout) -> MCA_TensorBuffer:
    main_buf = MCA_TensorBuffer(shape=l1_buf.tensor_shape, dtype=l1_buf.tensor_dtype, layout=main_layout, device=device)
    
    core = device.get_npu_core(core_id=core_id)
    with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
        core.mem_buffer_copy(main_buf.reference, l1_buf.reference, n_pages=l1_buf.n_pages)        
    return main_buf

@MCA_RT_OPERATOR
def TPU_RT_LINEAR(
    device: GoogleTPUDevice, 
    core_id: int,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    
    dtype:      torch.dtype = torch.float32,
    acc_dtype:  torch.dtype = torch.float32, 
) -> MCA_TensorBuffer:
    K, M = buf_ifm.tensor_shape
    KW, N = buf_wgt.tensor_shape
    NB, _ = buf_bias.tensor_shape if buf_bias is not None else (N, 1)
    
    assert K == KW, f"[ERROR] The second dimension of input tensor (K={K}) must match the second dimension of weight tensor (KW={KW})."
    assert N == NB, f"[ERROR] The first dimension of weight tensor (N={N}) must match the first dimension of bias tensor (NB={NB})."
    
    if buf_ifm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Input feature map buffer must be allocated in L1 memory.")
    if buf_wgt.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Weight buffer must be allocated in L1 memory.")
    if buf_bias is not None and buf_bias.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Bias buffer must be allocated in L1 memory.")
    
    k_tile_num = buf_ifm.y_shard_grid * buf_ifm.y_page_num_per_shard
    m_tile_num = buf_ifm.x_shard_grid * buf_ifm.x_page_num_per_shard
    n_tile_num = buf_wgt.x_shard_grid * buf_wgt.x_page_num_per_shard
    
    ofm_layout = MCA_TensorMemoryLayout(
        mem_type=MCA_TensorMemoryType.L1,
        grid_shape=(1, 1),
        page_shape=(buf_ifm.layout.y_page_size, buf_ifm.layout.x_page_size),
    )
    
    buf_ofm = MCA_TensorBuffer(shape=(N, M), dtype=acc_dtype, layout=ofm_layout, device=device, core_ids=[core_id,])
    
    TPU_RT_KERNEL_TILED_LINEAR_BURST_NKM(
        core = device.get_npu_core(core_id=core_id),
        
        buf_ifm  = buf_ifm.reference, 
        buf_wgt  = buf_wgt.reference, 
        buf_bias = None if buf_bias is None else buf_bias.reference,
        buf_ofm  = buf_ofm.reference,
        
        n_tile_num=n_tile_num, k_tile_num=k_tile_num, m_tile_num=m_tile_num, 
        dtype=dtype, acc_dtype=acc_dtype,
    )
    
    return buf_ofm
