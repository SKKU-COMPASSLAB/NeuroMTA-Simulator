import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.google_tpu.architecture import *
from neuromta.ip.google_tpu.runtime_kernel import *


__all__ = [
    "TPU_RT_LINEAR"
]


@MCA_RT_OPERATOR
def TPU_RT_LINEAR(
    device: GoogleTPUDevice, 
    core_id: int,
    
    buf_ifm:  MCA_TensorBuffer,
    buf_wgt:  MCA_TensorBuffer,
    buf_bias: MCA_TensorBuffer | None,
    buf_ofm:  MCA_TensorBuffer,
    
    dtype:      torch.dtype = torch.float32,
    acc_dtype:  torch.dtype = torch.float32, 
) -> None:
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
    
    k_tile_num = buf_ifm.y_n_pages
    m_tile_num = buf_ifm.x_n_pages
    n_tile_num = buf_wgt.x_n_pages
    
    # ofm_layout = MCA_TensorMemoryLayout(
    #     mem_type=MCA_TensorMemoryType.L1,
    #     page_shape=(buf_ifm.layout.y_page_size, buf_ifm.layout.x_page_size),
    # )
    
    # buf_ofm = MCA_TensorBuffer(shape=(N, M), dtype=acc_dtype, layout=ofm_layout, device=device, core_ids=[core_id,])
    
    if buf_ofm.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] Output feature map buffer must be allocated in L1 memory.")
    if buf_ofm.tensor_shape != (N, M):
        raise Exception(f"[ERROR] Output feature map buffer shape {buf_ofm.tensor_shape} does not match the expected shape {(N, M)}.")
    if buf_ofm.tensor_dtype != acc_dtype:
        raise Exception(f"[ERROR] Output feature map buffer dtype {buf_ofm.dtype} does not match the expected dtype {acc_dtype}.")
    
    TPU_RT_KERNEL_TILED_LINEAR_BURST_NKM(
        core = device.get_npu_core(core_id=core_id),
        
        buf_ifm  = buf_ifm.reference, 
        buf_wgt  = buf_wgt.reference, 
        buf_bias = None if buf_bias is None else buf_bias.reference,
        buf_ofm  = buf_ofm.reference,
        
        n_tile_num=n_tile_num, k_tile_num=k_tile_num, m_tile_num=m_tile_num, 
        dtype=dtype, acc_dtype=acc_dtype,
    )
    
    # return buf_ofm
