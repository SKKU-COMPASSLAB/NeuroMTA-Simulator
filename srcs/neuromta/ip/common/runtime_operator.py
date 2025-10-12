import torch

from neuromta.framework import *
from neuromta.hardware import *


__all__ = [
    "MCA_RT_GLOBAL_SYNC",
    "MCA_RT_DMA_LOAD",
    "MCA_RT_DMA_STORE",
]


@MCA_RT_OPERATOR
def MCA_RT_GLOBAL_SYNC(device: MCA_DeviceBase, core_ids: list[int]):
    for core_id in core_ids:
        core = device.get_npu_core(core_id=core_id)
        with MCA_RT_JIT_COMPILE_REGION(core, "BARRIER"):
            core.inter_core_sync_barrier(core_ids)
            

@MCA_RT_OPERATOR
def MCA_RT_DMA_LOAD(device: MCA_DeviceBase, src_buf: MCA_TensorBuffer, dst_buf: MCA_TensorBuffer):
    if src_buf.layout.mem_type == MCA_TensorMemoryType.L1:
        raise Exception(f"[ERROR] Main buffer must be allocated in MAIN memory.")
    if dst_buf.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] L1 buffer must be allocated in L1 memory.")
    if src_buf.tensor_shape != dst_buf.tensor_shape:
        raise Exception(f"[ERROR] The shape of main buffer {src_buf.tensor_shape} does not match the shape of L1 buffer {dst_buf.tensor_shape}.")
    
    for core_id in dst_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)

        with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
            page_indice = dst_buf.get_page_idx_by_owner(core.core_id)
            buf_src = src_buf.get_reference_by_page_idx(*page_indice)
            buf_dst = dst_buf.get_reference_by_page_idx(*page_indice)

            core.mem_buffer_copy(buf_dst, buf_src, n_pages=len(page_indice))


@MCA_RT_OPERATOR
def MCA_RT_DMA_STORE(device: MCA_DeviceBase, src_buf: MCA_TensorBuffer, dst_buf: MCA_TensorBuffer):
    if dst_buf.layout.mem_type == MCA_TensorMemoryType.L1:
        raise Exception(f"[ERROR] Main buffer must be allocated in MAIN memory.")
    if src_buf.layout.mem_type == MCA_TensorMemoryType.MAIN:
        raise Exception(f"[ERROR] L1 buffer must be allocated in L1 memory.")
    if dst_buf.tensor_shape != src_buf.tensor_shape:
        raise Exception(f"[ERROR] The shape of main buffer {dst_buf.tensor_shape} does not match the shape of L1 buffer {src_buf.tensor_shape}.")

    for core_id in src_buf.core_ids:
        core = device.get_npu_core(core_id=core_id)
        
        with MCA_RT_JIT_COMPILE_REGION(core, "MEM_COPY"):
            page_indice = src_buf.get_page_idx_by_owner(core_id)
            buf_src = src_buf.get_reference_by_page_idx(*page_indice)
            buf_dst = dst_buf.get_reference_by_page_idx(*page_indice)

            core.mem_buffer_copy(buf_dst, buf_src, n_pages=len(page_indice))
