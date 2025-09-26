import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "MCA_RT_GLOBAL_SYNC",
]


@MCA_RT_OPERATOR
def MCA_RT_GLOBAL_SYNC(device: MCA_DeviceBase, core_ids: list[int]):
    for core_id in core_ids:
        core = device.get_npu_core(core_id=core_id)
        with MCA_RT_JIT_COMPILE_REGION(core, "BARRIER"):
            core.inter_core_sync_barrier(core_ids)