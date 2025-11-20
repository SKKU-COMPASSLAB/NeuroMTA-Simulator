import abc
from typing import Callable

from neuromta.framework import *

from neuromta.component.core import *
from neuromta.component.context import *

from neuromta.component.implementation.common.hardware import *
from neuromta.component.implementation.common.tensor_buffer import *


__all__ = [
    # "MCA_RuntimeEnvironment",
]


# _global_mca_rt_env_pool: list['MCA_RuntimeEnvironment'] = []

# def global_mca_rt_env_pool_add_opened_env(env: 'MCA_RuntimeEnvironment'):
#     global _global_mca_rt_env_pool
#     _global_mca_rt_env_pool.append(env)
    
# def global_mca_rt_env_pool_remove_closed():
#     global _global_mca_rt_env_pool
#     i = len(_global_mca_rt_env_pool) - 1
#     while i >= 0:
#         if not _global_mca_rt_env_pool[i].is_opened:
#             del _global_mca_rt_env_pool[i]
#         else:
#             break
#         i -= 1
    
# def global_mca_rt_env_pool_count_all() -> int:
#     global _global_mca_rt_env_pool
#     return len(_global_mca_rt_env_pool)

# def global_mca_rt_env_pool_count_opened() -> int:
#     global _global_mca_rt_env_pool
#     return sum(1 for env in _global_mca_rt_env_pool if env.is_opened)

# def global_mca_rt_env_pool_count_closed() -> int:
#     global _global_mca_rt_env_pool
#     return sum(1 for env in _global_mca_rt_env_pool if not env.is_opened)


# """
# MCA Runtime Environment

# Description:
#     This class represents the runtime environment for executing kernels on the MCA device. It manages memory spaces
#     including L1 scratchpad memory, L1 I/O memory spaces, and main memory.
    
# How Memory Space Configured?
#     1. L1 Scratchpad Space:
#         - This space is for temporary data storage during kernel execution.
#         - The size of the scratchpad space per bank is specified during the initialization of the runtime environment.
        
#     2. L1 I/O Spaces:
#         - These spaces are used for buffering intermediate input and output data buffers.
#         - The total size allocated for I/O spaces per bank is calculated by subtracting the scratchpad space size from
#           the total L1 memory bank size.
#         - Assumes that the intermediate input and output data buffers are stored in L1 I/O spaces.
        
#     3. Main Memory Space:
#         - This space is used for storing larger datasets that do not fit in L1 memory.
#         - The size of the main memory space per channel is specified during initialization.
#         - The channel interleaving strategy is applied across all available main memory channels by default.
#         - Assumes that DNN parameters (e.g., weights, biases) are stored in main memory space.
# """

# class MCA_RuntimeEnvironment:
#     def __init__(
#         self, 
        
#         device: MCA_DeviceBase,
#         core_group: MCA_CoreGroup,
        
#         l1_spad_space_size_per_bank: int=parse_mem_cap_str("64KB"),
#         l1_io_space_size_per_bank: list[int]=None,
#         l1_io_space_bank_num: int=1,
#         main_mem_size_per_channel: int=parse_mem_cap_str("1GB"),
#     ):  
#         self._device = device
#         self._core_group = core_group
        
#         self._l1_spad_space_size_per_bank = l1_spad_space_size_per_bank
#         self._l1_io_space_size_per_bank = l1_io_space_size_per_bank
#         self._l1_io_space_bank_num = l1_io_space_bank_num
#         self._main_mem_size_per_channel = main_mem_size_per_channel
        
#         self._l1_spad_space:    MCA_L1MemorySpace       = None
#         self._l1_io_spaces:     list[MCA_L1MemorySpace] = []
#         self._main_mem_space:   MCA_MainMemorySpace     = None
        
#         self._is_opened = False
        
#         total_io_space_per_bank = self._device.global_context.config._l1_mem_bank_size - self._l1_spad_space_size_per_bank
        
#         if self._l1_io_space_size_per_bank is None:
#             self._l1_io_space_size_per_bank = [
#                 total_io_space_per_bank // 2,
#                 total_io_space_per_bank // 2,
#             ]
#         else:
#             if isinstance(self._l1_io_space_size_per_bank, int):
#                 self._l1_io_space_size_per_bank = [self._l1_io_space_size_per_bank for _ in range(self._l1_io_space_bank_num)]
            
#             reserved_io_space_per_bank = sum(self._l1_io_space_size_per_bank)
#             if reserved_io_space_per_bank > total_io_space_per_bank:
#                 raise ValueError(f"Total L1 I/O space size per bank ({reserved_io_space_per_bank} bytes) exceeds available space ({total_io_space_per_bank} bytes) after reserving L1 scratchpad space ({self._l1_spad_space_size_per_bank} bytes).")
#             remaining_io_space_per_bank = total_io_space_per_bank - reserved_io_space_per_bank
#             if remaining_io_space_per_bank >= parse_mem_cap_str("1KB"):
#                 self._l1_io_space_size_per_bank.append(remaining_io_space_per_bank)
        
#     def __enter__(self):
#         return self.open()
    
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.close()
    
#     def open(self):
#         if global_mca_rt_env_pool_count_closed() > 0:
#             logger.warning(f"There are {global_mca_rt_env_pool_count_closed()} closed runtime environments in the global pool. Consider closing them before creating new runtime environment to avoid memory fragmentation.")
        
#         # create memory spaces
#         main_mem_channel_ids = list(range(self._device.global_context.n_main_mem_channels))
        
#         if self._l1_spad_space_size_per_bank > 0:
#             self._l1_spad_space = self._device.create_l1_mem_space(
#                 size_per_bank=self._l1_spad_space_size_per_bank,
#                 core_group=self._core_group
#             )
        
#         self._l1_io_spaces = [
#             self._device.create_l1_mem_space(
#                 size_per_bank=self._l1_io_space_size_per_bank[i],
#                 core_group=self._core_group
#             )
#             for i in range(len(self._l1_io_space_size_per_bank))
#         ]
        
#         if self._main_mem_size_per_channel > 0:
#             self._main_mem_space = self._device.create_main_mem_space(
#                 size_per_channel=self._main_mem_size_per_channel,
#                 channel_ids=main_mem_channel_ids
#             )
        
#         self._is_opened = True
        
#         # register to global pool
#         global_mca_rt_env_pool_add_opened_env(self)
        
#         return self
    
#     def close(self):
#         # remove in reverse order (avoid fragmentation issues)
#         if self._main_mem_space is not None:
#             self._main_mem_space.remove()
#             self._main_mem_space = None
        
#         for io_space in self._l1_io_spaces:
#             io_space.remove()
#         self._l1_io_spaces = []
        
#         if self._l1_spad_space is not None:
#             self._l1_spad_space.remove()
#             self._l1_spad_space = None

#         self._is_opened = False
        
#         # remove all closed envs from global pool
#         global_mca_rt_env_pool_remove_closed()
        
#     def get_l1_spad_space(self) -> MCA_L1MemorySpace:
#         if not self.is_opened:
#             raise RuntimeError("Runtime environment is not opened.")
#         return self._l1_spad_space
        
#     def get_l1_io_space(self, bank_id: int=0) -> MCA_L1MemorySpace:
#         if not self.is_opened:
#             raise RuntimeError("Runtime environment is not opened.")
#         if bank_id < 0 or bank_id >= len(self._l1_io_spaces):
#             raise ValueError(f"Invalid L1 I/O space bank_id {bank_id}, must be in range [0, {len(self._l1_io_spaces)}).")
#         return self._l1_io_spaces[bank_id]
    
#     def get_main_mem_space(self) -> MCA_MainMemorySpace:
#         if not self.is_opened:
#             raise RuntimeError("Runtime environment is not opened.")
#         return self._main_mem_space
    
#     def get_l1_io_space_bank_num(self) -> int:
#         return len(self._l1_io_spaces)
        
#     @property
#     def is_opened(self) -> bool:
#         return self._is_opened
    
#     @property
#     def device(self) -> MCA_DeviceBase:
#         return self._device
    
#     @property
#     def core_group(self) -> MCA_CoreGroup:
#         return self._core_group