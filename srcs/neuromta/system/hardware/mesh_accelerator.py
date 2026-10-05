import functools
import os

import numpy as np

from neuromta.framework import *

from neuromta.component.companions.booksim import BookSim2
from neuromta.component.companions.dramsim import DRAMSim3

from neuromta.component.context.icnt_context import IcntContext, IcntConfig
from neuromta.component.context.global_context import GlobalContext, GlobalConfig
from neuromta.component.context.compute_tile_context import ComputeTileContext, ComputeTileConfig
from neuromta.component.context.mem_context import MemoryContext, MemoryConfig

from neuromta.component.core.ccg_tile import CCGTile
from neuromta.component.core.dma_tile import DMATile

from neuromta.system.hardware.base_accelerator import BaseAccelerator


__all__ = [
    "MeshAccelerator",
    "MeshAcceleratorConfig",
    "MeshAcceleratorRuntimeContext",
    "MESH_TILE_COORDS",
]

IP_ROOT = os.path.abspath(os.path.dirname(__file__))
IP_CACHE_DIR = os.path.join(IP_ROOT, ".cache")
IP_DRAMSIM_CONFIG_CACHE_FMT = os.path.join(IP_CACHE_DIR, "dramsim_mesh_accelerator_{config_name}.ini")


class MeshAccelerator(BaseAccelerator):
    def __init__(self, global_config: GlobalConfig, icnt_config: IcntConfig, ccg_config: ComputeTileConfig, mem_config: MemoryConfig,):
        super().__init__()

        self.global_context     = GlobalContext(config=global_config)
        self.icnt_context       = IcntContext(config=icnt_config)
        self.ccg_context        = ComputeTileContext(config=ccg_config)
        self.mem_context        = MemoryContext(config=mem_config)

        if self.icnt_context.config.booksim2_enable:
            self.companion_core.register_companion_module(
                self.icnt_context.config.booksim2_module_id,
                module=BookSim2(config=self.icnt_context.config.booksim2_config)
            )

        if self.mem_context.config.dramsim3_enable:
            self.companion_core.register_companion_module(
                self.mem_context.config.dramsim3_module_id,
                module=DRAMSim3(config=self.mem_context.config.dramsim3_config)
            )

        self.ccg_tiles: list[CCGTile] = [
            CCGTile(
                core_id=core_id,
                ctile_context=self.ccg_context,
                icnt_context=self.icnt_context,
                dma_context=self.mem_context,
                global_context=self.global_context,
            )
            for core_id in self.global_context.config.ccg_tile_ids
        ]

        self.mem_tiles: list[DMATile] = [
            DMATile(
                core_id=core_id,
                icnt_context=self.icnt_context,
                mem_context=self.mem_context,
                global_context=self.global_context,
            )
            for core_id in self.global_context.config.dma_tile_ids
        ]

        self._ccg_tile_mesh: np.ndarray = (np.zeros(icnt_config.shape, dtype=int) - 1)
        for core_id in self.global_context.config.ccg_tile_ids:
            y, x = self.icnt_context.core_id_to_coord(core_id)
            self._ccg_tile_mesh[y, x] = core_id

        valid_rows = np.any(self._ccg_tile_mesh != -1, axis=1)
        valid_cols = np.any(self._ccg_tile_mesh != -1, axis=0)
        self._ccg_tile_mesh = self._ccg_tile_mesh[np.ix_(valid_rows, valid_cols)]

        self._dma_tile_mesh: np.ndarray = (np.zeros(icnt_config.shape, dtype=int) - 1)
        for core_id in self.global_context.config.dma_tile_ids:
            y, x = self.icnt_context.core_id_to_coord(core_id)
            self._dma_tile_mesh[y, x] = core_id

        valid_rows = np.any(self._dma_tile_mesh != -1, axis=1)
        valid_cols = np.any(self._dma_tile_mesh != -1, axis=0)
        self._dma_tile_mesh = self._dma_tile_mesh[np.ix_(valid_rows, valid_cols)]

    def get_ccg_tile(self, core_id: int, rel: bool=False) -> CCGTile:
        if rel:
            core_id = self.global_context.config.ccg_tile_ids[core_id]

        for tile in self.ccg_tiles:
            if tile.core_id == core_id:
                return tile

        raise ValueError(f"Compute tile with core_id {core_id} not found.")

    def get_ccg_tile_mesh(self):
        return self._ccg_tile_mesh

    def get_dma_tile(self, core_id: int, rel: bool=False) -> DMATile:
        if rel:
            core_id = self.global_context.config.dma_tile_ids[core_id]

        for tile in self.mem_tiles:
            if tile.core_id == core_id:
                return tile

        raise ValueError(f"DMA tile with core_id {core_id} not found.")

    def get_dma_tile_mesh(self):
        return self._dma_tile_mesh


class MeshAcceleratorRuntimeContext:
    def __init__(self, device: 'MeshAccelerator'):
        if not isinstance(device, MeshAccelerator):
            raise Exception(f"The device should be a MeshAccelerator instance")

        self.device = device

        self.dma_tile_ids = device.global_context.config.dma_tile_ids
        self.ccg_tile_ids = device.global_context.config.ccg_tile_ids

        # Memory Allocation / Deallocation (DEVICE MEMORY and LOCAL CACHE)
        self.device_memory_vacancy: dict[int, list[tuple[int, int]]] = {}   # key: DMA ID / value: [(start address, end address), ...]
        self.local_cache_vacancy:   dict[int, list[tuple[int, int]]] = {}   # key: CCG ID / value: [(start address, end address), ...]

        for dma_id in self.dma_tile_ids:
            instance_id = device.mem_context.get_instance_id_with_dma_id(dma_id)
            offset = device.mem_context.get_instance_addr(instance_id, 0)
            self.device_memory_vacancy[dma_id] = [(offset, offset + device.mem_context.channel_size_per_instance)]

        for ccg_id in self.ccg_tile_ids:
            self.local_cache_vacancy[ccg_id] = [(0, device.ccg_context.config.local_cache)]

        # CCG Tile Allocation / Deallocation
        self.ccg_kernel_vacancy:    dict[int, bool] = {ccg_id: True for ccg_id in self.ccg_tile_ids}    # key: CCG ID / value: kernel vacancy flag

        self.icnt_ccg_flag = np.zeros(self.device.icnt_context.config.shape, dtype=bool)
        self.icnt_dma_flag = np.zeros(self.device.icnt_context.config.shape, dtype=bool)
        self.icnt_ccg_map  = np.zeros(self.device.icnt_context.config.shape, dtype=int) - 1
        self.icnt_dma_map  = np.zeros(self.device.icnt_context.config.shape, dtype=int) - 1

        for ccg_id in self.ccg_tile_ids:
            y, x = self.device.icnt_context.core_id_to_coord(ccg_id)
            self.icnt_ccg_flag[y, x] = True
            self.icnt_ccg_map[y, x] = ccg_id

        for dma_id in self.dma_tile_ids:
            y, x = self.device.icnt_context.core_id_to_coord(dma_id)
            self.icnt_dma_flag[y, x] = True
            self.icnt_dma_map[y, x] = dma_id

        # Roofline Model
        self.total_flops = len(self.ccg_tile_ids) * self.device.ccg_context.config.tops * 1e12 / self.device.ccg_context.config.processor_clock_freq
        self.total_mem_bandwidth = self.device.mem_context.peak_bandwidth_per_cycle
        self.total_roofline_ridgepoint = self.total_flops / self.total_mem_bandwidth

        # Allocate LD/ST Buffers
        _ld_buffer_info = self.allocate_local_cache(self.ccg_tile_ids, self.device.ccg_context.config.ld_buffer_size, len(self.ccg_tile_ids))
        _st_buffer_info = self.allocate_local_cache(self.ccg_tile_ids, self.device.ccg_context.config.st_buffer_size, len(self.ccg_tile_ids))

        if _ld_buffer_info is None:
            raise Exception(f"Failed to allocate LD buffers for all CCG tiles with size {self.device.ccg_context.config.ld_buffer_size}")
        if _st_buffer_info is None:
            raise Exception(f"Failed to allocate ST buffers for all CCG tiles with size {self.device.ccg_context.config.st_buffer_size}")

        self.ld_buffer_ptrs = {ccg_id: (addr, size) for ccg_id, addr, size in _ld_buffer_info}
        self.st_buffer_ptrs = {ccg_id: (addr, size) for ccg_id, addr, size in _st_buffer_info}

    @staticmethod
    def _allocate_memory_from_vacancy_info(vacancy_info: list[tuple[int, int]], size: int) -> int | None:
        if size <= 0:
            raise ValueError(f"Memory allocation size must be positive, got {size}")
        return next((start for start, end in vacancy_info if end - start >= size), None)

    @staticmethod
    def _update_memory_allocation_info(vacancy_info: list[tuple[int, int]], addr: int, size: int):
        if size <= 0:
            raise ValueError(f"Memory allocation size must be positive, got {size}")
        allocation_end = addr + size
        for index, (start, end) in enumerate(vacancy_info):
            if start <= addr and allocation_end <= end:
                vacancy_info[index:index + 1] = ([(start, addr)] if start < addr else []) + ([(allocation_end, end)] if allocation_end < end else [])
                return
        raise ValueError(f"Memory range [{addr}, {allocation_end}) is not vacant")

    @staticmethod
    def _update_memory_deallocation_info(vacancy_info: list[tuple[int, int]], addr: int, size: int):
        if size <= 0:
            raise ValueError(f"Memory deallocation size must be positive, got {size}")
        vacancy_info.append((addr, addr + size))
        vacancy_info.sort()
        merged = []
        for start, end in vacancy_info:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        vacancy_info[:] = merged

    def allocate_device_memory(self, dma_ids: list[int], size: int, n_banks: int) -> list[tuple[int, int, int]] | None:  # returns the list of (DMA tile ID, start address, bank size)
        if not isinstance(dma_ids, (list, tuple, np.ndarray)):
            raise Exception(f"dma_ids should be a list, tuple, or numpy array, got {type(dma_ids).__name__}")
        if len(dma_ids) == 0:
            raise Exception(f"dma_ids should not be empty")

        allocated_banks: list[tuple[int, int, int]] = []     # list of (DMA tile ID, start address, bank size)
        vacancy = {dma_id: list(self.device_memory_vacancy[dma_id]) for dma_id in dma_ids}
        # allocated_dma_ids: list[int] = []               # list of DMA tile IDs

        for i in range(n_banks):
            dma_idx = i % len(dma_ids)
            dma_id = dma_ids[dma_idx]
            addr = self._allocate_memory_from_vacancy_info(vacancy[dma_id], size)

            if addr is None:
                return None

            allocated_banks.append((dma_id, addr, size))
            self._update_memory_allocation_info(vacancy[dma_id], addr, size)
            # allocated_dma_ids.append(dma_id)

        for dma_id in vacancy:
            self.device_memory_vacancy[dma_id] = vacancy[dma_id]

        return allocated_banks

    def deallocate_device_memory(self, addr: int, size: int):
        dma_id = self.device.mem_context.get_dma_id_with_address(addr)
        self._update_memory_deallocation_info(self.device_memory_vacancy[dma_id], addr, size)
        return self

    def allocate_local_cache(self, ccg_ids: list[int], size: int, n_banks) -> list[tuple[int, int, int]] | None:    # returns the list of (CCG tile ID, address, bank size)
        if not isinstance(ccg_ids, (list, tuple, np.ndarray)):
            raise Exception(f"ccg_ids should be a list, tuple, or numpy array, got {type(ccg_ids).__name__}")
        if len(ccg_ids) == 0:
            raise Exception(f"ccg_ids should not be empty")

        allocated_banks: list[tuple[int, int, int]] = []    # list of (CCG tile ID, start address, bank size)
        vacancy = {ccg_id: list(self.local_cache_vacancy[ccg_id]) for ccg_id in ccg_ids}

        for i in range(n_banks):
            ccg_idx = i % len(ccg_ids)
            ccg_id = ccg_ids[ccg_idx]
            addr = self._allocate_memory_from_vacancy_info(vacancy[ccg_id], size)

            if addr is None:
                return None

            allocated_banks.append((ccg_id, addr, size))
            self._update_memory_allocation_info(vacancy[ccg_id], addr, size)

        for ccg_id in vacancy:
            self.local_cache_vacancy[ccg_id] = vacancy[ccg_id]

        return allocated_banks

    def deallocate_local_cache(self, ccg_id: int, addr: int, size: int):
        self._update_memory_deallocation_info(self.local_cache_vacancy[ccg_id], addr, size)
        return self

    def allocate_ccg_kernel(self, ccg_mesh_shape: tuple[int, int], close_to: list[int]) -> np.ndarray | None:       # returns the CCG tile mesh (2D numpy array with each element indicating the CCG tile ID)
        mesh_h, mesh_w = ccg_mesh_shape
        icnt_h, icnt_w = self.device.icnt_context.config.shape

        if mesh_h <= 0 or mesh_w <= 0:
            raise ValueError(f"Invalid CCG mesh shape: {ccg_mesh_shape}")
        if mesh_h > icnt_h or mesh_w > icnt_w:
            return None

        icnt_ccg_vacancy_flag = self.icnt_ccg_flag.copy()
        for ccg_id, is_vacant in self.ccg_kernel_vacancy.items():
            if not is_vacant:
                y, x = self.device.icnt_context.core_id_to_coord(ccg_id)
                icnt_ccg_vacancy_flag[y, x] = False

        close_to_coords = np.asarray([self.device.icnt_context.core_id_to_coord(core_id) for core_id in close_to], dtype=np.int64).reshape(-1, 2)
        y_distance = np.abs(np.arange(icnt_h)[:, None] - close_to_coords[:, 0]).sum(axis=1)
        x_distance = np.abs(np.arange(icnt_w)[:, None] - close_to_coords[:, 1]).sum(axis=1)
        distance_map = y_distance[:, None] + x_distance[None, :]
        distance_integral = np.pad(distance_map.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        distance_scores = distance_integral[mesh_h:, mesh_w:] - distance_integral[:-mesh_h, mesh_w:] - distance_integral[mesh_h:, :-mesh_w] + distance_integral[:-mesh_h, :-mesh_w]

        occupied_integral = np.pad((~icnt_ccg_vacancy_flag).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        occupied_counts = occupied_integral[mesh_h:, mesh_w:] - occupied_integral[:-mesh_h, mesh_w:] - occupied_integral[mesh_h:, :-mesh_w] + occupied_integral[:-mesh_h, :-mesh_w]
        distance_scores = np.where(occupied_counts == 0, distance_scores, np.inf)

        if not np.isfinite(distance_scores).any():
            return None

        best_offset = np.unravel_index(np.argmin(distance_scores), distance_scores.shape)
        allocated_ccg_tile_mesh = self.icnt_ccg_map[best_offset[0]:best_offset[0] + mesh_h, best_offset[1]:best_offset[1] + mesh_w]

        for ccg_id in allocated_ccg_tile_mesh.flatten().tolist():
            self.ccg_kernel_vacancy[ccg_id] = False

        return allocated_ccg_tile_mesh

    def deallocate_ccg_kernel(self, ccg_tile_mesh: np.ndarray):
        for ccg_id in ccg_tile_mesh.flatten().tolist():
            self.ccg_kernel_vacancy[ccg_id] = True


def MESH_TILE_COORDS(shape: tuple[int, int], offset: tuple[int, int]=(0, 0), stride: tuple[int, int]=(1, 1)) -> list[tuple[int, int]]:
    return [
        (y + offset[0], x + offset[1])
        for y in range(0, shape[0], stride[0])
        for x in range(0, shape[1], stride[1])
    ]


class MeshAcceleratorConfig(dict):
    global_config: GlobalConfig
    icnt_config: IcntConfig
    ccg_config: ComputeTileConfig
    mem_config: MemoryConfig

    @classmethod
    def large(
        cls,
        processor_clock_freq: int=parse_freq_str("1GHz"),
        icnt_mesh_shape: tuple[int, int]=(12, 12),
        icnt_flit_size: int=parse_mem_cap_str("128B"),
        icnt_max_payload_size: int=32,
        icnt_subnets: int=2,
        booksim2_enable: bool=False,

        ccg_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((12, 10), offset=(0, 1)),
        ccg_tops: int=4,
        ccg_x_dim: int=32,
        ccg_y_dim: int=32,
        ccg_local_cache: int=parse_mem_cap_str("1MB"),
        ccg_ld_buffer_size: int=parse_mem_cap_str("128KB"),
        ccg_st_buffer_size: int=parse_mem_cap_str("128KB"),

        dma_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((4, 4), offset=(0, 0), stride=(3, 11)),
        dma_ch_per_instance: int=4,
        dma_channel_size: int=parse_mem_cap_str("1GB"),
        dma_mem_config_path: str="HBM2_8Gb_x128",
        dma_mem_config_cache_fmt: str=IP_DRAMSIM_CONFIG_CACHE_FMT,
        dramsim3_enable: bool=False,
    ) -> "MeshAcceleratorConfig":
        return cls.create(
            config_name="large",
            processor_clock_freq=processor_clock_freq,
            icnt_mesh_shape=icnt_mesh_shape,
            icnt_flit_size=icnt_flit_size,
            icnt_max_payload_size=icnt_max_payload_size,
            icnt_subnets=icnt_subnets,
            booksim2_enable=booksim2_enable,
            ccg_tile_coords=ccg_tile_coords,
            ccg_tops=ccg_tops,
            ccg_x_dim=ccg_x_dim,
            ccg_y_dim=ccg_y_dim,
            ccg_local_cache=ccg_local_cache,
            ccg_ld_buffer_size=ccg_ld_buffer_size,
            ccg_st_buffer_size=ccg_st_buffer_size,
            dma_tile_coords=dma_tile_coords,
            dma_ch_per_instance=dma_ch_per_instance,
            dma_channel_size=dma_channel_size,
            dma_mem_config_path=dma_mem_config_path,
            dma_mem_config_cache_fmt=dma_mem_config_cache_fmt,
            dramsim3_enable=dramsim3_enable
        )

    @classmethod
    def medium(
        cls,
        processor_clock_freq: int=parse_freq_str("1GHz"),
        icnt_mesh_shape: tuple[int, int]=(4, 13),
        icnt_flit_size: int=parse_mem_cap_str("64B"),
        icnt_max_payload_size: int=32,
        icnt_subnets: int=2,
        booksim2_enable: bool=False,

        ccg_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((4, 11), offset=(0, 1)),
        ccg_tops: int=4,
        ccg_x_dim: int=32,
        ccg_y_dim: int=32,
        ccg_local_cache: int=parse_mem_cap_str("1MB"),
        ccg_ld_buffer_size: int=parse_mem_cap_str("128KB"),
        ccg_st_buffer_size: int=parse_mem_cap_str("128KB"),

        dma_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((4, 4), offset=(0, 0), stride=(1, 12)),
        dma_ch_per_instance: int=8,
        dma_channel_size: int=parse_mem_cap_str("1GB"),
        dma_mem_config_path: str="HBM2_8Gb_x128",
        dma_mem_config_cache_fmt: str=IP_DRAMSIM_CONFIG_CACHE_FMT,
        dramsim3_enable: bool=False,
    ) -> "MeshAcceleratorConfig":
        return cls.create(
            config_name="medium",
            processor_clock_freq=processor_clock_freq,
            icnt_mesh_shape=icnt_mesh_shape,
            icnt_flit_size=icnt_flit_size,
            icnt_max_payload_size=icnt_max_payload_size,
            icnt_subnets=icnt_subnets,
            booksim2_enable=booksim2_enable,
            ccg_tile_coords=ccg_tile_coords,
            ccg_tops=ccg_tops,
            ccg_x_dim=ccg_x_dim,
            ccg_y_dim=ccg_y_dim,
            ccg_local_cache=ccg_local_cache,
            ccg_ld_buffer_size=ccg_ld_buffer_size,
            ccg_st_buffer_size=ccg_st_buffer_size,
            dma_tile_coords=dma_tile_coords,
            dma_ch_per_instance=dma_ch_per_instance,
            dma_channel_size=dma_channel_size,
            dma_mem_config_path=dma_mem_config_path,
            dma_mem_config_cache_fmt=dma_mem_config_cache_fmt,
            dramsim3_enable=dramsim3_enable
        )

    @classmethod
    def small(
        cls,
        processor_clock_freq: int=parse_freq_str("1GHz"),
        icnt_mesh_shape: tuple[int, int]=(4, 4),
        icnt_flit_size: int=parse_mem_cap_str("64B"),
        icnt_max_payload_size: int=32,
        icnt_subnets: int=2,
        booksim2_enable: bool=False,

        ccg_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((4, 2), offset=(0, 1)),
        ccg_tops: int=4,
        ccg_x_dim: int=32,
        ccg_y_dim: int=32,
        ccg_local_cache: int=parse_mem_cap_str("1MB"),
        ccg_ld_buffer_size: int=parse_mem_cap_str("128KB"),
        ccg_st_buffer_size: int=parse_mem_cap_str("128KB"),

        dma_tile_coords: list[tuple[int, int]]=MESH_TILE_COORDS((4, 4), offset=(0, 0), stride=(2, 3)),
        dma_ch_per_instance: int=1,
        dma_channel_size: int=parse_mem_cap_str("1GB"),
        dma_mem_config_path: str="LPDDR4_8Gb_x16_2400",
        dma_mem_config_cache_fmt: str=IP_DRAMSIM_CONFIG_CACHE_FMT,
        dramsim3_enable: bool=False,
    ) -> "MeshAcceleratorConfig":
        return cls.create(
            config_name="small",
            processor_clock_freq=processor_clock_freq,
            icnt_mesh_shape=icnt_mesh_shape,
            icnt_flit_size=icnt_flit_size,
            icnt_max_payload_size=icnt_max_payload_size,
            icnt_subnets=icnt_subnets,
            booksim2_enable=booksim2_enable,
            ccg_tile_coords=ccg_tile_coords,
            ccg_tops=ccg_tops,
            ccg_x_dim=ccg_x_dim,
            ccg_y_dim=ccg_y_dim,
            ccg_local_cache=ccg_local_cache,
            ccg_ld_buffer_size=ccg_ld_buffer_size,
            ccg_st_buffer_size=ccg_st_buffer_size,
            dma_tile_coords=dma_tile_coords,
            dma_ch_per_instance=dma_ch_per_instance,
            dma_channel_size=dma_channel_size,
            dma_mem_config_path=dma_mem_config_path,
            dma_mem_config_cache_fmt=dma_mem_config_cache_fmt,
            dramsim3_enable=dramsim3_enable
        )

    @classmethod
    def stacked_3d_dram_npu(
        cls,
        processor_clock_freq: int=parse_freq_str("1GHz"),
        icnt_mesh_shape: tuple[int, int]=(4, 4),
        icnt_flit_size: int=parse_mem_cap_str("64B"),
        icnt_max_payload_size: int=32,
        icnt_subnets: int=2,
        booksim2_enable: bool=False,

        ccg_tops: int=64,
        ccg_x_dim: int=128,
        ccg_y_dim: int=128,
        ccg_local_cache: int=parse_mem_cap_str("1MB"),          # no remaining space for local cache, all local cache is used for LD/ST buffers
        ccg_ld_buffer_size: int=parse_mem_cap_str("512KB"),
        ccg_st_buffer_size: int=parse_mem_cap_str("512KB"),

        dma_ch_per_instance: int=8,
        dma_channel_size: int=parse_mem_cap_str("1GB"),
        dma_mem_config_path: str="HBM2_8Gb_x128",
        dma_mem_config_cache_fmt: str=IP_DRAMSIM_CONFIG_CACHE_FMT,
        dramsim3_enable: bool=False,
    ) -> "MeshAcceleratorConfig":
        ccg_tile_coords = MESH_TILE_COORDS(icnt_mesh_shape)
        dma_tile_coords = MESH_TILE_COORDS(icnt_mesh_shape)

        return cls.create(
            config_name="stacked_3d_dram_npu",
            processor_clock_freq=processor_clock_freq,
            icnt_mesh_shape=icnt_mesh_shape,
            icnt_flit_size=icnt_flit_size,
            icnt_max_payload_size=icnt_max_payload_size,
            icnt_subnets=icnt_subnets,
            booksim2_enable=booksim2_enable,
            ccg_tile_coords=ccg_tile_coords,
            ccg_tops=ccg_tops,
            ccg_x_dim=ccg_x_dim,
            ccg_y_dim=ccg_y_dim,
            ccg_local_cache=ccg_local_cache,
            ccg_ld_buffer_size=ccg_ld_buffer_size,
            ccg_st_buffer_size=ccg_st_buffer_size,
            dma_tile_coords=dma_tile_coords,
            dma_ch_per_instance=dma_ch_per_instance,
            dma_channel_size=dma_channel_size,
            dma_mem_config_path=dma_mem_config_path,
            dma_mem_config_cache_fmt=dma_mem_config_cache_fmt,
            dramsim3_enable=dramsim3_enable
        )

    @classmethod
    def create(
        cls,

        config_name: str,

        processor_clock_freq: int,

        icnt_mesh_shape: tuple[int, int],
        icnt_flit_size: int,
        icnt_max_payload_size: int,
        icnt_subnets: int,
        booksim2_enable: bool,

        ccg_tile_coords: list[tuple[int, int]],
        ccg_tops: int,
        ccg_x_dim: int,
        ccg_y_dim: int,
        ccg_local_cache: int,
        ccg_ld_buffer_size: int,
        ccg_st_buffer_size: int,

        dma_tile_coords: list[tuple[int, int]],
        dma_ch_per_instance: int,
        dma_channel_size: int,
        dma_mem_config_path: str,
        dma_mem_config_cache_fmt: str,
        dramsim3_enable: bool,
    ):
        n_comp_tiles = len(ccg_tile_coords)
        n_dma_tiles = len(dma_tile_coords)

        global_config = GlobalConfig(
            processor_clock_freq=processor_clock_freq,
            ccg_tile_ids=list(range(n_comp_tiles)),
            dma_tile_ids=list(range(n_comp_tiles, n_comp_tiles + n_dma_tiles)),
        )

        icnt_config = IcntConfig(
            processor_clock_freq=processor_clock_freq,
            shape=icnt_mesh_shape,
            flit_size=icnt_flit_size,
            max_payload_size=icnt_max_payload_size,
            subnets=icnt_subnets,
            booksim2_enable=booksim2_enable,
            booksim2_kwargs={
                "routing_delay": 1,
                "vc_alloc_delay": 1,
                "sw_alloc_delay": 1,
                "st_prepare_delay": 0,
                "st_final_delay": 1,

                "input_speedup": 1,
                "output_speedup": 1,
                "internal_speedup": 1.0,

                "num_vcs": 16,
                "vc_buf_size": 8,
            }
        )

        for ctile_idx, (ctile_y, ctile_x) in enumerate(ccg_tile_coords):
            ctile_core_id = global_config.ccg_tile_ids[ctile_idx]
            icnt_coord = (ctile_y, ctile_x)
            icnt_config.update_core_map(coord=icnt_coord, core_id=ctile_core_id)

        for dma_idx, (dma_y, dma_x) in enumerate(dma_tile_coords):
            dma_core_id = global_config.dma_tile_ids[dma_idx]
            icnt_coord = (dma_y, dma_x)
            icnt_config.update_core_map(coord=icnt_coord, core_id=dma_core_id)

        ccg_config = ComputeTileConfig(
            processor_clock_freq=processor_clock_freq,
            tops=ccg_tops,
            tile_x_dim=ccg_x_dim,
            tile_y_dim=ccg_y_dim,
            local_cache=ccg_local_cache,
            ld_buffer_size=ccg_ld_buffer_size,
            st_buffer_size=ccg_st_buffer_size,
        )

        if dramsim3_enable:
            mem_sim_config = dict(
                dramsim3_enable=dramsim3_enable,
                dramsim3_src_config_path=dma_mem_config_path,
                dramsim3_dst_config_path=dma_mem_config_cache_fmt.format(config_name=config_name),
                dramsim3_max_issue_per_cmd_q_per_cycle=32,
            )
        else:
            mem_sim_config = dict(
                dramsim3_enable=dramsim3_enable,
                lightweight_read_latency_cycles=17,
                lightweight_write_latency_cycles=34,
                lightweight_write_accept_latency_cycles=2,
                lightweight_write_completion_policy="accept",
                lightweight_enable_latency_amortization=False,
                lightweight_channel_bandwidth_bytes_per_cycle=64,
                lightweight_dma_granularity=parse_mem_cap_str("256B"),
                lightweight_address_mapping="dramsim3",
                lightweight_channel_interleave_bytes=parse_mem_cap_str("64B"),
                lightweight_address_mapping_scheme="rorabgbachco",
                lightweight_dram_rows=32768,
                lightweight_dram_columns=64,
                lightweight_dram_burst_length=4,
                lightweight_instance_command_issue_gap_cycles=1,
                lightweight_command_issue_gap_cycles=1,
                lightweight_read_to_write_turnaround_cycles=6,
                lightweight_write_to_read_turnaround_cycles=8,
                lightweight_burst_size_bytes=parse_mem_cap_str("64B"),
                lightweight_row_size_bytes=parse_mem_cap_str("2KB"),
                lightweight_row_hit_latency_cycles=1,
                lightweight_row_miss_penalty_cycles=14,
                lightweight_row_conflict_penalty_cycles=14,
                lightweight_bank_group_penalty_cycles=1,
                lightweight_dma_max_outstanding_bursts=64,
                lightweight_channel_max_outstanding_bursts=16,
                lightweight_request_queue_depth=32,
                lightweight_concurrent_request_command_gap_cycles=0,
                lightweight_concurrent_request_command_gap_threshold=4,
                lightweight_concurrent_request_command_gap_limit=1,
            )

        mem_config = MemoryConfig(
            mem_addr_offset=0,
            processor_clock_freq=processor_clock_freq,
            n_instance=n_dma_tiles,
            channel_size=dma_channel_size,
            n_channel_per_instance=dma_ch_per_instance,
            **mem_sim_config,
            instance_dma_map={
                i: global_config.dma_tile_ids[i] for i in range(n_dma_tiles)
            }
        )

        return cls(
            global_config=global_config,
            icnt_config=icnt_config,
            ccg_config=ccg_config,
            mem_config=mem_config,
        )
