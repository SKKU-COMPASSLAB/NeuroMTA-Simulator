import pprint
import torch
from typing import Any, Sequence

from neuromta.framework import *

from neuromta.component.context.icnt_context import *
from neuromta.component.context.global_context import *
from neuromta.component.context.mxu_context import *
from neuromta.component.context.vpu_context import *

from neuromta.component.core.npu_core import *
from neuromta.component.core.dma_core import *

from neuromta.component.companions.booksim import BookSim2
from neuromta.component.companions.dramsim import DRAMSim3


__all__ = [
    "MCA_CoreGroup",
    "MCA_MemorySpace",
    "MCA_MainMemorySpace",
    "MCA_L1MemorySpace",
    "MCA_DeviceBase",
    
    "MTA_CoreGrid",
    "MTA_DeviceBase",
]


class MCA_CoreGroup(list):
    def __init__(self, core_ids: Sequence[int]):
        super().__init__(core_ids)
    
    def merge(self, other: 'MCA_CoreGroup') -> 'MCA_CoreGroup':
        new_core_ids = list(set(self) | set(other))
        return MCA_CoreGroup(new_core_ids)
    
    def split(self, shape: int) -> list['MCA_CoreGroup']:
        if shape <= 0:
            raise ValueError("Core group split shape must be greater than 0.")
        n_subgroups = (len(self) + shape - 1) // shape
        subgroups = []
        for i in range(n_subgroups):
            start_idx = i * shape
            end_idx = min((i + 1) * shape, len(self))
            sub_core_ids = self[start_idx:end_idx]
            subgroups.append(MCA_CoreGroup(sub_core_ids))
        return subgroups
    
    def intersection(self, other: 'MCA_CoreGroup') -> 'MCA_CoreGroup':
        new_core_ids = list(set(self).intersection(set(other)))
        return MCA_CoreGroup(new_core_ids)
    
    def __add__(self, other: 'MCA_CoreGroup') -> 'MCA_CoreGroup':
        return self.merge(other)
    
    def __eq__(self, value):
        if not isinstance(value, MCA_CoreGroup):
            return False
        
        c1 = sorted(list(self))
        c2 = sorted(list(value))
        return c1 == c2
    
    def __getitem__(self, idx: int) -> int:
        item = super().__getitem__(idx)
        if isinstance(item, list):
            return MCA_CoreGroup(item)
        return item
    
    @classmethod
    def merge_core_groups(cls, core_groups: Sequence['MCA_CoreGroup']) -> 'MCA_CoreGroup':
        merged_core_ids = set()
        for cg in core_groups:
            merged_core_ids.update(cg)
        return MCA_CoreGroup(sorted(list(merged_core_ids)))
        
    @property
    def core_ids(self) -> Sequence[int]:
        return list(self)
    
    @property
    def n_cores(self) -> int:
        return len(self)
    
    def __str__(self):
        return f"MCA_CoreGroup(n_cores: {self.n_cores}, core_ids: {self.core_ids})"
    
    
class MCA_MemorySpace:
    def __init__(self, device: 'MCA_DeviceBase', mem_type: GlobalContextMemType, size_per_owner: int, owner_ids: Sequence[int]):
        self._device = device
        self._mem_type = mem_type
        self._owner_ids = owner_ids
        self._size_per_owner = size_per_owner
        self._owner_id_to_mem_id_mappings: dict[int, int] = {}
        self._mem_id_to_stack_id_mappings: dict[int, int] = {}
        
        if self.mem_type == GlobalContextMemType.MAIN:
            for owner_id in self._owner_ids:
                self._owner_id_to_mem_id_mappings[owner_id] = owner_id
        elif self.mem_type == GlobalContextMemType.L1:
            for core_id in self._owner_ids:
                core_info = device.global_context.get_core_info(core_type=GlobalContextCoreType.NPU, core_id=core_id)
                mem_info = core_info.owned_mem_info
                self._owner_id_to_mem_id_mappings[core_id] = mem_info.mem_id
        
        for owner_id, mem_id in self._owner_id_to_mem_id_mappings.items():
            mem_info = device.global_context.get_mem_info(mem_type=self._mem_type, mem_id=mem_id)
            stack_id = mem_info.create_stack(stack_size=size_per_owner)
            self._mem_id_to_stack_id_mappings[mem_id] = stack_id
            
        self._is_removed = False
            
    def empty_space(self, owner_id: int) -> int:
        if self._is_removed:
            raise ValueError(f"Memory space is already removed.")
        if owner_id not in self._owner_ids:
            raise ValueError(f"Owner ID {owner_id} is not part of this MCA_MainMemorySpace.")
        
        mem_id = self._owner_id_to_mem_id_mappings[owner_id]
        mem_info = self._device.global_context.get_mem_info(mem_type=self._mem_type, mem_id=mem_id)
        stack_id = self._mem_id_to_stack_id_mappings[mem_id]
        return mem_info.empty_space(stack_id=stack_id)
    
    def allocate(self, owner_id: int, size: int) -> Pointer:
        if self._is_removed:
            raise ValueError(f"Memory space is already removed.")
        mem_id = self._owner_id_to_mem_id_mappings[owner_id]
        mem_info = self._device.global_context.get_mem_info(mem_type=self._mem_type, mem_id=mem_id)
        stack_id = self._mem_id_to_stack_id_mappings[mem_id]
        return mem_info.allocate_data(size=size, stack_id=stack_id)
            
    def remove(self):
        if self._is_removed:
            return
        
        for owner_id in self._owner_ids:
            mem_id = self._owner_id_to_mem_id_mappings[owner_id]
            mem_info = self._device.global_context.get_mem_info(mem_type=self._mem_type, mem_id=mem_id)
            stack_id = self._mem_id_to_stack_id_mappings[mem_id]
            mem_info.remove_stack(stack_id=stack_id)
        self._is_removed = True
            
    @property
    def device(self) -> 'MCA_DeviceBase':
        return self._device
    
    @property
    def mem_type(self) -> GlobalContextMemType:
        return self._mem_type
    
    @property
    def owner_ids(self) -> Sequence[int] | MCA_CoreGroup:
        return self._owner_ids
    
    @property
    def size_per_owner(self) -> int:
        return self._size_per_owner
    
    @property
    def is_removed(self) -> bool:
        return self._is_removed
    
    def override(self, new_owners) -> '_MCA_MemorySpaceOverrided':
        return _MCA_MemorySpaceOverrided(original_mem_space=self, new_owners=new_owners)
    
class _MCA_MemorySpaceOverrided(MCA_MemorySpace):
    def __init__(self, original_mem_space: MCA_MemorySpace, new_owners: Sequence[int]):
        # do not call super().__init__ since we want to override the original memory space without creating a new one in the global context
        self._original_mem_space = original_mem_space
        
        self._device = self._original_mem_space.device
        self._mem_type = self._original_mem_space.mem_type
        self._owner_ids = new_owners
        self._size_per_owner = self._original_mem_space.size_per_owner    
        self._is_removed = False
        
    def empty_space(self, owner_id: int) -> int:
        return self._original_mem_space.empty_space(owner_id)
    
    def allocate(self, owner_id: int, size: int) -> Pointer:
        return self._original_mem_space.allocate(owner_id, size)
            
    def remove(self):
        raise ValueError(f"Overrided memory space cannot be removed directly. Please remove the original memory space instead.")
    
    def override(self, new_owners) -> '_MCA_MemorySpaceOverrided':
        return _MCA_MemorySpaceOverrided(original_mem_space=self._original_mem_space, new_owners=new_owners)

class MCA_MainMemorySpace(MCA_MemorySpace):
    def __init__(self, device: 'MCA_DeviceBase', size_per_channel: int, channel_ids: Sequence[int]=None,):
        if channel_ids is None:
            channel_ids = list(range(device.global_context.n_main_mem_instances))
        super().__init__(device=device, mem_type=GlobalContextMemType.MAIN, size_per_owner=size_per_channel, owner_ids=channel_ids)
        
    def empty_space(self, channel_id):
        return super().empty_space(channel_id)
    
    def allocate(self, channel_id, size):
        return super().allocate(channel_id, size)

class MCA_L1MemorySpace(MCA_MemorySpace):
    def __init__(self, device: 'MCA_DeviceBase', size_per_bank: int, core_group: MCA_CoreGroup):
        super().__init__(device=device, mem_type=GlobalContextMemType.L1, size_per_owner=size_per_bank, owner_ids=core_group)
        
    def empty_space(self, core_id):
        return super().empty_space(core_id)
    
    def allocate(self, core_id, size):
        return super().allocate(core_id, size)

class MCA_DeviceBase(Device):
    def __init__(
        self, 
        
        global_config: GlobalContextConfig,
        icnt_config: IcntConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        self.global_context = GlobalContext(config=global_config)
        
        if icnt_config is not None:
            self.icnt_context = IcntContext(config=icnt_config)
            
            self.companion_core.register_companion_module(
                self.global_context.config.booksim_module_id,
                module=BookSim2(config=self.icnt_context.config.booksim2_config)
            )
        else:
            self.icnt_context = None
        
        self.mxu_config = mxu_config
        self.vpu_config = vpu_config
        
        self.npu_core_ids = MCA_CoreGroup(self.global_context.npu_core_ids)
        self.dma_core_ids = MCA_CoreGroup(self.global_context.dma_core_ids)

        self.npu_core_id_to_idx_mappings = {core_id: idx for idx, core_id in enumerate(self.npu_core_ids)}
        self.dma_core_id_to_idx_mappings = {core_id: idx for idx, core_id in enumerate(self.dma_core_ids)}

        self.npu_cores: list[NPUCore] = [
            NPUCore(core_id=core_id, global_context=self.global_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for core_id in self.npu_core_ids
        ]
        
        self.dma_cores: list[DMACore] = [
            DMACore(core_id=core_id, global_context=self.global_context)
            for core_id in self.dma_core_ids
        ]
        
        if self.global_context.config.main_mem_config.dramsim3_config is not None:
            self.companion_core.register_companion_module(
                self.global_context.config.dramsim_module_id,
                module=DRAMSim3(config=self.global_context.config.main_mem_config.dramsim3_config)
            )
            
        self._main_mem_spaces: list[MCA_MainMemorySpace] = []
        self._l1_mem_spaces: list[MCA_L1MemorySpace] = []
    
    ################################################################
    # NPU Hardware Resource Access API
    ################################################################
    
    def get_npu_core(self, core_id: int) -> NPUCore:
        core_idx = self.npu_core_id_to_idx_mappings[core_id]
        return self.npu_cores[core_idx]
    
    def get_npu_core_group(self, offset: int=None, n_cores: int=None) -> MCA_CoreGroup:
        if offset is None:
            offset = 0
        if n_cores is None:
            n_cores = len(self.npu_core_ids) - offset
        core_ids = self.npu_core_ids[offset:offset+n_cores]
        return MCA_CoreGroup(core_ids)
    
    ################################################################
    # Wrapper API: Global Context Memory Space Management
    ################################################################
    
    def create_main_mem_space(self, size_per_channel: int, channel_ids: Sequence[int]=None) -> MCA_MainMemorySpace:
        if size_per_channel <= 0:
            raise ValueError("Main memory space size per channel must be greater than 0.")
        mem_space = MCA_MainMemorySpace(device=self, size_per_channel=size_per_channel, channel_ids=channel_ids)
        self._main_mem_spaces.append(mem_space)
        return mem_space
    
    def create_l1_mem_space(self, size_per_bank: int, core_group: MCA_CoreGroup) -> MCA_L1MemorySpace:
        if size_per_bank <= 0:
            raise ValueError("L1 memory space size per bank must be greater than 0.")
        mem_space = MCA_L1MemorySpace(device=self, size_per_bank=size_per_bank, core_group=core_group)
        self._l1_mem_spaces.append(mem_space)
        return mem_space
    
    def remove_all_main_mem_space(self):
        for i in range(len(self._main_mem_spaces)-1, -1, -1):
            if not self._main_mem_spaces[i].is_removed:
                self._main_mem_spaces[i].remove()
            self._main_mem_spaces.pop(i)
    
    def remove_all_l1_mem_space(self):
        for i in range(len(self._l1_mem_spaces)-1, -1, -1):
            if not self._l1_mem_spaces[i].is_removed:
                self._l1_mem_spaces[i].remove()
            self._l1_mem_spaces.pop(i)
            
    def clear_all_mem_spaces(self):
        self.remove_all_l1_mem_space()
        self.remove_all_main_mem_space()
    
    def mem_get_data(self, ptr: Pointer, size: int, dtype: torch.dtype, native_python_type: bool=False) -> Any:
        if isinstance(ptr, Pointer):
            mem_info = self.global_context.get_mem_info_by_address(ptr.addr)
            return mem_info.mem_handle.get_data(ptr, size=size, dtype=dtype, native_python_type=native_python_type)
        else:
            raise Exception(f"Unsupported pointer type {type(ptr)} for mem_get_data.")
        
    def mem_set_data(self, ptr: Pointer, size: int, data: Any):
        if isinstance(ptr, Pointer):
            mem_info = self.global_context.get_mem_info_by_address(ptr.addr)
            mem_info.mem_handle.set_data(ptr, size=size, data=data)
        else:
            raise Exception(f"Unsupported pointer type {type(ptr)} for mem_set_data.")
    
    def summary(self) -> dict[str, Any]:
        return {
            "device_type": type(self).__name__,
            "npu_cores": len(self.npu_cores),
            "dma_cores": len(self.dma_cores),
            "global_config": self.global_context.config.summary(),
            "mxu_config": self.mxu_config,
            "vpu_config": self.vpu_config,
        }
        
    def print_summary(self):
        pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
        pp.pprint(self.summary())


class MTA_CoreGrid(MCA_CoreGroup):
    def __init__(self, offset: tuple[int, int], shape: tuple[int, int], core_ids: list[int]):
        super().__init__(core_ids)
        
        self.offset = offset
        self.shape = shape
        
    def lower(self) -> MCA_CoreGroup:
        return MCA_CoreGroup(self.core_ids)
    
    def split(self, shape: tuple[int, int]) -> list['MTA_CoreGrid']:
        n_rows, n_cols = self.shape
        sub_n_rows, sub_n_cols = shape
        
        if sub_n_rows <= 0 or sub_n_cols <= 0:
            raise ValueError("Core grid split shape dimensions must be greater than 0.")
        
        n_row_grids = (n_rows + sub_n_rows - 1) // sub_n_rows
        n_col_grids = (n_cols + sub_n_cols - 1) // sub_n_cols
        
        subgrids = []
        for c in range(n_col_grids):
            for r in range(n_row_grids):
                start_row = r * sub_n_rows
                end_row = min((r + 1) * sub_n_rows, n_rows)
                start_col = c * sub_n_cols
                end_col = min((c + 1) * sub_n_cols, n_cols)
                
                grid_core_ids = []
                for rr in range(start_row, end_row):
                    for cc in range(start_col, end_col):
                        idx = rr * n_cols + cc
                        grid_core_ids.append(self.core_ids[idx])
                
                grid_offset = (self.offset[0] + start_row, self.offset[1] + start_col)
                grid_shape = (end_row - start_row, end_col - start_col)
                subgrids.append(MTA_CoreGrid(offset=grid_offset, shape=grid_shape, core_ids=grid_core_ids))
        
        return subgrids
        
    def __getitem__(self, idx: int) -> int:
        if isinstance(idx, tuple):
            grid = torch.arange(len(self.core_ids)).view(self.shape)
            grid = grid[*idx]
            core_ids = [self.core_ids[i] for i in grid.flatten().tolist()]
            if len(core_ids) > 1:
                return MTA_CoreGrid(offset=(0, 0), shape=grid.shape, core_ids=core_ids)
            return core_ids[0]
        return super().__getitem__(idx)
    
    def __str__(self):
        return f"MTA_CoreGroup(n_cores: {self.n_cores}, shape: {self.shape}, core_ids: {self.core_ids})"


class MTA_DeviceBase(MCA_DeviceBase):
    def __init__(
        self, 
        
        global_config: GlobalContextConfig, 
        icnt_config: IcntConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__(global_config=global_config, icnt_config=icnt_config, mxu_config=mxu_config, vpu_config=vpu_config)
            
        npu_core_rows, npu_core_cols = [], []
        
        for core_id in self.npu_core_ids:
            coord = self.icnt_context.core_id_to_coord(core_id)
            npu_core_rows.append(coord[0])
            npu_core_cols.append(coord[1])
            
        npu_core_rows = sorted(list(set(npu_core_rows)))
        npu_core_cols = sorted(list(set(npu_core_cols)))
        
        self._npu_core_grid = torch.tensor([[self.icnt_context.coord_to_core_id((r, c)) for c in npu_core_cols]for r in npu_core_rows])
        self._npu_core_grid_enabled = True
        
        for core_id in torch.unique(self._npu_core_grid):
            if core_id not in self.npu_core_ids:
                self._npu_core_grid_enabled = False  # the accelerator does not have a full mesh of NPU cores
                break

    def get_npu_core_group(self, offset: tuple[int, int]=None, shape: tuple[int, int]=None) -> MTA_CoreGrid:
        if not self._npu_core_grid_enabled:
            raise Exception("[ERROR] Unable to get npu core grid since the accelerator does not have a full mesh of NPU cores.")

        if offset is None and shape is None:
            return MTA_CoreGrid(offset=(0, 0), shape=self._npu_core_grid.shape, core_ids=self._npu_core_grid.flatten().tolist())
        
        if shape[0] >= self._npu_core_grid.shape[0] - offset[0]:
            shape = list(shape)
            shape[0] = self._npu_core_grid.shape[0] - offset[0]  # make sure the shape does not exceed the grid boundary
        if shape[1] >= self._npu_core_grid.shape[1] - offset[1]:
            shape = list(shape)
            shape[1] = self._npu_core_grid.shape[1] - offset[1]  # make sure the shape does not exceed the grid boundary
        
        grid = self._npu_core_grid[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]]
        return MTA_CoreGrid(offset=offset, shape=tuple(shape), core_ids=grid.flatten().tolist())
    
    def summary(self):
        s = super().summary()
        s["icnt_config"] = self.icnt_context.config.summary()
        return s
