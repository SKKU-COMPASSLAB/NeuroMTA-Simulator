import torch
from typing import Any, Sequence

from neuromta.framework.logger import logger
from neuromta.framework.debug_utils import *


__all__ = [
    "Pointer",
    "ReferencePointer",
    "MemoryBankHandle",
    "MemoryHandle",
]


class Pointer:
    def __init__(self, addr: int=None):
        self._addr = addr
        
        if self._addr is not None and not isinstance(self._addr, int):
            raise ValueError(f"Pointer address must be an integer or None, but got {type(self._addr)}.")
    
    @property
    def addr(self) -> int | None:
        return self._addr
    
    @addr.setter
    def addr(self, value: int | None):
        if value is not None and not isinstance(value, int):
            raise ValueError(f"Pointer address must be an integer or None, but got {type(value)}.")
        self._addr = value
    
    def __add__(self, offset: int) -> 'Pointer':
        if isinstance(offset, torch.Tensor):
            offset = offset.item()
        if not isinstance(offset, int):
            raise ValueError(f"Offset must be an integer, but got {type(offset)}.")
        if self._addr is None:
            return ReferencePointer(ref=self, offset=offset)
        return Pointer(addr=self._addr + offset)
    
    def __sub__(self, offset: int):
        if isinstance(offset, torch.Tensor):
            offset = offset.item()
        if not isinstance(offset, int):
            raise ValueError(f"Subtraction only supports Pointer or integer types, but got {type(offset)}.")
        if self._addr is None:
            return ReferencePointer(ref=self, offset=-offset)
        return Pointer(addr=self._addr - offset)    
        
    def __hash__(self):
        return hash((self._addr,))
    
    def __eq__(self, other):
        if isinstance(other, int): other = Pointer(addr=other)
        if not isinstance(other, Pointer): return NotImplemented
        return self._addr == other._addr
    
    def __lt__(self, other):
        if isinstance(other, int): other = Pointer(addr=other)
        if not isinstance(other, Pointer): return NotImplemented
        return self._addr < other._addr
    
    def __le__(self, other):
        if isinstance(other, int): other = Pointer(addr=other)
        if not isinstance(other, Pointer): return NotImplemented
        return self._addr <= other._addr
    
    def __gt__(self, other):
        if isinstance(other, int): other = Pointer(addr=other)
        if not isinstance(other, Pointer): return NotImplemented
        return self._addr > other._addr
    
    def __ge__(self, other):
        if isinstance(other, int): other = Pointer(addr=other)
        if not isinstance(other, Pointer): return NotImplemented
        return self._addr >= other._addr
    
    def __repr__(self):
        return f"Pointer(id={hex(id(self))}, addr={self._addr})"
    
    def __str__(self):
        return self.__repr__()
    
class ReferencePointer(Pointer):
    def __init__(self, ref: Pointer, offset: int=0):
        super().__init__(addr=None)
        
        self.ref = ref
        self.offset = offset
        
    @property
    def addr(self) -> int | None:
        if self.ref.addr is None:
            return None
        return self.ref.addr + self.offset
    
    @addr.setter
    def addr(self, value: int | None):
        if value is not None and not isinstance(value, int):
            raise ValueError(f"Pointer address must be an integer or None, but got {type(value)}.")
        if self.ref.addr is None:
            raise ValueError("Cannot set address of ReferencePointer when the reference Pointer's address is None.")
        self.offset = value - self.ref.addr
        
    def __add__(self, offset: int) -> 'ReferencePointer':
        if not isinstance(offset, int):
            raise ValueError(f"Offset must be an integer, but got {type(offset)}.")
        return ReferencePointer(ref=self.ref, offset=self.offset + offset)
        
    def __repr__(self):
        return f"ReferencePointer(ref={self.ref}, offset={self.offset})"
        

class MemoryBankHandle:
    MAX_BANK_SIZE = 32 * (2 ** 20)  # 32 MB
    MAX_BANK_WARNING_PRINTED = False
    
    def __init__(self, base_addr: int, size: int):
        self._base_addr  = base_addr
        self._size       = size 
        self._addr_space = torch.zeros(size, dtype=torch.uint8)
        
        if self._size > MemoryBankHandle.MAX_BANK_SIZE and not MemoryBankHandle.MAX_BANK_WARNING_PRINTED:
            logger.warning(f"MemoryHandle size {self._size} exceeds maximum size {MemoryBankHandle.MAX_BANK_SIZE}. This may lead to high memory usage.")
            logger.warning("If you want to suppress this warning, consider using BankedMemoryHandle for large memory regions, or increase the MAX_SIZE limit in MemoryHandle class.")
            MemoryBankHandle.MAX_BANK_WARNING_PRINTED = True
        
    def get_data(self, key: int | Pointer, size: int, dtype: torch.dtype=torch.uint8, native_python_type: bool=False) -> torch.Tensor:
        if isinstance(key, Pointer):
            offset = key.addr - self._base_addr
        elif isinstance(key, int):
            offset = key - self._base_addr
        else:
            raise ValueError(f"Key must be an integer or Pointer, but got {type(key)}.")
            
        if offset + size > self._size:
            raise ValueError(f"Requested data exceeds memory handle size: offset {offset}, size {size}, handle size {self._size}")
        
        data = self._addr_space[offset:offset+size].view(dtype).clone()
        
        if torch.numel(data) == 1:
            data = data.flatten()[0]
        
        if native_python_type:
            if torch.numel(data) == 1:
                data = data.item()
            else:
                data = data.tolist()
        
        return data
    
    def set_data(self, key: int | Pointer, size: int, data: Any):
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                data = data.unsqueeze(0)
        else:
            if isinstance(data, Sequence):
                data = torch.tensor(data)
            else:
                data = torch.tensor([data])
        
        data = data.flatten().view(dtype=torch.uint8)
        
        if size <= data.numel():
            data = data[:size]
        else:
            raise ValueError(f"Data size {data.numel()} exceeds the specified size {size} to set.")
        
        if isinstance(key, Pointer):
            offset = key.addr - self._base_addr
        elif isinstance(key, int):
            offset = key - self._base_addr
        else:
            raise ValueError(f"Key must be an integer or Pointer, but got {type(key)}.")
        
        if offset + size > self._size:
            raise ValueError(f"Data to set exceeds memory handle size: offset {offset}, size {size}, handle size {self._size}")

        self._addr_space[offset:offset+size] = data
        
    @property
    def base_addr(self) -> int:
        return self._base_addr
    
    @property
    def size(self) -> int:
        return self._size
    
    
class MemoryHandle:
    MAX_BANK_SIZE = 32 * (2 ** 20)  # 32 MB
    MAX_BANK_WARNING_PRINTED = False
    
    def __init__(self, base_addr: int, bank_size: int, n_banks: int, dynamic_space_size: int, static_space_size: int=None):
        
        self._base_addr  = base_addr
        self._bank_size  = bank_size
        self._n_banks    = n_banks
        
        self._bank_handles: dict[int, MemoryBankHandle] = {}
        self._bank_mask = torch.zeros(self._n_banks, dtype=torch.bool)
        
        # DLM: Dynamic Local Memory (for dynamic allocation via Core)
        self._dlm_space_size   = dynamic_space_size
        self._dlm_space_addr   = self._base_addr
        self._dlm_space_offset = 0
        self._dlm_space_allocation_map: dict[int, int] = {}  # Maps allocated pointer addresses to their sizes
        
        # SLM: Scheduled Local Memory (for static allocation via manual set/get data method or Compiler)
        self._slm_space_size   = static_space_size if static_space_size is not None else (self._bank_size * self._n_banks - dynamic_space_size)
        self._slm_space_addr   = self._base_addr + self._dlm_space_size
        
        if self._bank_size > MemoryHandle.MAX_BANK_SIZE and not MemoryHandle.MAX_BANK_WARNING_PRINTED:
            logger.warning(f"Bank size {self._bank_size} exceeds maximum bank size {MemoryHandle.MAX_BANK_SIZE}. This may lead to high memory usage.")
            logger.warning("If you want to suppress this warning, consider increasing the MAX_BANK_SIZE limit in BankedMemoryHandle class.")
            MemoryHandle.MAX_BANK_WARNING_PRINTED = True
        
    def get_data(self, key: int | Pointer, size: int, dtype: torch.dtype=torch.uint8, native_python_type: bool=False) -> torch.Tensor:
        if isinstance(key, Pointer):
            offset = key.addr - self._base_addr
        else:
            offset = key
            
        if offset + size > self.size:
            raise ValueError(f"Requested data exceeds memory handle size: offset {offset}, size {size}, handle size {self.size}")
        
        data_chunks = []
        remaining_size = size
        current_offset = offset
        
        while remaining_size > 0:
            bank_index = current_offset // self._bank_size
            bank_offset = current_offset % self._bank_size
            chunk_size = min(remaining_size, self._bank_size - bank_offset)
            
            if not self._bank_mask[bank_index]:
                self._bank_handles[bank_index] = MemoryBankHandle(
                    base_addr=self._base_addr + bank_index * self._bank_size,
                    size=self._bank_size
                )
                self._bank_mask[bank_index] = True
            
            bank_handle = self._bank_handles[bank_index]
            chunk_data = bank_handle.get_data(bank_handle.base_addr + bank_offset, chunk_size, dtype=torch.uint8)
            data_chunks.append(chunk_data)
            
            current_offset += chunk_size
            remaining_size -= chunk_size
        
        data = torch.cat(data_chunks).view(dtype)
        
        if torch.numel(data) == 1:
            data = data.flatten()[0]
        
        if native_python_type:
            if torch.numel(data) == 1:
                data = data.item()
            else:
                data = data.tolist()
        
        return data
    
    def set_data(self, key: int | Pointer, size: int, data: Any):
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                data = data.unsqueeze(0)
        else:
            if isinstance(data, Sequence):
                data = torch.tensor(data)
            else:
                data = torch.tensor([data])
        
        data = data.flatten().view(dtype=torch.uint8)
        data = data[:size]  # TODO: Truncate if data is larger than size (this may not be a desired behavior)
        
        if isinstance(key, Pointer):
            offset = key.addr - self._base_addr
        elif isinstance(key, int):
            offset = key - self._base_addr
        else:
            raise ValueError(f"Key must be an integer or Pointer, but got {type(key)}.")
        
        if offset + size > self.size:
            raise ValueError(f"Data to set exceeds memory handle size: offset {offset}, size {size}, handle size {self.size}")
        
        remaining_size = size
        current_offset = offset
        data_offset = 0
        
        while remaining_size > 0:
            bank_index = current_offset // self._bank_size
            bank_offset = current_offset % self._bank_size
            chunk_size = min(remaining_size, self._bank_size - bank_offset)
            
            if not self._bank_mask[bank_index]:
                self._bank_handles[bank_index] = MemoryBankHandle(
                    base_addr=self._base_addr + bank_index * self._bank_size,
                    size=self._bank_size
                )
                self._bank_mask[bank_index] = True
            
            bank_handle = self._bank_handles[bank_index]
            chunk_data = data[data_offset:data_offset+chunk_size]
            bank_handle.set_data(bank_handle.base_addr + bank_offset, chunk_size, chunk_data)
            
            current_offset += chunk_size
            data_offset += chunk_size
            remaining_size -= chunk_size
            
    def remove_data(self, key: int | Pointer, size: int):
        if isinstance(key, Pointer):
            offset = key.addr - self._base_addr
        elif isinstance(key, int):
            offset = key - self._base_addr
        else:
            raise ValueError(f"Key must be an integer or Pointer, but got {type(key)}.")
        
        if offset + size > self.size:
            raise ValueError(f"Data to remove exceeds memory handle size: offset {offset}, size {size}, handle size {self.size}")
        
        remaining_size = size
        current_offset = offset
        
        while remaining_size > 0:
            bank_index = current_offset // self._bank_size
            bank_offset = current_offset % self._bank_size
            chunk_size = min(remaining_size, self._bank_size - bank_offset)
            
            if self._bank_mask[bank_index]:
                del self._bank_handles[bank_index]
                self._bank_mask[bank_index] = False
            
            current_offset += chunk_size
            remaining_size -= chunk_size
            
    def clear_pages(self):
        self._bank_handles.clear()
        self._bank_mask.fill_(False)
        
    def allocate_static_mem_space(self, ptr: Pointer, size: int) -> Pointer:
        if ptr.addr is not None:
            raise ValueError(f"Pointer already has an address {ptr.addr}, cannot allocate static memory space.")
        
        if size + self._dlm_space_offset > self._dlm_space_size:
            raise ValueError(f"Requested static memory size {size} exceeds static space size {self._dlm_space_size}.")
        
        ptr.addr = self._dlm_space_addr + self._dlm_space_offset
        self._dlm_space_offset += size
        self._dlm_space_allocation_map[ptr.addr] = size
        return ptr
    
    def deallocate_static_mem_space(self, ptr: Pointer):
        if ptr.addr is None:
            raise ValueError("Pointer address is None, cannot deallocate.")
        
        if ptr.addr in self._dlm_space_allocation_map:
            self._dlm_space_allocation_map.pop(ptr.addr)
        
        self._dlm_space_offset = 0
        for addr, size in self._dlm_space_allocation_map.items():
            self._dlm_space_offset = max(self._dlm_space_offset, addr + size - self._dlm_space_addr)
        ptr.addr = None

    @property
    def base_addr(self) -> int:
        return self._base_addr
    
    @property
    def size(self) -> int:
        return self._bank_size * self._n_banks
            
    @property
    def page_size(self) -> int:
        return self._bank_size
    
    @property
    def n_pages(self) -> int:
        return self._n_banks
    
    @property
    def dynamic_space_size(self) -> int:
        return self._dlm_space_size
    
    @property
    def scheduled_space_size(self) -> int:
        return self._slm_space_size
    
    @property
    def dynamic_space_addr(self) -> int:
        return self._dlm_space_addr
    
    @property
    def scheduled_space_addr(self) -> int:
        return self._slm_space_addr
    
    @property
    def vacant_dynamic_space_size(self) -> int:
        return self._dlm_space_size - self._dlm_space_offset

            
if __name__ == "__main__":
    original_data = torch.arange(30, dtype=torch.int32)
    original_size = original_data.numel() * original_data.element_size()
    
    base_addr = 0x1000
    bank_size = 32
    n_banks   = original_size // bank_size * 2
    
    ptr = Pointer(addr=base_addr + 10)
    
    paged_mem = MemoryHandle(base_addr=base_addr, bank_size=bank_size, n_banks=n_banks)
    paged_mem.set_data(ptr, original_data)
    data = paged_mem.get_data(ptr, original_size, dtype=torch.int32)
    
    print("Retrieved Data:", data.tolist())
    assert data.tolist() == original_data.tolist(), "Data mismatch!"
    print("All tests passed!")
