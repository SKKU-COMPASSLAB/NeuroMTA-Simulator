import enum
import math
import torch
from typing import Any, Sequence, TypeVar, Generic

from neuromta.framework.logger import logger
from neuromta.framework.debug_utils import *


__all__ = [
    "DataContainer",
    "Variable",
    "Page",
    "PointerType",
    "Pointer",
    "BufferPointer",
    "BufferHandle",
    "CircularBufferHandle",
    "MemoryHandle",
    
    "create_var_ptr",
    "create_page_ptr",
    "create_uniform_buffer",
    "create_distributed_buffer",
]


T = TypeVar('T')

class DataContainer(Generic[T]):
    def __init__(self, data: T=None):
        self.data: T = data


class _DataElement:
    def __init__(self, addr: int, size: int, content: Any=None):
        self._addr = addr
        self._size = size
        self._content = content

    @property
    def addr(self) -> int:
        return self._addr
    
    @addr.setter
    def addr(self, value: int):
        if self._addr is not None:
            raise Exception(f"Address is already set to {self._addr}, cannot be changed to {value}.")
        self._addr = value

    @property
    def size(self) -> int:
        return self._size
    
    @property
    def content(self) -> Any:
        return self._content
    
    @content.setter
    def content(self, value: Any):
        self._content = value
    

class Variable(_DataElement):
    def __init__(self, addr: int, size: int, content: Any=None):
        super().__init__(addr, size, content)


class Page(_DataElement):
    def __init__(self, addr: int, size: int, content: torch.Tensor=None):
        super().__init__(addr, size, content)
        
    def content_view(self, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        view = self.content
        
        if dtype is not None:
            view = view.view(dtype=dtype)
        if shape is not None:
            view = view.reshape(shape=shape)
            
        return view
    
    def set_content(self, value: torch.Tensor, offset: int=0):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Page content must be a torch.Tensor, got {type(value)}.")
        
        if self._content is None:
            self._content = torch.zeros(self.size, dtype=torch.uint8)
        
        value = value.view(dtype=torch.uint8).flatten()
        try:
            self._content[offset:offset + value.numel()] = value
        except Exception as e:
            logger.error(f"Failed to set content to page at address {self.addr} with size {self.size}. The provided value has size {value.numel()} and offset {offset}.")
            raise Exception(f"Failed to set content: {e}")

    @property
    def content(self) -> torch.Tensor:
        if self._content is None:
            self._content = torch.zeros(self.size, dtype=torch.uint8)
        return self._content
    
    @content.setter
    def content(self, value: torch.Tensor):
        if value is None:
            return  # if the functional model is not used, the value can be None
        if not isinstance(value, torch.Tensor):
            raise Exception(f"Page content must be a torch.Tensor, got {type(value)}.")
        self._content = value


class PointerType(enum.Enum):
    UNDEFINED   = enum.auto()
    PAGE        = enum.auto()
    VARIABLE    = enum.auto()
    
    @classmethod
    def get_pointer_type_with_handle(cls, handle: Any) -> 'PointerType':
        if isinstance(handle, Variable):
            return cls.VARIABLE
        elif isinstance(handle, Page):
            return cls.PAGE
        return cls.UNDEFINED
    
class Pointer:
    def __init__(self, data_element: _DataElement=None):
        if data_element is not None:
            self._addr = data_element.addr
            self._size = data_element.size
            self._ptr_type = PointerType.get_pointer_type_with_handle(data_element)
        else:
            self._addr = 0
            self._size = 0
            self._ptr_type = PointerType.UNDEFINED
            
    def initialize(self, data_element: _DataElement=None):
        if data_element is None:
            self._addr = 0
            self._size = 0
            self._ptr_type = PointerType.UNDEFINED
        else:
            self._addr = data_element.addr
            self._size = data_element.size
            self._ptr_type = PointerType.get_pointer_type_with_handle(data_element)
        
    @property
    def addr(self) -> int:
        return self._addr
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def ptr_type(self) -> PointerType:
        return self._ptr_type
    
    def __str__(self):
        return f"Pointer(addr={self._addr}, size={self._size}, type={self._ptr_type})"
    
    
class BufferPointer:
    def __init__(self, handle: 'BufferHandle'=None, item: int | slice | tuple[int, ...] | None=None):
        self._handle = handle
        self._item = item
        
        if self._item is None and self._handle is not None:
            self._item = slice(0, handle.n_pages, 1)
            
    def initialize(self, handle: 'BufferHandle'=None, item: int | slice | tuple[int, ...] | None=None):
        self._handle = handle
        self._item = item
        
        if self._item is None and self._handle is not None:
            self._item = slice(0, handle.n_pages, 1)
            
    @property
    def size(self) -> int:
        return self._handle.size
        
    @property
    def raw_handle(self) -> 'BufferHandle':
        if self._item != slice(0, self._handle.n_pages, 1) and self._item != slice(0, self._handle.n_pages, None):
            if self.is_circular:
                raise Exception(f"Cannot access raw_handle of a CircularBufferPointer that is not pointing to the entire buffer. The current item is {self._item}. Use 'resolve' method instead.")
            return self.resolve(is_read=True)
        return self._handle
    
    @property
    def is_circular(self) -> bool:
        return isinstance(self._handle, CircularBufferHandle)
    
    @property
    def n_pages(self) -> int:
        return self.resolve(is_read=True).n_pages
    
    @property
    def page_size(self) -> int:
        return self._handle.page_size
        
    def __getitem__(self, new_item) -> 'BufferPointer':
        if isinstance(new_item, (int, slice, tuple)):
            if self._item is None:
                return BufferPointer(handle=self._handle, item=new_item)
            elif isinstance(self._item, int):
                if new_item != 0 and new_item != slice(0, 1, 1) and new_item != slice(0, 1, None):
                    raise Exception(f"Cannot slice a BufferPointer that points to a single page.")
                return BufferPointer(handle=self._handle, item=self._item)
            elif isinstance(self._item, slice):
                if isinstance(new_item, int):
                    return BufferPointer(handle=self._handle, item=self._item.start + new_item)
                elif isinstance(new_item, slice):
                    start = self._item.start + (new_item.start or 0)
                    stop = self._item.start + (new_item.stop or (self._handle.n_pages - self._item.start))
                    return BufferPointer(handle=self._handle, item=slice(start, stop, new_item.step))
                elif isinstance(new_item, tuple):
                    return BufferPointer(handle=self._handle, item=tuple(self._item.start + i for i in new_item))
            elif isinstance(self._item, tuple):
                if isinstance(new_item, int):
                    return BufferPointer(handle=self._handle, item=self._item[new_item])
                elif isinstance(new_item, slice):
                    return BufferPointer(handle=self._handle, item=self._item[new_item])
                elif isinstance(new_item, tuple):
                    return BufferPointer(handle=self._handle, item=tuple(self._item[i] for i in new_item))
        return super().__getitem__(new_item)

    def resolve(self, is_read: bool=None) -> 'BufferHandle':
        if isinstance(self._handle, CircularBufferHandle):
            if is_read is None:
                raise ValueError(f"Cannot resolve the reference since is_read is not specified for CircularBufferHandle.")
            elif is_read:
                offset = self._handle._rd_ptr
            else:
                offset = self._handle._wr_ptr
        else:
            offset = 0
        
        page_ptrs = self._handle.page_ptrs
        
        if isinstance(self._item, int):
            idx  = (self._item + offset) % self._handle.n_pages
            page_ptrs = [page_ptrs[idx]]
        elif isinstance(self._item, slice):
            start = (self._item.start + offset) % self._handle.n_pages
            stop = (self._item.stop + offset) % self._handle.n_pages
            
            if start < stop:
                page_ptrs = page_ptrs[start:stop]
            else:
                page_ptrs = page_ptrs[start:] + page_ptrs[:stop]
        elif isinstance(self._item, tuple):
            page_ptrs = [page_ptrs[(i + offset) % self._handle.n_pages] for i in self._item]
        
        return BufferHandle(page_size=self._handle.page_size, n_pages=len(page_ptrs), page_ptrs=page_ptrs)


class BufferHandle:
    def __init__(self, page_size: int, n_pages: int, page_ptrs: list[Pointer]):
        self._page_size: int = page_size
        self._n_pages: int = n_pages
        self._page_ptrs: list[Pointer] = page_ptrs
        
        if isinstance(self._page_ptrs, Pointer):
            self._page_ptrs = [self._page_ptrs]
        
        for ptr in self._page_ptrs:
            if not isinstance(ptr, Pointer):
                raise ValueError(f"All pages must be able to be converted to Pointer.")
            if ptr.size != page_size:
                raise ValueError(f"All pages must have the same size of {page_size}, but found page with size {ptr.size}.")
            
        if len(self._page_ptrs) != n_pages:
            raise ValueError(f"Expected {n_pages} pages, but got {len(self._page_ptrs)}.")
    
    def __setstate__(self, state: dict):
        self._page_size = state["page_size"]
        self._n_pages = state["n_pages"]
        self._page_ptrs = state["page_ptrs"]

    @property
    def page_size(self) -> int:
        return self._page_size
    
    @property
    def n_pages(self) -> int:
        return self._n_pages
    
    @property
    def page_ptrs(self) -> list[Pointer]:
        return self._page_ptrs
    
    @property
    def size(self) -> int:
        return self._page_size * self._n_pages
    
    
class CircularBufferHandle(BufferHandle):
    def __init__(self, page_size: int, n_pages: int, page_ptrs: list[Pointer]):
        super().__init__(page_size, n_pages, page_ptrs)
        
        self._rd_ptr    = 0
        self._wr_ptr    = 0
        self._rsvd_ptr  = 0
        
        self._rd_ptr_phase = False
        self._wr_ptr_phase = False
        self._rsvd_ptr_phase = False

    def __getitem__(self, item) -> BufferHandle:
        raise Exception(f"Cannot create reference for CircularBufferPointer with slicing. Use specialized reference methods 'rd_ref' or 'wr_ref' instead.")

    @property
    def _alloc_space(self) -> int:
        if self._rsvd_ptr == self._rd_ptr:
            if self._rd_ptr_phase == self._rsvd_ptr_phase:
                return 0
            else:
                return self._n_pages
        elif self._rsvd_ptr > self._rd_ptr:
            return self._rsvd_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._rsvd_ptr)
        
    @property
    def _real_space(self) -> int:
        if self._wr_ptr == self._rd_ptr:
            if self._rd_ptr_phase == self._wr_ptr_phase:
                return 0
            else:
                return self._n_pages
        elif self._wr_ptr > self._rd_ptr:
            return self._wr_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._wr_ptr)
        
    def check_vacancy(self, n_pages: int) -> bool:
        return self._alloc_space + n_pages <= self._n_pages
    
    def check_occupancy(self, n_pages: int) -> bool:
        return self._real_space >= n_pages
    
    def allocate_cb_space(self, n_pages: int):
        if self._rsvd_ptr + n_pages >= self._n_pages:
            self._rsvd_ptr_phase = not self._rsvd_ptr_phase
        self._rsvd_ptr = (self._rsvd_ptr + n_pages) % self._n_pages
        
    def occupy_cb_space(self, n_pages: int):
        if self._wr_ptr + n_pages >= self._n_pages:
            self._wr_ptr_phase = not self._wr_ptr_phase
        self._wr_ptr = (self._wr_ptr + n_pages) % self._n_pages
        
    def deallocate_cb_space(self, n_pages: int):
        if self._rd_ptr + n_pages >= self._n_pages:
            self._rd_ptr_phase = not self._rd_ptr_phase
        self._rd_ptr = (self._rd_ptr + n_pages) % self._n_pages


class _MemoryHandleDataEntry:
    def __init__(self, addr: int, elem: _DataElement):
        self.addr = addr
        self.elem = elem
        self.is_expired = False
        
        self.nxt_entry: _MemoryHandleDataEntry = None
        self.prv_entry: _MemoryHandleDataEntry = None
        
class _MemoryHandleChannelSpaceTracker:
    def __init__(self, base_addr: int, size: int):
        self._base_addr = base_addr
        self._size = size
        
        self._head: _MemoryHandleDataEntry = None
        self._tail: _MemoryHandleDataEntry = None
        self._addr_map: dict[int, _MemoryHandleDataEntry] = {}
        
    def empty_space(self) -> int:
        if self._head is None:
            return self._size
        
        head_addr = self._head.addr
        tail_addr = self._tail.addr + self._tail.elem.size
        
        if head_addr < tail_addr:
            return (self._base_addr + self._size - tail_addr) + (head_addr - self._base_addr)
        else:
            return head_addr - tail_addr
        
    def allocate_space(self, elem: _DataElement) -> int | None:
        if self._tail is None:
            if elem.size > self._size:
                return None
            
            elem.addr = self._base_addr
            
            self._head = _MemoryHandleDataEntry(self._base_addr, elem)
            self._tail = self._head
            self._addr_map[self._base_addr] = self._head
            
            return self._base_addr
        
        else:
            head_addr = self._head.addr
            tail_addr = self._tail.addr + self._tail.elem.size
            
            if head_addr < tail_addr:
                st, ed = tail_addr, self._base_addr + self._size
                if ed - st < elem.size:
                    st, ed = self._base_addr, head_addr
            else:
                st, ed = tail_addr, head_addr
                
            if (ed - st) > elem.size:
                elem.addr = st
                
                new_entry = _MemoryHandleDataEntry(st, elem)
                
                new_entry.prv_entry = self._tail
                self._tail.nxt_entry = new_entry
                
                self._tail = new_entry
                self._addr_map[st] = new_entry
                
                return st
        
            return None
        
    def deallocate_space(self, addr: int) -> bool:
        if self.search_elem(addr) is None:
            return False
        
        if addr not in self._addr_map:
            raise Exception(f"Cannot find the allocated address {addr} to deallocate.")
        
        entry = self._addr_map[addr]
        entry.is_expired = True
        
        while self._head is not None and self._head.is_expired:
            del self._addr_map[self._head.addr]
            self._head = self._head.nxt_entry
            if self._head is not None:
                self._head.prv_entry = None
            else:
                self._tail = None
        
        while self._tail is not None and self._tail.is_expired:
            del self._addr_map[self._tail.addr]
            self._tail = self._tail.prv_entry
            if self._tail is not None:
                self._tail.nxt_entry = None
            else:
                self._head = None
        
        return True
            
    def search_elem(self, addr: int) -> _DataElement | None:
        if addr not in self._addr_map:
            return None
        return self._addr_map[addr].elem

class MemoryHandle:
    def __init__(self, mem_id: str, base_addr: int, size: int, n_channels: int=1):
        self._mem_id = mem_id
        self._base_addr = base_addr
        self._size = size
        self._n_channels = n_channels
        self._channel_size = self._size // self._n_channels
        
        if self._size % self._n_channels != 0:
            raise Exception(f"Memory size {self._size} is not divisible by number of channels {self._n_channels}.")
        
        self._ch_trackers: list[_MemoryHandleChannelSpaceTracker] = [
            _MemoryHandleChannelSpaceTracker(base_addr + i * self._channel_size, self._channel_size)
            for i in range(self._n_channels)
        ]
                
    def get_data_element(self, key: Any) -> _DataElement:
        if isinstance(key, int):
            ch_id = (key - self._base_addr) // self._channel_size
            if ch_id < 0 or ch_id >= self._n_channels:
                raise Exception(f"Address {key} is out of range for memory handle {self}.")
            elem = self._ch_trackers[ch_id].search_elem(key)
            if elem is None:
                raise Exception(f"Cannot find the data element at address {key} in memory handle {self}.")
            return elem
        elif isinstance(key, Pointer):
            return self.get_data_element(key.addr)
        else:
            raise TypeError(f"Key must be an int or Pointer, got {type(key)}.")
        
    def get_content(self, key: Any, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> Any:
        if isinstance(key, BufferPointer):
            key = key.resolve(is_read=True)

        if isinstance(key, int):
            content = self.get_data_element(key).content
        elif isinstance(key, Pointer):
            content = self.get_data_element(key).content
        elif isinstance(key, BufferHandle):
            page_contents = []
            for page_ptr in key.page_ptrs:
                page: Page = self.get_data_element(page_ptr)
                page_content = page.content_view(shape=(-1,), dtype=torch.uint8)
                page_contents.append(page_content)
            content = torch.concat(page_contents, dim=0)
        
        if isinstance(content, torch.Tensor):
            if dtype is not None:
                content = content.view(dtype=dtype)
            if shape is not None:
                content = content.reshape(shape=shape)
            content = content.clone()
        
        return content
    
    def set_content(self, key: Any, value: Any, page_offset: int=0):
        if isinstance(key, BufferPointer):
            key = key.resolve(is_read=False)
        
        if isinstance(key, int):
            self.get_data_element(key).content = value
        elif isinstance(key, Pointer):
            page: Page = self.get_data_element(key)
            page.set_content(value=value, offset=0)
        elif isinstance(key, BufferHandle):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Buffer content must be a torch.Tensor, got {type(value)}.")
            
            paged_value = value.view(dtype=torch.uint8).reshape((key.n_pages, -1)).clone()
            
            for page_idx, page_ptr in enumerate(key.page_ptrs):
                page: Page = self.get_data_element(page_ptr)
                page.set_content(value=paged_value[page_idx, :], offset=page_offset)
        else:
            raise TypeError(f"Key must be an int or Pointer, got {type(key)}.")
        
    def allocate_var_ptr(self, var_size: int, initial_value: Any, channel_id: int=0, dst_ptr: Pointer=None) -> Pointer | None:
        if channel_id >= self._n_channels:
            raise Exception(f"Invalid channel id {channel_id} which exceeds the number of channels {self._n_channels}")
        
        elem = Variable(addr=None, size=var_size, content=initial_value)
        
        if self._ch_trackers[channel_id].allocate_space(elem) is None:
            return None
        
        if dst_ptr is not None:
            dst_ptr.initialize(elem)
            return dst_ptr
        return Pointer(data_element=elem)
    
    def allocate_page_ptr(self, page_size: int, channel_id: int=0, dst_ptr: Pointer=None) -> Pointer | None:
        if channel_id >= self._n_channels:
            raise Exception(f"Invalid channel id {channel_id} which exceeds the number of channels {self._n_channels}")
        
        elem = Page(addr=None, size=page_size, content=None)
        
        if self._ch_trackers[channel_id].allocate_space(elem) is None:
            return None
        
        if dst_ptr is not None:
            dst_ptr.initialize(elem)
            return dst_ptr
        return Pointer(data_element=elem)

    def allocate_buffer_ptr(self, page_size: int, n_pages: int, is_circular: bool, channel_id: int | tuple[int]=0) -> CircularBufferHandle | BufferHandle | None:
        is_channel_sharded = isinstance(channel_id, Sequence)
        page_ptrs = []

        for i in range(n_pages):
            if is_channel_sharded:
                # channel_id = i % self._n_channels
                selected_channel_id = channel_id[i % len(channel_id)]
            else:
                selected_channel_id = channel_id
                
            if selected_channel_id >= self._n_channels:
                raise Exception(f"Invalid channel id {selected_channel_id} which exceeds the number of channels {self._n_channels}")

            page_ptr = self.allocate_page_ptr(page_size, channel_id=selected_channel_id)
            if page_ptr is None:
                self.deallocate_ptr(*page_ptrs)
                return None
            page_ptrs.append(page_ptr)

        if is_circular:
            return CircularBufferHandle(page_size=page_size, n_pages=n_pages, page_ptrs=page_ptrs)
        else:
            return BufferHandle(page_size=page_size, n_pages=n_pages, page_ptrs=page_ptrs)

    def deallocate_ptr(self, *ptrs: Pointer | BufferHandle):
        for ptr in ptrs:
            if isinstance(ptr, Pointer):
                addr = ptr.addr
                ch_id = (addr - self._base_addr) // self._channel_size
                
                if ch_id < 0 or ch_id >= self._n_channels:
                    raise Exception(f"Address {addr} is out of range for memory handle {self}.")
                
                if not self._ch_trackers[ch_id].deallocate_space(addr):
                    raise KeyError(f"No data element found at address {addr} in memory handle with base address {self._base_addr}.")
            elif isinstance(ptr, BufferHandle):
                for page_ptr in ptr.page_ptrs:
                    self.deallocate_ptr(page_ptr)
            else:
                raise TypeError(f"Expected Pointer or BufferPointer, got {type(ptr)}.")

    @property
    def mem_id(self) -> str:
        return self._mem_id

    @property
    def base_addr(self) -> int:
        return self._base_addr
    
    @property
    def size(self) -> int:
        return self._size
    
    def __str__(self) -> str:
        return f"MemoryHandle(mem_id={self._mem_id}, base_addr={self._base_addr}, size={self._size})"


def create_var_ptr(mem_handle: MemoryHandle, var_size: int, initial_value: Any) -> Pointer | None:
    return mem_handle.allocate_var_ptr(var_size, initial_value)

def create_page_ptr(mem_handle: MemoryHandle, page_size: int) -> Pointer | None:
    return mem_handle.allocate_page_ptr(page_size)

def create_uniform_buffer(mem_handle: MemoryHandle, page_size: int, n_pages: int, is_circular: bool, channel_id: int | Sequence[int]=0) -> BufferPointer | None:
    bf_handle = mem_handle.allocate_buffer_ptr(page_size, n_pages, is_circular=is_circular, channel_id=channel_id)
    if bf_handle is None:
        return None
    return BufferPointer(handle=bf_handle, item=None)

def create_distributed_buffer(mem_handles: list[MemoryHandle], page_size: int, n_pages: int, channel_id: int=0, contiguous_n_pages: int=1) -> BufferPointer | None:
    if n_pages % (len(mem_handles) * contiguous_n_pages) != 0:
        n_pages += (len(mem_handles) * contiguous_n_pages - (n_pages % (len(mem_handles) * contiguous_n_pages)))
    n_page_group_per_handle = n_pages // len(mem_handles) // contiguous_n_pages
    
    page_ptrs: list[Pointer] = []
    page_ptr_to_handle: list[MemoryHandle] = []
    
    for _ in range(n_page_group_per_handle):
        for mem_handle in mem_handles:
            for _ in range(contiguous_n_pages):
                if len(page_ptrs) >= n_pages:
                    break
                
                page_ptr = mem_handle.allocate_page_ptr(page_size, channel_id=channel_id)
                
                if page_ptr is None:
                    for h, p in zip(page_ptr_to_handle, page_ptrs):
                        h.deallocate_ptr(p)   # deallocate previously allocated pages
                    return None
                
                page_ptrs.append(page_ptr)
                page_ptr_to_handle.append(mem_handle)
            
    bf_handle = BufferHandle(page_size=page_size, n_pages=n_pages, page_ptrs=page_ptrs)
    if bf_handle is None:
        return None
    return BufferPointer(handle=bf_handle, item=None)