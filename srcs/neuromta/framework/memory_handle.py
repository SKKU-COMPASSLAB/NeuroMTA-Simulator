import enum
import math
import torch
from typing import Any, Sequence


__all__ = [
    "Variable",
    "Page",
    "PointerType",
    "Pointer",
    "Reference",
    "BufferHandle",
    "CircularBufferHandle",
    "MemoryHandle",
    
    "create_var_ptr",
    "create_page_ptr",
    "create_uniform_buffer",
    "create_distributed_buffer",
]


class _DataElement:
    def __init__(self, addr: int, size: int, content: Any=None):
        self._addr = addr
        self._size = size
        self._content = content
        
    def copy_from(self, other: '_DataElement'):
        if not isinstance(other, _DataElement):
            raise TypeError(f"Expected _DataElement, got {type(other)}")
        
        self._addr = other.addr
        self._size = other.size
        
        if isinstance(self._content, torch.Tensor):
            if isinstance(other.content, torch.Tensor):
                self._content.copy_(other.content)
            else:
                self._content.fill_(other.content)
        else:            
            self._content = other.content

    @property
    def addr(self) -> int:
        return self._addr

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
            raise TypeError(f"[ERROR] Page content must be a torch.Tensor, got {type(value)}.")
        
        if self._content is None:
            self._content = torch.zeros(self.size, dtype=torch.uint8)
        
        value = value.view(dtype=torch.uint8).flatten()
        self._content[offset:offset + value.numel()] = value
        
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
            raise Exception(f"[ERROR] Page content must be a torch.Tensor, got {type(value)}.")
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
    
    
class Reference:
    def __init__(self, handle: 'BufferHandle', item: int | slice | tuple[int, ...] | None=None):
        self._handle = handle
        self._item = item
        
    @property
    def raw_handle(self) -> 'BufferHandle':
        return self._handle
        
    def __getitem__(self, new_item) -> 'Reference':
        if isinstance(new_item, (int, slice, tuple)):
            if self._item is None:
                return Reference(handle=self._handle, item=new_item)
            elif isinstance(self._item, int):
                if new_item != 0:
                    raise Exception(f"[ERROR] Cannot slice a Reference that points to a single page.")
                return Reference(handle=self._handle, item=self._item)
            elif isinstance(self._item, slice):
                if isinstance(new_item, int):
                    return Reference(handle=self._handle, item=self._item.start + new_item)
                elif isinstance(new_item, slice):
                    start = self._item.start + (new_item.start or 0)
                    stop = self._item.start + (new_item.stop or (self._handle.n_pages - self._item.start))
                    return Reference(handle=self._handle, item=slice(start, stop, new_item.step))
                elif isinstance(new_item, tuple):
                    return Reference(handle=self._handle, item=tuple(self._item.start + i for i in new_item))
            elif isinstance(self._item, tuple):
                if isinstance(new_item, int):
                    return Reference(handle=self._handle, item=self._item[new_item])
                elif isinstance(new_item, slice):
                    return Reference(handle=self._handle, item=self._item[new_item])
                elif isinstance(new_item, tuple):
                    return Reference(handle=self._handle, item=tuple(self._item[i] for i in new_item))
        return super().__getitem__(new_item)

    def resolve(self, is_read: bool=None) -> 'BufferHandle':
        if isinstance(self._handle, CircularBufferHandle):
            if is_read is None:
                raise ValueError(f"[ERROR] Cannot resolve the reference since is_read is not specified for CircularBufferHandle.")
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
            page_ptrs = page_ptrs[start:stop]
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
                raise ValueError(f"[ERROR] All pages must be able to be converted to Pointer.")
            if ptr.size != page_size:
                raise ValueError(f"[ERROR] All pages must have the same size of {page_size}, but found page with size {ptr.size}.")
            
        if len(self._page_ptrs) != n_pages:
            raise ValueError(f"[ERROR] Expected {n_pages} pages, but got {len(self._page_ptrs)}.")
    
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
    
    
class CircularBufferHandle(BufferHandle):
    def __init__(self, page_size: int, n_pages: int, page_ptrs: list[Pointer]):
        super().__init__(page_size, n_pages, page_ptrs)
        
        self._rd_ptr    = 0
        self._wr_ptr    = 0
        self._rsvd_ptr  = 0
    
    def __getitem__(self, item) -> BufferHandle:
        raise Exception(f"[ERROR] Cannot create reference for CircularBufferPointer with slicing. Use specialized reference methods 'rd_ref' or 'wr_ref' instead.")

    @property
    def _alloc_space(self) -> int:
        if self._rsvd_ptr >= self._rd_ptr:
            return self._rsvd_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._rsvd_ptr)
        
    @property
    def _real_space(self) -> int:
        if self._wr_ptr >= self._rd_ptr:
            return self._wr_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._wr_ptr)
        
    def check_vacancy(self, n_pages: int) -> bool:
        return self._alloc_space + n_pages <= self._n_pages
    
    def check_occupancy(self, n_pages: int) -> bool:
        return self._real_space >= n_pages
    
    def allocate_cb_space(self, n_pages: int):
        self._rsvd_ptr = (self._rsvd_ptr + n_pages) % self._n_pages
        
    def occupy_cb_space(self, n_pages: int):
        self._wr_ptr = (self._wr_ptr + n_pages) % self._n_pages
        
    def deallocate_cb_space(self, n_pages: int):
        self._rd_ptr = (self._rd_ptr + n_pages) % self._n_pages


class MemoryHandle:
    def __init__(self, mem_id: str, base_addr: int, size: int, n_channels: int=1):
        self._mem_id = mem_id
        self._base_addr = base_addr
        self._size = size
        self._n_channels = n_channels
        self._channel_size = self._size // self._n_channels
        
        if self._size % self._n_channels != 0:
            raise Exception(f"[ERROR] Memory size {self._size} is not divisible by number of channels {self._n_channels}.")
        
        self._data_elements: dict[int, _DataElement] = {}
                
    def get_data_element(self, key: Any) -> _DataElement:
        if isinstance(key, int):
            return self._data_elements[key]
        elif isinstance(key, Pointer):
            return self._data_elements[key.addr]
        else:
            raise TypeError(f"[ERROR] Key must be an int or Pointer, got {type(key)}.")
        
    def get_content(self, key: Any, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> Any:
        if isinstance(key, Reference):
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
        
        return content
    
    def set_content(self, key: Any, value: Any, offset: int=0):
        if isinstance(key, Reference):
            key = key.resolve(is_read=False)
        
        if isinstance(key, int):
            self.get_data_element(key).content = value
        elif isinstance(key, Pointer):
            page: Page = self.get_data_element(key)
            page.set_content(value=value, offset=0)
        elif isinstance(key, BufferHandle):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"[ERROR] Buffer content must be a torch.Tensor, got {type(value)}.")
            
            paged_value = value.view(dtype=torch.uint8).reshape((key.n_pages, -1))
            
            for page_idx, page_ptr in enumerate(key.page_ptrs):
                page: Page = self.get_data_element(page_ptr)
                page.set_content(value=paged_value[page_idx, :], offset=offset)
        else:
            raise TypeError(f"[ERROR] Key must be an int or Pointer, got {type(key)}.")

    def get_overlapping_data_addr(self, addr: int, size: int=1) -> int: # returns False if the address space is not overlapping with any memory handles
        keys = sorted(self._data_elements.keys())
        left, right = 0, len(keys) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_addr = keys[mid]
            page = self._data_elements[mid_addr]

            if not (addr + size <= page.addr or addr >= page.addr + page.size):
                return mid_addr
            if addr < page.addr:
                right = mid - 1
            else:
                left = mid + 1
                
        return -1

    def allocate_var_ptr(self, var_size: int, initial_value: Any, channel_id: int=0, dst_ptr: Pointer=None) -> Pointer | None:
        ch_st_addr = self.base_addr + channel_id * self._channel_size
        ch_ed_addr = self.base_addr + (channel_id + 1) * self._channel_size
        
        addr = ch_st_addr
        
        # for addr in range(ch_st_addr, ch_ed_addr - var_size + 1, var_size):
        while addr <= ch_ed_addr - var_size:
            overlap = self.get_overlapping_data_addr(addr, size=var_size)
            
            if overlap < 0:
                var = Variable(addr=addr, size=var_size, content=initial_value)
                self._data_elements[addr] = var
                
                if dst_ptr is not None:
                    dst_ptr.initialize(var)
                    return dst_ptr
                else:
                    return Pointer(data_element=var)
            else:
                # continue
                while addr < overlap + self._data_elements[overlap].size:
                    addr += var_size
        return None
    
    def allocate_page_ptr(self, page_size: int, channel_id: int=0, dst_ptr: Pointer=None) -> Pointer | None:
        ch_st_addr = self.base_addr + channel_id * self._channel_size
        ch_ed_addr = self.base_addr + (channel_id + 1) * self._channel_size
        
        addr = ch_st_addr
        
        # for addr in range(ch_st_addr, ch_ed_addr - page_size + 1, page_size):
        while addr <= ch_ed_addr - page_size:
            overlap = self.get_overlapping_data_addr(addr, size=page_size)
            
            if overlap < 0:
                page = Page(addr=addr, size=page_size)
                self._data_elements[addr] = page
                
                if dst_ptr is not None:
                    dst_ptr.initialize(page)
                    return dst_ptr
                else:
                    return Pointer(data_element=page)
            else:
                # continue
                while addr < overlap + self._data_elements[overlap].size:
                    addr += page_size
        return None

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
                raise Exception(f"[ERROR] Invalid channel id {selected_channel_id} which exceeds the number of channels {self._n_channels}")

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
                if addr in self._data_elements:
                    del self._data_elements[addr]
                else:
                    raise KeyError(f"[ERROR] No data element found at address {addr} in memory handle with base address {self._base_addr}.")
            elif isinstance(ptr, BufferHandle):
                for page_ptr in ptr.page_ptrs:
                    self.deallocate_ptr(page_ptr)
            else:
                raise TypeError(f"[ERROR] Expected Pointer or BufferPointer, got {type(ptr)}.")
    
    def clear(self):
        self._data_elements.clear()

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

def create_uniform_buffer(mem_handle: MemoryHandle, page_size: int, n_pages: int, is_circular: bool, channel_id: int | Sequence[int]=0) -> Reference | None:
    bf_handle = mem_handle.allocate_buffer_ptr(page_size, n_pages, is_circular=is_circular, channel_id=channel_id)
    if bf_handle is None:
        return None
    return Reference(handle=bf_handle, item=None)

def create_distributed_buffer(mem_handles: list[MemoryHandle], page_size: int, n_pages: int, channel_id: int=0) -> Reference | None:
    n_page_per_handle = math.ceil(n_pages / len(mem_handles))
    page_ptrs = []
    
    for mem_handle in mem_handles:
        for i in range(n_page_per_handle):
            if len(page_ptrs) >= n_pages:
                break
            page_ptr = mem_handle.allocate_page_ptr(page_size, channel_id=channel_id)
            if page_ptr is None:
                return None
            page_ptrs.append(page_ptr)
            
    bf_handle = BufferHandle(page_size=page_size, n_pages=n_pages, page_ptrs=page_ptrs)
    if bf_handle is None:
        return None
    return Reference(handle=bf_handle, item=None)