import torch
from typing import TypeVar, Generic, Sequence
import functools

from neuromta.framework.simulation_mode import is_global_performance_mode


__all__ = [
    "DataContainer",
]


T = TypeVar('T')

class DataContainer(Generic[T]):
    def __init__(self, data: T=None, shape: Sequence[int]=None, dtype: torch.dtype=None, initial_value: int=0, allocate_payload: bool=None):
        self._data = data
        self._shape = tuple(shape) if shape is not None else None
        self._dtype = dtype
        self._initial_value = initial_value
        self._logical_size = self._calc_logical_size(self._shape, self._dtype)
        
        if self.data is None:
            if shape is not None and dtype is not None:
                if allocate_payload is None:
                    allocate_payload = not is_global_performance_mode()
                if allocate_payload:
                    self.data = torch.full(size=shape, fill_value=initial_value, dtype=dtype)
        else:
            assert shape is None and dtype is None
            if isinstance(self.data, torch.Tensor):
                self._shape = tuple(self.data.size())
                self._dtype = self.data.dtype
                self._logical_size = self._calc_logical_size(self._shape, self._dtype)
            else:
                self._shape = None
                self._dtype = None
                self._logical_size = None

    @staticmethod
    def _calc_logical_size(shape: Sequence[int], dtype: torch.dtype) -> int:
        if shape is None or dtype is None:
            return None
        numel = functools.reduce(lambda x, y: x * y, shape, 1)
        return numel * dtype.itemsize

    def set_metadata(self, shape: Sequence[int]=None, dtype: torch.dtype=None, logical_size: int=None):
        if shape is not None:
            self._shape = tuple(shape)
        if dtype is not None:
            self._dtype = dtype
        self._logical_size = logical_size if logical_size is not None else self._calc_logical_size(self._shape, self._dtype)
        return self

    def ensure_payload(self, initial_value: int=None) -> T:
        if self.data is None and self.shape is not None and self.dtype is not None:
            if initial_value is None:
                initial_value = self._initial_value
            self.data = torch.full(size=self.shape, fill_value=initial_value, dtype=self.dtype)
        return self.data
    
    @property
    def data(self) -> T:
        return self._data
    
    @data.setter
    def data(self, value: T):
        if isinstance(value, torch.Tensor):
            self._dtype = value.dtype
            self._shape = tuple(value.size())
            self._logical_size = self._calc_logical_size(self._shape, self._dtype)
            self._data = value
        else:
            self._data = value
        
    @property
    def shape(self) -> Sequence[int]:
        return self._shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def size(self) -> int:
        if self.has_payload:
            return self.data.numel() * self.data.element_size()
        return self._logical_size
    
    @property
    def is_mem_segment(self) -> bool:
        return isinstance(self.data, torch.Tensor)  # TODO: More robust check

    @property
    def has_payload(self) -> bool:
        return isinstance(self.data, torch.Tensor)
    
    def __repr__(self):
        return f"DataContainer(shape={self.shape}, dtype={self.dtype}, size={self.size}, is_mem_segment={self.is_mem_segment})"
    
    def __str__(self):
        return self.__repr__()
