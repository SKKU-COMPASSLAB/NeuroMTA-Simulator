import torch
from typing import TypeVar, Generic, Sequence
from neuromta.framework.serializer import *


__all__ = [
    "DataContainer",
]


T = TypeVar('T')

class DataContainer(Generic[T], SerializableCoreObject):
    def __init__(self, data: T=None, shape: Sequence[int]=None, dtype: torch.dtype=None, initial_value: int=0):
        self._data = data
        self._shape = shape
        self._dtype = dtype
        self._initial_value = initial_value
        
        if self.data is None:
            if shape is not None and dtype is not None:
                self.data = torch.full(size=shape, fill_value=initial_value, dtype=dtype)
        else:
            assert shape is None and dtype is None
            if isinstance(self.data, torch.Tensor):
                self._shape = tuple(self.data.size())
                self._dtype = self.data.dtype
            else:
                self._shape = None
                self._dtype = None
                
    def get_state(self):
        return {
            "data": self.data,
            "shape": self.shape,
            "dtype": self.dtype,
            "initial_value": self._initial_value,
        }
    
    @classmethod
    def from_state(cls, core, state: dict) -> 'DataContainer':
        return cls(
            data=state.get("data", None),
            shape=state.get("shape", None),
            dtype=state.get("dtype", None),
            initial_value=state.get("initial_value", 0),
        )
    
    @property
    def data(self) -> T:
        return self._data
    
    @data.setter
    def data(self, value: T):
        if isinstance(value, torch.Tensor):
            self._dtype = value.dtype
            self._shape = tuple(value.size())
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
        if self.is_mem_segment or isinstance(self.data, torch.Tensor):
            return self.data.numel() * self.data.element_size()
        return None
    
    @property
    def is_mem_segment(self) -> bool:
        return isinstance(self.data, torch.Tensor)  # TODO: More robust check
    
    def __repr__(self):
        return f"DataContainer(shape={self.shape}, dtype={self.dtype}, size={self.size}, is_mem_segment={self.is_mem_segment})"
    
    def __str__(self):
        return self.__repr__()