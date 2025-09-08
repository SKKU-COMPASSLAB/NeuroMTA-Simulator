import math
import copy
import functools
from typing import Sequence


__all__ = [
    "DimensionDescriptor",
    "VariableDescriptor",
    "BufferDescriptor",
]


class DimensionDescriptor:
    def __init__(self, dim_id: str, size: int, tile: int=None):
        self.dim_id = dim_id
        self._size  = size
        self._tile  = tile
        
        if self._tile is None:
            self._tile = self._size
        
    def set_tile(self, tile: int):
        if tile > self._size:
            raise Exception(f"[ERROR] Cannot size tiling factor of the dimension '{self.dim_id}' as '{tile}' since it exceeds the size of the dimension '{self._size}'")
        
        self._tile = tile
        
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def tile(self) -> int:
        return self._tile
    
    @property
    def n_tile(self) -> int:
        return math.ceil(self.size / self.tile)
    

class VariableDescriptor:
    def __init__(self, var_id: str, dims: list[DimensionDescriptor]):
        self.var_id = var_id
        self._dims  = list(copy.deepcopy(dims))

        self._tile_offsets: list[int] = None
        self._update_offsets()
            
    def _update_offsets(self):
        self._tile_offsets  = [1]
        for d in list(reversed(self._dims))[:-1]:
            self._tile_offsets.insert(0, self._tile_offsets[0] * d.n_tile)
        
    def transpose(self, dim_orders: Sequence[int]):
        new_dims = [self._dims[di] for di in dim_orders]
        self._dims = new_dims
        self._update_offsets()
    
    def dim(self, dim_idx: int):
        return self._dims[dim_idx]

    def tile_id(self, *tile_idx: int):
        return sum([ti*do for ti, do in zip(tile_idx, self._tile_offsets)])
    
    def create_buffer(self, tile_id: int | Sequence[int]):
        if isinstance(tile_id, Sequence):
            tile_id = self.tile_id(*tile_id)
            
        return BufferDescriptor(var_desc=self, tile_id=tile_id)
    
    @property
    def n_tile(self) -> int:
        return functools.reduce(lambda a, b: a*b, map(lambda x: x.n_tile, self._dims), 1)
    
    @property
    def tile_size(self) -> int:
        return functools.reduce(lambda a, b: a*b, map(lambda x: x.tile, self._dims), 1)
    
    def __eq__(self, value):
        if not isinstance(value, VariableDescriptor):
            raise Exception(f"[ERROR] Cannot compare variable with '{type(value).__name__}'")
        
        return self.var_id == value.var_id

    def __ne__(self, value):
        return not self.__eq__(value)
    
    def __hash__(self):
        return hash(id(self))
    
    def __str__(self):
        return f"Variable(var={self.var_id}, dims={tuple(map(lambda x: x.dim_id, self._dims))})"
        

class BufferDescriptor:
    def __init__(self, var_desc: VariableDescriptor, tile_id: int):
        self.var_desc   = var_desc
        self.tile_id    = tile_id
        
    def __eq__(self, value):
        if not isinstance(value, BufferDescriptor):
            raise Exception(f"[ERROR] Cannot compare buffer with '{type(value).__name__}'")
        
        return self.var_desc.__eq__(value.var_desc) and self.tile_id == value.tile_id

    def __ne__(self, value):
        return not self.__eq__(value)
    
    def __hash__(self):
        return hash((hash(self.var_desc), self.tile_id))
    
    def __str__(self):
        return f"Buffer(var={self.var_desc.var_id}, tile_id={self.tile_id})"


if __name__ == "__main__":
    M = DimensionDescriptor("M", 128, 32)
    N = DimensionDescriptor("N", 128, 32)
    K = DimensionDescriptor("K", 128, 32)
    
    ifm = VariableDescriptor("IFM", (M, K))
    wgt = VariableDescriptor("WGT", (K, N))
    ofm = VariableDescriptor("OFM", (M, N))
    
    print(ifm)
    print(wgt)
    print(ofm)
    
    for m in range(M.n_tile):
        for n in range(N.n_tile):
            for k in range(K.n_tile):
                print(f"STEP {(m, n, k)}")
                print(f"  - ifm tile buffer: {ifm.create_buffer(ifm.tile_id(m, k))}")
                print(f"  - wgt tile buffer: {wgt.create_buffer(wgt.tile_id(k, n))}")
                print(f"  - ofm tile buffer: {ofm.create_buffer(ofm.tile_id(m, n))}")