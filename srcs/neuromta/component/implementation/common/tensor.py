import abc
import functools
import torch
from typing import Iterator


__all__ = [
    "TensorDimension",
    "TensorDimensionPool",
    "TensorDimensionCursor",
    
    "_TensorPreprocessing",
    "TensorPermute",
    "TensorReshape",
    
    "TensorLayout",
    "TensorLayoutPool",
]


#################################################################
# Tensor Dimension Management
#################################################################

class TensorDimension:
    def __init__(self, dim_id: str):
        self._dim_id = dim_id
        
        self._size: int = None
        self._tile: int = None
        self._pad: int  = 0
        
        self._tiling_enabled: bool = True
        
    def initialize(self, size: int):
        if self.is_initialized and self._size != size:
            raise RuntimeError(f"Dimension '{self._dim_id}' is already initialized with different size '{self._size}' or tile '{self._tile}'.")
        
        self._size = size
        self._tile = self._tile if (self._tile is not None) else self._size
        self._pad  = (self._tile - (self._size % self._tile)) % self._tile
        
    def tiling(self, tile: int):
        if not self.is_initialized:
            raise RuntimeError(f"Dimension must be initialized before setting tiling.")
        if not self.is_tiling_enabled:
            raise RuntimeError(f"Cannot set tiling on a dimension associated with other dimensions.")
        
        self._tile = tile
        self._pad  = (self._tile - (self._size % self._tile)) % self._tile

    def disable_tiling(self):
        self._tiling_enabled = False
        
    @property
    def is_initialized(self) -> bool:
        return self._size is not None and self._tile is not None
    
    @property
    def is_tiling_enabled(self) -> bool:
        return self._tiling_enabled

    @property
    def dim_id(self) -> str:
        return self._dim_id
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def tile(self) -> int:
        return self._tile
    
    @property
    def pad(self) -> int:
        return self._pad
    
    @property
    def n_tiles(self) -> int:
        return (self.size + self.pad) // self.tile  # padding is included in size
    
    def __repr__(self):
        size_repr = f"{self.size}+{self.pad}" if self.pad > 0 else f"{self.size}"
        return f"{self._dim_id}(size={size_repr}, tile={self.tile})"
    
    def __eq__(self, value):
        if not isinstance(value, TensorDimension):
            return False
        return self.dim_id == value.dim_id

class TensorDimensionCursor:
    def __init__(self):
        self.cursors: dict[str, int] = {}
        
    def add(self, dim: TensorDimension, pos: int) -> 'TensorDimensionCursor':
        if not dim.is_initialized:
            raise RuntimeError(f"Dimension '{dim.dim_id}' must be initialized before creating a cursor.")
        if pos < 0 or pos >= dim.n_tiles:
            raise ValueError(f"Cursor position {pos} is out of bounds for dimension '{dim.dim_id}' with {dim.n_tiles} tiles.")
        
        new_cursor = TensorDimensionCursor()
        new_cursor.cursors = self.cursors.copy()
        new_cursor.cursors[dim.dim_id] = pos
        
        return new_cursor
    
    def get(self, dim: TensorDimension) -> int:
        if isinstance(dim, TensorDimension):
            dim_id = dim.dim_id
        else:
            dim_id = dim
        
        return self.cursors.get(dim_id, None)
    
    def has(self, dim: TensorDimension) -> bool:
        if isinstance(dim, TensorDimension):
            dim_id = dim.dim_id
        else:
            dim_id = dim
        
        return dim_id in self.cursors
    
    def to_string(self) -> str:
        return ''.join([f'{dim_id}[{pos}]' for dim_id, pos in self.cursors.items()])
    
    def filter(self, dim_ids: list[str]) -> 'TensorDimensionCursor':
        if isinstance(dim_ids, TensorLayout):
            dim_ids = [dim.dim_id for dim in dim_ids._dims]

        filtered_cursor = TensorDimensionCursor()
        for dim_id in dim_ids:
            if dim_id in self.cursors:
                filtered_cursor.cursors[dim_id] = self.cursors[dim_id]
        return filtered_cursor

    def __repr__(self):
        return f"TensorDimensionCursor({self.to_string()})"

class TensorDimensionPool(dict[str, TensorDimension]):
    def __init__(self):
        super().__init__()
        
    def add(self, *dim_id: str):
        for n in dim_id:
            if n in self:
                raise KeyError(f"Dimension with name '{n}' already exists in the pool.")

            dim = TensorDimension(dim_id=n)
            super().__setitem__(n, dim)
        return self
    
    def get(self, *dim_id: str) -> TensorDimension | list[TensorDimension]:
        if len(dim_id) == 1:
            if dim_id[0] not in self:
                raise KeyError(f"Dimension with name '{dim_id[0]}' does not exist in the pool.")
            return super().get(dim_id[0])
        return [self[dim_id] for dim_id in dim_id]
    
    def has(self, dim_id: str) -> bool:
        return dim_id in self
    
    def disable_tiling(self, *dim_id: str) -> TensorDimension:
        dims = [(dim_id if isinstance(dim_id, TensorDimension) else self[dim_id]) for dim_id in dim_id]
        for dim in dims:
            dim.disable_tiling()
        return self
    
    def __getattribute__(self, name):
        if name in self:
            return self[name]
        return super().__getattribute__(name)
    
    def __setitem__(self, key, value):
        raise RuntimeError("Direct assignment to TensorDimensionPool is not allowed. Use the 'add' method instead.")
    
    @property
    def is_initialized(self) -> bool:
        return all(dim.is_initialized for dim in self.values())
    
    def get_cursors(self, fixed: TensorDimensionCursor=None) -> Iterator[TensorDimensionCursor]:
        if not self.is_initialized:
            raise RuntimeError("TensorDimensionPool must be initialized before accessing tile indices.")
        if fixed is None:
            fixed = TensorDimensionCursor()

        def generate_indices(dimensions: list[TensorDimension], current_index: TensorDimensionCursor) -> Iterator[TensorDimensionCursor]:
            if not dimensions:
                yield current_index
            else:
                for i in range(dimensions[0].n_tiles):
                    yield from generate_indices(dimensions[1:], current_index.add(dimensions[0], i))

        return generate_indices(list(dim for dim in self.values() if not fixed.has(dim)), current_index=fixed)


#################################################################
# Tensor Preprocessing Operations
#################################################################

class _TensorPreprocessing:
    @abc.abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        pass
    
    @abc.abstractmethod
    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        pass
    
class TensorPermute(_TensorPreprocessing):
    def __init__(self, order: list[int]):
        super().__init__()
        self.order = order
        
    def forward(self, tensor):
        return tensor.permute(self.order)

    def backward(self, tensor):
        return tensor.permute(*[self.order.index(i) for i in range(len(self.order))])

class TensorReshape(_TensorPreprocessing):
    def __init__(self, shape: tuple[int]):
        super().__init__()
        self.shape = shape
        self.orig_shape = None
        
    def forward(self, tensor):
        self.orig_shape = tensor.shape
        return tensor.reshape(self.shape)

    def backward(self, tensor):
        return tensor.reshape(self.orig_shape)


#################################################################
# Tensor Layout
#################################################################
    
class TensorLayout:
    def __init__(self, dims: list[TensorDimension], preprocessing: list[_TensorPreprocessing]=None):
        self._dims = dims
        self._preprocessing = preprocessing if preprocessing is not None else []
        self._orig_tensor: torch.Tensor = None

        if isinstance(self._dims, TensorDimension):
            self._dims = [self._dims]

        for dim in self._dims:
            if not isinstance(dim, TensorDimension):
                raise TypeError("All elements of dims must be TensorDimension instances.")
        
        if isinstance(self._preprocessing, _TensorPreprocessing):
            self._preprocessing = [self._preprocessing]
        
        for p in self._preprocessing:
            if not isinstance(p, _TensorPreprocessing):
                raise TypeError("All elements of preprocessing must be _TensorPreprocessing instances.")
        
    def update_tensor(self, tensor: torch.Tensor=None, shape: tuple[int]=None, dtype: torch.dtype=None) -> 'TensorLayout':
        if tensor is not None:
            pass
        elif (shape is not None) and (dtype is not None):
            tensor = torch.zeros(shape, dtype=dtype)
        else:
            raise RuntimeError("Either tensor or (shape and dtype) must be provided to initialize the TensorLayout.")
        
        for p in self._preprocessing:
            tensor = p.forward(tensor)
            
        if tensor.dim() != len(self._dims):
            raise ValueError(f"Tensor has {tensor.dim()} dimensions, but format expects {len(self._dims)} dimensions.")
        
        for i, dim in enumerate(self._dims):
            dim.initialize(size=tensor.size(i))
            
        self._orig_tensor = tensor
        
        return self
    
    def restore_tensor(self) -> torch.Tensor:
        for p in self._preprocessing[::-1]:
            self._orig_tensor = p.backward(self._orig_tensor)
        return self._orig_tensor

    def __getitem__(self, cursor: TensorDimensionCursor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("TensorLayout must be initialized before accessing tiles.")
        
        slices = []
        pads   = []
        
        for dim in self._dims:
            pos = cursor.get(dim)
            if pos is None:
                raise ValueError(f"Missing cursor for dimension '{dim.dim_id}'.")
            
            start = pos * dim.tile
            end   = min(start + dim.tile, dim.size)
            pad   = dim.pad if end == dim.size else 0
            
            slices.append(slice(start, end))
            pads = [0, pad] + pads  # pad is added in reverse order for torch.nn.functional.pad

        tile = self._orig_tensor[tuple(slices)]
        tile = torch.nn.functional.pad(tile, pads)
        
        return tile
    
    def __setitem__(self, cursor: TensorDimensionCursor, value: torch.Tensor):
        if not self.is_initialized:
            raise RuntimeError("TensorLayout must be initialized before accessing tiles.")

        value = value.reshape(self.tile_shape)
        
        slices = []
        tile_slices = []
        
        for dim in self._dims:
            pos = cursor.get(dim)
            if pos is None:
                raise ValueError(f"Missing cursor for dimension '{dim.dim_id}'.")
            
            start = pos * dim.tile
            end   = min(start + dim.tile, dim.size)
            pad   = dim.pad if end == dim.size else 0
            
            slices.append(slice(start, end))
            tile_slices.append(slice(0, dim.tile - pad))

        self._orig_tensor[tuple(slices)] = value[tuple(tile_slices)]
    
    def get_cursors(self, fixed: TensorDimensionCursor=None) -> Iterator[TensorDimensionCursor]:
        if not self.is_initialized:
            raise RuntimeError("TensorLayout must be initialized before accessing tile indices.")

        if fixed is not None:
            old_fixed = fixed
            fixed = TensorDimensionCursor()
            
            for dim in self._dims:
                if old_fixed.has(dim):
                    fixed = fixed.add(dim, old_fixed.get(dim))
        else:
            fixed = TensorDimensionCursor()

        def generate_indices(dimensions: list[TensorDimension], current_index: TensorDimensionCursor) -> Iterator[TensorDimensionCursor]:
            if not dimensions:
                yield current_index
            else:
                for i in range(dimensions[0].n_tiles):
                    yield from generate_indices(dimensions[1:], current_index.add(dimensions[0], i))

        return generate_indices(list(dim for dim in self._dims if not fixed.has(dim)), current_index=fixed)

    @property
    def is_initialized(self) -> bool:
        return all(dim.is_initialized for dim in self._dims) and self._orig_tensor is not None

    @property
    def tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing size.")
        return functools.reduce(lambda x, y: x * y, (dim.size for dim in self._dims), 1) * self._orig_tensor.dtype.itemsize
    
    @property
    def padded_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing padded_size.")
        return functools.reduce(lambda x, y: x * y, (dim.size + dim.pad for dim in self._dims), 1) * self._orig_tensor.dtype.itemsize
    
    @property
    def tile_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_size.")
        return functools.reduce(lambda x, y: x * y, (dim.tile for dim in self._dims), 1) * self._orig_tensor.dtype.itemsize
    
    @property
    def tensor_shape(self) -> tuple[int]:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing shape.")
        return tuple(dim.size for dim in self._dims)
    
    @property
    def tensor_dtype(self) -> torch.dtype:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing dtype.")
        return self._orig_tensor.dtype
    
    @property
    def padded_tensor_shape(self) -> tuple[int]:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing padded_shape.")
        return tuple(dim.size + dim.pad for dim in self._dims)
    
    @property
    def tile_shape(self) -> tuple[int]:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_shape.")
        return tuple(dim.tile for dim in self._dims)
    
    @property
    def n_tiles(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing n_tiles.")
        return functools.reduce(lambda x, y: x * y, (dim.n_tiles for dim in self._dims), 1)
    
    @property
    def tile_grid(self) -> tuple[int]:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_grid.")
        return tuple(dim.n_tiles for dim in self._dims)

    def __repr__(self):
        return f"TensorFormat(dims={self._dims})"

class TensorLayoutPool(dict[str, TensorLayout]):
    def __init__(self):
        super().__init__()
        
        self._input_layout_ids: list[str] = []
        self._output_layout_id: str = None

    def add(
        self, 
        name: str, dims: list[TensorDimension], preprocessing: list[_TensorPreprocessing]=None, 
        shape: tuple[int, ...]=None, dtype: torch.dtype=None, initial_tensor: torch.Tensor=None, 
        is_input: bool=False, is_output: bool=False,
    ) -> 'TensorLayoutPool':
        if name in self:
            raise KeyError(f"Layout with name '{name}' already exists in the pool.")
        
        layout = TensorLayout(dims=dims, preprocessing=preprocessing)
        layout.update_tensor(initial_tensor, shape, dtype)
        
        super().__setitem__(name, layout)
        
        if is_output:
            if self._output_layout_id is not None:
                raise RuntimeError(f"Output layout is already set to '{self._output_layout_id}'. Only one output layout is allowed.")
            self._output_layout_id = name
        elif is_input:
            self._input_layout_ids.append(name)
        return self

    def get(self, layout_id: str) -> TensorLayout:
        if layout_id not in self:
            raise KeyError(f"Layout with name '{layout_id}' does not exist in the pool.")
        return super().get(layout_id)

    def has(self, layout_id: str) -> bool:
        return layout_id in self
    
    def clear(self):
        self._input_layout_ids.clear()
        self._output_layout_id = None
        return super().clear()

    def __getattribute__(self, name):
        if name in self:
            return self[name]
        return super().__getattribute__(name)
    
    def __setitem__(self, key, value):
        raise RuntimeError("Direct assignment to TensorLayoutPool is not allowed. Use the 'add' method instead.")
    
    @property
    def is_initialized(self) -> bool:
        return all(layout.is_initialized for layout in self.values())
    
    @property
    def output_layout(self) -> TensorLayout:
        if self._output_layout_id is None:
            return None
        return self[self._output_layout_id]
    
    @property
    def input_layouts(self) -> list[TensorLayout]:
        return [self[layout_id] for layout_id in self._input_layout_ids]
    
    @property
    def param_layouts(self) -> list[TensorLayout]:
        return [layout for layout_id, layout in self.items() if layout_id not in self._input_layout_ids and layout_id != self._output_layout_id]
    
    @property
    def input_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing size.")
        return sum(layout.tensor_size for layout_id, layout in self.items() if layout_id in self._input_layout_ids)
    
    @property
    def input_padded_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing padded_size.")
        return sum(layout.padded_tensor_size for layout_id, layout in self.items() if layout_id in self._input_layout_ids)
    
    @property
    def input_tile_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_size.")
        return sum(layout.tile_size for layout_id, layout in self.items() if layout_id in self._input_layout_ids)
    
    @property
    def param_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing size.")
        return sum(layout.tensor_size for layout_id, layout in self.items() if layout_id not in self._input_layout_ids and layout_id != self._output_layout_id)
    
    @property
    def param_padded_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing padded_size.")
        return sum(layout.padded_tensor_size for layout_id, layout in self.items() if layout_id not in self._input_layout_ids and layout_id != self._output_layout_id)
    
    @property
    def param_tile_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_size.")
        return sum(layout.tile_size for layout_id, layout in self.items() if layout_id not in self._input_layout_ids and layout_id != self._output_layout_id)
    
    @property
    def output_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing size.")
        if self._output_layout_id is None:
            return 0
        return self[self._output_layout_id].tensor_size
    
    @property
    def output_padded_tensor_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing padded_size.")
        if self._output_layout_id is None:
            return 0
        return self[self._output_layout_id].padded_tensor_size
    
    @property
    def output_tile_size(self) -> int:
        if not self.is_initialized:
            raise RuntimeError("TensorFormat must be initialized before accessing tile_size.")
        if self._output_layout_id is None:
            return 0
        return self[self._output_layout_id].tile_size
    
    @property
    def tiling_target_dims(self) -> list[TensorDimension]:
        target_dims: dict[str, TensorDimension] = {}
        
        for layout in self.input_layouts:
            for dim in layout._dims:
                if dim not in target_dims.values() and dim.is_tiling_enabled:
                    target_dims[dim.dim_id] = dim

        if self.output_layout is not None:
            for dim in self.output_layout._dims:
                if dim not in target_dims.values() and dim.is_tiling_enabled:
                    target_dims[dim.dim_id] = dim

        for layout in self.param_layouts:
            for dim in layout._dims:
                if dim not in target_dims.values() and dim.is_tiling_enabled:
                    target_dims[dim.dim_id] = dim
                    
        return list(target_dims.values())
    
    def set_tiling_factor_with_mem_usage(self, max_mem_usage: int) -> bool:
        target_dims = self.tiling_target_dims
        target_dim_idx = 0
        _skip_cnt = 0
        
        while (self.input_tile_size + self.param_tile_size + self.output_tile_size) > max_mem_usage:
            d = target_dims[target_dim_idx]
            
            if d.tile == 1:
                _skip_cnt += 1
            else:
                _skip_cnt = 0
                d.tiling(max(1, d.tile // 2))
                
            if _skip_cnt >= len(target_dims):
                return False  # cannot reduce any further
                
            target_dim_idx += 1
            if target_dim_idx >= len(target_dims):
                target_dim_idx = 0
        
        return True
         
            
if __name__ == "__main__":
    from neuromta.framework import *
    
    ###################################################################################################################
    
    ifm  = torch.randint(0, 16, (1*32*32*32,)).reshape(1, 32, 32, 32).to(dtype=torch.int32)
    wgt  = torch.randint(0, 16, (64*32*3*3,)).reshape(64, 32, 3, 3).to(dtype=torch.int32)
    bias = torch.randint(0, 16, (64,)).reshape(64).to(dtype=torch.int32)
    ofm  = torch.nn.functional.conv2d(ifm, wgt, bias=bias, stride=1, padding=1, dilation=1)
    
    dims = TensorDimensionPool()
    dims.add("N", "K", "C", "H", "W", "OH", "OW", "FH", "FW")
    dims.disable_tiling("H", "OH", "FH")  # Disable tiling on height-related dimensions (image height cannot be simply tiled due to halo regions)
    dims.disable_tiling("W", "OW", "FW")  # Disable tiling on width-related dimensions (image width cannot be simply tiled due to halo regions)
    
    layouts = TensorLayoutPool()
    layouts.add(name="IFM",  dims=dims.get("N", "C", "H", "W"),     initial_tensor=ifm, is_input=True   )
    layouts.add(name="WGT",  dims=dims.get("K", "C", "FH", "FW"),   initial_tensor=wgt                  )
    layouts.add(name="BIAS", dims=dims.get("K"),                    initial_tensor=bias                 )
    layouts.add(name="OFM",  dims=dims.get("N", "K", "OH", "OW"),   initial_tensor=ofm, is_output=True  )

    max_mem_usage = parse_mem_cap_str("128KB")
    flag = layouts.set_tiling_factor_with_mem_usage(max_mem_usage=max_mem_usage)

    if not flag:
        print(f"Cannot meet the memory constraint with any tiling configuration: {max_mem_usage} bytes")
        exit(1)
    else:
        print(f"Tiling configuration found to meet the memory constraint")
        print(f"  - Max Memory Usage:    {max_mem_usage} bytes")
        print(f"  - Actual Memory Usage: {layouts.input_tile_size + layouts.param_tile_size + layouts.output_tile_size} bytes")
        print(f"\nTiling Target Dimensions:")
        for d in layouts.tiling_target_dims:
            print(f"  - {d}")
    
    ###################################################################################################################

    for layout_id, layout in layouts.items():
        print(f"\nLayout {layout_id}:")
        print(f"  - Tensor Shape:       {layout.tensor_shape}")
        print(f"  - Padded Shape:       {layout.padded_tensor_shape}")
        print(f"  - Tile Shape:         {layout.tile_shape}")
        print(f"  - Tensor Size:        {layout.tensor_size} bytes")
        print(f"  - Padded Tensor Size: {layout.padded_tensor_size} bytes")
        print(f"  - Tile Size:          {layout.tile_size} bytes")
        print(f"  - Number of Tiles:    {layout.n_tiles}")
        print(f"  - Tile Grid:          {layout.tile_grid}")
        
    print(f"\nLayout Summary")
    print(f"  - Input/Param Tensor Size:        {layouts.input_tensor_size} bytes + {layouts.param_tensor_size} bytes")
    print(f"  - Input/Param Padded Tensor Size: {layouts.input_padded_tensor_size} bytes + {layouts.param_padded_tensor_size} bytes")
    print(f"  - Input/Param Tile Size:          {layouts.input_tile_size} bytes + {layouts.param_tile_size} bytes")
    print(f"  - Output Tensor Size:             {layouts.output_tensor_size} bytes")
    print(f"  - Output Padded Tensor Size:      {layouts.output_padded_tensor_size} bytes")
    print(f"  - Output Tile Size:               {layouts.output_tile_size} bytes")
    
    for fixed_cursor in layouts.output_layout.get_cursors():
        print(f"\nOFM Tile Cursor: {fixed_cursor}")
        for i, cursor in enumerate(dims.get_cursors(fixed=fixed_cursor)):
            # print(f"  - {i} -> {cursor}")
            print(f"  - {i} ->", end="")
            print(f" IFM({list(layouts.IFM.get_cursors(fixed=cursor))})", end=", ")
            print(f" WGT({list(layouts.WGT.get_cursors(fixed=cursor))})", end=", ")
            print(f" BIAS({list(layouts.BIAS.get_cursors(fixed=cursor))})")

    ###################################################################################################################
    
    print("\nIntegrity Check")
    for fixed_cursor in layouts.output_layout.get_cursors():
        ofm_tile = layouts.output_layout[fixed_cursor]
        reconstructed_ofm_tile = torch.zeros_like(ofm_tile)
        
        for i, cursor in enumerate(dims.get_cursors(fixed=fixed_cursor)):
            ifm_tile  = layouts.IFM[cursor]
            wgt_tile  = layouts.WGT[cursor]
            if i == 0:
                bias_tile = layouts.BIAS[cursor]
            else:
                bias_tile = None
            
            partial_ofm_tile = torch.nn.functional.conv2d(ifm_tile, wgt_tile, bias=bias_tile, stride=1, padding=1, dilation=1)
            reconstructed_ofm_tile = partial_ofm_tile + reconstructed_ofm_tile

        if not torch.allclose(ofm_tile, reconstructed_ofm_tile, atol=1e-6):
            print(f"Integrity check failed at tile {fixed_cursor}!")
            print("\nOriginal OFM Tile:")
            print(ofm_tile)
            print("\nReconstructed OFM Tile:")
            print(reconstructed_ofm_tile)
            break
        
        print(f"OFM Tile {fixed_cursor} passed.")
