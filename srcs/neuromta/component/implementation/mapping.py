import functools

import torch
import math
from typing import Any, Sequence, Dict, List, Callable

from neuromta.framework import *
from neuromta.component.core import *
from neuromta.component.context.global_context import GlobalContextMemInfo
from neuromta.component.implementation.tensor_buffer import *
from neuromta.component.implementation.hardware import *


__all__ = [
    "TileSignature",
    "TiledOperatorSignature",
]


class TileSignature(SerializableCoreObject):
    def __init__(self, buf_name: str, tile_shape: tuple[int, int], dtype: torch.dtype, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.tile_shape = tile_shape
        self.dtype = dtype
        self.coords: tuple[int, int, int, int] = (y_s, x_s, y_t, x_t)
        
        if isinstance(self.tile_shape, int):
            self.tile_shape = (1, self.tile_shape)
        elif len(self.tile_shape) == 1:
            self.tile_shape = (1, self.tile_shape[0])
        if len(self.tile_shape) != 2:
            raise ValueError("Shape must be a tuple of (height, width).")
        
    def depends_on(self, other: 'TileSignature') -> bool:
        return self.buf_name == other.buf_name and self.coords == other.coords

    @property
    def signature(self) -> str:
        return f"{self.buf_name}{self.coords}"
    
    def __hash__(self):
        return hash((self.buf_name, self.coords))
    
    def __eq__(self, other):
        if not isinstance(other, TileSignature):
            return NotImplemented
        return self.buf_name == other.buf_name and self.coords == other.coords
    
    def get_state(self):
        return {
            "buf_name": self.buf_name,
            "tile_shape": self.tile_shape,
            "dtype": self.dtype,
            "coords": self.coords,
        }
        
    @classmethod
    def from_state(cls, core, state):
        return cls(
            buf_name=state.get("buf_name", ""),
            tile_shape=state.get("tile_shape", (0, 0)),
            dtype=state.get("dtype", torch.float32),
            y_s=state.get("coords", (0, 0, 0, 0))[0],
            x_s=state.get("coords", (0, 0, 0, 0))[1],
            y_t=state.get("coords", (0, 0, 0, 0))[2],
            x_t=state.get("coords", (0, 0, 0, 0))[3],
        )
        
    @property
    def tile_size(self) -> int:
        return functools.reduce(lambda a, b: a * b, self.tile_shape, 1) * self.dtype.itemsize
    
class TiledOperatorSignature(SerializableCoreObject):
    def __init__(self):
        self.i_tiles:   list[list[TileSignature]]   = []
        self.o_tile:    TileSignature               = None
        self.op_kwargs: list[dict[str, Any]]        = []
        
        if not (len(self.i_tiles) == len(self.op_kwargs)):
            raise ValueError("Length of input tiles and operation kwargs must match.")
        
    def get_state(self):
        return {
            "i_tiles": [tile.get_state() for tile in self.i_tiles],
            "o_tile": self.o_tile.get_state() if self.o_tile is not None else None,
            "op_kwargs": self.op_kwargs,
        }

    @classmethod
    def from_state(cls, core, state):
        instance = cls()
        instance.i_tiles = [TileSignature.from_state(core, tile_state) for tile_state in state.get("i_tiles", [])]
        instance.o_tile = TileSignature.from_state(core, state.get("o_tile", None)) if state.get("o_tile") is not None else None
        instance.op_kwargs = state.get("op_kwargs", [])
        return instance
        
    def add_uop(self, i_tiles: list[TileSignature], o_tile: TileSignature, op_kwargs: dict[str, Any]=None):
        self.i_tiles.append(i_tiles)
        if self.o_tile is None:
            self.o_tile = o_tile
        else:
            if self.o_tile.buf_name != o_tile.buf_name or self.o_tile.coords != o_tile.coords:
                raise ValueError("Output tile signature does not match existing output tile signature.")
        self.op_kwargs.append(op_kwargs if op_kwargs is not None else {})
        
    def reorder_uops(self, target_buf_name: str):
        def tile_key_fn(i_tiles: list[TileSignature]):
            for tile in i_tiles:
                if tile.buf_name == target_buf_name:
                    return tile.coords
            return (math.inf, math.inf, math.inf, math.inf)
        
        combined = list(zip(self.i_tiles, self.op_kwargs))
        combined.sort(key=lambda x: tile_key_fn(x[0]))
        self.i_tiles, self.op_kwargs = zip(*combined)
        self.i_tiles = list(self.i_tiles)
        self.op_kwargs = list(self.op_kwargs)
        
    @property
    def signature(self) -> str:
        i_sigs = [
            "[" + ", ".join([t.signature for t in tile_pair]) + "]"
            for tile_pair in self.i_tiles
        ]
        i_sig_str = " + ".join(i_sigs)
        o_sig_str = self.o_tile.signature
        return f"{i_sig_str} -> {o_sig_str}"
    
    @property
    def n_uops(self) -> int:
        return len(self.i_tiles)
