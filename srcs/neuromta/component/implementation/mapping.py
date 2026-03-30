import abc
import enum
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


class TileSignature:
    def __init__(self, buf_name: str, tile_size: int, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.tile_size = tile_size
        self.coords: tuple[int, int, int, int] = (y_s, x_s, y_t, x_t)
        
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
    
    def __repr__(self):
        return self.signature
    
class TiledOperatorSignature:
    def __init__(self):
        self.i_tiles:   list[list[TileSignature]]   = []
        self.o_tile:    TileSignature               = None
        self.op_kwargs: list[dict[str, Any]]        = []
        
        if not (len(self.i_tiles) == len(self.op_kwargs)):
            raise ValueError("Length of input tiles and operation kwargs must match.")
        
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
