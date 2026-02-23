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
    "CollectiveTileSignature",
    "TiledOperatorSignature",
    # "MCA_OperatorSignature",
    # "MCA_CompiledOperatorGraph",
    # "MCA_OperatorGraphCompiler",
    # "mca_operator_method",
]


class TileSignature:
    def __init__(self, buf_name: str, y_s: int, x_s: int, y_t: int, x_t: int):
        self.buf_name = buf_name
        self.coords: tuple[int, int, int, int] = (y_s, x_s, y_t, x_t)
        
    def depends_on(self, other: 'TileSignature') -> bool:
        return self.buf_name == other.buf_name and self.coords == other.coords

    @property
    def signature(self) -> str:
        return f"{self.buf_name}{self.coords}"
    

class CollectiveTileSignature(TileSignature):
    def __init__(self, buf_name: str, src_tiles: Sequence[TileSignature], memcpy_patterns: Sequence[dict[int, int]]):
        super().__init__(buf_name, 0, 0, 0, 0)
        
        self.src_tiles = list(src_tiles)
        self.memcpy_patterns = list(memcpy_patterns)
        self.coords = None  # override coords to None for collective tile signature
        
        for src_tile in self.src_tiles:
            if src_tile.buf_name != buf_name:
                raise ValueError("Source tile buffer names do not match collective buffer name.")
            
    def depends_on(self, other):
        return any(src_tile.depends_on(other) for src_tile in self.src_tiles)

    @property
    def signature(self) -> str:
        def tile_signature_with_pattern(tile: TileSignature, pattern: dict[int, int]) -> str:
            pattern_str = "{" + ",".join([f"{k}:{v}" for k, v in pattern.items()]) + "}"
            return f"{tile.signature}{pattern_str}"
        return f"{self.buf_name}[COLLECTIVE {', '.join([tile_signature_with_pattern(tile, pattern) for tile, pattern in zip(self.src_tiles, self.memcpy_patterns)])}]"
    
    
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
