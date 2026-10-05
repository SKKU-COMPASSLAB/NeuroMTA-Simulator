from neuromta.framework import *


__all__ = [
    "ComputeTileContext",
    "ComputeTileConfig",
]


class ComputeTileConfig:
    def __init__(
        self,
        processor_clock_freq: int,
        tops: float,
        tile_x_dim: int,
        tile_y_dim: int,
        local_cache: int,
        ld_buffer_size: int,
        st_buffer_size: int,
    ):
        self.processor_clock_freq = processor_clock_freq
        self.tops = tops
        self.tile_x_dim = tile_x_dim
        self.tile_y_dim = tile_y_dim
        self.local_cache = local_cache
        self.ld_buffer_size = ld_buffer_size
        self.st_buffer_size = st_buffer_size


class ComputeTileContext:
    def __init__(
        self,
        config: ComputeTileConfig,
    ):
        self._config = config
        
    def get_compute_cycles(self, n_ops: int) -> float:
        return n_ops / ((self._config.tops * 1e12) / self._config.processor_clock_freq)    
    
    @property
    def config(self) -> ComputeTileConfig:
        return self._config
