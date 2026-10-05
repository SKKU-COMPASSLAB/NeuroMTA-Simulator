__all__ = [
    "GlobalConfig",
    "GlobalContext",
]


class GlobalConfig:
    def __init__(
        self,
        processor_clock_freq: int,
        ccg_tile_ids: list[int],
        dma_tile_ids: list[int],
    ):
        self.processor_clock_freq = processor_clock_freq
        self.ccg_tile_ids = ccg_tile_ids
        self.dma_tile_ids = dma_tile_ids
        
        if not isinstance(self.ccg_tile_ids, (list, tuple)):
            raise ValueError("ccg_tile_ids must be a list or tuple of integers.")
        if not isinstance(self.dma_tile_ids, (list, tuple)):
            raise ValueError("dma_tile_ids must be a list or tuple of integers.")

        tl = [self.ccg_tile_ids, self.dma_tile_ids]
        for a in tl:
            for b in tl:
                if a is not b and set(a).intersection(set(b)):
                    raise ValueError("Tile IDs must be unique across all tile types.")

class GlobalContext:
    def __init__(
        self,
        config: GlobalConfig,
    ):
        self._config = config
        
    @property
    def config(self) -> GlobalConfig:
        return self._config
