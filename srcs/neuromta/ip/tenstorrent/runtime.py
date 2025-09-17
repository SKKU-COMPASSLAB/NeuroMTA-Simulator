import torch

from neuromta.framework import *
from neuromta.hardware import *

from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "TenstorrentTensor",
]


class TenstorrentTensor(torch.Tensor):
    def __init__(self, data: torch.Tensor, device: TenstorrentDevice):
        assert isinstance(device, TenstorrentDevice)
        self.device = device
        super().__init__(data)