import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.common.operator import *
from neuromta.ip.tenstorrent.architecture import *
from neuromta.ip.tenstorrent.runtime_operator import *


class ProxyContext:
    def __init__(self):
        pass
    
    
class TenstorrentNetwork:
    def __init__(self, device: TenstorrentDevice, core_grid: MTA_CoreGrid):
        self.device = device
        self.core_grid = core_grid