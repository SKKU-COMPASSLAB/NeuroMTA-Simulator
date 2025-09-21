import torch

from neuromta.framework import *
from neuromta.hardware import *

from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "",
]


class TT_RT_LINEAR(RuntimeOperator):
    def __init__(
        self, 
        
        batch_size: int, 
        in_features: int, 
        out_features: int, 
        bias: bool=True,
        
        mem_type: TensorMemoryType=TensorMemoryType.MAIN,
    ):
        super().__init__()
        
        self.batch_size = batch_size    
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        