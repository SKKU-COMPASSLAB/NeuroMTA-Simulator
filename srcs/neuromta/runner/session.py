import os
import multiprocessing as mp

from neuromta.framework import *
from neuromta.component import *


class Session(mp.Process):
    def __init__(self, device_preset: str, model_preset: str):
        super().__init__()
        
        self.device_preset = device_preset
        self.model_preset = model_preset
        
    def run(self):
        pass
