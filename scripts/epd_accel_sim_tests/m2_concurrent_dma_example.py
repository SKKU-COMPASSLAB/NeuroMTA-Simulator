import os

from neuromta.framework import *
from neuromta.component import *
from neuromta.system import *


IP_ROOT = os.path.abspath(os.path.dirname(__file__))
IP_CACHE_DIR = os.path.join(IP_ROOT, ".cache")
IP_DRAMSIM_CONFIG_FMT = os.path.join(IP_CACHE_DIR, "dramsim_{config_name}.ini").format


def core_checkpoint(core: Core, header: str="CORE CHECKPOINT") -> None:
    logger.info(f"{header:30s}: {core.__class__.__name__} (ID: {core.core_id}) @ timestamp: {core.timestamp} cycles")


@jit_prototype
def dma_kernel(core: CCGTile):
    # TEST 2: Asynchronous DMA reads
    core.dma_read_memory(addr=0,   size=128, sync=False)
    core.dma_read_memory(addr=128, size=128, sync=False)
    core.dma_read_memory(addr=256, size=128, sync=False)
    core.dma_read_memory(addr=384, size=128, sync=False)
    core.dma_read_memory(addr=512, size=128, sync=False)
    
    core.async_rpc_wait_all()
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "DMA READ COMPLETE")
    

@jit_program_prototype
def main(device: MeshAccelerator):
    tile0 = device.get_ccg_tile(core_id=0)
    tile1 = device.get_ccg_tile(core_id=1)
    
    dma_kernel(tile0)
    dma_kernel(tile1)


if __name__ == "__main__":
    config_name = os.path.split(os.path.splitext(__file__)[0])[-1]
    
    config = config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    
    logger.set_print_options(log_level=LogLevel.DEBUG)
    device.set_command_debug_verbosity(verbose=False)
    
    program = main(device)
    program.dispatch()
    
    device.run_kernels()
    
    logger.info(f"simulation terminated with timestamp: {device.timestamp}")