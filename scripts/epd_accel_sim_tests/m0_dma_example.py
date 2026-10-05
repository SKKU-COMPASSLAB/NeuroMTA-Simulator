import os

from neuromta.framework import *
from neuromta.component import *
from neuromta.system import *


IP_ROOT = os.path.abspath(os.path.dirname(__file__))
IP_CACHE_DIR = os.path.join(IP_ROOT, ".cache")
IP_DRAMSIM_CONFIG_FMT = os.path.join(IP_CACHE_DIR, "dramsim_{config_name}.ini").format


def core_checkpoint(core: Core, header: str="CORE CHECKPOINT") -> None:
    logger.info(f"{header:30s}: {core.__class__.__name__} (ID: {core.core_id}) @ timestamp: {core.timestamp} cycles")
    core._timestamp = 0


@jit_prototype
def main(core: CCGTile):
    # TEST 1: Sequential and Synchronous DMA reads
    core.dma_read_memory(addr=0,   size=128, sync=True)
    core.dma_read_memory(addr=128, size=128, sync=True)
    core.dma_read_memory(addr=256, size=128, sync=True)
    core.dma_read_memory(addr=384, size=128, sync=True)
    core.dma_read_memory(addr=512, size=128, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "SEQUENTIAL DMA READS")
    
    # TEST 2: Asynchronous DMA reads
    core.dma_read_memory(addr=0,   size=128, sync=False)
    core.dma_read_memory(addr=128, size=128, sync=False)
    core.dma_read_memory(addr=256, size=128, sync=False)
    core.dma_read_memory(addr=384, size=128, sync=False)
    core.dma_read_memory(addr=512, size=128, sync=False)
    
    core.async_rpc_wait_all()
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "ASYNCHRONOUS DMA READS")
    
    # TEST 3: Burst DMA reads
    core.dma_read_memory(addr=0,   size=512, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "BURST DMA READS")
    
    # TEST 4: Sequential and Synchronous DMA writes
    core.dma_write_memory(addr=0,   size=128, sync=True)
    core.dma_write_memory(addr=128, size=128, sync=True)
    core.dma_write_memory(addr=256, size=128, sync=True)
    core.dma_write_memory(addr=384, size=128, sync=True)
    core.dma_write_memory(addr=512, size=128, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "SEQUENTIAL DMA WRITES")
    
    # TEST 5: Asynchronous DMA writes
    core.dma_write_memory(addr=0,   size=128, sync=False)
    core.dma_write_memory(addr=128, size=128, sync=False)
    core.dma_write_memory(addr=256, size=128, sync=False)
    core.dma_write_memory(addr=384, size=128, sync=False)
    core.dma_write_memory(addr=512, size=128, sync=False)
    
    core.async_rpc_wait_all()
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "ASYNCHRONOUS DMA WRITES")
    
    # TEST 6: Burst DMA writes
    core.dma_write_memory(addr=0,   size=512, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "BURST DMA WRITES")


if __name__ == "__main__":
    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    
    logger.set_print_options(log_level=LogLevel.DEBUG)
    device.set_command_debug_verbosity(verbose=False)
    
    tile0 = device.get_ccg_tile(core_id=0)
    
    kernel0 = main(tile0)
    
    kernel0.dispatch()

    device.run_kernels()
    
    logger.info(f"simulation terminated with timestamp: {device.timestamp}")