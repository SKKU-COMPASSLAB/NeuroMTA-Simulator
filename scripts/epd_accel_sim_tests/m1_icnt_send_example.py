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
    # TEST 1: Sequential and synchronous broadcasts
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=True)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=True)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=True)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "SEQUENTIAL BROADCASTS")
    
    # TEST 2: Asynchronous broadcasts
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=False)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=False)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=False)
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=128, sync=False)
    
    core.async_rpc_wait_all()
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "ASYNCHRONOUS BROADCASTS")
    
    # TEST 3: Burst broadcasts
    core.icnt_send_data(target_core_ids=[1, 2, 3], size=512, sync=True)
    
    core.debug_core_with_ambiguous_func(core_checkpoint, core, "BURST BROADCASTS")


if __name__ == "__main__":
    config_name = os.path.split(os.path.splitext(__file__)[0])[-1]
    
    config = config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()
    
    logger.set_print_options(log_level=LogLevel.DEBUG)
    device.set_command_debug_verbosity(verbose=False)
    
    tile0 = device.get_ccg_tile(core_id=0)
    
    kernel0 = main(tile0)
    
    kernel0.dispatch()

    device.run_kernels()
    
    logger.info(f"simulation terminated with timestamp: {device.timestamp}")