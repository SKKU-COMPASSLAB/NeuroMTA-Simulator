import os
import math
import configparser
from typing import Any, Callable

from neuromta.framework import *

try:
    import pydramsim3
    PYDRAMSIM3_AVAILABLE = True
except ImportError as e:
    PYDRAMSIM3_AVAILABLE = False

PYDRAMSIM3_AVAILABLE = os.environ.get("NEUROMTA_DISABLE_DRAMSIM", "0") != "1" and PYDRAMSIM3_AVAILABLE

logger.debug(f"pydramsim3 availability: {PYDRAMSIM3_AVAILABLE}")
if not PYDRAMSIM3_AVAILABLE:
    logger.debug(f"pydramsim3 is not available. Using a simplified DRAMSim3 model instead. To enable pydramsim3, please install it via 'pip install externals/pydramsim3'")

__all__ = [
    "PYDRAMSIM3_AVAILABLE",
    "DRAMSim3",
    "DRAMSim3Config",
]

if PYDRAMSIM3_AVAILABLE:
    class DRAMSim3Config:
        def __init__(
            self, 
            src_config_path: str,
            dst_config_path: str,
            processor_clock_freq: int,
            n_instance: int,
            channel_size: int,
            n_channel_per_instance: int,
            n_cmd_q_per_instance: int,
        ):  
            if not os.path.isfile(src_config_path):
                src_config_path = pydramsim3.PYDRAMSIM_MSYS_CONFIG_PATH(src_config_path)
            if not os.path.isfile(src_config_path):
                raise FileNotFoundError(f"DRAMSim3 config file '{src_config_path}' not found.")
            
            pydramsim3.create_new_dramsim_config_file(
                src_config_path=src_config_path,
                new_config_path=dst_config_path,
                system_params={
                    "channel_size": channel_size // (1024 * 1024),  # GB -> MB
                    "channels": n_channel_per_instance,
                    # "address_mapping": "rorababgchco",
                },
                dram_structure_params={
                    "bankgroups": 1  # TODO: more authentic way of doing this..?
                }
            )

            self.config_path = dst_config_path
            self.processor_clock_freq = processor_clock_freq
            self.n_instance = n_instance
            self.channel_size = channel_size
            self.n_channel_per_instance = n_channel_per_instance
            self.n_cmd_q_per_instance = n_cmd_q_per_instance
            
        def peak_bandwidth(self) -> float:
            return pydramsim3.get_bandwidth_from_dramsim_config(self.config_path) * self.n_instance

        def summary(self) -> dict[str, Any]:
            return {
                "config_path": os.path.abspath(self.config_path),
                "processor_clock_freq": self.processor_clock_freq,
            }

    class DRAMSim3(CompanionModule):
        def __init__(self, config: DRAMSim3Config):
            super().__init__()
            
            self.config = config
            
            if self.config.n_instance <= 0:
                raise ValueError("DRAMSim3Config.n_instance must be greater than 0.")
            
            self._msys_instances = [
                pydramsim3.create_msys(
                    config_file=self.config.config_path,
                    output_dir=pydramsim3.PYDRAMSIM_DEFAULT_OUT_DIR,
                    cmd_queue_num=self.config.n_cmd_q_per_instance,
                ) for _ in range(self.config.n_instance)
            ]
            
            self._mem_clock_time = pydramsim3.msys_get_tck(self._msys_instances[0])
            self._ref_clock_time = 1 / (self.config.processor_clock_freq * (1e-9))
            self._rem_clock_sync_time = 0
            
        def update_cycle_time(self, cycle_time):
            self._rem_clock_sync_time += cycle_time * self._ref_clock_time
            
            mem_cycles = math.floor(self._rem_clock_sync_time / self._mem_clock_time)
            self._rem_clock_sync_time -= mem_cycles * self._mem_clock_time
            
            for msys in self._msys_instances:
                pydramsim3.msys_cycle_step(msys=msys, cycles=mem_cycles)

        def create_command(self, inst_id: int, cmd_q_id: int, addr: int, size: int, is_write: bool) -> CompanionCommandSignature:
            capsule = pydramsim3.create_msys_cmd(cmd_q_id=cmd_q_id, addr=addr, size=size, is_write=is_write)
            cmd = CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs={
                "inst_id": inst_id,
                "cmd_q_id": cmd_q_id,
                "addr": addr,
                "size": size,
                "is_write": is_write,
            })
            return cmd

        def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
            inst_id = cmd.kwargs["inst_id"]
            msys = self._msys_instances[inst_id]
            return pydramsim3.msys_dispatch_cmd(msys=msys, cmd=cmd.capsule, dispatch_callback=dispatch_callback, execute_callback=execute_callback)
else:
    class DRAMSim3Config:
        def __init__(
            self, 
            processor_clock_freq: int,
            n_instance: int,
            channel_size: int,
            n_channel_per_instance: int,
            n_cmd_q_per_instance: int,
            bandwidth_per_instance: int,
        ):  
            self.processor_clock_freq = processor_clock_freq
            self.n_instance = n_instance
            self.channel_size = channel_size
            self.n_channel_per_instance = n_channel_per_instance
            self.n_cmd_q_per_instance = n_cmd_q_per_instance
            self.bandwidth_per_instance = bandwidth_per_instance
            
        def peak_bandwidth(self) -> float:
            return self.bandwidth_per_instance * self.n_instance

        def summary(self) -> dict[str, Any]:
            return {
                "processor_clock_freq": self.processor_clock_freq,
                "n_instance": self.n_instance,
                "channel_size": self.channel_size,
                "n_channel_per_instance": self.n_channel_per_instance,
                "n_cmd_q_per_instance": self.n_cmd_q_per_instance,
                "bandwidth_per_instance": self.bandwidth_per_instance,
            }

    class DRAMSim3(CompanionModule):
        def __init__(self, config: DRAMSim3Config):
            super().__init__()
            self.config = config
            
            # Buffer for commands waiting to be sorted into queues
            self._suspended_cmds: list[tuple[CompanionCommandSignature, Callable]] = []
            
            # Internal Queues per (instance, cmd_q)
            self._cmd_queues: dict[tuple[int, int], list[tuple[CompanionCommandSignature, Callable]]] = {}
            
            # Commands currently being processed by DRAM
            self._in_flight_cmds: list[list] = []
            
            self._base_latency_cycles = 40
            self._max_queue_depth = 16

        def _get_transfer_cycles(self, size_bytes: int) -> int:
            # Calculate burst duration based on bandwidth
            return max(1, size_bytes // max(1, self.config.bandwidth_per_instance))

        def update_cycle_time(self, cycle_time: int):
            """
            Processes DRAM simulation cycles. 
            Order: Clear Finished -> Issue from Queues -> Inject from Suspended.
            """
            
            # STEP 1: Progress Time and Clear Completed Transactions
            # Do this first to free up 'is_queue_busy' status for the current cycle.
            completed_indices = []
            for i, status in enumerate(self._in_flight_cmds):
                status[0] -= cycle_time
                if status[0] <= 0:
                    completed_indices.append(i)
                    
            for i in sorted(completed_indices, reverse=True):
                _, cmd, callback = self._in_flight_cmds.pop(i)
                # Pass the whole 'cmd' object back to the callback
                callback(cmd.capsule)

            # STEP 2: Issue commands from internal queues to DRAM (In-Flight)
            # We model HOL blocking here: only one active command per (inst, q_id).
            for key, queue in self._cmd_queues.items():
                if not queue:
                    continue
                    
                # Check if this specific queue is already busy in-flight
                is_queue_busy = any(
                    c[1].capsule['inst_id'] == key[0] and 
                    c[1].capsule['cmd_q_id'] == key[1] 
                    for c in self._in_flight_cmds
                )
                
                if not is_queue_busy:
                    cmd, callback = queue.pop(0)
                    latency = self._base_latency_cycles + self._get_transfer_cycles(cmd.capsule['size'])
                    self._in_flight_cmds.append([latency, cmd, callback])

            # STEP 3: Inject from _suspended_cmds to _cmd_queues
            # FIX: Only pop if there is room in the target queue to prevent data loss.
            remaining_suspended = []
            for cmd, callback in self._suspended_cmds:
                inst_id = cmd.capsule['inst_id']
                q_id = cmd.capsule['cmd_q_id']
                key = (inst_id, q_id)
                
                if key not in self._cmd_queues:
                    self._cmd_queues[key] = []
                
                if len(self._cmd_queues[key]) < self._max_queue_depth:
                    self._cmd_queues[key].append((cmd, callback))
                else:
                    # If the queue is full, keep it in the suspended list for the next cycle
                    remaining_suspended.append((cmd, callback))
            
            self._suspended_cmds = remaining_suspended

        def create_command(self, inst_id: int, cmd_q_id: int, addr: int, size: int, is_write: bool) -> CompanionCommandSignature:
            capsule = {
                "inst_id": inst_id, "cmd_q_id": cmd_q_id,
                "addr": addr, "size": size, "is_write": is_write,
            }
            return CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs=capsule)

        def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
            if dispatch_callback is not None:
                dispatch_callback(cmd.capsule)
                
            self._suspended_cmds.append((cmd, execute_callback))
            return True