import os
import functools
from typing import Any
from neuromta.framework import *
import math
from pydramsim3 import *


TENSTORRENT_IP_ROOT = os.path.abspath(os.path.dirname(__file__))
TENSTORRENT_IP_CACHE_DIR = os.path.join(TENSTORRENT_IP_ROOT, ".cache")
TENSTORRENT_IP_DRAMSIM_CONFIG_FMT = os.path.join(TENSTORRENT_IP_CACHE_DIR, "dramsim_{config_name}.ini").format



class DRAMTestContext:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        processor_clock_freq    = parse_freq_str("1GHz")
        main_mem_channel_size   = parse_mem_cap_str("4GB")
        dramsim3_config_path    = TENSTORRENT_IP_DRAMSIM_CONFIG_FMT(config_name='blackhole')
        dramsim3_channel_size   = main_mem_channel_size // (1024 * 1024)    # GB -> MB
        
        create_new_dramsim_config_file(
            src_config_path="GDDR6_8Gb_x16.ini",
            new_config_path=dramsim3_config_path,
            system_params={
                "channel_size": dramsim3_channel_size,
                "channels": 3,  # original config is for 3 channels
                "address_mapping": "rorababgchco",
            },
            dram_structure_params={
                "bankgroups": 1  # TODO: more authentic way of doing this..?
            }
        )
               
        self.msys = create_msys(
            config_file=dramsim3_config_path,
            output_dir=os.path.join(TENSTORRENT_IP_CACHE_DIR),
            cmd_queue_num=3
        )

        self.ongoing_cmds = []
        
        self.cycles = 0
        self.summary: list[list[tuple[int, int, int, int, int]]] = []  # src_id, dst_id, packet_size, cycles, timestamp
        
        self._mem_clock_time = pydramsim3.msys_get_tck(self.msys)
        self._ref_clock_time = 1 / (processor_clock_freq * (1e-9))
        self._rem_clock_sync_time = 0
        
    def execute_callback(self, addr, size, is_write, cmd):
        self.summary[-1].append((addr, size, is_write, self.cycles))
    
        if self.verbose:
            print(f"  * [CYCLES {self.cycles:<3d}] addr {addr:<2d} size {size:<3d} {'WRITE' if is_write else 'READ '}")
            
    def update_cycle_time(self, cycles):
        self._rem_clock_sync_time += cycles * self._ref_clock_time
        
        mem_cycles = math.floor(self._rem_clock_sync_time / self._mem_clock_time)
        self._rem_clock_sync_time -= mem_cycles * self._mem_clock_time
        
        self.cycles += cycles
        pydramsim3.msys_cycle_step(msys=self.msys, cycles=mem_cycles)

    def dispatch_single_cmd(self, addr, size, is_write) -> Any:
        cmd = create_msys_cmd(
            cmd_q_id=0,
            addr=addr,
            size=size,
            is_write=is_write
        )
        
        self.ongoing_cmds.append(cmd)
        
        msys_dispatch_cmd(
            msys=self.msys, 
            cmd=cmd, 
            dispatch_callback=None, 
            execute_callback=functools.partial(self.execute_callback, addr, size, is_write)
        )
        
        return cmd

    def run_test(self):
        self.cycles = 0
        self.summary.append([])
        
        if self.verbose:    
            print(f"=== TEST {len(self.summary)-1} START ===")
            
        print(f"Ongoing cmds: {sum(1 for c in self.ongoing_cmds if not check_msys_cmd_executed(c))}")
        
        while not all(check_msys_cmd_executed(c) for c in self.ongoing_cmds):
            self.update_cycle_time(1)
            self.cycles += 1
            
        self.ongoing_cmds.clear()
            
    def save_summary(self, file_path: str):
        with open(file_path, "w") as f:
            f.write("test,addr,size,is_write,cycles\n")
            for record_id, records in enumerate(self.summary):
                for record in records:
                    addr, size, is_write, cycles = record
                    f.write(f"{record_id},{addr},{size},{is_write},{cycles}\n")