import functools
from typing import Any

from pybooksim2 import *


class IcntTestContext:
    def __init__(self, x: int, y: int, xr: int, yr: int, subnets: int, verbose: bool = False):
        self.verbose = verbose
               
        self.icnt_config = create_config_torus_2d(subnets=subnets, x=x, y=y, xr=xr, yr=yr)
        self.icnt = create_icnt(config=self.icnt_config)

        self.ongoing_cmds = []
        
        self.cycles = 0
        self.summary: list[list[tuple[int, int, int, int, int]]] = []  # src_id, dst_id, packet_size, cycles, timestamp
        
    def execute_callback(self, src_id, dst_id, packet_size, cmd_ptr):
        self.summary[-1].append((src_id, dst_id, packet_size, self.cycles, get_sim_time()))
    
        if self.verbose:
            print(f"  * [CYCLES {self.cycles:<3d}] src {src_id:<2d} -> dst {dst_id:<2d} : packet size {packet_size} flits")

    def dispatch_single_cmd(self, src_id: int, dst_id: int, packet_size: int) -> Any:
        cmd = create_icnt_cmd_data_packet(
            src_id=src_id,
            dst_id=dst_id,
            subnet=0,
            size=packet_size,
            is_write=True,
            is_response=False
        )
        
        self.ongoing_cmds.append(cmd)
        
        icnt_dispatch_cmd(
            icnt=self.icnt, 
            cmd=cmd, 
            dispatch_callback=None, 
            execute_callback=functools.partial(self.execute_callback, src_id, dst_id, packet_size)
        )
        
        return cmd

    def run_test(self):
        self.cycles = 0
        self.summary.append([])
        
        if self.verbose:    
            print(f"=== TEST {len(self.summary)-1} START ===")
        
        while not all(check_icnt_cmd_received(c) for c in self.ongoing_cmds):
            icnt_cycle_step(icnt=self.icnt, cycles=1)
            self.cycles += 1
            
        self.ongoing_cmds.clear()
            
    def save_summary(self, file_path: str):
        with open(file_path, "w") as f:
            f.write("test,src_id,dst_id,packet_size,cycles,timestamp\n")
            for record_id, records in enumerate(self.summary):
                for record in records:
                    src_id, dst_id, packet_size, cycles, timestamp = record
                    f.write(f"{record_id},{src_id},{dst_id},{packet_size},{cycles},{timestamp}\n")