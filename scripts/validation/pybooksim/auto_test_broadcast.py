import subprocess
import sys
import os
import math


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

script_path = os.path.join(ROOT_DIR, "test_broadcast.py")

default_configs = {
    '-xr': 1,
    '-yr': 1,
    '-subnets': 2,
    '-src': 0,
    '-packet-size': 16,  # 16 flits x 32B = 512B packet size
}

processes = []

for x in [4, 8, 16]:
    for y in [4, 8, 16]:
        n_nodes = x * y
        
        for n_concurrent in [2**i for i in range(0, int(math.log2(n_nodes))+1)]:
            additional_options = {
                '-x': x,
                '-y': y,
                '-n-concurrent': n_concurrent,
            }
            
            cmd = [sys.executable, script_path]
            
            for key, value in {**default_configs, **additional_options}.items():
                cmd.extend([key, str(value)])
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(proc)
            print(f"start running: {' '.join(cmd[2:])} (PID: {proc.pid})")

for proc in processes:
    proc.wait()

print("All processes completed.")