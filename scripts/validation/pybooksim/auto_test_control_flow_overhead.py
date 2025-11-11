import subprocess
import sys
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

script_path = os.path.join(ROOT_DIR, "test_control_flow_overhead.py")

default_configs = {
    '-x': 8,
    '-y': 8,
    '-xr': 1,
    '-yr': 1,
    '-subnets': 2,
    '-packet-size': 16,  # 16 flits x 32B = 512B packet size
}

processes = []

for src_id in range(0, 16, 4):
    for dst_id in range(1, 16, 4):
        for transaction_size in [64, 256, 1024, 4096]:  # in flits
            for packet_size in [4, 16, 64]:  # in flits
                if src_id == dst_id:
                    continue
                
                additional_options = {
                    '-src': src_id,
                    '-dst': dst_id,
                    '-transaction-size': transaction_size,
                    '-packet-size': packet_size,
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