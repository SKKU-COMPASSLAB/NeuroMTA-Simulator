import subprocess
import sys
import os
import argparse


parser = argparse.ArgumentParser(description="Auto Runner for Multiple Scripts")
parser.add_argument("--test", type=str, required=True, dest="test_name", help="Test name to run")
parser.add_argument("--noc-flit-size", nargs="*", type=str, dest="noc_flit_sizes", help="List of flit sizes for each script")
args = parser.parse_args()

test_name = args.test_name
noc_flit_sizes = args.noc_flit_sizes if args.noc_flit_sizes is not None else []

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

script_path = os.path.join(EXAMPLES_DIR, test_name, "main.py")
processes = []

for noc_flit_size in noc_flit_sizes:
    pargs = ["--noc-flit-size", noc_flit_size, "--log-dir", f"flit{noc_flit_size}"]
    proc = subprocess.Popen(
        [sys.executable, script_path] + pargs,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    processes.append(proc)
    print(f"start running: {script_path} {' '.join(pargs)} (PID: {proc.pid})")
    
for proc in processes:
    proc.wait()

print("All processes completed.")