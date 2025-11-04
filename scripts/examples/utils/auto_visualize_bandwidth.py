import subprocess
import sys
import os
import argparse


parser = argparse.ArgumentParser(description="Auto Runner for Multiple Scripts")
parser.add_argument("--test", type=str, required=True, dest="test_name", help="Test name to run")
parser.add_argument("--log-dir", nargs="*", type=str, dest="log_dirs", help="List of log directories for each script")
args = parser.parse_args()

test_name = args.test_name
log_dirs = args.log_dirs if args.log_dirs is not None else []

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

script_path = os.path.join(EXAMPLES_DIR, "utils", "visualize_bandwidth.py")

for log_dir in log_dirs:
    test_dir = os.path.join(EXAMPLES_DIR, test_name, log_dir)

    pargs = ["--test-dir", test_dir, "--booksim2-rolling", "1000", "--dramsim3-rolling", "10"]
    proc = subprocess.Popen(
        [sys.executable, script_path] + pargs,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    
    print(f"start running: {script_path} {' '.join(pargs)} (PID: {proc.pid})")
    
    proc.wait()

print("All processes completed.")