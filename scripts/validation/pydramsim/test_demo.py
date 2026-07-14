import os
import sys
import argparse

ROOT_DIR_NAME = os.path.dirname(os.path.abspath(__file__))
ROOT_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

sys.path.append(ROOT_DIR_NAME)

from neuromta.framework import *
from utils import DRAMTestContext


def parse_args():
    parser = argparse.ArgumentParser(description="PyDRAMSim Test")
    parser.add_argument("-save-dir", type=str, default=os.path.join(ROOT_DIR_NAME, ".logs", ROOT_FILE_NAME), help="Directory to save outputs")
    parser.add_argument("-verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    save_dir = args.save_dir
    verbose = args.verbose
    
    test_context = DRAMTestContext(verbose=verbose)
    
    test_context.dispatch_single_cmd(addr=0x00, size=parse_mem_cap_str("1MB"), is_write=False)
    test_context.run_test()
    
    os.makedirs(save_dir, exist_ok=True)

    summary_file_name = f"summary.csv"
    summary_file_path = os.path.join(save_dir, summary_file_name)
    
    test_context.save_summary(file_path=summary_file_path)
    print(f"Summary saved to: '{summary_file_path}'")


if __name__ == "__main__":
    main()
    