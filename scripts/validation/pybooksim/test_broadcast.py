import os
import sys
import argparse

ROOT_DIR_NAME = os.path.dirname(os.path.abspath(__file__))
ROOT_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

sys.path.append(ROOT_DIR_NAME)

from utils import IcntTestContext


def parse_args():
    parser = argparse.ArgumentParser(description="PyBookSim ICNT Test")
    parser.add_argument("-x", type=int, default=4, help="Number of nodes in X dimension")
    parser.add_argument("-y", type=int, default=4, help="Number of nodes in Y dimension")
    parser.add_argument("-xr", type=int, default=1, help="Torus wraparounds in X dimension")
    parser.add_argument("-yr", type=int, default=1, help="Torus wraparounds in Y dimension")
    parser.add_argument("-subnets", type=int, default=2, help="Number of subnets")
    parser.add_argument("-src", type=int, default=0, help="Source node ID")
    parser.add_argument("-packet-size", type=int, default=1, help="Packet size in flits")
    parser.add_argument("-n-concurrent", type=int, default=1, help="Number of concurrent commands to dispatch")
    parser.add_argument("-save-dir", type=str, default=os.path.join(ROOT_DIR_NAME, ".logs", ROOT_FILE_NAME), help="Directory to save outputs")
    parser.add_argument("-verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    x = args.x
    y = args.y
    xr = args.xr
    yr = args.yr
    subnets = args.subnets
    src_id = args.src
    packet_size = args.packet_size
    n_concurrent_cmds = args.n_concurrent
    save_dir = args.save_dir
    verbose = args.verbose
    
    if verbose:
        print(f"ICNT Broadcast Test with parameters:")
        print(f"  - Topology: {x}x{y} with {xr}x{yr} wraparounds")
        print(f"  - Source node ID: {src_id}")
        print(f"  - Packet size: {packet_size} flits")
        print(f"  - Number of concurrent commands: {n_concurrent_cmds}")
    
    test_context = IcntTestContext(x=x, y=y, xr=xr, yr=yr, subnets=subnets, verbose=verbose)
    
    for t, dst in enumerate(range(0, x * y, n_concurrent_cmds)):
        for i in range(n_concurrent_cmds):
            test_context.dispatch_single_cmd(src_id=src_id, dst_id=dst+i, packet_size=packet_size)
        test_context.run_test()
    
    os.makedirs(save_dir, exist_ok=True)

    summary_file_name = f"topo_{x}x{y}x{xr}x{yr}_subnet_{subnets}_concurrent_{n_concurrent_cmds}.csv"
    summary_file_path = os.path.join(save_dir, summary_file_name)
    
    test_context.save_summary(file_path=summary_file_path)
    print(f"Summary saved to: '{summary_file_path}'")


if __name__ == "__main__":
    main()
    