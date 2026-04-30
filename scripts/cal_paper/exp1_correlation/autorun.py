import os
import argparse
import multiprocessing as mp
from neuromta.framework import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-run all linear roofline experiments for Tenstorrent IP1.")
    parser.add_argument("-n", "--num_processes", type=int, default=12, help="Number of parallel processes to run experiments.", dest="num_processes")
    parser.add_argument("--monitor", action="store_true", help="Whether to monitor the experiments in real-time.", dest="monitor")
    parser.add_argument('--skip-execution', action="store_true", help="Whether to skip kernel execution and only perform compilation and profiling setup", dest="skip_execution")
    args = parser.parse_args()
    
    ROOT = os.path.abspath(os.path.dirname(__file__))
    
    commands = [
        f"python3 {ROOT}/run.py -n {args.num_processes} {'--monitor' if args.monitor else ''} {'--skip-execution' if args.skip_execution else ''}",
    ]
    
    processes: list[mp.Process] = []
    for cmd in commands:
        p = mp.Process(target=os.system, args=(cmd,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    logger.info("All experiments completed.")