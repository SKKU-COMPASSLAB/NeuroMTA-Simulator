import os
import sys
import multiprocessing as mp

from neuromta.framework import logger

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.dirname(__file__))
    
    commands = [
        f"python3 {ROOT}/conv2d_all_main.py -n 12",
        f"python3 {ROOT}/conv2d_l1_feature_map.py -n 12",
    ]
    
    processes: list[mp.Process] = []
    for cmd in commands:
        p = mp.Process(target=os.system, args=(cmd,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    logger.info("All experiments completed.")