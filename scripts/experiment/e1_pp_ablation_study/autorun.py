import os
import sys
import multiprocessing as mp

from neuromta.framework import logger
from neuromta.framework.parser_utils import parse_mem_cap_str

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.dirname(__file__))
    
    commands = []
    
    l1_buf_sizes = list(range(parse_mem_cap_str("128KB"), parse_mem_cap_str("1.2MB"), parse_mem_cap_str("128KB")))

    # Run with PP
    cmd_fmt = f"python3 {ROOT}/run_with_pp.py --l1-buf-size {{l1_buf_size}}"
    for l1_buf_size in l1_buf_sizes:
        cmd = cmd_fmt.format(l1_buf_size=l1_buf_size)
        commands.append(cmd)
        
    cmd_fmt = f"python3 {ROOT}/run_wo_pp.py --l1-buf-size {{l1_buf_size}}"
    for l1_buf_size in l1_buf_sizes:
        cmd = cmd_fmt.format(l1_buf_size=l1_buf_size)
        commands.append(cmd)
        
    cmd_fmt = f"python3 {ROOT}/run_wo_pp.py --l1-buf-size {{l1_buf_size}} --l1-interm"
    for l1_buf_size in l1_buf_sizes:
        cmd = cmd_fmt.format(l1_buf_size=l1_buf_size)
        commands.append(cmd)
    
    processes: list[mp.Process] = []
    for cmd in commands:
        p = mp.Process(target=lambda: os.system(f"{cmd} > /dev/null 2>&1"))
        p.start()
        processes.append(p)
        logger.info(f"Started process for command: {cmd}")
        
    for p in processes:
        p.join()
        
    logger.info("All experiments completed.")