import time

from neuromta.framework.logger import logger

__all__ = [
    "print_log_execution_time",
]

class print_log_execution_time:
    def __init__(self, desc="", disable=False):
        self.desc = desc
        self.disable = disable
        self.st_time_ns = time.perf_counter_ns()

    def open(self):
        self.st_time_ns = time.perf_counter_ns()
        
    def close(self):
        if self.disable:
            return
        elapsed_time_ns = time.perf_counter_ns() - self.st_time_ns
        logger.debug(f"{self.desc} \tExecution time: {elapsed_time_ns} ns")
        
    def get(self):
        if self.disable:
            return 0
        return time.perf_counter_ns() - self.st_time_ns
    
    def __enter__(self):
        self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()