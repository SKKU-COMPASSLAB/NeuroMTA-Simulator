import time

from neuromta.framework.logger import logger

__all__ = [
    "print_log_execution_time",
]

class print_log_execution_time:
    def __init__(self, desc=""):
        self.desc = desc
        self.st_time = time.time()

    def __enter__(self):
        self.st_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.st_time
        logger.debug(f"{self.desc} Execution time: {elapsed_time:.6f} seconds")