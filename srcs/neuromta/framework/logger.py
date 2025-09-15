import time
import sys
import os
import enum
import threading as th
import multiprocessing as mp
from typing import Any


__all__ = [
    "LogLevel",
    "logger",
    "set_global_monitoring_window",
    "unset_global_monitoring_window",
    "_LOG_LEVEL_COLORS",
    "_COLOR_RESET",
]


class LogLevel(enum.Enum):
    DEBUG       = 0
    INFO        = 1
    WARNING     = 2
    ERROR       = 3
    CRITICAL    = 4
    
_LOG_LEVEL_COLORS = {
    LogLevel.DEBUG:    "\033[94m",      # light blue
    LogLevel.INFO:     "\033[92m",      # light green
    LogLevel.WARNING:  "\033[93m",      # light yellow
    LogLevel.ERROR:    "\033[91m",      # light red
    LogLevel.CRITICAL: "\033[1;91m",    # bold light red
}
_COLOR_RESET = "\033[0m"


_global_current_log_level: LogLevel = LogLevel.INFO
_global_monitoring_window = None


def set_global_monitoring_window(monitoring_window):
    global _global_monitoring_window
    
    if _global_monitoring_window is not None:
        raise RuntimeError("Global monitoring window has already been set.")
    
    _global_monitoring_window = monitoring_window

def unset_global_monitoring_window():
    global _global_monitoring_window
    _global_monitoring_window = None


class logger:
    @classmethod
    def set_print_options(cls, log_level: LogLevel):
        global _global_current_log_level
        
        if isinstance(log_level, int):
            log_level = LogLevel(log_level)
        elif isinstance(log_level, str):
            log_level = LogLevel[log_level.upper()]
        
        _global_current_log_level = log_level

    @classmethod
    def log(cls, message: str, level: LogLevel = LogLevel.INFO):
        global _global_current_log_level
        global _global_monitoring_window

        if isinstance(level, int):
            level = LogLevel(level)
        elif isinstance(level, str):
            level = LogLevel[level.upper()]
        
        if level.value >= _global_current_log_level.value:
            if _global_monitoring_window is not None:
                _global_monitoring_window.add_log(message, level)   # use monitoring window to print log
            else:
                header = f"[{level.name}] "
                sys.stdout.write(f"{_LOG_LEVEL_COLORS[level]}{header}{message}{_COLOR_RESET}" + "\n")

    @classmethod
    def debug(cls, message: str):
        cls.log(message, LogLevel.DEBUG)
    
    @classmethod
    def info(cls, message: str):
        cls.log(message, LogLevel.INFO)

    @classmethod
    def warning(cls, message: str):
        cls.log(message, LogLevel.WARNING)

    @classmethod
    def error(cls, message: str):
        cls.log(message, LogLevel.ERROR)

    @classmethod
    def critical(cls, message: str):
        cls.log(message, LogLevel.CRITICAL)

