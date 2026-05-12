import sys
import enum


__all__ = [
    "LogLevel",
    "logger",
    "set_global_monitoring_window",
    "unset_global_monitoring_window",
    "get_global_monitoring_window",
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
_global_current_monitor_log_level: LogLevel = LogLevel.INFO
_global_monitoring_window = None


def set_global_monitoring_window(monitoring_window):
    global _global_monitoring_window
    
    if _global_monitoring_window is not None:
        raise RuntimeError("Global monitoring window has already been set.")
    
    _global_monitoring_window = monitoring_window

def unset_global_monitoring_window():
    global _global_monitoring_window
    _global_monitoring_window = None
    
def get_global_monitoring_window():
    return _global_monitoring_window


class logger:
    @classmethod
    def set_print_options(cls, log_level: LogLevel=None, monitor_log_level: LogLevel=None):
        global _global_current_log_level
        global _global_current_monitor_log_level

        if isinstance(log_level, int):
            log_level = LogLevel(log_level)
        elif isinstance(log_level, str):
            log_level = LogLevel[log_level.upper()]
        
        if log_level is not None:
            _global_current_log_level = log_level

        if isinstance(monitor_log_level, int):
            monitor_log_level = LogLevel(monitor_log_level)
        elif isinstance(monitor_log_level, str):
            monitor_log_level = LogLevel[monitor_log_level.upper()]

        if monitor_log_level is not None:
            _global_current_monitor_log_level = monitor_log_level

    @classmethod
    def log(cls, message: str, level: LogLevel = LogLevel.INFO, end: str = "\n"):
        global _global_current_log_level
        global _global_monitoring_window

        if isinstance(level, int):
            level = LogLevel(level)
        elif isinstance(level, str):
            level = LogLevel[level.upper()]
        
        if level.value >= _global_current_log_level.value:
            header = f"[{level.name}] "
            if level.value >= LogLevel.ERROR.value:
                sys.stderr.write(f"{_LOG_LEVEL_COLORS[level]}{header}{message}{_COLOR_RESET}" + end)
                sys.stderr.flush()   # ensure the log is printed immediately
            else:
                sys.stdout.write(f"{_LOG_LEVEL_COLORS[level]}{header}{message}{_COLOR_RESET}" + end)
                sys.stdout.flush()   # ensure the log is printed immediately
                
        if level.value >= _global_current_monitor_log_level.value:
            if _global_monitoring_window is not None:
                if _global_monitoring_window.is_initialized:
                    _global_monitoring_window.add_log(message, level)   # also print to monitoring window for better visibility when log level is high enough

    @classmethod
    def debug(cls, message: str, end: str = "\n"):
        cls.log(message, LogLevel.DEBUG, end=end)

    @classmethod
    def info(cls, message: str, end: str = "\n"):
        cls.log(message, LogLevel.INFO, end=end)

    @classmethod
    def warning(cls, message: str, end: str = "\n"):
        cls.log(message, LogLevel.WARNING, end=end)

    @classmethod
    def error(cls, message: str, end: str = "\n"):
        cls.log(message, LogLevel.ERROR, end=end)

    @classmethod
    def critical(cls, message: str, end: str = "\n"):
        cls.log(message, LogLevel.CRITICAL, end=end)

    @classmethod
    def get_current_log_level(cls) -> LogLevel:
        global _global_current_log_level
        return _global_current_log_level

    @classmethod
    def is_current_debug_log_level(cls) -> bool:
        global _global_current_log_level
        return _global_current_log_level == LogLevel.DEBUG

