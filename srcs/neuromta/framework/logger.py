import sys
import enum


__all__ = [
    "LogLevel",
    "logger",
]


class LogLevel(enum.Enum):
    DEBUG       = 0
    INFO        = 1
    WARNING     = 2
    ERROR       = 3
    CRITICAL    = 4

_global_current_log_level: LogLevel = LogLevel.INFO

class logger:
    @classmethod
    def set_log_level(cls, level: LogLevel):
        global _global_current_log_level
        _global_current_log_level = level

    @classmethod
    def log(cls, message: str, level: LogLevel = LogLevel.INFO):
        if isinstance(level, int):
            level = LogLevel(level)
        
        if level.value >= _global_current_log_level.value:
            sys.stdout.write(f"[{level.name}] {message}\n")
    
    @classmethod
    def debug(cls, message: str):
        logger.log(message, LogLevel.DEBUG)
    
    @classmethod
    def info(cls, message: str):
        logger.log(message, LogLevel.INFO)
        
    @classmethod
    def warning(cls, message: str):
        logger.log(message, LogLevel.WARNING)
        
    @classmethod
    def error(cls, message: str):
        logger.log(message, LogLevel.ERROR)
        
    @classmethod
    def critical(cls, message: str):
        logger.log(message, LogLevel.CRITICAL)
