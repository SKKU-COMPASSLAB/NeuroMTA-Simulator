import enum


__all__ = [
    "MemoryType",
    "DataType",
]
    

class MemoryType(enum.Enum):
    L1   = enum.auto()
    L2   = enum.auto()
    MAIN = enum.auto()
    
    
class DataType(enum.Enum):
    INT8  = enum.auto()
    INT16 = enum.auto()
    INT32 = enum.auto()
    BF16  = enum.auto()
    FP16  = enum.auto()
    FP32  = enum.auto()
    FP64  = enum.auto()
    
    @property
    def size(self) -> int:
        if self == DataType.INT8:
            return 1
        elif self == DataType.INT16:
            return 2
        elif self == DataType.INT32:
            return 4
        elif self == DataType.BF16:
            return 2
        elif self == DataType.FP16:
            return 2
        elif self == DataType.FP32:
            return 4
        elif self == DataType.FP64:
            return 8
        else:
            raise ValueError(f"Unknown data type: {self}")