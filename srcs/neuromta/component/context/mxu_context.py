import enum
import torch


__all__ = [
    "MXUConfig",
    "MXUContext",
    "MXUElementwiseOp",
]
    

class MXUElementwiseOp(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    CMP_MAX = enum.auto()
    CMP_MIN = enum.auto()
    
    
class MXUConfig(dict):
    def __init__(
        self,
        
        pe_arr_height: int = 32,
        pe_arr_width: int = 32,
        seq_len: int = 32,
        dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        op_latency_per_byte: int = 1,
    ):
        super().__init__()
        
        self["pe_arr_height"] = pe_arr_height
        self["pe_arr_width"] = pe_arr_width
        self["seq_len"] = seq_len
        self["dtype"] = dtype
        self["acc_dtype"] = acc_dtype
        self["op_latency_per_byte"] = op_latency_per_byte
        
    def create_context(self) -> "MXUContext":
        return MXUContext(**self)


class MXUContext:
    def __init__(
        self,
        
        pe_arr_height: int,
        pe_arr_width: int,
        seq_len: int,
        dtype: torch.dtype,
        acc_dtype: torch.dtype,
        op_latency_per_byte: int,
    ):
        self.pe_arr_height  = pe_arr_height
        self.pe_arr_width   = pe_arr_width
        self.seq_len        = seq_len
        self._dtype         = dtype
        self._acc_dtype     = acc_dtype
        self.op_latency_per_byte = op_latency_per_byte
        
        # # Determine the tile shape
        if self.seq_len != self.pe_arr_height:
            raise Exception(f"The sequence length should be the same with the PE array height for OS dataflow (input output tile shape consistency)")
        
        # Initialize registers
        self._pe_arr_regs: torch.Tensor = torch.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)

    def reconfigure_dtype(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self._dtype = dtype
        self._acc_dtype = acc_dtype
        
        self._pe_arr_regs: torch.Tensor = torch.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)

    def get_preload_pe_arr_cycles(self) -> int:
        return self.pe_arr_width
    
    def get_preload_acc_regs_cycles(self) -> int:
        return self.seq_len
    
    def get_execute_cycles(self) -> int:
        return self.seq_len * self.op_latency_per_byte * self.dtype.itemsize
    
    def get_flush_pe_arr_cycles(self) -> int:
        return self.pe_arr_width
    
    def get_flush_acc_regs_cycles(self) -> int:
        return self.seq_len

    def get_pe_arr_regs(self, clear_regs: bool=True) -> torch.Tensor:
        regs = self._pe_arr_regs
        if clear_regs:
            self._pe_arr_regs = torch.zeros_like(self._pe_arr_regs)
        return regs

    def load_tile_pe_arr(self, tile: torch.Tensor):
        self._pe_arr_regs[:, :] = tile.to(dtype=self._acc_dtype)

    def execute_gemm(self, ifm_tile: torch.Tensor, wgt_tile: torch.Tensor=None, psum_tile: torch.Tensor=None) -> torch.Tensor:
        if wgt_tile is None:
            raise Exception("[ERROR] WGT tile must be provided for OS dataflow.")
        if ifm_tile.shape != self.ifm_tile_shape:
            raise Exception(f"IFM tile shape {ifm_tile.shape} does not match expected shape {self.ifm_tile_shape}.")
        if wgt_tile.shape != self.wgt_tile_shape:
            raise Exception(f"WGT tile shape {wgt_tile.shape} does not match expected shape {self.wgt_tile_shape}.")
        
        self._pe_arr_regs = (ifm_tile.to(dtype=self._acc_dtype) @ wgt_tile.to(dtype=self._acc_dtype)) + self._pe_arr_regs
        
    def execute_maxpool(self, ifm_tile: torch.Tensor, psum_tile: torch.Tensor=None) -> torch.Tensor:
        if psum_tile is not None:
            raise Exception("[ERROR] PSUM tile must not be provided for OS dataflow.")
        if ifm_tile.shape != self.ofm_tile_shape:
            raise Exception(f"IFM tile shape {ifm_tile.shape} does not match expected shape {self.ofm_tile_shape}.")
        
        self._pe_arr_regs = torch.maximum(ifm_tile.to(dtype=self._acc_dtype), self._pe_arr_regs)

    def execute_elemwise(self, ifm_tile: torch.Tensor, op: MXUElementwiseOp) -> torch.Tensor:
        if ifm_tile.shape != self.ofm_tile_shape:
            raise Exception(f"IFM tile shape {ifm_tile.shape} does not match expected shape {self.ofm_tile_shape}.")

        if op == MXUElementwiseOp.ADD:
            self._pe_arr_regs = ifm_tile.to(dtype=self._acc_dtype) + self._pe_arr_regs
        elif op == MXUElementwiseOp.SUB:
            self._pe_arr_regs = ifm_tile.to(dtype=self._acc_dtype) - self._pe_arr_regs
        elif op == MXUElementwiseOp.MUL:
            self._pe_arr_regs = ifm_tile.to(dtype=self._acc_dtype) * self._pe_arr_regs
        elif op == MXUElementwiseOp.DIV:
            self._pe_arr_regs = ifm_tile.to(dtype=self._acc_dtype) / self._pe_arr_regs
        else:
            raise Exception(f"Unsupported elementwise operation: {op}.")
        
    @property
    def acc_dtype(self) -> torch.dtype:
        return self._acc_dtype
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
        
    @property
    def pe_arr_shape(self) -> tuple[int, int]:
        return (self.pe_arr_height, self.pe_arr_width)
        
    @property
    def m_tile(self) -> int:
        return self.pe_arr_height
        
    @property
    def n_tile(self) -> int:
        return self.pe_arr_width
        
    @property
    def k_tile(self) -> int:
        return self.seq_len
    
    @property
    def ifm_tile_numel(self) -> int:
        return self.m_tile * self.k_tile
    
    @property
    def wgt_tile_numel(self) -> int:
        return self.k_tile * self.n_tile
    
    @property
    def ofm_tile_numel(self) -> int:
        return self.m_tile * self.n_tile
    
    @property
    def ifm_tile_shape(self) -> tuple[int, int]:
        return (self.m_tile, self.k_tile)
    
    @property
    def wgt_tile_shape(self) -> tuple[int, int]:
        return (self.k_tile, self.n_tile)

    @property
    def ofm_tile_shape(self) -> tuple[int, int]:
        return (self.m_tile, self.n_tile)
    
    @property
    def ifm_tile_size(self) -> int:
        return self.ifm_tile_numel * self.dtype.itemsize
    
    @property
    def wgt_tile_size(self) -> int:
        return self.wgt_tile_numel * self.dtype.itemsize
    
    @property
    def ofm_tile_size(self) -> int:
        return self.ofm_tile_numel * self.acc_dtype.itemsize
