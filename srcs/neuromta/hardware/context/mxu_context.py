import enum
import torch


__all__ = [
    "MXUDataflow",
    "MXUConfig",
    "MXUContext",
]


class MXUDataflow(enum.Enum):
    OS = enum.auto()  # Output Stationary
    WS = enum.auto()  # Weight Stationary
    
    
class MXUConfig(dict):
    def __init__(
        self,
        
        pe_arr_height: int = 32,
        pe_arr_width: int = 32,
        seq_len: int = 32,
        dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        dataflow: MXUDataflow = MXUDataflow.OS,
        op_latency_per_byte: int = 1,
    ):
        super().__init__()
        
        self["pe_arr_height"] = pe_arr_height
        self["pe_arr_width"] = pe_arr_width
        self["seq_len"] = seq_len
        self["dtype"] = dtype
        self["acc_dtype"] = acc_dtype
        self["dataflow"] = dataflow
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
        dataflow: MXUDataflow,
        op_latency_per_byte: int,
    ):
        self.pe_arr_height  = pe_arr_height
        self.pe_arr_width   = pe_arr_width
        self.seq_len        = seq_len
        self._dtype         = dtype
        self._acc_dtype     = acc_dtype
        self._dataflow      = dataflow
        self.op_latency_per_byte = op_latency_per_byte
        
        # Determine the tile shape
        if self._dataflow == MXUDataflow.OS:
            if self.seq_len != self.pe_arr_height:
                raise Exception(f"[ERROR] The sequence length should be the same with the PE array height for OS dataflow (input output tile shape consistency)")
        
        # Initialize registers
        self._pe_arr_regs: torch.Tensor = torch.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)
        self._acc_regs:    torch.Tensor = torch.zeros((self.seq_len, self.pe_arr_width), dtype=self._acc_dtype) if self._dataflow == MXUDataflow.WS else None

    def reconfigure_dtype(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        self._dtype = dtype
        self._acc_dtype = acc_dtype
        
        self._pe_arr_regs: torch.Tensor = torch.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)
        self._acc_regs:    torch.Tensor = torch.zeros((self.seq_len, self.pe_arr_width), dtype=self._acc_dtype) if self._dataflow == MXUDataflow.WS else None

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

    def get_acc_regs(self, clear_regs: bool=True) -> torch.Tensor:
        if self._acc_regs is None:
            return self.get_pe_arr_regs()
        regs = self._acc_regs
        if clear_regs:
            self._acc_regs = torch.zeros_like(self._acc_regs)
        return regs

    def load_tile_pe_arr(self, tile: torch.Tensor):
        if tile.shape != self.pe_arr_shape:
            raise Exception(f"[ERROR] Tile shape {tile.shape} does not match PE array shape {(self.pe_arr_height, self.pe_arr_width)}.")
        
        self._pe_arr_regs[:, :] = tile.to(dtype=self._acc_dtype)
        
    def load_tile_acc_regs(self, tile: torch.Tensor):
        if self._acc_regs is None:
            raise Exception("[ERROR] Accumulator registers are not available in this dataflow.")
        if tile.shape != self.acc_regs_shape:
            raise Exception(f"[ERROR] Tile shape {tile.shape} does not match accumulator registers shape {self.acc_regs_shape}.")
        self._acc_regs[:, :] = tile.to(dtype=self._acc_dtype)

    def execute_gemm(self, ifm_tile: torch.Tensor, wgt_tile: torch.Tensor=None, psum_tile: torch.Tensor=None) -> torch.Tensor:
        if self._dataflow == MXUDataflow.OS:
            if wgt_tile is None:
                raise Exception("[ERROR] WGT tile must be provided for OS dataflow.")
            if ifm_tile.shape != self.ifm_tile_shape:
                raise Exception(f"[ERROR] IFM tile shape {ifm_tile.shape} does not match expected shape {self.ifm_tile_shape}.")
            if wgt_tile.shape != self.wgt_tile_shape:
                raise Exception(f"[ERROR] WGT tile shape {wgt_tile.shape} does not match expected shape {self.wgt_tile_shape}.")
            
            self._pe_arr_regs = (ifm_tile @ wgt_tile) + self._pe_arr_regs
        elif self._dataflow == MXUDataflow.WS:
            if wgt_tile is not None:
                raise Exception("[ERROR] WGT tile should not be provided for WS dataflow.")
            if ifm_tile.shape != self.ifm_tile_shape:
                raise Exception(f"[ERROR] IFM tile shape {ifm_tile.shape} does not match expected shape {self.ifm_tile_shape}.")
            if psum_tile is None:
                psum_tile = self._acc_regs
            
            self._acc_regs[:, :] = (ifm_tile @ self._pe_arr_regs) + psum_tile
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    def flush_pe_arr(self):
        self._pe_arr_regs = torch.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)

    def flush_acc_regs(self) -> torch.Tensor:
        if self._acc_regs is None:
            raise Exception("[ERROR] Accumulator registers are not available in this dataflow.")

        self._acc_regs = torch.zeros((self.seq_len, self.pe_arr_width), dtype=self._acc_dtype)
        
    @property
    def acc_dtype(self) -> torch.dtype:
        return self._acc_dtype
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def dataflow(self) -> MXUDataflow:
        return self._dataflow
        
    @property
    def pe_arr_shape(self) -> tuple[int, int]:
        return (self.pe_arr_height, self.pe_arr_width)
    
    @property
    def acc_regs_shape(self) -> tuple[int, int]:
        if self._acc_regs is None:
            return self.pe_arr_shape
        return (self.seq_len, self.pe_arr_width)
        
    @property
    def m_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.pe_arr_height
        elif self._dataflow == MXUDataflow.WS:
            return self.seq_len
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    @property
    def n_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.pe_arr_width
        elif self._dataflow == MXUDataflow.WS:
            return self.pe_arr_width
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    @property
    def k_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.seq_len
        elif self._dataflow == MXUDataflow.WS:
            return self.pe_arr_height
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
    
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
