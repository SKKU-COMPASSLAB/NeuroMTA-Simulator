import enum
import torch

from neuromta.framework.parser_utils import parse_mem_cap_str


__all__ = [
    "VPUConfig",
    "VPUOperator",
    "VPUContext",
]


class VPUConfig(dict):
    def __init__(
        self, 
        
        vreg_len: int = parse_mem_cap_str("128B"),
        vreg_num: int = 32,
        vdtype: torch.dtype = torch.float32,
        
        vlen_max: int = 1024,
        vlen_min: int = 32,
        
        unary_op_latency: int = 2,
        arith_op_latency: int = 4,
    ):
        super().__init__()
        
        self["vreg_len"] = vreg_len
        self["vreg_num"] = vreg_num
        self["vdtype"]   = vdtype
        self["vlen_max"] = vlen_max
        self["vlen_min"] = vlen_min
        self["unary_op_latency"] = unary_op_latency
        self["arith_op_latency"] = arith_op_latency
        
    def create_context(self) -> "VPUContext":
        return VPUContext(**self)
    

class VPUOperator(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    
    RELU = enum.auto()
    
    @property
    def is_unary(self) -> bool:
        return self in (VPUOperator.RELU,)


class VPUContext:
    def __init__(
        self,
        
        vreg_len: int = parse_mem_cap_str("128B"),
        vreg_num: int = 32,
        vdtype: torch.dtype = torch.float32,
        
        vlen_max: int = 1024,
        vlen_min: int = 32,
        
        unary_op_latency: int = 2,
        arith_op_latency: int = 4,
    ):
        self._physical_vreg_len = vreg_len    # physical vector register length in bytes
        self._physical_vreg_num = vreg_num    # number of physical vector registers
        
        self.vlen_max = vlen_max    # maximum vector length in elements
        self.vlen_min = vlen_min    # minimum vector length in elements

        self.unary_op_latency = unary_op_latency
        self.arith_op_latency = arith_op_latency

        self._physical_vrf: torch.Tensor = torch.zeros((self._physical_vreg_num * self._physical_vreg_len,), dtype=torch.uint8)

        self._vdtype:   torch.dtype = vdtype
        self._vlen:     int         = self._physical_vreg_len // self._vdtype.itemsize
        self._n_vregs:  int         = self._physical_vreg_num
        
        self._vreg_view:    torch.Tensor = self._physical_vrf.view(dtype=self._vdtype).reshape(self._n_vregs, self._vlen)

    def reconfigure_vector_reg_file(self, vlen: int, vdtype: torch.dtype):
        if isinstance(vdtype, str):
            vdtype = torch.dtype(vdtype)
            
        if vlen < self.vlen_min or vlen > self.vlen_max:
            raise Exception(f"[ERROR] Vector length {vlen} is out of bounds ({self.vlen_min}, {self.vlen_max}).")
            
        self._vdtype    = vdtype
        self._vlen      = vlen
        self._n_vregs   = self._physical_vreg_len // (self._vlen * self._vdtype.itemsize) * self._physical_vreg_num
        
        self._vreg_view = self._physical_vrf.view(dtype=self._vdtype).reshape(self._n_vregs, self._vlen)
        
    def set_vector_reg(self, vreg_idx: int, data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            raise Exception("[ERROR] Data must be a numpy array.")
        if data.dtype != self._vdtype:
            raise Exception(f"[ERROR] Data type {data.dtype} does not match vector register type {self._vdtype}.")
        if len(data.flatten()) != self._vlen:
            raise Exception(f"[ERROR] Data size {data.size()} does not match vector length {self._vlen}.")
        if vreg_idx < 0 or vreg_idx >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_idx} out of bounds (0, {self._n_vregs}).")

        self._vreg_view[vreg_idx, :] = data
        
    def get_vector_reg(self, vreg_idx: int) -> torch.Tensor:
        if vreg_idx < 0 or vreg_idx >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_idx} out of bounds (0, {self._n_vregs}).")
        
        return self._vreg_view[vreg_idx, :].clone()
    
    def execute_vector_op(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False) -> None:     
        if inplace:
            vreg_dest = vreg_a
        
        if opcode.is_unary:
            is_unary = True
            vreg_b = 0
        else:
            is_unary = False
        
        if vreg_a < 0 or vreg_a >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_a} out of bounds (0, {self._n_vregs}).")
        if not is_unary and vreg_b is None:
            raise Exception(f"[ERROR] Vector register B must be provided for binary operations.")
        elif vreg_b < 0 or vreg_b >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_b} out of bounds (0, {self._n_vregs}).")
        if vreg_dest is not None and (vreg_dest < 0 or vreg_dest >= self._n_vregs):
            raise Exception(f"[ERROR] Vector register index {vreg_dest} out of bounds (0, {self._n_vregs}).")

        if opcode == VPUOperator.ADD:
            self._vreg_view[vreg_dest, :] = self._vreg_view[vreg_a, :] + self._vreg_view[vreg_b, :]
        elif opcode == VPUOperator.SUB:
            self._vreg_view[vreg_dest, :] = self._vreg_view[vreg_a, :] - self._vreg_view[vreg_b, :]
        elif opcode == VPUOperator.MUL:
            self._vreg_view[vreg_dest, :] = self._vreg_view[vreg_a, :] * self._vreg_view[vreg_b, :]
        elif opcode == VPUOperator.DIV:
            self._vreg_view[vreg_dest, :] = self._vreg_view[vreg_a, :] / self._vreg_view[vreg_b, :]
        elif opcode == VPUOperator.RELU:
            torch.maximum(self._vreg_view[vreg_a, :], 0, out=self._vreg_view[vreg_dest, :])

    @property
    def vdtype(self) -> torch.dtype:
        return self._vdtype
    
    @property
    def vlen(self) -> int:
        return self._vlen