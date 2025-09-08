import math


__all__ = [
    "K_UNIT",
    "M_UNIT",
    "G_UNIT",
    "T_UNIT",
    
    "parse_freq_str",
    "parse_mem_cap_str",
]


K_UNIT = 1024
M_UNIT = K_UNIT * 1024
G_UNIT = M_UNIT * 1024
T_UNIT = G_UNIT * 1024
    
    
def parse_freq_str(expr: str) -> int:
    if expr.lower().endswith("khz"):
        expr = math.floor(float(expr[:-3]) * K_UNIT)
    elif expr.lower().endswith("mhz"):
        expr = math.floor(float(expr[:-3]) * M_UNIT)
    elif expr.lower().endswith("ghz"):
        expr = math.floor(float(expr[:-3]) * G_UNIT)
    elif expr.lower().endswith("hz"):
        expr = int(expr[:-2])
    else:
        try:
            expr = int(expr)
        except:
            raise Exception(f"[ERROR] Invalid frequency expression: {expr}")
    return expr


def parse_mem_cap_str(expr: str) -> int:
    if expr.lower().endswith("bytes"):
        expr = expr[:-4]
    elif expr.lower().endswith("byte"):
        expr = expr[:-3]
    
    if expr.lower().endswith("kb"):
        expr = math.floor(float(expr[:-2]) * K_UNIT)
    elif expr.lower().endswith("mb"):
        expr = math.floor(float(expr[:-2]) * M_UNIT)
    elif expr.lower().endswith("gb"):
        expr = math.floor(float(expr[:-2]) * G_UNIT)
    elif expr.lower().endswith("b"):
        expr = int(expr[:-1])
    else:
        try:
            expr = int(expr)
        except:
            raise Exception(f"[ERROR] Invalid memory capacity expression: {expr}")
    return expr