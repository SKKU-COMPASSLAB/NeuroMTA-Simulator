import enum


__all__ = [
    "SimulationMode",
    "normalize_simulation_mode",
    "set_global_simulation_mode",
    "get_global_simulation_mode",
    "is_global_performance_mode",
]


class SimulationMode(enum.Enum):
    CORRECTNESS = enum.auto()
    PERFORMANCE = enum.auto()


_global_simulation_mode: SimulationMode = SimulationMode.CORRECTNESS


def normalize_simulation_mode(mode: SimulationMode | str | bool) -> SimulationMode:
    if isinstance(mode, SimulationMode):
        return mode
    if isinstance(mode, bool):
        return SimulationMode.PERFORMANCE if mode else SimulationMode.CORRECTNESS
    if isinstance(mode, str):
        normalized = mode.strip().upper()
        aliases = {
            "CORRECTNESS": SimulationMode.CORRECTNESS,
            "FUNCTIONAL": SimulationMode.CORRECTNESS,
            "VALIDATION": SimulationMode.CORRECTNESS,
            "PERFORMANCE": SimulationMode.PERFORMANCE,
            "PERF": SimulationMode.PERFORMANCE,
        }
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError(f"Invalid simulation mode: {mode}")


def set_global_simulation_mode(mode: SimulationMode | str | bool) -> SimulationMode:
    global _global_simulation_mode
    _global_simulation_mode = normalize_simulation_mode(mode)
    return _global_simulation_mode


def get_global_simulation_mode() -> SimulationMode:
    return _global_simulation_mode


def is_global_performance_mode() -> bool:
    return _global_simulation_mode == SimulationMode.PERFORMANCE
