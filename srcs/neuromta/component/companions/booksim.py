import os
from typing import Any, Callable
from ctypes import c_void_p
from neuromta.framework import *

try:
    import pybooksim2
    PYBOOKSIM2_AVAILABLE = True
except ImportError as e:
    PYBOOKSIM2_AVAILABLE = False
    
PYBOOKSIM2_AVAILABLE = os.environ.get("NEUROMTA_DISABLE_BOOKSIM", "0") != "1" and PYBOOKSIM2_AVAILABLE

logger.debug(f"pybooksim2 availability: {PYBOOKSIM2_AVAILABLE}")
if not PYBOOKSIM2_AVAILABLE:
    logger.debug(f"pybooksim2 is not available. Using a simplified BookSim2 model instead. To enable pybooksim2, please install it via 'pip install externals/pybooksim2'")
 
    
__all__ = [
    "PYBOOKSIM2_AVAILABLE",
    "BookSim2",
    "BookSim2Config",
]


if PYBOOKSIM2_AVAILABLE:
    class BookSim2Config:
        def __init__(self, processor_clock_freq: float, flit_size: int, subnets: int, x: int, y: int, xr: int, yr: int):
            if not PYBOOKSIM2_AVAILABLE:
                raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")
            
            self.processor_clock_freq: float = processor_clock_freq
            self._flit_size: int = flit_size
            self._subnets: int = subnets
            self._x: int = x
            self._y: int = y
            self._xr: int = xr
            self._yr: int = yr

            self._config: c_void_p = pybooksim2.create_config_torus_2d(subnets, x, y, xr, yr)
            
        def peak_bandwidth_per_router(self) -> float:
            return self._flit_size * self._subnets * self.processor_clock_freq
        
        def peak_bisection_bandwidth(self) -> float:
            return self.peak_bandwidth_per_router() * self._y * 2

        def create_icnt(self) -> c_void_p:
            return pybooksim2.create_icnt(config=self._config)
        
        def update_field(self, field: str, value: Any):
            pybooksim2.update_config(self._config, field, value)
        
        def summary(self) -> dict[str, Any]:
            return {
                "subnets": self._subnets,
                "x": self._x,
                "y": self._y,
                "xr": self._xr,
                "yr": self._yr,
            }

    class BookSim2(CompanionModule):
        def __init__(self, config: BookSim2Config):
            super().__init__()
            
            if not PYBOOKSIM2_AVAILABLE:
                raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")

            self._icnt = config.create_icnt()
            self.config = config

        def update_cycle_time(self, cycle_time):
            pybooksim2.icnt_cycle_step(icnt=self._icnt, cycles=cycle_time)

        def create_command(self, src_id: int, dst_id: int, subnet: int, n_flits: int, is_write: bool, is_response: bool) -> CompanionCommandSignature:
            capsule = pybooksim2.create_icnt_cmd_data_packet(src_id, dst_id, subnet, n_flits, is_write, is_response)
            cmd = CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs={
                "src_id": src_id,
                "dst_id": dst_id,
                "subnet": subnet,
                "n_flits": n_flits,
                "is_write": is_write,
                "is_response": is_response,
            })
            return cmd

        def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
            return pybooksim2.icnt_dispatch_cmd(icnt=self._icnt, cmd=cmd.capsule, dispatch_callback=dispatch_callback, execute_callback=execute_callback)

        def get_stats(self) -> dict[int, dict[str, float | int]]:
            return pybooksim2.get_icnt_router_stats(icnt=self._icnt)
else:
    class BookSim2Config:
        def __init__(self, processor_clock_freq: float, flit_size: int, subnets: int, x: int, y: int, xr: int, yr: int):
            self.processor_clock_freq: float = processor_clock_freq
            self._flit_size: int = flit_size
            self._subnets: int = subnets
            self._x: int = x
            self._y: int = y
            self._xr: int = xr
            self._yr: int = yr
            
        @property
        def max_in_flight_per_subnet(self) -> int:
            return self._x * self._y * self._xr * self._yr * 2 * 16
        
        def peak_bandwidth_per_router(self) -> float:
            return self._flit_size * self._subnets * self.processor_clock_freq
        
        def peak_bisection_bandwidth(self) -> float:
            return self.peak_bandwidth_per_router() * self._y * 2

    class BookSim2(CompanionModule):
        def __init__(self, config: BookSim2Config):
            super().__init__()
            
            self.config = config
            
            self._vcs_per_link = 4 
            
            # Structure: _link_occupancy[subnet_id][link_id] = current_used_vcs
            # link_id is a tuple (node_a, node_b)
            self._link_occupancy: dict[int, dict[tuple[int, int], int]] = {
                s: {} for s in range(config._subnets)
            }
            
            # Commands waiting at the source injection buffer
            # Organized by (src_id, subnet) to model per-source HOL blocking
            self._injection_buffers: dict[tuple[int, int], list[tuple[CompanionCommandSignature, Callable]]] = {}
            
            # Packets currently in the network
            # [remaining_cycles, command, callback, reserved_links]
            self._in_flight_cmds: list[list] = []

        def _get_torus_path_links(self, src_id: int, dst_id: int) -> list[tuple[int, int]]:
            """
            Generates a list of directed links (u, v) using DOR (X-then-Y) 
            and shortest path wrap-around for Torus.
            """
            links = []
            x_sz, y_sz = self.config._x, self.config._y
            
            curr_x, curr_y = src_id % x_sz, src_id // x_sz
            dst_x, dst_y = dst_id % x_sz, dst_id // x_sz

            # 1. X-Dimension Routing
            while curr_x != dst_x:
                # Calculate shortest direction on Torus
                dist_right = (dst_x - curr_x + x_sz) % x_sz
                dist_left = (curr_x - dst_x + x_sz) % x_sz
                
                step = 1 if dist_right <= dist_left else -1
                next_x = (curr_x + step + x_sz) % x_sz
                
                u = curr_y * x_sz + curr_x
                v = curr_y * x_sz + next_x
                links.append((u, v))
                curr_x = next_x

            # 2. Y-Dimension Routing
            while curr_y != dst_y:
                # Calculate shortest direction on Torus
                dist_down = (dst_y - curr_y + y_sz) % y_sz
                dist_up = (curr_y - dst_y + y_sz) % y_sz
                
                step = 1 if dist_down <= dist_up else -1
                next_y = (curr_y + step + y_sz) % y_sz
                
                u = curr_y * x_sz + curr_x
                v = next_y * x_sz + curr_x
                links.append((u, v))
                curr_y = next_y
                
            return links

        def _can_reserve_path(self, subnet_id: int, links: list[tuple[int, int]]) -> bool:
            """Checks if all links in the path have an available Virtual Channel."""
            subnet_links = self._link_occupancy[subnet_id]
            for l in links:
                if subnet_links.get(l, 0) >= self._vcs_per_link:
                    return False
            return True

        def _reserve_path(self, subnet_id: int, links: list[tuple[int, int]]):
            """Increments VC usage for all links in the path."""
            for l in links:
                self._link_occupancy[subnet_id][l] = self._link_occupancy[subnet_id].get(l, 0) + 1

        def _release_path(self, subnet_id: int, links: list[tuple[int, int]]):
            """Decrements VC usage after packet arrival."""
            for l in links:
                self._link_occupancy[subnet_id][l] -= 1

        def update_cycle_time(self, cycle_time: int):
            # 1. Process In-Flight progress and release resources
            completed_indices = []
            for i, status in enumerate(self._in_flight_cmds):
                status[0] -= cycle_time
                if status[0] <= 0:
                    completed_indices.append(i)

            for i in sorted(completed_indices, reverse=True):
                remaining, cmd, callback, reserved_links = self._in_flight_cmds.pop(i)
                self._release_path(cmd.capsule['subnet'], reserved_links)
                callback(cmd.capsule)  # Notify command completion

            # 2. Injection with HOL Blocking (Per source, per subnet)
            for (src_id, subnet_id), buffer in self._injection_buffers.items():
                while buffer:
                    cmd, callback = buffer[0] # Peek at the head of the line
                    path_links = self._get_torus_path_links(cmd.capsule['src_id'], cmd.capsule['dst_id'])
                    
                    # Check for Link Contention
                    if self._can_reserve_path(subnet_id, path_links):
                        buffer.pop(0) # Remove from buffer
                        self._reserve_path(subnet_id, path_links)
                        
                        # Latency = Hops + Serialization
                        latency = len(path_links) + cmd.capsule['n_flits']
                        self._in_flight_cmds.append([latency, cmd, callback, path_links])
                    else:
                        # HOL BLOCKING: Head of line cannot proceed due to link/VC contention.
                        # Subsequent packets in this specific buffer are blocked.
                        break

        def create_command(self, src_id: int, dst_id: int, subnet: int, n_flits: int, is_write: bool, is_response: bool) -> CompanionCommandSignature:
            capsule = {"src_id": src_id, "dst_id": dst_id, "subnet": subnet, "n_flits": n_flits, "is_write": is_write, "is_response": is_response}
            return CompanionCommandSignature(module_id=self.module_id, capsule=capsule, kwargs=capsule)

        def dispatch_command(self, cmd: CompanionCommandSignature, dispatch_callback: Callable, execute_callback: Callable) -> bool:
            if dispatch_callback is not None:
                dispatch_callback(cmd.capsule)
            
            src_id = cmd.capsule['src_id']
            subnet_id = cmd.capsule['subnet']
            
            key = (src_id, subnet_id)
            if key not in self._injection_buffers:
                self._injection_buffers[key] = []
            
            self._injection_buffers[key].append((cmd, execute_callback))
            return True

        def get_stats(self) -> dict[int, dict[str, float | int]]:
            return {}