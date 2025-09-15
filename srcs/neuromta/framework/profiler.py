import os
import abc
import math
import shutil
from collections import deque

from neuromta.framework.core import Core, Kernel, Command
from neuromta.framework.device import Device


__all__ = [
    "Profiler",
    "CommandUtilizationProfiler",
    "ProfilerHub",
]


NEUROMTA_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_TRACE_DIR = os.path.join(NEUROMTA_ROOT_DIR, ".logs", "profiles")


class Profiler(metaclass=abc.ABCMeta):
    def __init__(self, core: Core):
        self._core = core
        self._hook_id = core.register_command_debug_hook(self.profile_step)
        
    def __del__(self):
        if self._core is not None and self._hook_id is not None:
            self._core.unregister_command_debug_hook(self._hook_id)
    
    def save_as_file(self, filename: str):
        with open(filename, "wt") as file:
            file.write(self.dump())
            
    @abc.abstractmethod
    def profile_step(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        pass
    
    @abc.abstractmethod
    def dump(self) -> str:
        pass
    
    @abc.abstractmethod
    def clear(self):
        pass
    
    @abc.abstractmethod
    def check_convergence(self) -> bool:
        pass
    

class CommandUtilizationProfiler(Profiler):
    def __init__(self, core: Core, window_size: int = 8, thres: float=0.05, cmd_ids: list[str] = None):
        super().__init__(core)
        
        if window_size < 2:
            raise Exception("[ERROR] window_size must be at least 2.")
        
        self._window_size = window_size
        self._thres = thres
        self._history: dict[str, deque[tuple[int, int]]] = {}
        self._convergence_flags: dict[str, bool] = {}
        
        if cmd_ids is None:
            cmd_ids = [cmd_id for cmd_id in dir(core) if hasattr(getattr(core, cmd_id), "_is_command_method")]

        if isinstance(cmd_ids, str):
            cmd_ids = [cmd_ids]
            
        for cmd_id in cmd_ids:
            self._history[cmd_id] = deque(maxlen=window_size)
            self._convergence_flags[cmd_id] = False
        
    def profile_step(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        if cmd.cmd_id not in self._history.keys():
            return

        history = self._history[cmd.cmd_id]

        if len(history):
            _, last_duration = history[-1]
            history.append((commit_time, last_duration + (commit_time - issue_time)))
        else:
            history.append((commit_time, commit_time - issue_time))

        if len(history) == self._window_size:
            util_max = -1
            util_min = math.inf
            
            for i in range(1, len(history)):
                util = history[i][1] / history[i][0]
                
                util_max = max(util_max, util)
                util_min = min(util_min, util)
            
            self._convergence_flags[cmd.cmd_id] = ((util_max - util_min) < (util_max * self._thres))
    
    def dump(self):
        content = ["command_id,last_commit_time,duration"]
        for cmd_id, history in self._history.items():
            if len(history):
                content.append(f"{cmd_id},{history[-1][0]},{history[-1][1]}")
        return "\n".join(content)

    def clear(self):
        self._history.clear()
        
    def check_convergence(self):
        if self._core.is_idle:
            return True
        return all(self._convergence_flags.values())


class ProfilerHub:
    def __init__(self):
        self._profilers: dict[str, Profiler] = {}
        
    def register_profiler(self, profiler_id: str, profiler: Profiler):
        if profiler_id in self._profilers.keys():
            raise Exception(f"[ERROR] Profiler with name '{profiler_id}' already exists. Please use a unique name.")
        self._profilers[profiler_id] = profiler
        
    def unregister_profiler(self, profiler_id: str):
        if profiler_id not in self._profilers.keys():
            raise Exception(f"[ERROR] Profiler with name '{profiler_id}' does not exist.")
        del self._profilers[profiler_id]
        
    def run_profile(self, device: Device, verbose: bool=False, cycle_resolution: int=1):
        if not device.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        device.verbose = verbose
        
        while True:
            if all(core.is_idle for core in device.cores.values()):
                break
            
            if all(profiler.check_convergence() for profiler in self._profilers.values()):
                if verbose:
                    print("[INFO] All profilers have detected convergence. Stopping simulation.")
                break
            
            device.run_single_step(cycle_resolution=cycle_resolution)

    def save_profiles(self, save_profile_dir: str = DEFAULT_TRACE_DIR):
        if os.path.isdir(save_profile_dir):
            shutil.rmtree(save_profile_dir)  # Remove existing directory
        os.makedirs(save_profile_dir, exist_ok=True)

        for profiler_id, profiler in self._profilers.items():
            profile_path = os.path.join(save_profile_dir, f"{profiler_id}.csv")
            profiler.save_as_file(profile_path)

            print(f"[INFO] Profile {profiler_id} saved to \"{profile_path}\"")

