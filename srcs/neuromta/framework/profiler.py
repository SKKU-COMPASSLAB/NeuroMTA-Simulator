import os
import abc
import math
import shutil
from collections import deque

from neuromta.framework.core import Core, Kernel, Command
from neuromta.framework.logger import logger


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
    

class CommandUtilizationProfiler(Profiler):
    def __init__(self, core: Core, window_size: int = 8, thres: float=0.05, cmd_ids: list[str] = None):
        super().__init__(core)
        
        if window_size < 2:
            raise Exception("[ERROR] window_size must be at least 2.")
        
        if cmd_ids is None:
            cmd_ids = [cmd_id for cmd_id in dir(core) if hasattr(getattr(core, cmd_id), "_is_command_method")]

        if isinstance(cmd_ids, str):
            cmd_ids = [cmd_ids]
        
        self._profiles: dict[str, list[int, int]] = {cmd_id: [0, 0] for cmd_id in cmd_ids}
        
    def profile_step(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        if cmd.cmd_id not in self._profiles.keys():
            return

        history = self._profiles[cmd.cmd_id]
        
        history[0] = commit_time
        history[1] += (commit_time - issue_time)
    
    def dump(self):
        content = ["command_id,last_commit_time,duration"]
        for cmd_id, history in self._profiles.items():
            if len(history):
                content.append(f"{cmd_id},{history[0]},{history[1]}")
        return "\n".join(content)

    def clear(self):
        self._profiles.clear()


class ProfilerHub:
    def __init__(self):
        self._profilers: dict[str, Profiler] = {}
        
    def register_profiler(self, profiler_id: str, profiler: Profiler):
        if profiler_id in self._profilers.keys():
            raise Exception(f"Profiler with name '{profiler_id}' already exists. Please use a unique name.")
        self._profilers[profiler_id] = profiler
        
    def unregister_profiler(self, profiler_id: str):
        if profiler_id not in self._profilers.keys():
            raise Exception(f"Profiler with name '{profiler_id}' does not exist.")
        del self._profilers[profiler_id]

    def save_profiles(self, save_profile_dir: str = DEFAULT_TRACE_DIR):
        if os.path.isdir(save_profile_dir):
            shutil.rmtree(save_profile_dir)  # Remove existing directory
        os.makedirs(save_profile_dir, exist_ok=True)

        for profiler_id, profiler in self._profilers.items():
            profile_path = os.path.join(save_profile_dir, f"{profiler_id}.csv")
            profiler.save_as_file(profile_path)

            logger.info(f"Profile {profiler_id} saved to \"{profile_path}\"")
            
    def clear_profiles(self):
        for profiler in self._profilers.values():
            profiler.clear()

