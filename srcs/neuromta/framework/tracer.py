import os
import shutil
from typing import Any

from neuromta.framework.core import *
from neuromta.framework.logger import logger


__all__ = [
    "TraceEntry",
    "Tracer",
    "TracerHub",
]


NEUROMTA_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_TRACE_DIR = os.path.join(NEUROMTA_ROOT_DIR, ".logs", "traces")


class TraceEntry:
    def __init__(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        self.core_id = TraceEntry.convert_valid_core_id(core.core_id)
        self.kernel_id = kernel.callstack
        self.cmd_id = cmd.cmd_id
        self.issue_time = issue_time
        self.commit_time = commit_time
        
    @staticmethod
    def convert_valid_core_id(core_id: Any) -> str:
        core_id: str = core_id.__str__()
        core_id = core_id.replace("/", "_").replace("\\", "_").replace(" ", "").replace(",", "_").replace("(", "").replace(")", ""
                        ).replace("::", "_").replace(":", "_").replace("*", "ptr").replace("?", "q").replace("<", "_").replace(">", "_")
        return core_id
        
    @staticmethod
    def get_csv_header() -> str:
        return "core_id,kernel_id,command_id,issue_time,commit_time,simulated_cycles"

    def to_csv_entry(self) -> str:
        simulated_cycles = self.commit_time - self.issue_time
        return f"{self.core_id},{self.kernel_id},{self.cmd_id},{self.issue_time},{self.commit_time},{simulated_cycles}"

    def __str__(self):
        return self.to_csv_entry()


class Tracer:
    def __init__(self):
        self._trace_entries: list[TraceEntry] = []
        self._debug_hook_handles: dict[str, str] = {}
        
    def __getstate__(self):
        return {
            "_trace_entries": self._trace_entries,
        }

    def __setstate__(self, state: dict):
        self._trace_entries = state["_trace_entries"]

    def core_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        entry = TraceEntry(core, kernel, cmd, issue_time, commit_time)
        self.add_trace(entry)
        
    def register_core(self, core: Core):
        hook = core.register_command_debug_hook(self.core_debug_hook)
        self._debug_hook_handles[core.core_id] = hook
        
    def unregister_core(self, core: Core):
        hook = self._debug_hook_handles.pop(core.core_id, None)
        if hook is not None:
            core.unregister_command_debug_hook(hook)
    
    def add_trace(self, trace: TraceEntry):
        self._trace_entries.append(trace)
            
    def save_traces_as_file(self, filename: str):
        with open(filename, "wt") as file:
            file.write(f"{TraceEntry.get_csv_header()}\n")
            for entry in self._trace_entries:
                file.write(entry.to_csv_entry() + "\n")
                
    def clear_traces(self):
        self._trace_entries.clear()
        
    @property
    def is_empty(self) -> bool:
        return len(self._trace_entries) == 0
    

class TracerHub:
    def __init__(self):
        self._tracers: dict[str, Tracer] = {}
        
    @classmethod
    def create_tracer_key(cls, device_id: str, core: Core) -> str:
        return f"{device_id}_{type(core).__name__}_CORE{core.core_id}"
    
    def register_tracer(self, tracer_id: str, tracer: Tracer):
        if tracer_id in self._tracers.keys():
            raise Exception(f"[ERROR] Tracer with ID '{tracer_id}' already exists. Please use a unique ID.")
        
        self._tracers[tracer_id] = tracer
        
    def unregister_tracer(self, tracer_id: str):
        if tracer_id not in self._tracers.keys():
            raise Exception(f"[ERROR] Tracer with ID '{tracer_id}' does not exist.")
        
        del self._tracers[tracer_id]
        
    def save_traces(self, save_trace_dir: str = DEFAULT_TRACE_DIR):
        if os.path.isdir(save_trace_dir):
            shutil.rmtree(save_trace_dir)  # Remove existing directory
        os.makedirs(save_trace_dir, exist_ok=True)
        
        for tracer_id, tracer in self._tracers.items():
            if not tracer.is_empty:
                trace_path = os.path.join(save_trace_dir, f"{tracer_id}.csv")
                tracer.save_traces_as_file(trace_path)

                logger.info(f"Trace for core {tracer_id} saved to \"{trace_path}\"")
