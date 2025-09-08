import os
import sys
import shutil
from typing import Sequence, Callable  #, Any

from neuromta.framework.core import Core, Kernel, Command, RPCMessage
from neuromta.framework.companion import CompanionCore
from neuromta.framework.tracer import Tracer, TraceEntry


__all__ = [
    "Device",
]


NEUROMTA_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_TRACE_DIR = os.path.join(NEUROMTA_ROOT_DIR, ".logs", "traces")


class Device:
    def __init__(self):
        self._cores: dict[str, Core] = None
        
        self._verbose:      bool = False

        self._rpc_req_send_inbox: dict[str, list[RPCMessage]] = {}
        self._rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = {}
        
        self._companion_core = CompanionCore()
        
    def get_core_from_id(self, core_id: int) -> Core:
        return self._cores[core_id]
        
    @property
    def companion_core(self) -> CompanionCore:
        return self._companion_core 
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        for core in self._cores.values():
            core.change_sim_model_options(use_cycle_model, use_functional_model)
        
    def _register_core(self, name: str, core: Core | Sequence[Core]):
        if isinstance(core, Sequence):
            for idx, item in enumerate(core):
                if isinstance(item, (Core, Sequence)):
                    self._register_core(f"{name}[{idx}]", item)
        elif isinstance(core, Core):
            if core.core_id in self._cores.keys():
                raise Exception(f"[ERROR] Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core
            core.register_command_debug_hook(self.default_command_debug_hook)

    def initialize(self, create_trace: bool = None):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._rpc_req_send_inbox = {core.core_id: [] for core in self._cores.values()}
        self._rpc_rsp_send_inbox = {core.core_id: [] for core in self._cores.values()}

        for core in self._cores.values():
            core.initialize_kernel_dispatch_queue()
            core.initialize_mp_queue_inbox(rpc_req_send_inbox=self._rpc_req_send_inbox, rpc_rsp_send_inbox=self._rpc_rsp_send_inbox)
        
        return self
    
    def default_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        if self._verbose:
            sys.stdout.write(f"[DEBUG] {issue_time:<4d} - {commit_time:<4d} | {core.core_id.__str__():<10s} | {kernel.kernel_id_full:<130s} | command: {cmd.cmd_id}\n")
            
    def run_kernels(
        self, 
        cycle_resolution:   int  = 1,                   # the number of cycles to update when all the cores are waiting and returning (0 | None) as the minimum remaining cycles
        verbose:            bool = False,               # whether to print verbose debug information
        max_steps:          int  = -1,                  # the maximum number of steps to run
        save_trace:         bool = False,               # whether to save the trace
        save_trace_dir:     str  = DEFAULT_TRACE_DIR    # the directory to save the trace
    ):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        if save_trace:
            if os.path.isdir(save_trace_dir):
                shutil.rmtree(save_trace_dir)  # Remove existing directory
            os.makedirs(save_trace_dir, exist_ok=True)
            
            tracers: dict[str, Tracer] = {}
            
            for core_id, core in self._cores.items():
                tracer = Tracer()
                tracer.register_core(core)
                tracers[core_id] = tracer

        self._verbose = verbose

        core_ids: list[str] = list(self._cores.keys())
        
        step_cnt = 0

        while not all(core.is_idle for core in self._cores.values()):  
            step_cnt += 1
            if step_cnt >= max_steps > 0:
                print(f"[INFO] Reached maximum steps: {max_steps}. Stopping simulation.")
                break
            
            remaining_cycles = None
            
            for core_id in core_ids:
                c = self._cores[core_id].get_remaining_cycles()
                
                if remaining_cycles is None:
                    remaining_cycles = c
                elif c is not None:
                    remaining_cycles = min(remaining_cycles, c)
                    
            for core_id in core_ids:
                self._cores[core_id].rpc_update_routine()
                    
            if remaining_cycles == 0 or remaining_cycles is None:
                remaining_cycles = self.companion_core.update_cycle_time_until_cmd_executed()
                    
                if remaining_cycles == 0 or remaining_cycles is None:
                    remaining_cycles = cycle_resolution
            else:
                self.companion_core.update_cycle_time_companion_modules(cycle_time=remaining_cycles)
            
            for core_id in core_ids:
                self._cores[core_id].update_cycle_time(cycle_time=remaining_cycles)
        
        if save_trace:
            for core_id, tracer in tracers.items():
                if not tracer.is_empty:
                    core_id_str_expr = TraceEntry.convert_valid_core_id(core_id)
                    trace_path = os.path.join(save_trace_dir, f"{core_id_str_expr}.csv")
                    tracer.save_traces_as_file(trace_path)

                    print(f"[INFO] Trace for core {core_id} saved to \"{trace_path}\"")

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
            
    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @property
    def timestamp(self) -> int:
        t = [core.timestamp for core in self._cores.values()]
        return max(t) if t else 0

    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())