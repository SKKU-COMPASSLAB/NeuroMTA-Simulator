import os
import json
import torch

from neuromta.framework.core import Core, Kernel, Command, RPCMessage
from neuromta.framework.logger import logger
from neuromta.framework.device import Device
from typing import Callable


__all__ = [
    "CommandTracer",
    "ExecutionTimeProfiler",
]


class CommandTracer:
    ALL_COMMANDS = lambda x: True
    
    def __init__(self, device: Device, core_ids: list[int]=None, command_filter: Callable[[Command], bool]=ALL_COMMANDS):
        if core_ids is None:
            core_ids = list(device.initialized_cores.keys())
            
        for core_id in core_ids:
            core = device.initialized_cores[core_id]
            core.register_command_debug_hook(self.command_debug_hook)
            
        self.command_filter = command_filter
        
        self.command_trace: dict[int, dict[str, list[Command]]] = {core_id: {} for core_id in core_ids}

    def command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        if not self.command_filter(cmd):
            return
        if core.core_id not in self.command_trace.keys():
            return
        if kernel.slot_id not in self.command_trace[core.core_id]:
            self.command_trace[core.core_id][kernel.slot_id] = []
            
        self.command_trace[core.core_id][kernel.slot_id].append(cmd)
        
    @staticmethod
    def _convert_arg_to_str(arg):
        if isinstance(arg, (int, float, str)):
            return str(arg)
        elif isinstance(arg, (list, tuple)):
            return "[" + ", ".join(CommandTracer._convert_arg_to_str(x) for x in arg) + "]"
        elif isinstance(arg, dict):
            return "{" + ", ".join(f"{CommandTracer._convert_arg_to_str(k)}: {CommandTracer._convert_arg_to_str(v)}" for k, v in arg.items()) + "}"
        elif isinstance(arg, torch.Tensor):
            if arg.ndim == 0:
                return str(arg.item())
            else:
                return f"Tensor(shape={list(arg.shape)}, dtype={arg.dtype})"
        else:
            return repr(arg)
        
    def save_as_file(self, dirname: str):
        os.makedirs(dirname, exist_ok=True)
        coredir_fmt = os.path.join(dirname, "core_{core_id}")
        filepath_fmt = os.path.join(dirname, "core_{core_id}", "core_{core_id}_slot_{slot_id}.json")
        
        for core_id, slot_traces in self.command_trace.items():
            for slot_id, traces in slot_traces.items():
                if len(traces) == 0:
                    continue
                
                os.makedirs(coredir_fmt.format(core_id=core_id), exist_ok=True)
                with open(filepath_fmt.format(core_id=core_id, slot_id=slot_id), "w") as f:
                    content = []
                    for command in traces:
                        content.append({
                            "issue_time": command.issue_time,
                            "commit_time": command.commit_time,
                            "command": {
                                "cmd_id": command.cmd_id,
                                "args": [self._convert_arg_to_str(arg) for arg in command.args],
                                "kwargs": {k: self._convert_arg_to_str(v) for k, v in command.kwargs.items()},
                            }
                        })
                        
                    json.dump(content, f, indent=4)
                    logger.info(f"Command trace for core {core_id} slot {slot_id} saved to '{filepath_fmt.format(core_id=core_id, slot_id=slot_id)}'.")

class ExecutionTimeProfiler:
    ALL_COMMANDS = lambda x: True
    NO_VAR_WAIT = lambda cmd: cmd.cmd_id not in [
        "var_atomic_barrier", "var_conditional_wait", "var_atomic_wait", "var_atomic_compare_and_swap", 
        "fifo_wait_until_valid", "fifo_wait_until_vacant",
    ]

    class ProcessState:
        def __init__(self):
            self.active_timelines: list[tuple[int, int]] = []
            self.wait_for_rpc_timestamp: int | None = None
            self.final_commit: int = 0
            
        def _add_active_timeline(self, start: int, end: int):
            if len(self.active_timelines) == 0:
                self.active_timelines.append((start, end))
            else:
                last_start, last_end = self.active_timelines[-1]
                if start <= last_end:
                    self.active_timelines[-1] = (last_start, max(last_end, end))
                else:
                    self.active_timelines.append((start, end))
            
        def add_cmd(self, issue_time: int, commit_time: int, is_rpc_wait: bool):
            if self.wait_for_rpc_timestamp is not None:
                if self.wait_for_rpc_timestamp < issue_time:
                    self._add_active_timeline(self.wait_for_rpc_timestamp, issue_time)
            
            self._add_active_timeline(issue_time, commit_time)

            self.final_commit = max(self.final_commit, commit_time)
            
            if is_rpc_wait:
                self.wait_for_rpc_timestamp = commit_time
            
        def add_kernel(self, commit_time: int):
            self.final_commit = max(self.final_commit, commit_time)
            
        @property
        def active_time(self) -> int:
            return sum(end - start for start, end in self.active_timelines)
        
        @property
        def active_utilization(self) -> float:
            if self.final_commit == 0:
                return 0.0
            return self.active_time / self.final_commit
            
    class CommandState:
        def __init__(self):
            self.command_counter: dict[str, list[int, int]] = {}
            
        def add_cmd(self, cmd: Command, issue_time: int, commit_time: int):
            cmd_id = cmd.cmd_id
            if cmd_id == "async_rpc_send_req_msg":
                msg: RPCMessage = cmd.args[0]
                cmd_id = f"{cmd_id}::{msg.cmd_id}"
            if cmd_id not in self.command_counter.keys():
                self.command_counter[cmd_id] = [0, 0]
            self.command_counter[cmd_id][0] += 1
            self.command_counter[cmd_id][1] += (commit_time - issue_time)
            
    def __init__(self, device: Device, core_ids: list[int], slot_ids: list[str], target_cmd_condition: Callable[[Command], bool]=NO_VAR_WAIT):
        self.proc_states: dict[int, dict[str, ExecutionTimeProfiler.ProcessState]] = {
            core_id: {
                slot_id: ExecutionTimeProfiler.ProcessState()
                for slot_id in slot_ids
            } 
            for core_id in core_ids
        }
        
        self.cmd_states: dict[int, dict[str, ExecutionTimeProfiler.CommandState]] = {
            core_id: {
                slot_id: ExecutionTimeProfiler.CommandState()
                for slot_id in slot_ids
            }
            for core_id in core_ids
        }
        
        for slot_id in slot_ids:
            device.register_kernel_debug_hook(self.kernel_debug_hook, slot_id, core_ids)
        
        for core_id in core_ids:
            core = device.initialized_cores[core_id]
            core.register_command_debug_hook(self.command_debug_hook)
            
        self.target_cmd_condition = target_cmd_condition

    def command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        if core.core_id not in self.proc_states.keys():
            return
        if kernel.slot_id not in self.proc_states[core.core_id].keys():
            return

        if not self.target_cmd_condition(cmd):
            return

        is_rpc_wait = False # (cmd.cmd_id == "async_rpc_wait_rsp_msg" and kernel.is_blocked)
        self.proc_states[core.core_id][kernel.slot_id].add_cmd(issue_time, commit_time, is_rpc_wait)
        self.cmd_states[core.core_id][kernel.slot_id].add_cmd(cmd, issue_time, commit_time)
        
    def kernel_debug_hook(self, core: Core, kernel: Kernel):
        if core.core_id not in self.proc_states.keys():
            return
        if kernel.slot_id not in self.proc_states[core.core_id].keys():
            return
        
        self.proc_states[core.core_id][kernel.slot_id].add_kernel(kernel.commit_time)
            
    def summary(self):
        report = {}
        for core_id, thread_states in self.proc_states.items():
            report[core_id] = {}
            for slot_id, thread_state in thread_states.items():
                report[core_id][slot_id] = {
                    "active_time_cycles": thread_state.active_time,
                    "final_commit_cycles": thread_state.final_commit,
                    "active_utilization": thread_state.active_utilization,
                    "command_count": self.cmd_states[core_id][slot_id].command_counter,
                }
        return report