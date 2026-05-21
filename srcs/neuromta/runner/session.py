import os
import torch
import sys
import importlib.util
import multiprocessing as mp
from typing import Any

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.software.common.network import MCA_NetworkRecipe


__all__ = ["SessionCommand", "SessionMessage", "Session"]


DEFAULT_RECIPE = dict(
    main_space_size_per_channel=parse_mem_cap_str("4GB"),
    data_space_size_per_core=parse_mem_cap_str("1MB"),
    spad_space_size_per_core=parse_mem_cap_str("512KB"),
    broadcast_optimize_queue_depth=8,
    broadcast_optimize_max_ref_cnt=4,
    context_buffer_slot_num=16,
    ld_ex_buffer_slot_num=16,
    ex_st_buffer_slot_num=8,
    concurrent_load_num=1,
    temporal_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.ALL,
    spatial_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_MAIN,
    greedy_temporal_reuse=True,
    
    dtype=torch.float16,
    acc_dtype=torch.float16,
)


class SessionCommand:
    def __init__(self, cmd_type: str, args: list[Any]=None):
        self.cmd_type = cmd_type
        self.args = args if args is not None else []
        
    @classmethod
    def exit(cls):
        return cls("exit")


class SessionMessage:
    def __init__(self, session_id: int, msg_type: str, payload: Any):
        self.session_id = session_id
        self.msg_type = msg_type
        self.payload  = payload
    
    @classmethod
    def error(cls, session_id: int, error_msg: str):
        return cls(session_id, "error", error_msg)
    
    @classmethod
    def done(cls, session_id: int, info: Any=None):
        return cls(session_id, "done", info)


class Session(mp.Process):
    def __init__(self, session_id: int, cmd_q: mp.Queue, msg_q: mp.Queue, device_lib_path: str, model_lib_path: str):
        super().__init__()
        
        self.session_id = session_id
        self.cmd_q = cmd_q
        self.msg_q = msg_q
        
        self.device_lib_path = device_lib_path
        self.model_lib_path = model_lib_path
        
    def change_recipe(self, device_lib, key: str, value: Any) -> SessionMessage:
        device_lib.RECIPE[key] = value
        return SessionMessage.done(self.session_id)
        
    def change_core_group_shape(self, device_lib, shape: tuple[int]) -> SessionMessage:
        device_lib.CORE_GROUP_SHAPE = shape
        return SessionMessage.done(self.session_id)

    def change_core_group_offset(self, device_lib, offset: tuple[int]) -> SessionMessage:
        device_lib.CORE_GROUP_OFFSET = offset
        return SessionMessage.done(self.session_id)

    def compile_graph(self, device_lib, model_lib) -> tuple[MCA_CompiledNetworkGraph | None, SessionMessage]:
        try:
            device: MCA_DeviceBase = device_lib.DEVICE
            core_group_offset: tuple[int] = device_lib.CORE_GROUP_OFFSET
            core_group_shape: tuple[int] = device_lib.CORE_GROUP_SHAPE
            
            module: torch.nn.Module = model_lib.MODULE
            dummy_inputs: list[Any] = model_lib.INPUTS
            
            recipe_kwargs = DEFAULT_RECIPE.copy()
            recipe_kwargs.update(device_lib.RECIPE)
            recipe_kwargs["device"] = device
            recipe_kwargs["core_groups"] = device.get_npu_core_group(core_group_offset, core_group_shape)
            
            graph_recipe = MCA_NetworkRecipe(**recipe_kwargs)
            
            graph = MCA_CompiledNetworkGraph.from_trace(module, graph_recipe, *dummy_inputs)
            
            summary = graph.compile_summary()
            payload = {}
            
            for group_idx, group in enumerate(summary['grouped_entries']):
                payload[group_idx] = []
                for entry_idx, entry in enumerate(group):
                    payload[group_idx].append({
                        "node": entry["node"],
                        "op_method": entry["op_method"],
                    })
            
            return graph, SessionMessage.done(self.session_id, payload)
        except Exception as e:
            return None, SessionMessage.error(self.session_id, str(e))

    def run_graph(self, graph: MCA_CompiledNetworkGraph, model_lib, group_idx: int, entry_idx: int) -> SessionMessage:
        try:
            dummy_inputs: list[Any] = model_lib.INPUTS
            result_dict = graph.run_compiled_graph(
                *dummy_inputs, group_idx=group_idx, entry_idx=entry_idx)
            payload = {
                "group_idx": group_idx,
                "entry_idx": entry_idx,
                "result": result_dict,
            }
            return SessionMessage.done(self.session_id, payload)
        except Exception as e:
            return SessionMessage.error(self.session_id, str(e))

    def run(self):
        # Dynamically load the device library
        device_lib_name = "device_lib"
        device_lib_spec = importlib.util.spec_from_file_location(device_lib_name, self.device_lib_path)
        device_lib = importlib.util.module_from_spec(device_lib_spec)
        sys.modules[device_lib_name] = device_lib
        device_lib_spec.loader.exec_module(device_lib)
        
        if not hasattr(device_lib, "DEVICE"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Device library does not have 'DEVICE' attribute."))
            return
        if not isinstance(device_lib.DEVICE, MCA_DeviceBase):
            self.msg_q.put(SessionMessage.error(self.session_id, "'DEVICE' attribute in device library is not an instance of 'MCA_DeviceBase' class."))
            return
        if not hasattr(device_lib, "RECIPE"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Device library does not have 'RECIPE' attribute."))
            return
        if not isinstance(device_lib.RECIPE, dict):
            self.msg_q.put(SessionMessage.error(self.session_id,    "'RECIPE' attribute in device library is not a dictionary."))
            return
        if not hasattr(device_lib, "CORE_GROUP_SHAPE"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Device library does not have 'CORE_GROUP_SHAPE' attribute."))
            return
        if not isinstance(device_lib.CORE_GROUP_SHAPE, tuple):
            self.msg_q.put(SessionMessage.error(self.session_id, "'CORE_GROUP_SHAPE' attribute in device library is not a tuple."))
            return
        if not hasattr(device_lib, "CORE_GROUP_OFFSET"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Device library does not have 'CORE_GROUP_OFFSET' attribute."))
            return
        if not isinstance(device_lib.CORE_GROUP_OFFSET, tuple):
            self.msg_q.put(SessionMessage.error(self.session_id, "'CORE_GROUP_OFFSET' attribute in device library is not a tuple."))
            return
        
        # Dynamically load the model library
        model_lib_name = "model_lib"
        model_lib_spec = importlib.util.spec_from_file_location(model_lib_name, self.model_lib_path)
        model_lib = importlib.util.module_from_spec(model_lib_spec)
        sys.modules[model_lib_name] = model_lib
        model_lib_spec.loader.exec_module(model_lib)
        
        if not hasattr(model_lib, "MODULE"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Model library does not have 'MODULE' attribute."))
            return
        if not isinstance(model_lib.MODULE, torch.nn.Module):
            self.msg_q.put(SessionMessage.error(self.session_id, "'MODULE' attribute in model library is not an instance of 'torch.nn.Module' class."))
            return
        if not hasattr(model_lib, "INPUTS"):
            self.msg_q.put(SessionMessage.error(self.session_id, "Model library does not have 'INPUTS' attribute."))
            return
        if not isinstance(model_lib.INPUTS, (list, tuple)):
            self.msg_q.put(SessionMessage.error(self.session_id, "'INPUTS' attribute in model library is not a list or tuple."))
            return
        
        self.msg_q.put(SessionMessage.done(self.session_id, f"Session {self.session_id} initialized successfully."))
        
        # Main loop to process commands
        graph: MCA_CompiledNetworkGraph | None = None
        
        while True:
            cmd: SessionCommand = self.cmd_q.get()
            
            if not isinstance(cmd, SessionCommand):
                self.msg_q.put(SessionMessage.error(self.session_id, "Received command is not an instance of 'SessionCommand' class."))
                continue
            
            if cmd.cmd_type == "exit":
                self.msg_q.put(SessionMessage.done(self.session_id, "Session exiting."))
                break
            elif cmd.cmd_type == "change_recipe":
                key, value = cmd.args
                msg = self.change_recipe(device_lib, key, value)
                self.msg_q.put(msg)
            elif cmd.cmd_type == "change_core_group_shape":
                shape, = cmd.args
                msg = self.change_core_group_shape(device_lib, shape)
                self.msg_q.put(msg)
            elif cmd.cmd_type == "change_core_group_offset":
                offset, = cmd.args
                msg = self.change_core_group_offset(device_lib, offset)
                self.msg_q.put(msg)
            elif cmd.cmd_type == "compile_graph":
                graph, msg = self.compile_graph(device_lib, model_lib)
                self.msg_q.put(msg)
            elif cmd.cmd_type == "run_graph":
                if graph is None:
                    self.msg_q.put(SessionMessage.error(self.session_id, "No compiled graph available. Please compile the graph before running."))
                    continue
                group_idx, entry_idx = cmd.args
                msg = self.run_graph(graph, model_lib, group_idx, entry_idx)
                self.msg_q.put(msg)
            else:
                self.msg_q.put(SessionMessage.error(self.session_id, f"Unknown command type: {cmd.cmd_type}"))
                continue