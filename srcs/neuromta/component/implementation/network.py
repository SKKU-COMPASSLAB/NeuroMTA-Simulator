import abc
import functools
import math
import warnings
import enum
from neuromta.component.implementation.operator import MCA_Operator
import torch

from collections import deque, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Callable

from neuromta.framework import *
from neuromta.component.implementation.hardware import MCA_DeviceBase, MCA_CoreGroup, MCA_L1MemorySpace, MCA_MainMemorySpace, MCA_MemorySpace
from neuromta.component.implementation.tensor_buffer import MCA_TensorBuffer
from neuromta.component.implementation.mapping import MCA_OperatorMapper
# from multiprocessing import Pool
from torch.multiprocessing import Pool



__all__ = [
    "Placeholder",
    
    "NetworkGraphEntry",
    "NetworkGraphCompiledEntry",
    "NetworkGraphContext",
    "NetworkGraphCompilationRecipe",
    "NetworkGraphCompiler",
]


class Placeholder:
    def __init__(self, name: str):
        self.name = name


def _check_jit_type_compatibility(s_type, m_arg: Any) -> tuple[bool, bool, Any]:  # flag_compatible, flag_optional, converted arg
    try:
        m_type = torch._C._jit_try_infer_type(m_arg).type()

        if s_type.isSubtypeOf(m_type):
            return (True, "Optional" in s_type.kind(), m_arg)

        if s_type.kind() == "BoolType" and m_type.kind() in ("BoolType", "IntType", "NumberType"):
            return (True, False, bool(m_arg))
        if s_type.kind() == "NumberType" and m_type.kind() in ("IntType", "FloatType"):
            return (True, False, m_arg)
        elif s_type.kind() == "DeviceObjType" and m_type.kind() in ("StringType"):
            return (True, False, torch.device(m_arg))
        elif s_type.kind() == "TensorType" and m_type.kind() in ("ListType"):
            return (True, False, torch.tensor(m_arg))
        elif "Optional" in s_type.kind():
            s_opt_type = s_type.getElementType()
            _flag_compatible, _, _converted_arg = _check_jit_type_compatibility(s_opt_type, m_arg)
            return (_flag_compatible, True, _converted_arg)
    except:
        return (False, False, m_arg)
    
    return (False, False, m_arg)

def _check_argument_compatibility(s_arg, m_arg: Any) -> tuple[bool, bool, Any]:
    s_type = s_arg.type

    return _check_jit_type_compatibility(s_type=s_type, m_arg=m_arg)

def _get_attr_from_node(node: torch.Node, attr_name: str) -> any:
    for attr_types in ['f', 'fs', 'c', 's', 'ss', 'i', 'g', 'gs', 'ival', 't', 'ts', 'ty', 'tys']:
        try:
            return getattr(node, attr_types)(attr_name)
        except:
            pass
    
    return None

def _find_nonprim_method(method_domain: str, method_name: str, args: list[Any]) -> tuple[Callable, list[Any], dict[str, Any]]:
    method = None
    
    for aten_ref in [getattr(torch.ops, method_domain), torch, torch.nn.functional]:
        try:
            method = getattr(aten_ref, method_name)
            break
        except:
            pass
        
    # check schema
    pp_args = []
    pp_kwargs = {}
    
    for overload_name in method._overload_names:
        schema: torch._C.FunctionSchema = torch._C._get_schema(method._qualified_op_name, overload_name)
        
        if len(schema.arguments) < len(args):
            continue
        
        flag_schema_compatible = True
        tmp_pp_args = []
        tmp_pp_kwargs = {}
        
        for s_arg, m_arg in zip(schema.arguments, args):
            flag_compatible, flag_optional, converted_arg = _check_argument_compatibility(s_arg=s_arg, m_arg=m_arg)

            if not flag_compatible and not flag_optional:
                flag_schema_compatible = False
            else:
                tmp_pp_kwargs[s_arg.name] = converted_arg
        
        if flag_schema_compatible:
            pp_args = tmp_pp_args
            pp_kwargs = tmp_pp_kwargs
            break
    
    return method, pp_args, pp_kwargs

def _kahn_topological_sort(graph: Dict[int, Iterable[int]]) -> List[int]:
    indeg = defaultdict(int)  # node -> in-degree
    nodes: Set[int] = set()

    # collect nodes and compute indegrees
    for u, vs in graph.items():
        nodes.add(u)
        for v in vs:
            nodes.add(v)
            indeg[v] += 1
        if u not in indeg:
            indeg.setdefault(u, indeg[u])

    q = deque([n for n in nodes if indeg.get(n, 0) == 0])
    order: List[int] = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        raise ValueError("Graph has at least one cycle; topological ordering not possible")

    return order


class NetworkGraphEntry:
    class Type(enum.Enum):
        PRIM            = enum.auto()  # prim operators (operators starting with "prim::")
        GRAPH           = enum.auto()  # graph node (submodule call, but not compiled as runtime kernel e.g., torch.nn.Sequential)
        NONPRIM         = enum.auto()  # nonprim operators (operators starting with "aten::", "quantized::", etc.)
        PIPELINED       = enum.auto()  # pipelined compiled graph
        
    def __init__(self, node_type: Type, node: torch.Node, **kwargs):
        self.node_type = node_type
        self.node = node
        
        self.subgraph: NetworkGraphCompiler = kwargs.get("subgraph", None)
        
    @property
    def is_prim(self) -> bool:
        return self.node_type in (NetworkGraphEntry.Type.PRIM, NetworkGraphEntry.Type.GRAPH)
    
    @property
    def is_nonprim(self) -> bool:
        return self.node_type == NetworkGraphEntry.Type.NONPRIM
    
    @property
    def is_pipelined(self) -> bool:
        return self.node_type == NetworkGraphEntry.Type.PIPELINED
    
    @property
    def is_compilation_target(self) -> bool:
        return False    # dummy implementation (for NetworkGraphCompiledEntry)
    
    @property
    def is_compilation_ready(self) -> bool:
        return False    # dummy implementation (for NetworkGraphCompiledEntry)
        
    @property
    def is_compiled(self):
        return False    # dummy implementation (for NetworkGraphCompiledEntry)
    
    @property
    def is_subgraph_available(self) -> bool:
        return self.subgraph is not None
        
    def __str__(self):
        r = f"{self.node_type.name}({self.node.kind()}"
        
        ivars = list('%'+i.debugName() for i in self.node.inputs())
        ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r += f", inputs={ivars}, outputs={ovars}"
        
        if self.is_subgraph_available:
            r += f", graph={type(self.subgraph.module).__name__}"
        
        return r + ")"
    
class NetworkGraphCompiledEntry(NetworkGraphEntry):
    class TensorProcessingType(enum.Enum):
        PERMUTE = enum.auto()
        
    class TensorProcessing:
        def __init__(self, proc_type: 'NetworkGraphCompiledEntry.TensorProcessingType', proc_params: Dict[str, Any]):
            self.proc_type = proc_type
            self.proc_params = proc_params
        
        @classmethod
        def permute(cls, *order):
            return cls(
                proc_type=NetworkGraphCompiledEntry.TensorProcessingType.PERMUTE,
                proc_params={'order': order}
            )
            
        def apply(self, tensor: torch.Tensor) -> torch.Tensor:
            if self.proc_type == NetworkGraphCompiledEntry.TensorProcessingType.PERMUTE:
                order = self.proc_params['order']
                return tensor.permute(*order)
            else:
                raise NotImplementedError(f"unsupported tensor processing type: {self.proc_type}")
    
    class BufferSource:
        GLOBAL_CONTEXT = "__global_context"
        
        def __init__(self, key: str | torch.Value, module: Any=GLOBAL_CONTEXT):
            self._key = key
            self.module = module
            
            if module == NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT:
                if not isinstance(self._key, torch.Value):
                    raise RuntimeError("For GLOBAL_CONTEXT buffer source, key must be a torch.Value instance.")
            else:
                if not isinstance(self._key, str):
                    raise RuntimeError("For module buffer source, key must be a string representing the attribute name.")
                
        @property
        def key(self) -> str:
            if self.module == NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT:
                return self._key.debugName()
            else:
                return self._key
            
            
    class BufferUsage(enum.Enum):
        INPUT   = enum.auto()
        OUTPUT  = enum.auto()
        INPLACE = enum.auto()
        PARAMS  = enum.auto()
        
    class BufferMemType(enum.Enum):
        MAIN    = enum.auto()
        L1      = enum.auto()
    
    class BufferSignature:
        def __init__(
            self, 
            src: 'NetworkGraphCompiledEntry.BufferSource'=None,
            dst: 'NetworkGraphCompiledEntry.BufferSource'=None, 
            shape: Sequence[int]=None, dtype: torch.dtype=None, shard_shape: Sequence[int]=None, blocked_mapping: bool=False, 
            orig_dtype: torch.dtype=None, 
            buffer_usage: 'NetworkGraphCompiledEntry.BufferUsage'=None,
            buffer_mem_type: 'NetworkGraphCompiledEntry.BufferMemType'=None,
            preprocessings: List['NetworkGraphCompiledEntry.TensorProcessing']=None,
            postprocessings: List['NetworkGraphCompiledEntry.TensorProcessing']=None,
        ):
            self.src = src
            self.dst = dst
            
            self.shape = shape
            self.dtype = dtype
            self.shard_shape = shard_shape
            self.blocked_mapping = blocked_mapping
            self.orig_dtype = orig_dtype if orig_dtype is not None else dtype
            self.buffer_usage = buffer_usage if buffer_usage is not None else NetworkGraphCompiledEntry.BufferUsage.PARAMS
            self.buffer_mem_type = buffer_mem_type if buffer_mem_type is not None else NetworkGraphCompiledEntry.BufferMemType.MAIN
            self.preprocessings = preprocessings if preprocessings is not None else []
            self.postprocessings = postprocessings if postprocessings is not None else []
            self.ignore: bool = False
            
            if self.shape is None or self.dtype is None or self.shard_shape is None:
                raise RuntimeError("Buffer signature must have valid shape, dtype, and shard_shape.")
            
            if self.is_input and self.src is None:
                raise RuntimeError("Input buffer must have a valid source.")
            if self.is_output and self.dst is None:
                raise RuntimeError("Output buffer must have a valid destination.")
            
        def get_shard_size(self) -> int:
            return self.shard_shape[0] * self.shard_shape[1] * self.dtype.itemsize
        
        def get_shard_num(self) -> int:
            n_height_shards = (self.shape[-2] + self.shard_shape[0] - 1) // self.shard_shape[0]
            n_width_shards  = (self.shape[-1] + self.shard_shape[1] - 1) // self.shard_shape[1]
            return n_height_shards * n_width_shards
        
        @property
        def is_input(self) -> bool:
            return self.buffer_usage in (NetworkGraphCompiledEntry.BufferUsage.INPUT, NetworkGraphCompiledEntry.BufferUsage.INPLACE, NetworkGraphCompiledEntry.BufferUsage.PARAMS)
        
        @property
        def is_output(self) -> bool:
            return self.buffer_usage in (NetworkGraphCompiledEntry.BufferUsage.OUTPUT, NetworkGraphCompiledEntry.BufferUsage.INPLACE)
        
        @property
        def is_params(self) -> bool:
            return self.buffer_usage == NetworkGraphCompiledEntry.BufferUsage.PARAMS
        
        @property
        def is_l1_buffer(self) -> bool:
            return self.buffer_mem_type == NetworkGraphCompiledEntry.BufferMemType.L1
        
        @property
        def is_main_buffer(self) -> bool:
            return self.buffer_mem_type == NetworkGraphCompiledEntry.BufferMemType.MAIN
    
    def __init__(
        self, 
        node: torch.Node, 
        submodule: torch.nn.Module,
        buffer_signatures: Dict[str, 'NetworkGraphCompiledEntry.BufferSignature'],
        runtime_kwargs: Dict[str, Any],
        target_op_method: Callable[..., MCA_Operator],
        total_ops: int,
    ):
        super().__init__(NetworkGraphEntry.Type.GRAPH, node)
        
        self.submodule: torch.nn.Module = submodule
        
        # Common runtime parameters required
        self.device:     MCA_DeviceBase = None
        self.exe_core_group: MCA_CoreGroup = None
        self.mem_core_group: MCA_CoreGroup = None
        
        self.main_data_mem_space: MCA_MainMemorySpace = None
        self.l1_data_mem_space_size_per_core: int = None
        self.spad_ld_pp_space_size_per_core:  int = None
        self.spad_st_pp_space_size_per_core:  int = None
        
        # Buffer signatures
        self.buffer_signatures  = buffer_signatures  # name -> BufferSignature
        self.runtime_kwargs     = runtime_kwargs
        self.target_op_method   = target_op_method
        self.total_ops          = total_ops
                
        self._buffers:     dict[str, MCA_TensorBuffer]   = {}
        self._runtime_ops: list[MCA_Operator]            = []

        # Common runtime configuration for operator mapping
        self.broadcast_optimize: bool = False
        self.broadcast_optimize_targets: list[str]=None
        self.auto_dispatch: bool = True
        self.mapping_strategy: MCA_OperatorMapper = MCA_OperatorMapper.CONTIGUOUS
        
        # Flags
        self._is_compiled: bool = False
    
    @property
    def is_compilation_target(self) -> bool:
        return True
    
    @property
    def is_compilation_ready(self) -> bool:
        return all([
            self.device is not None,
            self.exe_core_group is not None,
            self.mem_core_group is not None,
            self.main_data_mem_space is not None,
            self.l1_data_mem_space_size_per_core is not None,
            self.spad_ld_pp_space_size_per_core is not None,
            self.spad_st_pp_space_size_per_core is not None,
        ])
        
    @property
    def is_compiled(self):
        return self._is_compiled
    
    def get_buffer_name_by_key(self, key: str) -> str:
        for buf_name, buf_sig in self.buffer_signatures.items():
            if buf_sig.is_input and buf_sig.src.key == key:
                return buf_name
            if buf_sig.is_output and buf_sig.dst.key == key:
                return buf_name
        return None
    
    def entry_buffer_alloc_method(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compilation_ready:
            raise RuntimeError("Entry is not ready for compilation.")
        
        l1_data_mem_space   = self.device.create_l1_mem_space(self.l1_data_mem_space_size_per_core, core_group=self.mem_core_group)
        # spad_ld_mem_space   = self.device.create_l1_mem_space(self.spad_ld_pp_space_size_per_core, core_group=self.exe_core_group)
        # spad_st_mem_space   = self.device.create_l1_mem_space(self.spad_st_pp_space_size_per_core, core_group=self.exe_core_group)
        
        for buf_name, buf_sig in self.buffer_signatures.items():
            if buf_sig.ignore: continue
            if buf_name in self._buffers.keys(): continue
            
            self._buffers[buf_name] = MCA_TensorBuffer(
                mem_space=l1_data_mem_space if buf_sig.is_l1_buffer else self.main_data_mem_space,
                shape=buf_sig.shape,
                dtype=buf_sig.dtype,
                shard_shape=buf_sig.shard_shape,
                blocked_mapping=buf_sig.blocked_mapping,
            ).allocate()
        
        self.device.remove_all_l1_mem_space()    # clean up SPAD memory spaces after compilation
        
    def entry_compile_method(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compilation_ready:
            raise RuntimeError("Entry is not ready for compilation.")
        
        # l1_data_mem_space   = self.device.create_l1_mem_space(self.l1_data_mem_space_size_per_core, core_group=self.mem_core_group)
        spad_ld_mem_space   = self.device.create_l1_mem_space(self.spad_ld_pp_space_size_per_core, core_group=self.exe_core_group)
        spad_st_mem_space   = self.device.create_l1_mem_space(self.spad_st_pp_space_size_per_core, core_group=self.exe_core_group)
        
        # for buf_name, buf_sig in self.buffer_signatures.items():
        #     if buf_sig.ignore: continue
        #     if buf_name in self._buffers.keys(): continue
            
        #     self._buffers[buf_name] = MCA_TensorBuffer(
        #         mem_space=l1_data_mem_space if buf_sig.is_l1_buffer else self.main_data_mem_space,
        #         shape=buf_sig.shape,
        #         dtype=buf_sig.dtype,
        #         shard_shape=buf_sig.shard_shape,
        #         blocked_mapping=buf_sig.blocked_mapping,
        #     ).allocate()
            
        self._runtime_ops.append(
            self.target_op_method(
                self.device, self.exe_core_group, spad_ld_mem_space, spad_st_mem_space,
                **self._buffers,
                **self.runtime_kwargs,
                broadcast_optimize=True,
                mapping_strategy=MCA_OperatorMapper.OUTPUT_STATIONARY,
            )
        )
        
        self.device.remove_all_l1_mem_space()    # clean up SPAD memory spaces after compilation
        
        self._is_compiled = True
        
    def entry_buffer_init_method(self, graph_context: 'NetworkGraphContext', skip_pure_input_buffers: bool=False):
        for buf_name, buf_sig in self.buffer_signatures.items():
            if buf_sig.ignore: continue
            if not buf_sig.is_input: continue
            if skip_pure_input_buffers and buf_sig.buffer_usage == NetworkGraphCompiledEntry.BufferUsage.INPUT: continue
            
            if buf_sig.src.module == NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT:
                buf_data = graph_context[buf_sig.src.key]
            else:
                buf_data = getattr(buf_sig.src.module, buf_sig.src.key)
            
            for proc in buf_sig.preprocessings:
                buf_data = proc.apply(buf_data)
            
            buf = self._buffers[buf_name]
            buf.update(buf_data.to(buf_sig.dtype))
            
    def entry_buffer_finalize_method(self, graph_context: 'NetworkGraphContext'):
        for buf_name, buf_sig in self.buffer_signatures.items():
            if not buf_sig.is_output: continue
            
            if buf_sig.dst.module != NetworkGraphCompiledEntry.BufferSource.GLOBAL_CONTEXT:
                raise RuntimeError("Output buffer source must be GLOBAL_CONTEXT.")
            
            buf = self._buffers[buf_name]
            buf_data = buf.restore()
            
            for proc in buf_sig.postprocessings:
                buf_data = proc.apply(buf_data)
            
            graph_context[buf_sig.dst.key] = buf_data.to(buf_sig.orig_dtype)
    
    def execute(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compiled:
            raise RuntimeError("Compiled entry is not ready for execution.")
        
        self.entry_buffer_init_method(graph_context)
        
        for op in self._runtime_ops:
            op.dispatch()
            
        logger.debug(f"[TIMESTAMP {self.device.timestamp:<8d}] start executing compiled graph entry: {self}")

        while not self.device.is_idle:
            self.device.run_kernels(
                sync_target_core_groups=[self.exe_core_group.core_ids,],
            )
            
        self.entry_buffer_finalize_method(graph_context)

    def __str__(self):
        r = f"COMPILED_GRAPH({self.node.kind()}"
        
        ivars = list('%'+i.debugName() for i in self.node.inputs())
        ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r += f", inputs={ivars}, outputs={ovars}"
        r += f", submodule={type(self.submodule).__name__}"
        
        return r + ")"
    
class NetworkGraphCompiledEntryPipelined(NetworkGraphEntry):
    def __init__(
        self,
        *entries: NetworkGraphCompiledEntry,
    ):
        super().__init__(NetworkGraphEntry.Type.PIPELINED, None)
        
        self.device: MCA_DeviceBase = None
        self.entries: List[NetworkGraphCompiledEntry] = list(entries)
        
        for e in self.entries:
            if not isinstance(e, NetworkGraphCompiledEntry):
                raise RuntimeError("All entries in pipelined compiled entry must be NetworkGraphCompiledEntry instances.")
            if not e.is_compilation_target:
                raise RuntimeError("All entries in pipelined compiled entry must be compilation targets.")
            
            if self.device is None:
                self.device = e.device
            elif id(self.device) != id(e.device):
                raise RuntimeError("All entries in pipelined compiled entry must share the same device.")
    
    @property
    def is_compilation_target(self) -> bool:
        return True
    
    @property
    def is_compilation_ready(self) -> bool:
        return all([e.is_compilation_ready for e in self.entries])
        
    @property
    def is_compiled(self):
        return all([e.is_compiled for e in self.entries])
    
    def entry_buffer_alloc_method(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compilation_ready:
            raise RuntimeError("Entry is not ready for compilation.")
        
        # STEP 1: share buffers between dependent entries
        for p_idx, p_entry in enumerate(self.entries):
            for f_entry in self.entries[p_idx+1:]:
                for o in p_entry.node.outputs():
                    if o in f_entry.node.inputs():
                        p_buf_name = p_entry.get_buffer_name_by_key(o.debugName())
                        f_buf_name = f_entry.get_buffer_name_by_key(o.debugName())
                        
                        p_entry.buffer_signatures[p_buf_name].buffer_mem_type = NetworkGraphCompiledEntry.BufferMemType.L1
                        f_entry.buffer_signatures[f_buf_name].ignore = True
                        
            p_entry.entry_buffer_alloc_method(graph_context)
            
            for f_entry in self.entries[p_idx+1:]:
                for o in p_entry.node.outputs():
                    if o in f_entry.node.inputs():
                        f_entry._buffers[f_buf_name] = p_entry._buffers[p_buf_name]  # share the buffer object
    
    def entry_compile_method(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compilation_ready:
            raise RuntimeError("Entry is not ready for compilation.")
        
        # STEP 2: setup pipelining between dependent entries
        for p_idx, p_entry in enumerate(self.entries):
            p_entry.entry_compile_method(graph_context)
            
            for f_entry in self.entries[p_idx+1:]:
                for o in p_entry.node.outputs():
                    if o in f_entry.node.inputs():
                        p_buf_name = p_entry.get_buffer_name_by_key(o.debugName())
                        f_buf_name = f_entry.get_buffer_name_by_key(o.debugName())
                        
                        for p_op in p_entry._runtime_ops:
                            for f_op in f_entry._runtime_ops:
                                p_op.pipeline(dst_op=f_op, src_buf_name=p_buf_name, dst_buf_name=f_buf_name)
                                
    def entry_buffer_init_method(self, graph_context: 'NetworkGraphContext'):
        for e in self.entries:
            e.entry_buffer_init_method(graph_context)
            
    def entry_buffer_finalize_method(self, graph_context: 'NetworkGraphContext'):
        for e in self.entries:
            e.entry_buffer_finalize_method(graph_context)
    
    def execute(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compiled:
            raise RuntimeError("Compiled entry is not ready for execution.")
        
        self.entry_buffer_init_method(graph_context)
        
        sync_target_core_groups = []
        
        for e in self.entries:
            sync_target_core_groups.append(e.exe_core_group.core_ids)
            
            # for op in e._runtime_ops:
            op = e._runtime_ops[0]  # assume single op per compiled entry for pipelining
            op.dispatch()
                
        while not self.device.is_idle:
            self.device.run_kernels(
                sync_target_core_groups=sync_target_core_groups,
            )
            
        self.entry_buffer_finalize_method(graph_context)
        
    def __str__(self):
        graph_types = [type(e.submodule).__name__ for e in self.entries]
        r = f"PIPELINED_COMPILED_GRAPH({', '.join(graph_types)}"
        return r + ")"
    

class NetworkGraphContext(dict):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def _get_primitive_name(name: str) -> str:
        if isinstance(name, torch.Node):
            name = name.kind()
        elif isinstance(name, torch.nn.Module):
            name = type(name).__name__
        elif isinstance(name, type):
            name = name.__name__
        return name
    
    def __getitem__(self, key):
        if isinstance(key, torch.Value):
            key = key.debugName()
        elif isinstance(key, Placeholder):
            key = key.name
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, torch.Value):
            key = key.debugName()
        elif isinstance(key, Placeholder):
            key = key.name
        return super().__setitem__(key, value)
    
    def run_prim_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False):
        node = entry.node   
        node_domain, node_action = node.kind().split("::")
        
        if node_domain != "prim":
            raise Exception(f"only 'prim' domain is supported by the session\nexception occurred for the node: {node.kind()}")
        
        attrs = {attr_name: _get_attr_from_node(node, attr_name) for attr_name in node.attributeNames()}
        
        for o in node.outputs():
            if o.type().kind() == 'NoneType':
                self[o] = None

        if node_action == "CallMethod":
            submodule = self[list(node.inputs())[0]]
            args = [self[i] for i in list(node.inputs())[1:]]
            method_name = attrs['name']
            
            if isinstance(submodule, torch.nn.Module) and "forward" in method_name:
                if entry.is_compilation_target:
                    outputs = submodule(*args)
                elif entry.node_type == NetworkGraphEntry.Type.GRAPH:
                    outputs = entry.subgraph.run_graph(*args, trace_mode=trace_mode)
            else:
                method = getattr(submodule, method_name)
                outputs = method(*args)
            
            if len(list(node.outputs())) == 1:
                self[node.output()] = outputs
            else:
                for idx, o in enumerate(node.outputs()):
                    self[o] = outputs[idx]
        elif node_action == "GetAttr":
            self[node.output()] = getattr(self[node.input()], attrs['name'])
        elif node_action == "Constant":
            self[node.output()] = attrs.get('value', None)
        elif node_action == "ListConstruct":
            self[node.output()] = [self[i] for i in node.inputs()]
        elif node_action == "TupleConstruct":
            self[node.output()] = tuple([self[i] for i in node.inputs()])
        elif node_action == "NumToTensor":
            self[node.output()] = torch.tensor(self[node.inputsAt(0)])
        else:
            raise Exception(f"action '{node_action}' is not supported by the session in 'prim' domain\nexception occurred for the node: {node.kind()}")
        
    def run_nonprim_entry(self, entry: NetworkGraphEntry):
        node = entry.node
        node_domain, node_action = node.kind().split("::")
        
        args = [self[i] for i in node.inputs()]
        method, pp_args, pp_kwargs = _find_nonprim_method(node_domain, node_action, args=args)
        
        if method is None:
            raise Exception(f"method '{node_action}' in domain '{node_domain}' not found\nexception occurred for the node: {node.kind()}")
        
        outputs = method(*pp_args, **pp_kwargs)
        
        if len(list(node.outputs())) == 1:
            self[node.output()] = outputs
        else:
            for idx, o in enumerate(node.outputs()):
                self[o] = outputs[idx]
                
    def run_compiled_entry(self, entry: NetworkGraphCompiledEntry):
        entry.execute(self)
                
    def run_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False):
        # STEP 1: pre-run the entry to prepare inputs/outputs
        if entry.is_prim:
            self.run_prim_entry(entry, trace_mode=trace_mode)
        elif entry.is_nonprim:
            self.run_nonprim_entry(entry)
        elif entry.is_pipelined:
            for sub_entry in entry.entries:
                self.run_entry(sub_entry, trace_mode=True)
        else:
            raise Exception(f"unsupported entry type: {entry.node_type}")
        
        # STEP 2: run the entry with compiled runtime (if necessary)
        if entry.is_compilation_target and not trace_mode:
            if entry.is_compiled:
                logger.debug(f"running compiled graph entry: {entry}")
                self.run_compiled_entry(entry)
            else:
                logger.warning(f"compiled graph entry is not executable (missing runtime parameters): {entry}")
                pass


class NetworkGraphCompilationRecipe:
    DEFAULT = "_DEFAULT"
    
    def __init__(
        self, 
        device: MCA_DeviceBase,
        core_groups: List[MCA_CoreGroup],
        
        main_data_mem_space_size: int,
        l1_mem_space_size_per_core: int,
        l1_spad_ld_pp_space_ratio: float,
        l1_spad_st_pp_space_ratio: float,
        
        max_pipeline_window: int,
    ):
        self.device = device
        self.core_groups = core_groups
        
        self.main_data_mem_space_size = main_data_mem_space_size
        self.l1_mem_space_size_per_core = l1_mem_space_size_per_core
        self.l1_spad_ld_pp_space_ratio = l1_spad_ld_pp_space_ratio
        self.l1_spad_st_pp_space_ratio = l1_spad_st_pp_space_ratio
        
        self.max_pipeline_window = max_pipeline_window
            
    def supports(self, module_type: type | str) -> bool:
        if isinstance(module_type, type):
            module_type = module_type.__name__
        elif isinstance(module_type, torch.nn.Module):
            module_type = type(module_type).__name__
        return hasattr(self, module_type)
    
    def get_compiled_entry(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Module) -> NetworkGraphCompiledEntry:
        if not self.supports(type(submodule)):
            raise Exception(f"compilation recipe does not support module type: {type(submodule).__name__}")
        
        compile_method = getattr(self, type(submodule).__name__)
        compiled_entry = compile_method(graph_context, node, submodule)
        
        if not isinstance(compiled_entry, NetworkGraphCompiledEntry):
            raise Exception("compilation recipe method must return a NetworkGraphCompiledEntry instance")
        
        return compiled_entry
    
    @staticmethod
    def recipe(func: Callable):
        @functools.wraps(func)
        def _wrapper(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Module) -> NetworkGraphCompiledEntry:
            if not isinstance(submodule, torch.nn.Module):
                raise Exception("recipe methods can only compile torch.nn.Module submodules")
            entry = func(self, graph_context, node, submodule)
            if not isinstance(entry, NetworkGraphCompiledEntry):
                raise Exception("recipe methods must return a NetworkGraphCompiledEntry instance")
            return entry
        return _wrapper
    
    @property
    def l1_data_mem_space_size_per_core(self) -> int:
        _l1_data_mem_space_ratio = 1.0 - (self.l1_spad_ld_pp_space_ratio + self.l1_spad_st_pp_space_ratio)
        return math.floor(self.l1_mem_space_size_per_core * _l1_data_mem_space_ratio)
    
    @property
    def spad_ld_pp_space_size_per_core(self) -> int:
        return math.floor(self.l1_mem_space_size_per_core * self.l1_spad_ld_pp_space_ratio)
    
    @property
    def spad_st_pp_space_size_per_core(self) -> int:
        return math.floor(self.l1_mem_space_size_per_core * self.l1_spad_st_pp_space_ratio)


def _run_compiled_entry(args):
    entry: NetworkGraphCompiledEntry
    graph_context: NetworkGraphContext
    
    entry, graph_context = args
    logger.debug(f"running compiled graph entry in parallel: {entry}")
    graph_context.run_entry(entry, trace_mode=False)
    logger.debug(f"finished running compiled graph entry in parallel: {entry}")
    
def _distribute_resources(total: int, weights: list[float]) -> list[int]:
    n = len(weights)
    
    if total < n:
        raise ValueError("Insufficient total resources to allocate at least 1 unit per entity.")
    
    allocations = [1] * n
    remaining_resources = total - n
    
    sum_weights = sum(weights)

    additional_allocations = []
    for w in weights:
        share = (w / sum_weights) * remaining_resources
        additional_allocations.append(share)
    
    for i in range(n):
        int_share = int(additional_allocations[i])
        allocations[i] += int_share
    
    current_sum = sum(allocations)
    leftover = total - current_sum
    
    if leftover > 0:
        remainders = [(i, additional_allocations[i] - int(additional_allocations[i])) for i in range(n)]
        remainders.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(int(leftover)):
            idx = remainders[i][0]
            allocations[idx] += 1
            
    return allocations

class NetworkGraphCompiler:
    def __init__(
        self, 
        module: torch.nn.Module, 
        graph_ivars: list[torch.Value], 
        graph_ovars: list[torch.Value], 
        graph_nodes: list[torch.Node], 
        graph_context: NetworkGraphContext, 
        graph_entries: list[NetworkGraphEntry],
        graph_recipe: NetworkGraphCompilationRecipe,
    ):
        self.module = module
        self.graph_ivars = graph_ivars
        self.graph_ovars = graph_ovars
        self.graph_nodes = graph_nodes
        self.graph_context = graph_context
        self.graph_entries = graph_entries
        self.graph_recipe = graph_recipe
        
        self._lowered = False
    
    @classmethod
    def from_trace(
        cls, 
        module: torch.nn.Module,
        graph_recipe: NetworkGraphCompilationRecipe, 
        *dummy_inputs: torch.Tensor,
        disable_lowering: bool=False, disable_toposort: bool=False, disable_compile: bool=False,
    ):
        logger.debug(f"tracing module: {type(module).__name__} with dummy inputs: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in dummy_inputs]}")
        
        warnings.filterwarnings("ignore", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: suppress quantized model warning
        
        # STEP 1: create traced graph (torch.jit.trace)
        module = module.eval()
        
        graph_context = NetworkGraphContext()
        
        traced_module = torch.jit.trace(module, *dummy_inputs)
        traced_graph = traced_module.graph

        # STEP 2: parse traced graph to get input/output variables and nodes
        graph_ivars = list(traced_graph.inputs())
        graph_ovars = list(traced_graph.outputs())
        graph_nodes = list(traced_graph.nodes())
        
        entries: list[NetworkGraphEntry] = []
        
        # STEP 3-1: initialize input variables
        graph_context[graph_ivars[0]] = module
        for idx, ivar in enumerate(graph_ivars[1:]):
            graph_context[ivar] = dummy_inputs[idx]
        
        # STEP 3-2: construct entries and initialize intermediate variables
        for node in graph_nodes:
            node_domain, node_action = node.kind().split("::")
            
            # STEP 3-2-1: prim nodes
            if node_domain == "prim":
                entry = NetworkGraphEntry(NetworkGraphEntry.Type.PRIM, node)
                attrs = {attr_name: _get_attr_from_node(node, attr_name) for attr_name in node.attributeNames()}

                if node_action == "CallMethod":
                    submodule = graph_context[list(node.inputs())[0]]
                    method_name = attrs['name']
                    
                    # check if the submodule is a torch.nn.Module and method is "forward" -> subgraph or compiled graph
                    if isinstance(submodule, torch.nn.Module) and "forward" in method_name:
                        args = tuple(graph_context[i] for i in list(node.inputs())[1:])
                        subgraph = NetworkGraphCompiler.from_trace(submodule, graph_recipe, *args, disable_lowering=True, disable_toposort=True, disable_compile=True)

                        if graph_recipe.supports(submodule):
                            entry = graph_recipe.get_compiled_entry(graph_context, node, submodule)
                        else:
                            entry = NetworkGraphEntry(NetworkGraphEntry.Type.GRAPH, node, subgraph=subgraph)
                    
                    # otherwise, the entry remains as prim::CallMethod (not compiled as runtime kernel or operator)
                    else:
                        entry = NetworkGraphEntry(NetworkGraphEntry.Type.PRIM, node)
            
            # STEP 3-2-2: nonprim nodes (aten::, quantized::, ...)
            else:
                entry = NetworkGraphEntry(NetworkGraphEntry.Type.NONPRIM, node)
            
            entries.append(entry)
            graph_context.run_entry(entry, trace_mode=True)
            
        warnings.filterwarnings("default", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("default", category=FutureWarning)  # TODO: suppress quantized model warning 
            
        graph = cls(module, graph_ivars, graph_ovars, graph_nodes, graph_context, entries, graph_recipe)
        
        if not disable_lowering:
            graph.lowering()
        if not disable_toposort:
            graph.topological_sort()
        if not disable_compile:
            graph.compile_entries()
            
        return graph
    
    def rename_vars(self, var_rename_map: dict[str, str]):
        # STEP 1: rename input variables
        for ivar in self.graph_ivars:
            if ivar.debugName() in var_rename_map:
                ivar.setDebugName(var_rename_map[ivar.debugName()])
                
        # STEP 2: rename output variables
        for ovar in self.graph_ovars:
            if ovar.debugName() in var_rename_map:
                ovar.setDebugName(var_rename_map[ovar.debugName()])
                
        # STEP 3: rename variables in nodes
        for node in self.graph_nodes:
            for ivar in node.inputs():
                if ivar.debugName() in var_rename_map:
                    ivar.setDebugName(var_rename_map[ivar.debugName()])
            for ovar in node.outputs():
                if ovar.debugName() in var_rename_map:
                    ovar.setDebugName(var_rename_map[ovar.debugName()])
                    
    def entrywise_compile(self):
        raise NotImplementedError("entrywise compilation is not implemented yet.")

    def lowering(self, context_name=None):
        logger.debug(f"lowering graph of module: {type(self.module).__name__}")
        
        if context_name is None:
            context_name = self.module.__class__.__name__
            
        # STEP 1: rename all the variables
        var_rename_map = {vname: f"{context_name}::{vname}" for vname in self.graph_context.keys()}
        self.rename_vars(var_rename_map=var_rename_map)
        
        # STEP 2: lower subgraphs
        lowered_entries: list[NetworkGraphEntry] = []
        lowered_graph_nodes: list[torch.Node] = []
        
        for entry in self.graph_entries:
            if entry.is_subgraph_available:
                # STEP 2-1: lowering subgraph
                subgraph_context_name = context_name + "::" + entry.node.inputsAt(0).debugName().split("::")[-1]
                entry.subgraph.lowering(context_name=subgraph_context_name)

                # STEP 2-2: rename subgraph input variables
                var_rename_map = {cvar.debugName(): pvar.debugName() for cvar, pvar in zip(entry.subgraph.graph_ivars, entry.node.inputs())}
                entry.subgraph.rename_vars(var_rename_map=var_rename_map)
                
                # STEP 2-3: rename subgraph output variables
                var_rename_map = {cvar.debugName(): pvar.debugName() for cvar, pvar in zip(entry.subgraph.graph_ovars, entry.node.outputs())}
                entry.subgraph.rename_vars(var_rename_map=var_rename_map)
                
                # STEP 2-4: append lowered entries and nodes
                lowered_entries.extend(entry.subgraph.graph_entries)
                lowered_graph_nodes.extend(entry.subgraph.graph_nodes)
            else:  
                lowered_entries.append(entry)
                lowered_graph_nodes.append(entry.node)
        
        self.graph_entries = lowered_entries
        self.graph_nodes = lowered_graph_nodes
        
        self._lowered = True

        return self
        
    def topological_sort(self):
        logger.debug(f"topological sorting graph of module: {type(self.module).__name__}")
        
        # STEP 1: build dependency graph between entries
        entry_dept_graph: Dict[int, Set[int]] = {idx: set() for idx in range(len(self.graph_entries))}
        
        for entry_idx, entry in enumerate(self.graph_entries[:-1]):
            ovars = list(o.debugName() for o in entry.node.outputs())
            
            for check_idx, check_entry in enumerate(self.graph_entries[entry_idx+1:], start=entry_idx+1):
                ivars = list(i.debugName() for i in check_entry.node.inputs())
                
                if any(o in ivars for o in ovars):
                    entry_dept_graph[entry_idx].add(check_idx)
        
        # STEP 2: topological sort            
        sorted_entry_indices = _kahn_topological_sort(entry_dept_graph)
        self.graph_entries = [self.graph_entries[idx] for idx in sorted_entry_indices]
        
        return self
    
    def compile_entries(self):
        # STEP 1: allocate main memory space as specified in the recipe
        main_data_mem_space = self.graph_recipe.device.create_main_mem_space(self.graph_recipe.main_data_mem_space_size)
        
        # STEP 2: compile operators
        new_entries = []

        entry_idx = 0
        
        while entry_idx < len(self.graph_entries):
            entry = self.graph_entries[entry_idx]
            
            if entry.is_subgraph_available:
                logger.debug(f"start compiling subgraph entry: {entry}")
                entry.subgraph.compile_entries()
                logger.debug(f"finished compiling subgraph entry: {entry}")
            elif entry.is_compilation_target:
                logger.debug(f"compiling graph entry: {entry}")
                
                device = self.graph_recipe.device
                entry: NetworkGraphCompiledEntry = entry
                
                for pipeline_window in range(self.graph_recipe.max_pipeline_window, 0, -1):
                    _is_pipeline_succeed = True
                    
                    _pipelined_future_entries: list[NetworkGraphCompiledEntry] = []
                    _pipeline_st = entry_idx
                    _pipeline_ed = min(entry_idx + pipeline_window, len(self.graph_entries))
                    
                    _mem_subcore_groups: dict[int, list[MCA_CoreGroup]] = {}
                    _mem_core_group_cursor = 0
                    
                    _n_core_groups = len(self.graph_recipe.core_groups)

                    if _n_core_groups < pipeline_window:
                        _is_pipeline_succeed = False
                        break
                    
                    for p_idx in range(_pipeline_st, _pipeline_ed):
                        p_entry: NetworkGraphCompiledEntry = self.graph_entries[p_idx]
                        if not p_entry.is_compilation_target:
                            _is_pipeline_succeed = False
                            break
                        
                        _collected_pipelined_bufs: dict[str, list[tuple[int, str]]] = {}
                        
                        for f_idx in range(p_idx+1, _pipeline_ed):
                            f_entry: NetworkGraphCompiledEntry = self.graph_entries[f_idx]
                            if not f_entry.is_compilation_target:
                                _is_pipeline_succeed = False
                                break
                            
                            for p_ovar in p_entry.node.outputs():
                                for f_ivar in f_entry.node.inputs():
                                    if p_ovar.debugName() != f_ivar.debugName():
                                        continue
                                    
                                    p_buf_name = p_entry.get_buffer_name_by_key(p_ovar.debugName())
                                    f_buf_name = f_entry.get_buffer_name_by_key(f_ivar.debugName())
                                    
                                    if p_buf_name is None:
                                        raise RuntimeError(f"cannot find buffer signature name for pipelined output variable: entry_idx={p_idx}, var={p_ovar.debugName()}")
                                    if f_buf_name is None:
                                        raise RuntimeError(f"cannot find buffer signature name for pipelined input variable: entry_idx={f_idx}, var={f_ivar.debugName()}")
                                    
                                    if p_buf_name not in _collected_pipelined_bufs.keys():
                                        _collected_pipelined_bufs[p_buf_name] = []
                                        
                                    _collected_pipelined_bufs[p_buf_name].append((f_idx, f_buf_name)) 
                                    
                        if not _is_pipeline_succeed:
                            break
                        
                        _required_mem_segs: list[tuple[int, int]] = []  # [(shard_size, shard_num), ...]
                        
                        for p_buf_name in _collected_pipelined_bufs.keys():
                            _shard_size = p_entry.buffer_signatures[p_buf_name].get_shard_size()
                            _shard_num  = p_entry.buffer_signatures[p_buf_name].get_shard_num()
                            _required_mem_segs.append((_shard_size, _shard_num))
                            
                        if _mem_core_group_cursor >= _n_core_groups:
                            _is_pipeline_succeed = False
                            break
                        
                        _tmp_mem_core_group = self.graph_recipe.core_groups[_mem_core_group_cursor]
                        _mem_core_group_num = 1
                        
                        while True:
                            _tmp_n_cores = _tmp_mem_core_group.n_cores
                            _tmp_total_mem_usage_per_core = sum([shard_size * math.ceil(shard_num / _tmp_n_cores) for shard_size, shard_num in _required_mem_segs])
                            
                            if _tmp_total_mem_usage_per_core < self.graph_recipe.l1_data_mem_space_size_per_core:
                                break
                            
                            if _mem_core_group_cursor + _mem_core_group_num >= _n_core_groups:
                                _is_pipeline_succeed = False
                                break
                            
                            _tmp_mem_core_group = _tmp_mem_core_group.merge(self.graph_recipe.core_groups[_mem_core_group_cursor + _mem_core_group_num])
                            _mem_core_group_num += 1
                            
                        if _is_pipeline_succeed:
                            _pipelined_future_entries.append(p_entry)
                            _mem_subcore_groups[len(_pipelined_future_entries)-1] = [self.graph_recipe.core_groups[_mem_core_group_cursor + i] for i in range(_mem_core_group_num)]
                            _mem_core_group_cursor += _mem_core_group_num
                            
                    # end for p_idx in range(_pipeline_st, _pipeline_ed)
                    if _is_pipeline_succeed:    
                        total_ops_per_p_entry = [p_entry.total_ops for p_entry in _pipelined_future_entries]
                        
                        _exe_subcore_groups: dict[int, list[MCA_CoreGroup]] = {}
                        _exe_core_group_cursor = 0
                        
                        for p_idx, (p_entry, n_exe_core_groups) in enumerate(zip(_pipelined_future_entries, _distribute_resources(total=_n_core_groups, weights=total_ops_per_p_entry))):
                            _exe_subcore_groups[p_idx] = self.graph_recipe.core_groups[_exe_core_group_cursor:_exe_core_group_cursor + n_exe_core_groups]
                            _exe_core_group_cursor += n_exe_core_groups
                        
                        for p_idx, p_entry in enumerate(_pipelined_future_entries):
                            p_entry.device = device
                            p_entry.main_data_mem_space = main_data_mem_space
                            p_entry.mem_core_group = MCA_CoreGroup.merge_core_groups(_mem_subcore_groups[p_idx])
                            p_entry.exe_core_group = MCA_CoreGroup.merge_core_groups(_exe_subcore_groups[p_idx])
                            p_entry.l1_data_mem_space_size_per_core = self.graph_recipe.l1_data_mem_space_size_per_core
                            p_entry.spad_ld_pp_space_size_per_core  = self.graph_recipe.spad_ld_pp_space_size_per_core
                            p_entry.spad_st_pp_space_size_per_core  = self.graph_recipe.spad_st_pp_space_size_per_core
                        
                        entry = NetworkGraphCompiledEntryPipelined(*_pipelined_future_entries)
                        
                        logger.debug(f"pipelining succeeded with window size: {pipeline_window} for entry starting at index: {entry_idx}")
                        for p_entry in entry.entries:
                            # logger.debug(f"  - pipelined entry: {p_entry} on exe_core_group: {p_entry.exe_core_group}, mem_core_group: {p_entry.mem_core_group}")
                            logger.debug(f"  - pipelined entry: {p_entry} on exe_core_group: {p_entry.exe_core_group.n_cores}, mem_core_group: {p_entry.mem_core_group.n_cores}")
                        
                        break
                
                if not isinstance(entry, NetworkGraphCompiledEntryPipelined):
                    entry.device = device
                    entry.main_data_mem_space = main_data_mem_space
                    entry.exe_core_group = self.graph_recipe.core_groups[0]
                    entry.mem_core_group = self.graph_recipe.core_groups[0]
                    entry.l1_data_mem_space_size_per_core = self.graph_recipe.l1_data_mem_space_size_per_core
                    entry.spad_ld_pp_space_size_per_core  = self.graph_recipe.spad_ld_pp_space_size_per_core
                    entry.spad_st_pp_space_size_per_core  = self.graph_recipe.spad_st_pp_space_size_per_core
                    
                entry.entry_buffer_alloc_method(self.graph_context)
                entry.entry_compile_method(self.graph_context)
                logger.debug(f"finished compiling graph entry: {entry}")
                
            new_entries.append(entry)
            
            # print(f"{entry}: {entry.is_compiled}")
            # if isinstance(entry, NetworkGraphCompiledEntryPipelined):
            #     for sub_entry in entry.entries:
            #         print(f"  - {sub_entry}: {sub_entry.is_compiled}")
            
            if isinstance(entry, NetworkGraphCompiledEntryPipelined):
                entry_idx += len(entry.entries)
            else:
                entry_idx += 1
            
        self.graph_entries = new_entries
        
        self.graph_recipe.device.remove_all_main_mem_space()  # clean up main memory spaces after compilation
        
    def get_outputs(self):
        if len(self.graph_ovars) == 1:
            return self.graph_context[self.graph_ovars[0]]
        return [self.graph_context[o] for o in self.graph_ovars]
    
    def run_graph(self, *dummy_inputs, trace_mode: bool=False):
        self.graph_context[self.graph_ivars[0]] = self.module
        for idx, ivar in enumerate(self.graph_ivars[1:]):
            self.graph_context[ivar] = dummy_inputs[idx]
        
        for entry in self.graph_entries:
            try:
                self.graph_context.run_entry(entry, trace_mode=trace_mode)
            except Exception as e:
                logger.error(f"exception occurred while running the graph with node: {entry.node}")
                raise Exception(f"exception occurred while running the entry: {entry}\n{e}") from e
        
        return self.get_outputs()
    
    def run_graph_compiled_parallel(self, *dummy_inputs):
        self.run_graph(*dummy_inputs, trace_mode=True)

        with Pool() as pool:
            pool.map(_run_compiled_entry, [(entry, self.graph_context) for entry in self.graph_entries if entry.is_compiled])
            
        return self.get_outputs()
    
    def print_graph(self, indent: int=0):
        print(" " * indent + f"OPEN_GRAPH[type={type(self.module).__name__}]({', '.join(list('%'+i.debugName() for i in self.graph_ivars))}):")
        for entry in self.graph_entries:
            print(" " * (indent + 2) + str(entry))
            
            if entry.is_subgraph_available:
                entry.subgraph.print_graph(indent=indent+2)
        print(" " * (indent + 2) + f"return {', '.join(list('%'+o.debugName() for o in self.graph_ovars))}")
