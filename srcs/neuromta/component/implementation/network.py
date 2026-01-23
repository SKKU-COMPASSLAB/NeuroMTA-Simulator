import abc
import functools
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
    class BufferSignature:
        def __init__(self, shape: Sequence[int], dtype: torch.dtype, shard_shape: Sequence[int], blocked_mapping: bool=False, orig_dtype: torch.dtype=None):
            self.shape = shape
            self.dtype = dtype
            self.shard_shape = shard_shape
            self.blocked_mapping = blocked_mapping
            self.orig_dtype = orig_dtype if orig_dtype is not None else dtype
            
        def get_shard_size(self) -> int:
            return self.shard_shape[0] * self.shard_shape[1] * self.dtype.itemsize
        
        def get_shard_num(self) -> int:
            n_height_shards = (self.shape[-2] + self.shard_shape[0] - 1) // self.shard_shape[0]
            n_width_shards  = (self.shape[-1] + self.shard_shape[1] - 1) // self.shard_shape[1]
            return n_height_shards * n_width_shards
    
    def __init__(
        self, 
        node: torch.Node, 
        submodule: torch.nn.Module,
        buffer_signatures: Dict[str, 'NetworkGraphCompiledEntry.BufferSignature'],
        runtime_kwargs: Dict[str, Any],
        entry_compile_method: Callable[['NetworkGraphContext', 'NetworkGraphCompiledEntry'], None],
        entry_buffer_init_method: Callable[['NetworkGraphContext', 'NetworkGraphCompiledEntry'], None],
        entry_buffer_finalize_method: Callable[['NetworkGraphContext', 'NetworkGraphCompiledEntry'], None],
    ):
        super().__init__(NetworkGraphEntry.Type.GRAPH, node)
        
        self.submodule: torch.nn.Module = submodule
        
        # Common runtime parameters required
        self.device: MCA_DeviceBase = None
        self.core_group: MCA_CoreGroup = None
        
        # Buffer signatures
        self.buffer_signatures  = buffer_signatures  # name -> BufferSignature
        self.runtime_kwargs     = runtime_kwargs
        
        self.entry_compile_method = entry_compile_method
        self.entry_buffer_init_method = entry_buffer_init_method
        self.entry_buffer_finalize_method = entry_buffer_finalize_method
        
        self._mem_space:   dict[str, MCA_MemorySpace]    = None
        self._buffers:     dict[str, MCA_TensorBuffer]   = None
        self._runtime_ops: list[MCA_Operator]            = None
        
        # Common runtime configuration for operator mapping
        self.broadcast_optimize: bool = False
        self.broadcast_optimize_targets: list[str]=None
        self.auto_dispatch: bool = True
        self.mapping_strategy: MCA_OperatorMapper = MCA_OperatorMapper.CONTIGUOUS
    
    @property
    def is_compilation_target(self) -> bool:
        return True
    
    @property
    def is_compilation_ready(self) -> bool:
        return all([
            self.device is not None,
            self.core_group is not None,
        ])
        
    @property
    def is_compiled(self):
        return all([
            self._mem_space is not None,
            self._buffers is not None,
            self._runtime_ops is not None,
        ])
        
    def execute(self, graph_context: 'NetworkGraphContext'):
        if not self.is_compiled:
            raise RuntimeError("Compiled entry is not ready for execution.")
        
        self.entry_buffer_init_method(graph_context, self)
        
        for op in self._runtime_ops:
            op.dispatch()
            
        logger.debug(f"[TIMESTAMP {self.device.timestamp:<8d}] start executing compiled graph entry: {self}")
            
        while not self.device.is_idle:
            self.device.run_kernels(
                sync_target_core_groups=[self.core_group.core_ids,],
            )
            
        self.entry_buffer_finalize_method(graph_context, self)

    def __str__(self):
        r = f"COMPILED_GRAPH({self.node.kind()}"
        
        ivars = list('%'+i.debugName() for i in self.node.inputs())
        ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r += f", inputs={ivars}, outputs={ovars}"
        r += f", submodule={type(self.submodule).__name__}"
        
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
    def __init__(self, device: MCA_DeviceBase):
        self.device = device
    
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
    
    @classmethod
    def from_trace(
        cls, 
        module: torch.nn.Module, #dummy_inputs: tuple[torch.Tensor], 
        # graph_context: NetworkGraphContext, graph_recipe: NetworkGraphCompilationRecipe, 
        graph_recipe: NetworkGraphCompilationRecipe, 
        *dummy_inputs: torch.Tensor,
        disable_lowering: bool=False, disable_toposort: bool=False, disable_compile: bool=False,
    ):
        logger.debug(f"tracing module: {type(module).__name__} with dummy inputs: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in dummy_inputs]}")
        
        warnings.filterwarnings("ignore", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: suppress quantized model warning
        
        # STEP 1: create traced graph (torch.jit.trace)
        module = module.eval()
        
        # if not isinstance(dummy_inputs, tuple):
        #     dummy_inputs = (dummy_inputs,)
        
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
                # if host_context.rt_supports(node.kind()):
                #     entry = NetworkGraphEntry(NetworkGraphEntryType.COMPILED_NONPRIM, node)
                # else:
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
        for entry in self.graph_entries:
            if entry.is_subgraph_available:
                logger.debug(f"start compiling subgraph entry: {entry}")
                entry.subgraph.compile_entries()
                logger.debug(f"finished compiling subgraph entry: {entry}")
            elif entry.is_compilation_target:
                logger.debug(f"compiling graph entry: {entry}")
                
                device = self.graph_recipe.device
                core_group = device.get_npu_core_group((0, 0), (8, 8))  # TODO: implement dynamic core group selection algorithm here
                
                compiled_entry: NetworkGraphCompiledEntry = entry
                
                compiled_entry.device = device
                compiled_entry.core_group = core_group
                compiled_entry.entry_compile_method(self.graph_context, compiled_entry)
        
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
    
    def print_graph(self, indent: int=0):
        print(" " * indent + f"OPEN_GRAPH[type={type(self.module).__name__}]({', '.join(list('%'+i.debugName() for i in self.graph_ivars))}):")
        for entry in self.graph_entries:
            print(" " * (indent + 2) + str(entry))
            
            if entry.is_subgraph_available:
                entry.subgraph.print_graph(indent=indent+2)
        print(" " * (indent + 2) + f"return {', '.join(list('%'+o.debugName() for o in self.graph_ovars))}")
