import abc
import warnings
import enum
import torch

from collections import deque, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Set, Callable

from neuromta.framework import *
from neuromta.hardware.core.npu_core import NPUCore
from neuromta.hardware.implementation.common.software import *
from neuromta.hardware.implementation.common.hardware import *
from neuromta.hardware.implementation.common.tensor import *


__all__ = [
    "NetworkGraphEntryType",
    "NetworkGraphEntry",
    
    "get_global_network_context",
    "CompilationRecipe",
    "NetworkGraphContext",
    "NetworkGraph",
]


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

class NetworkGraphEntryType(enum.Enum):
    PRIM                = enum.auto()  # prim operators (operators starting with "prim::")
    GRAPH               = enum.auto()  # graph node (submodule call, but not compiled as runtime kernel e.g., torch.nn.Sequential)
    NONPRIM             = enum.auto()  # nonprim operators (operators starting with "aten::", "quantized::", etc.)
    COMPILED_GRAPH      = enum.auto()  # compiled graph nodes (submodule call compiled as runtime kernel e.g., torch.nn.Conv2d, torch.nn.Linear ...)
    COMPILED_NONPRIM    = enum.auto()  # compiled nonprim nodes (nonprim operators compiled as runtime kernel e.g., aten::matmul, aten::add_, ...)
    
    @property
    def is_prim(self) -> bool:
        return self in (NetworkGraphEntryType.PRIM, NetworkGraphEntryType.GRAPH, NetworkGraphEntryType.COMPILED_GRAPH)
    
    @property
    def is_nonprim(self) -> bool:
        return self in (NetworkGraphEntryType.NONPRIM, NetworkGraphEntryType.COMPILED_NONPRIM)
    
    @property
    def is_compiled_target(self) -> bool:
        return self in (NetworkGraphEntryType.COMPILED_GRAPH, NetworkGraphEntryType.COMPILED_NONPRIM)

class NetworkGraphEntry:
    def __init__(self, node_type: NetworkGraphEntryType, node: torch.Node, **kwargs):
        self.node_type = node_type
        self.node = node
        
        self.subgraph: NetworkGraph = kwargs.get("subgraph", None)
        self.submodule: torch.nn.Module = kwargs.get("submodule", None)
        self.recipe: type | CompilationRecipe = kwargs.get("recipe", None)
        
    def compile(self):
        self.recipe: CompilationRecipe = self.recipe(self.node)  # instantiate the recipe class
        # self.recipe.compile()
    
    @property
    def is_compiled(self) -> bool:
        # not compiled: recipe is a class type
        # compiled:     recipe is an instance of CompilationRecipe
        return isinstance(self.recipe, CompilationRecipe)
        
    def __str__(self):
        r = f"{self.node_type.name}({self.node.kind()}"
        
        ivars = list('%'+i.debugName() for i in self.node.inputs())
        ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r += f", inputs={ivars}, outputs={ovars}"
        
        if self.node_type == NetworkGraphEntryType.GRAPH:
            r += f", graph={type(self.subgraph.module).__name__}"
        if self.node_type == NetworkGraphEntryType.COMPILED_GRAPH:
            r += f", submodule={type(self.submodule).__name__}"
        if self.node_type == NetworkGraphEntryType.COMPILED_NONPRIM:
            r += f", method={self.node.kind()}"
        
        return r + ")"


_global_network_context: 'NetworkGraphContext' = None

def get_global_network_context() -> 'NetworkGraphContext':
    global _global_network_context
    return _global_network_context

class CompiledStep:
    def __init__(self):
        self._op_signatures:   dict[str, list[tuple[Callable, tuple, dict]]]    = {}  # operator signatures per slot ID (key: slot_id, value: list of (op, args, kwargs))
        self._layout_updates:  dict[str, str | torch.Value | torch.Tensor]      = {}  # layout updates (key: layout_name, value: variable name or tensor)
        self._buffer_allocs:   list[str]                                        = []  # buffer allocations (list of buffer names)
        self._buffer_loads:    dict[str, tuple[str, TensorDimensionCursor]]     = {}  # buffer loads (key: buffer_name, value: (layout_name, cursor))
        self._buffer_stores:   dict[str, tuple[str, TensorDimensionCursor]]     = {}  # buffer stores (key: buffer_name, value: (layout_name, cursor))
        self._buffer_deallocs: list[str]                                        = []  # buffer deallocations (list of buffer names)
        self._layout_restores: dict[str, str | torch.Value]                     = {}  # layout restores (key: layout_name, value: variable name or tensor)

    def op(self, slot_id: str, op: Callable, *args, **kwargs):
        if slot_id not in self._op_signatures:
            self._op_signatures[slot_id] = []
        signature = (op, args, kwargs)
        self._op_signatures[slot_id].append(signature)

    def layout_update(self, layout_name: str, var_name: str | torch.Value | torch.Tensor):
        self._layout_updates[layout_name] = var_name

    def layout_restore(self, layout_name: str, var_name: str | torch.Value):
        self._layout_restores[layout_name] = var_name
        
    def buffer_alloc(self, buffer_name: str):
        self._buffer_allocs.append(buffer_name)
        
    def buffer_load(self, buffer_name: str, layout_name: str, cursor: TensorDimensionCursor=None):
        self._buffer_loads[buffer_name] = (layout_name, cursor)

    def buffer_store(self, buffer_name: str, layout_name: str, cursor: TensorDimensionCursor=None):
        self._buffer_stores[buffer_name] = (layout_name, cursor)
        
    def buffer_dealloc(self, buffer_name: str):
        self._buffer_deallocs.append(buffer_name)

class CompilationRecipe(metaclass=abc.ABCMeta):
    def __init__(self, node: torch.Node):
        self._node = node
        self._dims = TensorDimensionPool()
        self._layouts = TensorLayoutPool()
        self._buffers: dict[str, MCA_TensorBuffer] = {}
        self._steps: list[CompiledStep] = []
    
    def new_step(self) -> CompiledStep:
        step = CompiledStep()
        self._steps.append(step)
        return step
    
    def clear_step(self):
        self._steps.clear()
        
    def run_step(self, step_idx: int):
        context: NetworkGraphContext = get_global_network_context()
        assert isinstance(context, NetworkGraphContext), "CompilationRecipe requires an active NetworkGraphContext."
        
        step = self._steps[step_idx]
        
        # update layouts
        for layout_name, var_name in step._layout_updates.items():
            if isinstance(var_name, (str, torch.Value)):
                var = context.variables[var_name]
            elif isinstance(var_name, torch.Tensor):
                var = var_name
            else:
                raise TypeError(f"layout_init variable must be either a string (variable name) or a torch.Tensor, not {type(var_name).__name__}.")
            self.layouts[layout_name].update_tensor(var)
        
        # allocate buffers
        for buffer_name in step._buffer_allocs:
            try:
                self._buffers[buffer_name].allocate()
            except Exception as e:
                logger.error(f"Failed to allocate buffer {buffer_name}: {e}")
                raise e

        # load to buffers
        for buffer_name, (layout_name, cursor) in step._buffer_loads.items():
            if cursor is None:
                var = self.layouts[layout_name].restore_tensor()
            else:
                var = self.layouts[layout_name][cursor]
                
            self._buffers[buffer_name].update(var)
            
        # dispatch ops
        for slot_id, ops in step._op_signatures.items():
            for op, args, kwargs in ops:
                op: RuntimeOperator = op(*args, **kwargs)
                if not isinstance(op, RuntimeOperator):
                    raise TypeError("Only RuntimeOperator instances can be dispatched in CompilationRecipe.")
                op.dispatch(slot_id=slot_id)
                
        # run ops
        context.device.run_kernels(sync_target_cores=context.core_ids)
        
        # store from buffers
        for buffer_name, (layout_name, cursor) in step._buffer_stores.items():
            if cursor is None:
                self.layouts[layout_name].update_tensor(self._buffers[buffer_name].restore())
            else:
                self.layouts[layout_name][cursor] = self._buffers[buffer_name].restore()
                
        # deallocate buffers
        for buffer_name in step._buffer_deallocs:
            self._buffers[buffer_name].deallocate()
                
        # restore layouts
        for layout_name, var_name in step._layout_restores.items():
            context.variables[var_name] = self.layouts[layout_name].restore_tensor()
            
    def run_all_steps(self):
        context: NetworkGraphContext = get_global_network_context()
        assert isinstance(context, NetworkGraphContext), "CompilationRecipe requires an active NetworkGraphContext."
        
        core_ids = context.core_ids
        npu_cores = [context.device.get_npu_core(core_id=core_id) for core_id in core_ids]
        
        if context.monitoring_window is not None:
            node_kind = self._node.kind()
            if node_kind == "prim::CallMethod":
                module = context.variables[self._node.inputsAt(0)]
                node_kind = f"compiled {type(module).__name__} module"
            else:
                node_kind = f"compiled {node_kind} operator"
                
            step_pbar_idx = context.monitoring_window.add_pbar(desc=f"{node_kind}")
            context.monitoring_window.pbar_handles[step_pbar_idx].update(len(self._steps), 0)
                
            logger.info(f"Running compiled recipe for node: {node_kind} with {len(self._steps)} steps.")
        else:
            step_pbar_idx = None
            
        logger.debug(f"total onc buffers: {sum(c.mem_handle.size for c in npu_cores)} bytes / empty onc buffers: {sum(c.mem_handle.empty_space() for c in npu_cores)} bytes.")
        
        for step_idx in range(len(self._steps)):
            self.run_step(step_idx=step_idx)
            
            if step_pbar_idx is not None:
                context.monitoring_window.pbar_handles[step_pbar_idx].update(len(self._steps), step_idx + 1)

        if step_pbar_idx is not None:
            context.monitoring_window.remove_pbar(step_pbar_idx)

    @property
    def node(self) -> torch.Node:
        return self._node
        
    @property
    def dims(self) -> TensorDimensionPool:
        return self._dims
    
    @property
    def layouts(self) -> TensorLayoutPool:
        return self._layouts
    
    @property
    def buffers(self) -> dict[str, MCA_TensorBuffer]:
        return self._buffers


class _VariableDict(dict):
    def __getitem__(self, key):
        if isinstance(key, torch.Value):
            key = key.debugName()
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, torch.Value):
            key = key.debugName()
        return super().__setitem__(key, value)

class NetworkGraphContext:
    def __init__(self, device: MCA_DeviceBase, core_ids: list[int]):
        self.device = device
        self.core_ids = core_ids
        
        self.variables: dict[str, Any] = _VariableDict()
        self.compile_recipes: dict[tuple[NetworkGraphEntryType, str], type] = {}
        
        self._monitoring_window: MonitoringWindow = None
        
    def __enter__(self):
        return self.open()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def open(self) -> 'NetworkGraphContext':
        global _global_network_context
        assert _global_network_context is None, "A HostContext is already active."
        _global_network_context = self
        return self
    
    def close(self):
        global _global_network_context
        assert _global_network_context is self, "Exiting a HostContext that is not active."
        _global_network_context = None

    @staticmethod
    def _get_primitive_name(name: str) -> str:
        if isinstance(name, torch.Node):
            name = name.kind()
        elif isinstance(name, torch.nn.Module):
            name = type(name).__name__
        elif isinstance(name, type):
            name = name.__name__
        return name
    
    @property
    def monitoring_window(self) -> MonitoringWindow:
        return self._monitoring_window

    def attach_monitoring_window(self):
        if get_global_monitoring_window() is None:
            raise Exception("No global MonitoringWindow found. Please set a global MonitoringWindow before attaching it to the NetworkGraphContext.")
        self._monitoring_window = get_global_monitoring_window()

    def detach_monitoring_window(self):
        self._monitoring_window = None

    def add_module_compilation_recipe(self, module_type: type, recipe: type):
        if not issubclass(module_type, torch.nn.Module):
            raise TypeError("module_type must be a subclass of torch.nn.Module.")
        if not issubclass(recipe, CompilationRecipe):
            raise TypeError("recipe must be a subclass of ModuleCompilationRecipe.")

        key = (NetworkGraphEntryType.COMPILED_GRAPH, module_type.__name__)
        self.compile_recipes[key] = recipe
        
    def add_nonprim_compilation_recipe(self, method_domain: str, method_name: str, recipe: CompilationRecipe):
        raise NotImplementedError("non-prim compilation recipe is not implemented yet")  # TODO: implement this method
    
    def search_module_compilation_recipe(self, module_type: type) -> type | None:
        key = (NetworkGraphEntryType.COMPILED_GRAPH, module_type.__name__)
        return self.compile_recipes.get(key, None)

    def search_nonprim_compilation_recipe(self, method_domain: str, method_name: str) -> CompilationRecipe | None:
        return None  # TODO: implement this method

    def run_prim_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False):
        node = entry.node   
        node_domain, node_action = node.kind().split("::")
        
        if node_domain != "prim":
            raise Exception(f"only 'prim' domain is supported by the session\nexception occurred for the node: {node.kind()}")
        
        attrs = {attr_name: _get_attr_from_node(node, attr_name) for attr_name in node.attributeNames()}
        
        for o in node.outputs():
            if o.type().kind() == 'NoneType':
                self.variables[o] = None

        if node_action == "CallMethod":
            submodule = self.variables[list(node.inputs())[0]]
            args = [self.variables[i] for i in list(node.inputs())[1:]]
            method_name = attrs['name']
            
            if isinstance(submodule, torch.nn.Module) and "forward" in method_name:
                if entry.node_type == NetworkGraphEntryType.COMPILED_GRAPH:
                    outputs = submodule(*args)
                elif entry.node_type == NetworkGraphEntryType.GRAPH:
                    outputs = entry.subgraph.run_graph(*args, trace_mode=trace_mode)
            else:
                method = getattr(submodule, method_name)
                outputs = method(*args)
            
            if len(list(node.outputs())) == 1:
                self.variables[node.output()] = outputs
            else:
                for idx, o in enumerate(node.outputs()):
                    self.variables[o] = outputs[idx]
        elif node_action == "GetAttr":
            self.variables[node.output()] = getattr(self.variables[node.input()], attrs['name'])
        elif node_action == "Constant":
            self.variables[node.output()] = attrs.get('value', None)
        elif node_action == "ListConstruct":
            self.variables[node.output()] = [self.variables[i] for i in node.inputs()]
        elif node_action == "TupleConstruct":
            self.variables[node.output()] = tuple([self.variables[i] for i in node.inputs()])
        elif node_action == "NumToTensor":
            self.variables[node.output()] = torch.tensor(self.variables[node.inputsAt(0)])
        else:
            raise Exception(f"action '{node_action}' is not supported by the session in 'prim' domain\nexception occurred for the node: {node.kind()}")
        
    def run_nonprim_entry(self, entry: NetworkGraphEntry):
        node = entry.node
        node_domain, node_action = node.kind().split("::")
        
        args = [self.variables[i] for i in node.inputs()]
        method, pp_args, pp_kwargs = _find_nonprim_method(node_domain, node_action, args=args)
        
        if method is None:
            raise Exception(f"method '{node_action}' in domain '{node_domain}' not found\nexception occurred for the node: {node.kind()}")
        
        outputs = method(*pp_args, **pp_kwargs)
        
        if len(list(node.outputs())) == 1:
            self.variables[node.output()] = outputs
        else:
            for idx, o in enumerate(node.outputs()):
                self.variables[o] = outputs[idx]
              
    def run_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False):
        # STEP 1: pre-run the entry to prepare inputs/outputs
        if entry.node_type.is_prim:
            self.run_prim_entry(entry, trace_mode=trace_mode)
        elif entry.node_type.is_nonprim:
            self.run_nonprim_entry(entry)
        else:
            raise Exception(f"unsupported entry type: {entry.node_type}")
        
        # STEP 2: run the entry with compiled runtime (if necessary)
        if entry.node_type.is_compiled_target:
            if not entry.is_compiled:
                entry.compile()
            if not trace_mode:
                entry.recipe.run_all_steps()
    
class NetworkGraph:
    def __init__(self, module: torch.nn.Module, graph_ivars: list[torch.Value], graph_ovars: list[torch.Value], graph_nodes: list[torch.Node], entries: list[NetworkGraphEntry]):
        self.module = module
        self.graph_ivars = graph_ivars
        self.graph_ovars = graph_ovars
        self.graph_nodes = graph_nodes
        self.entries = entries
    
    @classmethod
    def from_trace(cls, module: torch.nn.Module, *dummy_inputs, disable_lowering: bool=False, disable_toposort: bool=False):
        logger.debug(f"tracing module: {type(module).__name__} with dummy inputs: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in dummy_inputs]}")
        
        warnings.filterwarnings("ignore", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: suppress quantized model warning
        
        # STEP 0: get global network context
        network_context = get_global_network_context()
        assert network_context is not None, "No active NetworkGraphContext found. Please use 'with NetworkGraphContext(...) as context:' to create a context. Otherwise, open context manually using 'context.open()'." 
        
        # STEP 1: create traced graph (torch.jit.trace)
        module = module
        
        traced_module = torch.jit.trace(module, *dummy_inputs)
        traced_graph = traced_module.graph

        # STEP 2: parse traced graph to get input/output variables and nodes
        graph_ivars = list(traced_graph.inputs())
        graph_ovars = list(traced_graph.outputs())
        graph_nodes = list(traced_graph.nodes())
        
        entries: list[NetworkGraphEntry] = []
        
        # STEP 3-1: initialize input variables
        network_context.variables[graph_ivars[0]] = module
        for idx, ivar in enumerate(graph_ivars[1:]):
            network_context.variables[ivar] = dummy_inputs[idx]

        # STEP 3-2: construct entries and initialize intermediate variables
        for node in graph_nodes:
            node_domain, node_action = node.kind().split("::")
            
            # STEP 3-2-1: prim nodes
            if node_domain == "prim":
                entry = NetworkGraphEntry(NetworkGraphEntryType.PRIM, node)
                attrs = {attr_name: _get_attr_from_node(node, attr_name) for attr_name in node.attributeNames()}

                if node_action == "CallMethod":
                    submodule = network_context.variables[list(node.inputs())[0]]
                    method_name = attrs['name']
                    
                    # check if the submodule is a torch.nn.Module and method is "forward" -> subgraph or compiled graph
                    if isinstance(submodule, torch.nn.Module) and "forward" in method_name:
                        recipe = network_context.search_module_compilation_recipe(type(submodule))
                        
                        if recipe is not None:
                            entry = NetworkGraphEntry(NetworkGraphEntryType.COMPILED_GRAPH, node, submodule=submodule, recipe=recipe)
                        else:
                            args = [network_context.variables[i] for i in list(node.inputs())[1:]]
                            subgraph = NetworkGraph.from_trace(submodule, *args, disable_lowering=True, disable_toposort=True)
                            entry = NetworkGraphEntry(NetworkGraphEntryType.GRAPH, node, subgraph=subgraph)
                    
                    # otherwise, the entry remains as prim::CallMethod (not compiled as runtime kernel or operator)
                    else:
                        entry = NetworkGraphEntry(NetworkGraphEntryType.PRIM, node)
            
            # STEP 3-2-2: nonprim nodes (aten::, quantized::, ...)
            else:
                recipe = network_context.search_nonprim_compilation_recipe(node_domain, node_action)
                
                if recipe is not None:
                    entry = NetworkGraphEntry(NetworkGraphEntryType.COMPILED_NONPRIM, node, recipe=recipe)
                else:
                    entry = NetworkGraphEntry(NetworkGraphEntryType.NONPRIM, node)
            
            entries.append(entry)
            network_context.run_entry(entry, trace_mode=True)
            
        warnings.filterwarnings("default", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("default", category=FutureWarning)  # TODO: suppress quantized model warning 
            
        graph = cls(module, graph_ivars, graph_ovars, graph_nodes, entries)
        
        if not disable_lowering:
            graph.lowering()
        if not disable_toposort:
            graph.topological_sort()
            
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

    def lowering(self, context_name=None):
        logger.debug(f"lowering graph of module: {type(self.module).__name__}")
        
        network_context = get_global_network_context()
        assert network_context is not None, "No active NetworkGraphContext found. Please use 'with NetworkGraphContext(...) as context:' to create a context. Otherwise, open context manually using 'context.open()'." 
        
        if context_name is None:
            context_name = self.module.__class__.__name__
            
        # STEP 1: rename all the variables
        var_rename_map = {vname: f"{context_name}::{vname}" for vname in network_context.variables.keys()}
        self.rename_vars(var_rename_map=var_rename_map)
        
        # STEP 2: lower subgraphs
        lowered_entries: list[NetworkGraphEntry] = []
        lowered_graph_nodes: list[torch.Node] = []
        
        for entry in self.entries:
            if entry.node_type == NetworkGraphEntryType.GRAPH:
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
                lowered_entries.extend(entry.subgraph.entries)
                lowered_graph_nodes.extend(entry.subgraph.graph_nodes)
            else:  
                lowered_entries.append(entry)
                lowered_graph_nodes.append(entry.node)
        
        self.entries = lowered_entries
        self.graph_nodes = lowered_graph_nodes

        return self
        
    def topological_sort(self):
        logger.debug(f"topological sorting graph of module: {type(self.module).__name__}")
        
        # STEP 1: build dependency graph between entries
        entry_dept_graph: Dict[int, Set[int]] = {idx: set() for idx in range(len(self.entries))}
        
        for entry_idx, entry in enumerate(self.entries[:-1]):
            ovars = list(o.debugName() for o in entry.node.outputs())
            
            for check_idx, check_entry in enumerate(self.entries[entry_idx+1:], start=entry_idx+1):
                ivars = list(i.debugName() for i in check_entry.node.inputs())
                
                if any(o in ivars for o in ovars):
                    entry_dept_graph[entry_idx].add(check_idx)
        
        # STEP 2: topological sort            
        sorted_entry_indices = _kahn_topological_sort(entry_dept_graph)
        self.entries = [self.entries[idx] for idx in sorted_entry_indices]
        
        return self
        
    def get_outputs(self):
        network_context = get_global_network_context()
        assert network_context is not None, "No active NetworkGraphContext found. Please use 'with NetworkGraphContext(...) as context:' to create a context. Otherwise, open context manually using 'context.open()'."
        
        if len(self.graph_ovars) == 1:
            return network_context.variables[self.graph_ovars[0]]
        return [network_context.variables[o] for o in self.graph_ovars]

    def run_graph(self, *dummy_inputs, trace_mode: bool=False):
        network_context = get_global_network_context()
        assert network_context is not None, "No active NetworkGraphContext found. Please use 'with NetworkGraphContext(...) as context:' to create a context. Otherwise, open context manually using 'context.open()'."
        
        network_context.variables[self.graph_ivars[0]] = self.module
        for idx, ivar in enumerate(self.graph_ivars[1:]):
            network_context.variables[ivar] = dummy_inputs[idx]
        
        for entry in self.entries:
            try:
                network_context.run_entry(entry, trace_mode=trace_mode)
            except Exception as e:
                logger.error(f"exception occurred while running the graph with node: {entry.node}")
                raise Exception(f"exception occurred while running the entry: {entry}\n{e}") from e
        
        return self.get_outputs()
    
    def print_graph(self, indent: int=0):
        print(" " * indent + f"OPEN_GRAPH[type={type(self.module).__name__}]({', '.join(list('%'+i.debugName() for i in self.graph_ivars))}):")
        for entry in self.entries:
            print(" " * (indent + 2) + str(entry))
            
            if entry.node_type == NetworkGraphEntryType.GRAPH:
                entry.subgraph.print_graph(indent=indent+2)
        print(" " * (indent + 2) + f"return {', '.join(list('%'+o.debugName() for o in self.graph_ovars))}")
    
    
if __name__ == "__main__":
    class CNN(torch.nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, padding=1, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 7 * 7, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x
    
    logger.set_print_options(log_level=LogLevel.DEBUG)
    torch.set_printoptions(precision=4, linewidth=1024)

    network_context = NetworkGraphContext(device=None, core_ids=[0])  # Dummy host context for testing

    model = CNN().eval()
    
    with torch.no_grad():
        with network_context:
            print(f"=== Trace Graph ===")
            dummy_input = torch.randn(1, 1, 28, 28)
            graph = NetworkGraph.from_trace(model, dummy_input)
            graph.print_graph()
        
            dummy_input = torch.randn(1, 1, 28, 28)
            reference = model(dummy_input)
            simulated = graph.run_graph(dummy_input)
            
            print(f"=== Check Integrity ===")
            print(f"reference: {reference}")
            print(f"simulated: {simulated}")