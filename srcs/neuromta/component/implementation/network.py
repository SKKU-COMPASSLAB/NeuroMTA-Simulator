import functools
# import multiprocessing
import os
import warnings
import enum
import torch

from collections import deque, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Set, Callable

from neuromta.framework import *
from neuromta.component.context.global_context import GlobalContextMemType
from neuromta.component.implementation.hardware import MCA_DeviceBase, MCA_CoreGroup, MCA_L1MemorySpace, MCA_MainMemorySpace, MCA_MemorySpace
from neuromta.component.implementation.tensor_buffer import MCA_TensorBuffer
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *
from neuromta.component.utils.profiler import ProfilerTemplate, GroupedProfilerTemplate
from neuromta.component.utils.profiler_base import ProfilerFileSaverHub
from torch.multiprocessing import Pool



__all__ = [   
    "NetworkGraphEntry",
    "CompiledGraphEntry",
    "NetworkGraphContext",
    "NetworkRecipe",
    "MCA_CompiledNetworkGraph",
]


def _cast_to_jit_type(arg: Any, target_type: torch._C.Type) -> Any:
    if target_type.isSubtypeOf(torch._C.BoolType.get()):
        return bool(arg)
    elif target_type.isSubtypeOf(torch._C.IntType.get()):
        return int(arg)
    elif target_type.isSubtypeOf(torch._C.FloatType.get()):
        return float(arg)
    elif target_type.isSubtypeOf(torch._C.StringType.get()):
        return str(arg)
    elif target_type.isSubtypeOf(torch._C.TensorType.get()):
        return torch.tensor(arg) if not isinstance(arg, torch.Tensor) else arg
    elif target_type.isSubtypeOf(torch._C.DeviceObjType.get()):
        return torch.device(arg) if not isinstance(arg, torch.device) else arg
    return arg  # fallback: return as is if no specific conversion rule applies

def _get_attr_from_node(node: torch.Node, attr_name: str) -> any:
    for attr_types in ['f', 'fs', 'c', 's', 'ss', 'i', 'g', 'gs', 'ival', 't', 'ts', 'ty', 'tys']:
        try:
            return getattr(node, attr_types)(attr_name)
        except:
            pass
    
    return None

def resolve_args_kwargs(schema: torch._C.FunctionSchema, raw_inputs: list):
    args = []
    kwargs = {}
    
    for i, arg in enumerate(schema.arguments):
        if i < len(raw_inputs):
            val = raw_inputs[i]
        else:
            if arg.has_default_value():
                val = arg.default_value
            else:
                raise ValueError(f"No value provided for argument '{arg.name}'.")
                
        if arg.kwarg_only:
            kwargs[arg.name] = val
        else:
            args.append(val)
            
    return args, kwargs

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
        COMPILED        = enum.auto()  # compiled entry (entry compiled as runtime kernel by MCA_OperatorGraphCompiler)
        
    def __init__(self, node_type: Type, node: torch.Node, **kwargs):
        self.node_type = node_type
        self.node = node
        
        self.subgraph: MCA_CompiledNetworkGraph = kwargs.get("subgraph", None)
        self.submodule: torch.nn.Module = kwargs.get("submodule", None)
        
    def create_compiled_entry(self) -> 'CompiledGraphEntry':
        compiled_entry = CompiledGraphEntry(parent_entry=self)
        return compiled_entry
    
    @property
    def is_prim(self) -> bool:
        return self.node_type in (NetworkGraphEntry.Type.PRIM, NetworkGraphEntry.Type.GRAPH)
    
    @property
    def is_nonprim(self) -> bool:
        return self.node_type == NetworkGraphEntry.Type.NONPRIM
    
    @property
    def is_compiled(self) -> bool:
        return self.node_type == NetworkGraphEntry.Type.COMPILED
    
    @property
    def is_subgraph_available(self) -> bool:
        return self.subgraph is not None
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        node_kind = self.node.kind()
        ivars = list('%'+i.debugName() for i in self.node.inputs())
        ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r = f"{self.node_type.name}({node_kind}"
        r += f", inputs={ivars}, outputs={ovars}"
        if self.is_subgraph_available:
            r += f", graph={type(self.subgraph.module).__name__}"
        return r + ")"
    
    
class CompiledGraphEntry(NetworkGraphEntry):
    class BufferType(enum.Enum):
        PARAM  = enum.auto()
        INPUT  = enum.auto()
        OUTPUT = enum.auto()
        
    class TensorProcessing:
        def __init__(self, ptype: str, *args, **kwargs):
            self.ptype = ptype
            self.args = args
            self.kwargs = kwargs
            
        def apply(self, tensor: torch.Tensor) -> torch.Tensor:
            if self.ptype == 'permute':
                return tensor.permute(*self.args, **self.kwargs)
            elif self.ptype == 'to':
                return tensor.to(*self.args, **self.kwargs)
            else:
                raise NotImplementedError(f"unsupported tensor processing type: {self.ptype}")
    
    class Element:
        def __init__(self, etype: type):
            self.etype = etype
            
        def __repr__(self):
            return f"Element[type={self.etype}]"
    
    class BufferElement(Element):
        def __init__(
            self, 
            shape: tuple[int], 
            dtype: torch.dtype, 
            btype: 'CompiledGraphEntry.BufferType',
            preprocessings: list['CompiledGraphEntry.TensorProcessing']=None,
            postprocessings: list['CompiledGraphEntry.TensorProcessing']=None,
        ):
            super().__init__(etype=torch.Tensor)
            
            self.shape = shape
            self.dtype = dtype
            self.btype = btype
            
            self.preprocessings = preprocessings if preprocessings is not None else []
            self.postprocessings = postprocessings if postprocessings is not None else []
            
        def create(self, mem_space: MCA_MemorySpace) -> MCA_TensorBuffer:
            return MCA_TensorBuffer(mem_space=mem_space, shape=self.shape, dtype=self.dtype).allocate()
        
        def check_memory_requirement(self, mem_space: MCA_MemorySpace) -> bool:
            return MCA_TensorBuffer.check_memory_requirement(mem_space=mem_space, shape=self.shape, dtype=self.dtype)
            
        def permute(self, *order: int) -> 'CompiledGraphEntry.BufferElement':
            self.preprocessings.append(CompiledGraphEntry.TensorProcessing('permute', *order))
            self.postprocessings.insert(0, CompiledGraphEntry.TensorProcessing('permute', tuple(order.index(i) for i in range(len(order)))))
            return self

        def set_orig_dtype(self, orig_dtype: torch.dtype):
            if self.dtype != orig_dtype:
                self.preprocessings.append(CompiledGraphEntry.TensorProcessing('to', dtype=self.dtype))
                self.postprocessings.insert(0, CompiledGraphEntry.TensorProcessing('to', dtype=orig_dtype))
            return self
        
        def compatible_with(self, other: 'CompiledGraphEntry.BufferElement | MCA_TensorBuffer') -> bool:
            if isinstance(other, MCA_TensorBuffer):
                return self.shape == other.shape and self.dtype == other.dtype
            elif isinstance(other, CompiledGraphEntry.BufferElement):
                return self.shape == other.shape and self.dtype == other.dtype
            else:
                return False
        
        def __repr__(self):
            return f"BufferElement[shape={self.shape}, dtype={self.dtype}, btype={self.btype.name}]"
    
    class ContextEntry:
        def __init__(self, src: str, elem: 'CompiledGraphEntry.Element'):
            self.src = src
            self.elem = elem
            
        @property
        def is_constant(self) -> bool:
            return self.src is None
            
        def __repr__(self):
            return f"ContextEntry(src={self.src}, elem={self.elem}, is_constant={self.is_constant})"
    
    def __init__(self, parent_entry: NetworkGraphEntry):
        super().__init__(NetworkGraphEntry.Type.COMPILED, parent_entry.node, subgraph=parent_entry.subgraph, submodule=parent_entry.submodule)
        
        self._parent_entry = parent_entry
        
        self._ctx_entries:   dict[str, CompiledGraphEntry.ContextEntry] = {}
        self._local_context: dict[str, Any] = {}
        self._op_method: Callable = None
        
    def set_op_method(self, method: Callable):
        if not mca_operator_method_check(method):
            raise TypeError(f"The op_method given to the CompiledGraphEntry must be decorated with @mca_operator_method. (function: {method.__name__})")
        self._op_method = method
        
    def add_value_context(self, name: str, src: str, value: Any):
        self._ctx_entries[name] = self.ContextEntry(
            src=src,
            elem=self.Element(etype=type(value))
        )
        
    def add_constant_context(self, name: str, value: Any):
        self._ctx_entries[name] = self.ContextEntry(
            src=None,
            elem=value
        )
        
    def add_param_buffer_context(self, name: str, src: str, shape: tuple[int], dtype: torch.dtype) -> 'CompiledGraphEntry.BufferElement':
        elem = self.BufferElement(shape=shape, dtype=dtype, btype=self.BufferType.PARAM)
        self._ctx_entries[name] = self.ContextEntry(src=src, elem=elem)
        return elem
        
    def add_input_buffer_context(self, name: str, src: str, shape: tuple[int], dtype: torch.dtype) -> 'CompiledGraphEntry.BufferElement':
        elem = self.BufferElement(shape=shape, dtype=dtype, btype=self.BufferType.INPUT)
        self._ctx_entries[name] = self.ContextEntry(src=src, elem=elem)
        return elem

    def add_output_buffer_context(self, name: str, src: str, shape: tuple[int], dtype: torch.dtype) -> 'CompiledGraphEntry.BufferElement':
        elem = self.BufferElement(shape=shape, dtype=dtype, btype=self.BufferType.OUTPUT)
        self._ctx_entries[name] = self.ContextEntry(src=src, elem=elem)
        return elem
    
    def override_buffer_context(self, name: str, buffer: MCA_TensorBuffer):
        self._local_context[name] = buffer
        
    def allocate_buffer_context(self, name: str, mem_space: MCA_MemorySpace) -> MCA_TensorBuffer:
        ctx_entry = self._ctx_entries[name]
        if isinstance(ctx_entry.elem, self.BufferElement):
            buffer: MCA_TensorBuffer = ctx_entry.elem.create(mem_space=mem_space)
            self._local_context[name] = buffer
        return buffer
                
    def load_local_context(self, graph_context: 'NetworkGraphContext'):
        for name, ctx_entry in self._ctx_entries.items():
            if ctx_entry.src in graph_context:
                graph_value = graph_context[ctx_entry.src]
            elif ctx_entry.is_constant:
                graph_value = ctx_entry.elem
            else:
                graph_value = None
                
            if isinstance(ctx_entry.elem, self.BufferElement) and isinstance(graph_value, torch.Tensor):
                # apply preprocessings
                for proc in ctx_entry.elem.preprocessings:
                    graph_value = proc.apply(graph_value)
                
                # update buffer
                buffer: MCA_TensorBuffer = self._local_context[name]
                buffer.update(graph_value)
            else:
                self._local_context[name] = graph_value
                    
    def store_local_context(self, graph_context: 'NetworkGraphContext'):
        for name, ctx_entry in self._ctx_entries.items():
            if (not ctx_entry.is_constant) and (ctx_entry.src in graph_context):
                graph_value = graph_context[ctx_entry.src]
                if isinstance(ctx_entry.elem, self.BufferElement) and isinstance(graph_value, torch.Tensor):
                    # restore buffer
                    buffer: MCA_TensorBuffer = self._local_context[name]
                    buffer_value = buffer.restore()
                    
                    # apply postprocessings
                    for proc in ctx_entry.elem.postprocessings:
                        buffer_value = proc.apply(buffer_value)
                    
                    graph_context[ctx_entry.src] = buffer_value
                else:
                    graph_context[ctx_entry.src] = self._local_context[name]
                    
    def pcc_check(self, graph_context: 'NetworkGraphContext', atol: float=1e-4, rtol: float=1e-4) -> bool | float:
        graph_context.run_entry(self._parent_entry)   # ensure the graph context is updated with the latest value before PCC check
        
        for name, ctx_entry in self._ctx_entries.items():
            if ctx_entry.is_constant or ctx_entry.src not in graph_context:
                continue
            if not isinstance(ctx_entry.elem, self.BufferElement):
                continue
            if ctx_entry.elem.btype != self.BufferType.OUTPUT:
                continue
            
            ref = graph_context[ctx_entry.src]
            
            if isinstance(ctx_entry.elem, self.BufferElement) and isinstance(ref, torch.Tensor):
                buffer: MCA_TensorBuffer = self._local_context[name]
                sim = buffer.restore()
                
                for proc in ctx_entry.elem.postprocessings:
                    sim = proc.apply(sim)
            else:
                sim = self._local_context[name]
            
            try:
                if isinstance(ref, torch.Tensor) and isinstance(sim, torch.Tensor):
                    ref = ref.cpu()
                    sim = sim.cpu()
                    
                    if not torch.allclose(ref, sim, atol=atol, rtol=rtol):
                        print(f"ref: {ref.flatten()[:10]}")
                        print(f"sim: {sim.flatten()[:10]}")
                        absolute_error = torch.abs(ref - sim)
                        relative_error = absolute_error / (torch.abs(ref) + atol)
                        return torch.mean(relative_error).item()
            except:
                logger.error(f"PCC check failed for context entry '{name}' (ref: {ref}, sim: {sim}, atol: {atol}, rtol: {rtol})")
                return False
        return True
                    
    def get_op_sig(self):
        return self._op_method(**self._local_context)
    
    @property
    def buffer_contexts(self) -> 'dict[str, CompiledGraphEntry.ContextEntry]':
        return {name: ctx_entry for name, ctx_entry in self._ctx_entries.items() if isinstance(ctx_entry.elem, self.BufferElement)}
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"CompiledGraphEntry[op_method={self._op_method.__name__ if self._op_method else None}, ctx_entries={self._ctx_entries}]"

class NetworkRecipe:
    def __init__(
        self,
        device: MCA_DeviceBase,
        core_groups: list[MCA_CoreGroup],
        main_space_size_per_channel: int,
        data_space_size_per_core: int,
        spad_space_size_per_core: int,
        context_buffer_slot_num: int=16,
        fifo_buffer_slot_num: int=16,
        temporal_reuse_target: MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL,
        spatial_reuse_target: MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN,
    ):
        self.main_space_size_per_channel = main_space_size_per_channel
        self.data_space_size_per_core = data_space_size_per_core
        
        self._compile_recipe = MCA_OperatorGraphCompiler.CompileRecipe(
            device=device,
            core_groups=core_groups,
            spad_space_size_per_core=spad_space_size_per_core,
            context_buffer_slot_num=context_buffer_slot_num,
            fifo_buffer_slot_num=fifo_buffer_slot_num,
            temporal_reuse_target=temporal_reuse_target,
            spatial_reuse_target=spatial_reuse_target,
        )
        
    @property
    def spad_space_size_per_core(self) -> int:
        return self._compile_recipe.spad_space_size_per_core
    
    @property
    def device(self) -> MCA_DeviceBase:
        return self._compile_recipe.device
    
    def get_compile_method(self, alias_name: str | torch.Node) -> Callable:
        if isinstance(alias_name, torch.Node):
            alias_name = alias_name.kind()
        
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "alias_names") and alias_name in attr.alias_names:
                return attr
        return None
    
    def dispatch_entry(self, entry: CompiledGraphEntry):
        compiler = MCA_OperatorGraphCompiler()
        compiler.add_op(entry.get_op_sig())
        program = compiler.compile(self._compile_recipe)
        program.dispatch()
    
    @staticmethod
    def recipe(*alias_names: str):
        def _decorator(func: Callable[[CompiledGraphEntry, list[Any], list[Any]], CompiledGraphEntry]):
            @functools.wraps(func)
            def _wrapper(self, graph_context: 'NetworkGraphContext', entry: NetworkGraphEntry) -> CompiledGraphEntry:
                compiled_entry = entry.create_compiled_entry()
                i_args = [graph_context[i] for i in entry.node.inputs()]
                o_args = [graph_context[o] for o in entry.node.outputs()]
                ret = func(self, compiled_entry, i_args, o_args)
                if ret is not compiled_entry:
                    raise Exception(f"the compile method must return the given CompiledGraphEntry instance (alias names: {alias_names})")
                return ret
            _wrapper.alias_names = list(alias_names)
            return _wrapper
        return _decorator
    
    @property
    def compile_recipe(self) -> MCA_OperatorGraphCompiler.CompileRecipe:
        return self._compile_recipe
    

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
    
    def __getattr__(self, name):
        if isinstance(name, torch.Value):
            name = name.debugName()
        return super().__getattr__(name)
    
    def __getitem__(self, key):
        if isinstance(key, torch.Value):
            key = key.debugName()
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, torch.Value):
            key = key.debugName()
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

            if trace_mode:
                args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
            
            if entry.node_type == NetworkGraphEntry.Type.GRAPH:
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
            self[node.output()] = _cast_to_jit_type(attrs.get('value', None), node.output().type())
        elif node_action == "ListConstruct":
            self[node.output()] = [self[i] for i in node.inputs()]
        elif node_action == "TupleConstruct":
            self[node.output()] = tuple([self[i] for i in node.inputs()])
        elif node_action == "NumToTensor":
            self[node.output()] = torch.tensor(self[node.inputsAt(0)])
        elif node_action == "TupleUnpack":
            for idx, o in enumerate(node.outputs()):
                self[o] = self[node.input()][idx]
        else:
            logger.warning(f"{node}")
            raise Exception(f"action '{node_action}' is not supported by the session in 'prim' domain\nexception occurred for the node: {node.kind()}")
        
    def run_nonprim_entry(self, entry: NetworkGraphEntry):
        node = entry.node
        schema = torch._C.parse_schema(node.schema())
        
        _args, _kwargs = resolve_args_kwargs(schema=schema, raw_inputs=[self[i] for i in node.inputs()])
        _methods = torch._C._jit_get_operation(node.kind())
        _method_found = False
        
        for method in _methods:
            try:
                outputs = method(*_args, **_kwargs)
                _method_found = True
                break
            except Exception as e:
                pass

        if not _method_found:
            raise Exception(f"No suitable method found for non-primitive node: {node.kind()}")

        if len(list(node.outputs())) == 1:
            self[node.output()] = outputs.clone() if isinstance(outputs, torch.Tensor) else outputs
        else:
            for idx, o in enumerate(node.outputs()):
                self[o] = outputs[idx].clone() if isinstance(outputs[idx], torch.Tensor) else outputs[idx]
                
    def run_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False):
        # STEP 1: pre-run the entry to prepare inputs/outputs
        if entry.is_prim:
            self.run_prim_entry(entry, trace_mode=trace_mode)
        elif entry.is_nonprim:
            self.run_nonprim_entry(entry)
        else:
            raise Exception(f"unsupported entry type: {entry.node_type}")
    

class MCA_CompiledNetworkGraph:
    def __init__(
        self, 
        module: torch.nn.Module, 
        graph_ivars: list[torch.Value], 
        graph_ovars: list[torch.Value], 
        graph_nodes: list[torch.Node], 
        graph_context: NetworkGraphContext, 
        graph_entries: list[NetworkGraphEntry],
        graph_recipe: NetworkRecipe,
    ):
        self.module = module
        self.graph_ivars = graph_ivars
        self.graph_ovars = graph_ovars
        self.graph_nodes = graph_nodes
        self.graph_context = graph_context
        self.graph_entries = graph_entries
        self.graph_recipe = graph_recipe
        
        self.grouped_compiled_entries: list[list[CompiledGraphEntry]] = []
        
        self._lowered = False
    
    @classmethod
    def from_trace(
        cls, 
        module: torch.nn.Module,
        graph_recipe: NetworkRecipe, 
        *dummy_inputs: torch.Tensor,
        disable_lowering: bool=False, disable_toposort: bool=False, disable_compile: bool=False,
    ):
        logger.info(f"tracing module: {type(module).__name__} with dummy inputs: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in dummy_inputs]}")
        
        warnings.filterwarnings("ignore", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: suppress quantized model warning
        
        # STEP 1: create traced graph (torch.jit.trace)
        module = module.eval()
        
        graph_context = NetworkGraphContext()
        
        traced_module = torch.jit.trace(module, dummy_inputs)
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
                        subgraph = MCA_CompiledNetworkGraph.from_trace(submodule, graph_recipe, *args, disable_lowering=True, disable_toposort=True, disable_compile=True)
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
            graph.run_graph(*dummy_inputs, trace_mode=True)   # pre-run the graph to prepare runtime parameters for compilation
            graph.compile()
                    
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
        logger.info(f"lowering graph of module: {type(self.module).__name__}")
        
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
        logger.info(f"topological sorting graph of module: {type(self.module).__name__}")
        
        for entry in self.graph_entries:
            if entry.is_subgraph_available:
                entry.subgraph.topological_sort()
        
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
    
    def _compile_grouped_entries(self, entries: list[NetworkGraphEntry], main_mem_space: MCA_MainMemorySpace) -> list[list[CompiledGraphEntry]]:
        # STEP 1: create double buffering L1 data memory spaces  
        device = self.graph_recipe.device
        data_mem_space_size = self.graph_recipe.data_space_size_per_core
        core_group = self.graph_recipe.compile_recipe.global_core_group
        
        data_mem_pp_space = [device.create_l1_mem_space(data_mem_space_size // 2, core_group) for _ in range(2)]
        data_mem_pp_buffers = [{} for _ in range(2)]
        data_mem_pp_idx = 0
        main_mem_buffers = {}
        
        def switch_data_mem_pp():
            nonlocal data_mem_pp_idx
            data_mem_pp_idx = 1 - data_mem_pp_idx
            data_mem_pp_buffers[data_mem_pp_idx].clear()
            data_mem_pp_space[data_mem_pp_idx].remove()
            data_mem_pp_space[data_mem_pp_idx] = device.create_l1_mem_space(data_mem_space_size // 2, core_group)
        
        # STEP 2: compile entries and allocate buffers
        grouped_compiled_entries: list[list[CompiledGraphEntry]] = [[]]
        
        for entry in entries:
            compile_method = self.graph_recipe.get_compile_method(entry.node)
            
            if compile_method is None:
                raise Exception(f"compile method not found for the node: {entry.node}\nmake sure to define a compile method with @NetworkRecipe.recipe decorator for the node")
            
            compiled_entry: CompiledGraphEntry = compile_method(self.graph_context, entry)
            
            if not isinstance(compiled_entry, CompiledGraphEntry):
                raise Exception(f"compile method must return an instance of CompiledGraphEntry\nexception occurred for the node: {entry.node}")
            
            for buf_name, buf_ctx in compiled_entry.buffer_contexts.items():
                elem: CompiledGraphEntry.BufferElement = buf_ctx.elem

                if elem.btype == CompiledGraphEntry.BufferType.PARAM:
                    if not elem.check_memory_requirement(main_mem_space):
                        raise Exception(f"memory requirement for the parameter buffer context '{buf_name}' in the node: {entry.node} exceeds the main memory space size")
                    
                    buffer = elem.create(main_mem_space)
                    compiled_entry.override_buffer_context(buf_name, buffer)
                    
                elif elem.btype in (CompiledGraphEntry.BufferType.INPUT, CompiledGraphEntry.BufferType.OUTPUT):
                    if buf_ctx.src in data_mem_pp_buffers[data_mem_pp_idx] and elem.compatible_with(data_mem_pp_buffers[data_mem_pp_idx][buf_ctx.src]):
                        buffer = data_mem_pp_buffers[data_mem_pp_idx][buf_ctx.src]
                    elif buf_ctx.src in main_mem_buffers and elem.compatible_with(main_mem_buffers[buf_ctx.src]):
                        buffer = main_mem_buffers[buf_ctx.src]
                    else:
                        mem_space = data_mem_pp_space[data_mem_pp_idx]
                        if not elem.check_memory_requirement(mem_space):            # CASE 1: insufficient space in the current data memory space -> switch to the other data memory space
                            switch_data_mem_pp()
                            mem_space = data_mem_pp_space[data_mem_pp_idx]
                            
                            if not elem.check_memory_requirement(mem_space):        # CASE 2: insufficient space in the other data memory space -> try main memory space
                                mem_space = main_mem_space
                                if not elem.check_memory_requirement(mem_space):    # CASE 3: insufficient space in the main memory space -> raise exception
                                    raise Exception(f"memory requirement for the IO buffer context '{buf_name}' in the node: {entry.node} exceeds both the data memory space size and the main memory space size")
                    
                        buffer = elem.create(mem_space)
                        if mem_space.is_l1:
                            data_mem_pp_buffers[data_mem_pp_idx][buf_ctx.src] = buffer
                        else:
                            main_mem_buffers[buf_ctx.src] = buffer

                    compiled_entry.override_buffer_context(buf_name, buffer)
                    
                else:
                    raise Exception(f"unsupported buffer type: {elem.btype} in the node: {entry.node}")
                
            grouped_compiled_entries[-1].append(compiled_entry)
        
        # STEP 3: remove data memory spaces
        for data_mem_space in data_mem_pp_space:
            data_mem_space.remove()
            
        if len(grouped_compiled_entries[-1]) == 0:
            grouped_compiled_entries.pop()
        return grouped_compiled_entries

    def compile(self):
        device = self.graph_recipe.device
        main_mem_space_size = self.graph_recipe.main_space_size_per_channel
        main_mem_space = device.create_main_mem_space(main_mem_space_size)
        
        target_entries = []
        
        for entry in self.graph_entries:
            if self.graph_recipe.get_compile_method(entry.node) is not None:
                target_entries.append(entry)
            elif len(target_entries) > 0:
                self.grouped_compiled_entries.extend(self._compile_grouped_entries(target_entries, main_mem_space=main_mem_space))
                target_entries = []
                
        if len(target_entries) > 0:
            self.grouped_compiled_entries.extend(self._compile_grouped_entries(target_entries, main_mem_space=main_mem_space))
            target_entries = []

        return self
    
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
    
    def _run_compiled_entries_proc(
        self, sim_name: str, entries: list[CompiledGraphEntry], result_dict: dict, monitoring_window: bool=False, max_timestamp: int=None, 
        profilers: list[ProfilerTemplate | GroupedProfilerTemplate]=None, profiler_output_dir: str=None,
        pcc_check: bool=False, pcc_check_atol: float=1e-4, pcc_check_rtol: float=1e-4,
        execution_mode: SimulationMode | str=None,
    ):
        logger.info(f"running compiled entries for simulation: {sim_name}")
        
        # STEP 1: initialize environments (profilers, device reset, ...)
        if profiler_output_dir is not None:
            profilers = profilers if profilers is not None else []
            profiler_hub = ProfilerFileSaverHub(output_dir=os.path.join(profiler_output_dir, sim_name))
            profiler_hub.add_profilers(*profilers)
        
        device = self.graph_recipe.device
        core_groups = self.graph_recipe.compile_recipe.core_groups
        if execution_mode is not None:
            device.set_simulation_mode(execution_mode)
        if device.is_performance_mode and pcc_check:
            raise ValueError("PCC check requires correctness mode because performance mode does not compute tensor payloads.")
        
        device.reset_simulation()
        
        # STEP 2: load contexts
        for entry in entries:
            entry.load_local_context(self.graph_context)
            self.graph_recipe.dispatch_entry(entry)
        
        # STEP 3: run simulation and check PCC
        with MonitoringWindow(device, core_groups, sim_name=sim_name, disable=not monitoring_window):
            device.run_kernels(max_timestamp=max_timestamp)
            
        if pcc_check:
            for i, entry in enumerate(entries):
                pcc_flag = entry.pcc_check(self.graph_context, atol=pcc_check_atol, rtol=pcc_check_rtol)
                if pcc_flag is True:
                    logger.info(f"PCC check passed for the {sim_name}::{i} (entry_type: {entry.node.kind()})")
                else:
                    logger.warning(f"PCC check failed for the {sim_name}::{i} (entry_type: {entry.node.kind()}, MSE: {pcc_flag*100:.4f}%)")
        
        # STEP 5: save profiling results and simulation timestamp
        result_dict[sim_name] = {"timestamp": device.timestamp}
        
        if profiler_output_dir is not None:
            result_dict[sim_name]["profiles"] = [
                {
                    "metric_name": profiler.metric_name,
                    "metric_unit": profiler.metric_unit,
                    "profile": profiler.get_profile(),
                }
                for profiler in profilers
            ]
            profiler_hub.close()
    
    def run_compiled_graph(
        self, *dummy_inputs, 
        group_idx: int=None, entry_idx: int=None, monitoring_window: bool=False, max_timestamp: int=None,
        profilers: list[ProfilerTemplate | GroupedProfilerTemplate]=None, profiler_output_dir: str=None,
        pcc_check: bool=False, pcc_check_atol: float=1e-4, pcc_check_rtol: float=1e-4,
        execution_mode: SimulationMode | str=None,
    ) -> dict[str, Any]:
        if execution_mode is None:
            execution_mode = os.environ.get("NEUROMTA_SIMULATION_MODE", None)
        if execution_mode is not None:
            self.graph_recipe.device.set_simulation_mode(execution_mode)
        if self.graph_recipe.device.is_performance_mode and pcc_check:
            raise ValueError("PCC check requires correctness mode because performance mode does not compute tensor payloads.")
        
        self.run_graph(*dummy_inputs, trace_mode=True)
        
        result_dict = {}
        
        if group_idx is not None and entry_idx is not None:
            entry = self.grouped_compiled_entries[group_idx][entry_idx]
            self._run_compiled_entries_proc(
                f"{type(self.module).__name__}::group{group_idx}::entry{entry_idx}", [entry], 
                result_dict, monitoring_window, max_timestamp, profilers, profiler_output_dir, pcc_check, pcc_check_atol, pcc_check_rtol, execution_mode)
        elif group_idx is not None:
            entries = self.grouped_compiled_entries[group_idx]
            for entry_idx, entry in enumerate(entries):
                self._run_compiled_entries_proc(
                    f"{type(self.module).__name__}::group{group_idx}::entry{entry_idx}", [entry], 
                    result_dict, monitoring_window, max_timestamp, profilers, profiler_output_dir, pcc_check, pcc_check_atol, pcc_check_rtol, execution_mode)
        else:
            for group_idx, group in enumerate(self.grouped_compiled_entries):
                for entry_idx, entry in enumerate(group):
                    self._run_compiled_entries_proc(
                        f"{type(self.module).__name__}::group{group_idx}::entry{entry_idx}", [entry], 
                        result_dict, monitoring_window, max_timestamp, profilers, profiler_output_dir, pcc_check, pcc_check_atol, pcc_check_rtol, execution_mode)
            
        return dict(result_dict)
    
    def graph_summary(self) -> dict[str, Any]:
        summary = {
            "module": type(self.module).__name__,
            "input_vars": [ivar.debugName() for ivar in self.graph_ivars],
            "output_vars": [ovar.debugName() for ovar in self.graph_ovars],
            "entries": [
                {
                    "node": entry.node.kind(),
                    "inputs": [i.debugName() for i in entry.node.inputs()],
                    "outputs": [o.debugName() for o in entry.node.outputs()],
                    "is_subgraph_available": entry.is_subgraph_available,
                    "subgraph": entry.subgraph.graph_summary() if entry.is_subgraph_available else None,
                }
                for entry in self.graph_entries
            ]
        }
        return summary

    def compile_summary(self) -> dict[str, Any]:
        summary = {
            "module": type(self.module).__name__,
            "grouped_entries": []
        }
        for group_idx, group in enumerate(self.grouped_compiled_entries):
            group_summary = []
            for entry_idx, entry in enumerate(group):
                entry_summary = {
                    "node": entry.node.kind(),
                    "op_method": entry._op_method.__name__ if entry._op_method else None,
                    "local_context": {name: {"src": str(ctx_entry.src), "elem": ctx_entry.elem.__repr__()} for name, ctx_entry in entry._ctx_entries.items()},
                }
                group_summary.append(entry_summary)
            summary["grouped_entries"].append(group_summary)
        return summary
    
    def print_graph(self, indent: int=0):
        print(" " * indent + f"OPEN_GRAPH[type={type(self.module).__name__}]({', '.join(list('%'+i.debugName() for i in self.graph_ivars))}):")
        for entry in self.graph_entries:
            print(" " * (indent + 2) + str(entry))
            if entry.is_subgraph_available:
                entry.subgraph.print_graph(indent=indent+2)
        print(" " * (indent + 2) + f"return {', '.join(list('%'+o.debugName() for o in self.graph_ovars))}")
    
    def print_compile_summary(self):
        summary = self.compile_summary()
        print(f"COMPILE_SUMMARY for graph of module: {summary['module']}")
        for group_idx, group in enumerate(summary['grouped_entries']):
            print(f"  GROUP {group_idx}:")
            for entry_idx, entry in enumerate(group):
                print(f"    ENTRY {entry_idx}: node={entry['node']}, op_method={entry['op_method']}")
                for ctx_name, ctx_info in entry['local_context'].items():
                    print(f"      CONTEXT '{ctx_name:<7s}': src={ctx_info['src']}, elem={ctx_info['elem']}")
