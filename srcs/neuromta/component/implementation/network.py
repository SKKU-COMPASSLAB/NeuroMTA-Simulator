import abc
import functools
import math
import warnings
import enum
import torch

from collections import deque, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Callable

from neuromta.framework import *
from neuromta.component.context.global_context import GlobalContextMemType
from neuromta.component.implementation.hardware import MCA_DeviceBase, MCA_CoreGroup, MCA_L1MemorySpace, MCA_MainMemorySpace, MCA_MemorySpace
from neuromta.component.implementation.tensor_buffer import MCA_TensorBuffer
from neuromta.component.implementation.mapping import *
from neuromta.component.implementation.operator import *
# from multiprocessing import Pool
from torch.multiprocessing import Pool



__all__ = [
    "Placeholder",
    
    "NetworkGraphEntry",
    "NetworkGraphCompiledEntry",
    "NetworkGraphEntryCompileTarget",
    "NetworkGraphContext",
    "MCA_CompiledNetworkGraph",
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
        COMPILED        = enum.auto()  # pipelined compiled graph
        
    def __init__(self, node_type: Type, node: torch.Node, **kwargs):
        self.node_type = node_type
        self.node = node
        
        self.subgraph: MCA_CompiledNetworkGraph = kwargs.get("subgraph", None)
        self.submodule: torch.nn.Module = kwargs.get("submodule", None)
        
        self.is_compilation_target: bool = False
        
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
        if self.is_compiled:
            # node_kind = "nmta::compiled"
            # ivars = "COMPILED_INPUTS"
            # ovars = "COMPILED_OUTPUTS"
            return "COMPILED"
        else:
            node_kind = self.node.kind()
            ivars = list('%'+i.debugName() for i in self.node.inputs())
            ovars = list('%'+i.debugName() for i in self.node.outputs())
        
        r = f"{self.node_type.name}({node_kind}"
        r += f", inputs={ivars}, outputs={ovars}"
        if self.is_subgraph_available:
            r += f", graph={type(self.subgraph.module).__name__}"
        return r + ")"
    
class NetworkGraphCompiledEntry(NetworkGraphEntry):
    def __init__(
        self, 
        entries: 'list[NetworkGraphEntry]', 
        targets: 'list[NetworkGraphEntryCompileTarget]', 
        # core_groups: 'list[MCA_CoreGroup]',
        compiler_recipe: MCA_OperatorGraphCompiler.CompileRecipe, 
    ):
        super().__init__(NetworkGraphEntry.Type.COMPILED, None)
        
        logger.debug(f"creating compiled entry with {len(targets)} compile targets")
        for entry, target in zip(entries, targets):
            logger.debug(f"compiled entry target: {entry} -> {target.op_method.__name__} with {len(target.buf_sigs)} buffer signatures")
        
        self.entries = entries
        self.targets = targets
        self.compiler_recipe = compiler_recipe

        # if len(core_groups) != len(targets):
        #     raise ValueError("the number of core groups must match the number of compile targets")
        
        compiler = MCA_OperatorGraphCompiler()
            
        for entry, target, in zip(entries, targets):
            logger.debug(f"compiling entry: {entry} with target method: {len(target.buf_sigs)} buffer signatures")

            _op_method = target.op_method
            _op_bufs = [buf_sig.buf for buf_sig in target.buf_sigs]
            _op_kwargs = target.op_kwargs
            
            op = _op_method(*_op_bufs, **_op_kwargs)
            # op.initialize_core_group(core_group)
            
            compiler.add_op(op)

        self.compiled_ops = compiler.compile(
            recipe=compiler_recipe,
        )
        
    def dispatch(self, graph_context: 'NetworkGraphContext'):
        logger.debug(f"dispatching entry {self.__str__()}")
        
        # STEP 1: load inputs to buffers
        for target in self.targets:
            for buf_sig in target.buf_sigs:
                if buf_sig.buf is None or not buf_sig.buf.is_allocated:
                    continue
                buf_sig.load(graph_context)
        
        # STEP 2: execute compiled operators
        self.compiled_ops.dispatch()
        
    def execute(self, graph_context: 'NetworkGraphContext', pcc_check: bool=False):
        self.dispatch(graph_context)
        
        logger.debug(f"waiting for compiled operators to finish on device 0x{id(self.compiler_recipe.device):X}...")
        self.compiler_recipe.device.run_kernels()
        
        # STEP 3: store outputs from buffers to graph context
        for target in self.targets:
            for buf_sig in target.buf_sigs:
                if buf_sig.buf is None or not buf_sig.buf.is_allocated or not buf_sig.has_dst:
                    continue
                logger.debug(f"restoring buffer '{buf_sig.dst_name}' to graph context or destination module after executing compiled operator")
                buf_sig.store(graph_context, pcc_check=pcc_check)
                
    def summary(self) -> dict[str, Any]:
        return self.compiled_ops.summary()
        
    def __str__(self):
        return f"COMPILED_ENTRY(id=0x{id(self):X}, targets={[target.op_method.__name__ for target in self.targets]})"
    
class NetworkGraphEntryCompileTarget:
    class TensorProcessing:
        def __init__(self, proc_type: str, proc_params: Dict[str, Any]):
            self.proc_type = proc_type
            self.proc_params = proc_params
        
        @classmethod
        def permute(cls, *order):
            return cls(
                proc_type='permute',
                proc_params={'order': order}
            )
            
        @classmethod
        def to_dtype(cls, dtype: torch.dtype):
            return cls(
                proc_type='to_dtype',
                proc_params={'dtype': dtype}
            )
            
        def apply(self, tensor: torch.Tensor) -> torch.Tensor:
            if self.proc_type == 'permute':
                order = self.proc_params['order']
                return tensor.permute(*order)
            elif self.proc_type == 'to_dtype':
                dtype = self.proc_params['dtype']
                return tensor.to(dtype)
            else:
                raise NotImplementedError(f"unsupported tensor processing type: {self.proc_type}")
    
    class BufferSignature:
        CONTEXT = "CONTEXT"
        
        def __init__(
            self, 
            shape: Sequence[int], dtype: torch.dtype, shard_shape: Sequence[int], tile_shape: Sequence[int], blocked_mapping: bool=False, 
            orig_dtype: torch.dtype=None, 
            preprocessings: List['NetworkGraphEntryCompileTarget.TensorProcessing']=None,
            postprocessings: List['NetworkGraphEntryCompileTarget.TensorProcessing']=None,
        ):
            # Information from USER
            self.shape = shape
            self.dtype = dtype
            self.shard_shape = shard_shape
            self.tile_shape = tile_shape
            self.blocked_mapping = blocked_mapping
            self.orig_dtype = orig_dtype if orig_dtype is not None else dtype
            self.preprocessings = preprocessings if preprocessings is not None else []
            self.postprocessings = postprocessings if postprocessings is not None else []
            
            if self.shape is None or self.dtype is None or self.shard_shape is None:
                raise RuntimeError("Buffer signature must have valid shape, dtype, and shard_shape.")
            
            self._src: tuple[Any, str] = None
            self._dst: tuple[Any, str] = None
            
            # Information from COMPILER
            self.mem_space: MCA_MemorySpace = None
            
            # Information from RUNTIME
            self.buf: MCA_TensorBuffer = None
            
        def get_shard_size(self) -> int:
            return self.shard_shape[0] * self.shard_shape[1] * self.dtype.itemsize
        
        def get_shard_num(self) -> int:
            n_height_shards = (self.shape[-2] + self.shard_shape[0] - 1) // self.shard_shape[0]
            n_width_shards  = (self.shape[-1] + self.shard_shape[1] - 1) // self.shard_shape[1]
            return n_height_shards * n_width_shards
        
        def load_from(self, key: str, module: Any=CONTEXT):
            self._src = (module, key)
            return self
        
        def store_to(self, key: str, module: Any=CONTEXT):
            self._dst = (module, key)
            return self
        
        def load(self, graph_context: 'NetworkGraphContext'):
            if not self.is_allocated:
                raise Exception("buffer must be created before loading")

            if not self.buf.is_allocated:
                return
            
            if self.has_src:
                src_module, src_key = self.src
                if src_module == self.CONTEXT:
                    src_tensor = graph_context[src_key]
                else:
                    src_tensor = getattr(src_module, src_key)
                for preproc in self.preprocessings:
                    src_tensor = preproc.apply(src_tensor)
                src_tensor = src_tensor.to(self.dtype) if src_tensor.dtype != self.dtype else src_tensor   # convert to buffer dtype if necessary
                self.buf.update(src_tensor)
                
        def store(self, graph_context: 'NetworkGraphContext', pcc_check: bool=False):
            if not self.is_allocated:
                raise Exception("buffer must be created before storing")
            if not self.buf.is_allocated:
                logger.warning("buffer is not allocated, skipping store operation")
                return
            
            dst_tensor = self.buf.restore()
            for postproc in self.postprocessings:
                dst_tensor = postproc.apply(dst_tensor)
            dst_tensor = dst_tensor.to(self.orig_dtype) if self.orig_dtype != self.dtype else dst_tensor   # convert back to original dtype if necessary
                
            if self.has_dst:
                dst_module, dst_key = self.dst
                
                if pcc_check:
                    if dst_module == self.CONTEXT:
                        existing_tensor = graph_context.get(dst_key.debugName(), None)
                    else:
                        existing_tensor = getattr(dst_module, dst_key, None)
                    
                    if self.orig_dtype == torch.float32:
                        if self.dtype == torch.float32:
                            rtol, atol = 1e-2, 1e-3
                        elif self.dtype == torch.float16:
                            rtol, atol = 1e-2, 1e-3
                        elif self.dtype == torch.bfloat16:
                            rtol, atol = 1e-2, 1e-3
                        else:
                            rtol, atol = 1e-3, 1e-5
                    else:
                        rtol, atol = 1e-3, 1e-5
                        
                    if existing_tensor is None:
                        logger.error(f"skipping post-compilation check for buffer '{self.dst_name}' due to missing existing tensor in graph context or destination module")
                    elif not torch.allclose(existing_tensor, dst_tensor, rtol=rtol, atol=atol):
                        raise Exception(f"potential data inconsistency detected for buffer '{self.dst_name}' during post-compilation check: existing tensor in graph context or destination module is not close to the tensor restored from buffer\nexisting tensor: {existing_tensor}\ndst tensor: {dst_tensor}")
                    else:
                        logger.debug(f"post-compilation check passed for buffer '{self.dst_name}'")
                        
                if dst_module == self.CONTEXT:
                    graph_context[dst_key] = dst_tensor
                else:
                    setattr(dst_module, dst_key, dst_tensor)
                    
        def create(self, graph_context: 'NetworkGraphContext', mem_space: MCA_MemorySpace):
            if self.is_allocated:
                raise Exception("buffer is already allocated")
            self.mem_space = mem_space
            self.buf = MCA_TensorBuffer(
                mem_space=mem_space, 
                shape=self.shape, 
                dtype=self.dtype, 
                shard_shape=self.shard_shape, 
                blocked_mapping=self.blocked_mapping,
            ).tiling(tile_shape=self.tile_shape)
            
            cur_src_buf_name = f"BUFFER_SOURCED_FROM::{self.src_name}" if self.has_src else None
            cur_dst_buf_name = f"BUFFER_DESTINED_TO::{self.dst_name}" if self.has_dst else None
            existing_src_producer_buf_name = f"BUFFER_DESTINED_TO::{self.src_name}" if self.has_src else None
            existing_dst_consumer_buf_name = f"BUFFER_SOURCED_FROM::{self.dst_name}" if self.has_dst else None
            
            if cur_src_buf_name is not None:
                if cur_src_buf_name in graph_context.keys():
                    existing_buf_sig = graph_context[cur_src_buf_name]
                    if not isinstance(existing_buf_sig, NetworkGraphEntryCompileTarget.BufferSignature):
                        raise Exception(f"buffer name conflict in graph context: {cur_src_buf_name} is already used by a non-buffer entry")
                    if not self.compare_layout(existing_buf_sig):
                        raise Exception(f"buffer layout mismatch for buffer name {cur_src_buf_name} in graph context: existing buffer signature {existing_buf_sig} is not compatible with new buffer signature {self}")
                    self.buf = existing_buf_sig.buf
                elif existing_src_producer_buf_name in graph_context.keys():
                    self.buf = graph_context[existing_src_producer_buf_name].buf
                else:
                    graph_context[cur_src_buf_name] = self
            
            if cur_dst_buf_name is not None:
                if cur_dst_buf_name in graph_context.keys():
                    existing_buf_sig = graph_context[cur_dst_buf_name]
                    if not isinstance(existing_buf_sig, NetworkGraphEntryCompileTarget.BufferSignature):
                        raise Exception(f"buffer name conflict in graph context: {cur_dst_buf_name} is already used by a non-buffer entry")
                    if not self.compare_layout(existing_buf_sig):
                        raise Exception(f"buffer layout mismatch for buffer name {cur_dst_buf_name} in graph context: existing buffer signature {existing_buf_sig} is not compatible with new buffer signature {self}")
                    self.buf = existing_buf_sig.buf
                elif existing_dst_consumer_buf_name in graph_context.keys():
                    self.buf = graph_context[existing_dst_consumer_buf_name].buf
                else:
                    graph_context[cur_dst_buf_name] = self

        def allocate(self):
            if self.buf is None:
                raise Exception("buffer must be created before allocation")
            if self.buf.is_allocated:
                return self.buf
            self.buf.allocate()
            return self.buf
                
        def deallocate(self):
            if self.buf is not None:
                self.buf = None
        
        @property
        def src(self):
            return self._src
        
        @property
        def dst(self):
            return self._dst
        
        @property
        def has_src(self) -> bool:
            return self.src is not None
        
        @property
        def has_dst(self) -> bool:
            return self.dst is not None
        
        @property
        def is_allocated(self) -> bool:
            return self.buf is not None
        
        @property
        def src_name(self) -> str:
            if not self.has_src:
                raise RuntimeError("Buffer signature does not have a source")
            src_module, src_key = self.src
            if isinstance(src_key, torch.Value):
                src_key = '%'+src_key.debugName()
            src_module_name = "CONTEXT" if src_module == self.CONTEXT else f"module(id={id(src_module)}, type={type(src_module).__name__})"
            return f"{src_module_name}::{src_key}"
        
        @property
        def dst_name(self) -> str:
            if not self.has_dst:
                raise RuntimeError("Buffer signature does not have a destination")
            dst_module, dst_key = self.dst
            if isinstance(dst_key, torch.Value):
                dst_key = '%'+dst_key.debugName()
            dst_module_name = "CONTEXT" if dst_module == self.CONTEXT else f"module(id={id(dst_module)}, type={type(dst_module).__name__})"
            return f"{dst_module_name}::{dst_key}"
        
        @property
        def is_l1_buffer(self) -> bool:
            if self.mem_space is None:
                return False
            return self.mem_space.mem_type == GlobalContextMemType.L1
        
        @property
        def is_main_buffer(self) -> bool:
            if self.mem_space is None:
                return False
            return self.mem_space.mem_type == GlobalContextMemType.MAIN
        
        @property
        def n_tiles(self) -> int:
            remaining_dims = functools.reduce(lambda x, y: x * y, self.shape[:-2], 1)
            n_height_shards = self.shape[-2] // self.shard_shape[0]
            n_width_shards = self.shape[-1] // self.shard_shape[1]
            n_height_tiles_per_shard = (self.shard_shape[0] + self.tile_shape[0] - 1) // self.tile_shape[0]
            n_width_tiles_per_shard = (self.shard_shape[1] + self.tile_shape[1] - 1) // self.tile_shape[1]
            return remaining_dims * n_height_shards * n_width_shards * n_height_tiles_per_shard * n_width_tiles_per_shard
        
        def compare_layout(self, other: 'NetworkGraphEntryCompileTarget.BufferSignature') -> bool:
            return self.shape == other.shape and self.dtype == other.dtype and self.shard_shape == other.shard_shape and self.blocked_mapping == other.blocked_mapping and self.tile_shape == other.tile_shape
    
    def __init__(
        self, 
        op_method: Callable[..., MCA_OperatorSignature], 
        buf_sigs: 'list[NetworkGraphEntryCompileTarget.BufferSignature]', 
        op_kwargs: dict[str, Any],
        arith_intensity: float=0.0,
        max_n_cores: int=1,
    ):
        self.op_method: Callable[..., MCA_OperatorSignature] = op_method
        self.buf_sigs: list[NetworkGraphEntryCompileTarget.BufferSignature] = buf_sigs
        self.op_kwargs: dict[str, Any] = op_kwargs
        self.arith_intensity: float = arith_intensity
        self.max_n_cores: int = max_n_cores
        

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
        elif isinstance(name, Placeholder):
            name = name.name
        return super().__getattr__(name)
    
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

            if trace_mode:
                args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
            
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
                
    def run_compiled_entry(self, entry: NetworkGraphCompiledEntry, trace_mode: bool=False, pcc_check: bool=False):
        if trace_mode:
            for ee in entry.entries:
                self.run_entry(ee, trace_mode=trace_mode)
        else:
            entry.execute(self, pcc_check=pcc_check)
                
    def run_entry(self, entry: NetworkGraphEntry, trace_mode: bool=False, pcc_check: bool=False):
        # STEP 1: pre-run the entry to prepare inputs/outputs
        if entry.is_prim:
            self.run_prim_entry(entry, trace_mode=trace_mode)
        elif entry.is_nonprim:
            self.run_nonprim_entry(entry)
        elif entry.is_compiled:
            self.run_compiled_entry(entry, trace_mode=True)   # for compiled entry, pre-run with trace_mode=True to prepare inputs/outputs without executing the compiled kernels
        else:
            raise Exception(f"unsupported entry type: {entry.node_type}")
            
        # STEP 2: run the entry with compiled runtime (if necessary)
        if entry.is_compiled and not trace_mode:
            logger.debug(f"running compiled graph entry: {entry}")
            self.run_compiled_entry(entry, trace_mode=False, pcc_check=pcc_check)


class MCA_CompiledNetworkGraph:
    class NetworkRecipe:
        def __init__(
            self, 
            device: MCA_DeviceBase, 
            core_groups: list[MCA_CoreGroup],
            
            main_data_mem_space_size_per_channel: int,
            l1_data_mem_space_size_per_core: int,
            spad_mem_space_size_per_core: int,
            
            pipeline_granularity: int=8,
            broadcast_optimize_queue_depth: int=32,
            operator_pipelining: bool=False,
        ):
            self.main_data_mem_space_size_per_channel = main_data_mem_space_size_per_channel
            self.l1_data_mem_space_size_per_core = l1_data_mem_space_size_per_core
            self.spad_mem_space_size_per_core = spad_mem_space_size_per_core
            
            self.compiler_recipe = MCA_OperatorGraphCompiler.CompileRecipe(
                device=device,
                core_groups=core_groups,
                spad_space_size_per_core=spad_mem_space_size_per_core,
                pipeline_granularity=pipeline_granularity,
                broadcast_optimize_queue_depth=broadcast_optimize_queue_depth,
                operator_pipelining=operator_pipelining,
            )
            
        def supports(self, module_type: type | str) -> bool:
            if isinstance(module_type, type):
                module_type = module_type.__name__
            elif isinstance(module_type, torch.nn.Module):
                module_type = type(module_type).__name__
            return hasattr(self, module_type)
        
        def get_compile_target(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Module) -> NetworkGraphEntryCompileTarget:
            if not self.supports(type(submodule)):
                raise Exception(f"compilation recipe does not support module type: {type(submodule).__name__}")
            
            compile_method = getattr(self, type(submodule).__name__)
            compiled_entry = compile_method(graph_context, node, submodule)
            
            if not isinstance(compiled_entry, NetworkGraphEntryCompileTarget):
                raise Exception("compilation recipe method must return a NetworkGraphEntryCompileTarget instance")
            
            return compiled_entry
        
        @staticmethod
        def recipe(func: Callable):
            @functools.wraps(func)
            def _wrapper(self, graph_context: NetworkGraphContext, node: torch.Node, submodule: torch.nn.Module) -> NetworkGraphEntryCompileTarget:
                if not isinstance(submodule, torch.nn.Module):
                    raise Exception("recipe methods can only compile torch.nn.Module submodules")
                entry = func(self, graph_context, node, submodule)
                if not isinstance(entry, NetworkGraphEntryCompileTarget):
                    raise Exception("recipe methods must return a NetworkGraphEntryCompileTarget instance")
                return entry
            return _wrapper
        
    def __init__(
        self, 
        module: torch.nn.Module, 
        graph_ivars: list[torch.Value], 
        graph_ovars: list[torch.Value], 
        graph_nodes: list[torch.Node], 
        graph_context: NetworkGraphContext, 
        graph_entries: list[NetworkGraphEntry],
        graph_recipe: 'MCA_CompiledNetworkGraph.NetworkRecipe',
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
        graph_recipe: 'MCA_CompiledNetworkGraph.NetworkRecipe', 
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

                        if graph_recipe.supports(submodule):
                            entry = NetworkGraphEntry(NetworkGraphEntry.Type.PRIM, node, submodule=submodule)
                            entry.is_compilation_target = True
                        else:
                            subgraph = MCA_CompiledNetworkGraph.from_trace(submodule, graph_recipe, *args, disable_lowering=True, disable_toposort=True, disable_compile=True)
                            entry = NetworkGraphEntry(NetworkGraphEntry.Type.GRAPH, node, subgraph=subgraph)
                    
                    # otherwise, the entry remains as prim::CallMethod (not compiled as runtime kernel or operator)
                    else:
                        entry = NetworkGraphEntry(NetworkGraphEntry.Type.PRIM, node)
            
            # STEP 3-2-2: nonprim nodes (aten::, quantized::, ...)
            else:
                entry = NetworkGraphEntry(NetworkGraphEntry.Type.NONPRIM, node)
            
            entries.append(entry)
            graph_context.run_entry(entry, trace_mode=True, pcc_check=False)
            
        warnings.filterwarnings("default", category=UserWarning)    # TODO: suppress leaf Tensor access warning
        warnings.filterwarnings("default", category=FutureWarning)  # TODO: suppress quantized model warning 
            
        graph = cls(module, graph_ivars, graph_ovars, graph_nodes, graph_context, entries, graph_recipe)
        
        if not disable_lowering:
            graph.lowering()
        if not disable_toposort:
            graph.topological_sort()
        if not disable_compile:
            graph.run_graph(*dummy_inputs, trace_mode=True, pcc_check=False)   # pre-run the graph to prepare runtime parameters for compilation
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
    
    def _search_value_usage_as_input(self, value: torch.Value) -> list[tuple[int, torch.Node]]:
        usage_list = []
        value_name = value.debugName()
        for idx, entry in enumerate(self.graph_entries):
            if entry.is_subgraph_available:
                subgraph_usage_list = entry.subgraph._search_value_usage_as_input(value)
                usage_list.extend(subgraph_usage_list)
            elif entry.is_compiled:
                entry: NetworkGraphCompiledEntry
                # for compiled entry, we need to check the buffer signatures of the compile targets
                for target in entry.targets:
                    for buf_sig in target.buf_sigs:
                        if not buf_sig.has_src:
                            continue
                        src_key = buf_sig.src[1]
                        src_name = src_key.debugName() if isinstance(src_key, torch.Value) else str(src_key)
                        if src_name == value_name:
                            usage_list.append((idx, entry.node))
            elif value_name in [i.debugName() for i in entry.node.inputs()]:
                usage_list.append((idx, entry.node))
        return usage_list
    
    def _compile_targets(self, cp_targets: list[NetworkGraphEntryCompileTarget], main_mem_space: MCA_MainMemorySpace, l1_mem_space: MCA_L1MemorySpace) -> list[NetworkGraphCompiledEntry]:
        core_groups = self.graph_recipe.compiler_recipe.core_groups
        n_core_groups = len(core_groups)
        n_cores_per_group = core_groups[0].n_cores

        if n_core_groups <= 0:
            raise RuntimeError("no available core groups for compilation")

        normalized_targets: list[tuple[NetworkGraphEntry, NetworkGraphEntryCompileTarget]] = []

        for item in cp_targets:
            if isinstance(item, NetworkGraphEntry):
                if not item.is_compilation_target:
                    raise ValueError(f"non-compilation entry is passed to _compile_targets: {item}")
                if item.submodule is None:
                    raise ValueError(f"compilation entry does not have a valid submodule: {item}")

                target = self.graph_recipe.get_compile_target(self.graph_context, item.node, item.submodule)
                normalized_targets.append((item, target))
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], NetworkGraphEntry) and isinstance(item[1], NetworkGraphEntryCompileTarget):
                normalized_targets.append((item[0], item[1]))
            else:
                raise TypeError("_compile_targets expects compilation entries or (entry, compile_target) tuples")

        if len(normalized_targets) == 0:
            return []

        internal_entry_node_ids = {id(entry.node) for entry, _ in normalized_targets}
        graph_output_var_names = {ovar.debugName() for ovar in self.graph_ovars}

        def _is_internal_only_value(value: torch.Value) -> bool:
            usages = self._search_value_usage_as_input(value)
            if len(usages) == 0:
                return False

            for _, used_node in usages:
                if used_node is None or id(used_node) not in internal_entry_node_ids:
                    return False
            return True

        for _, target in normalized_targets:
            for buf_sig in target.buf_sigs:
                mem_space = main_mem_space
                buf_sig.create(self.graph_context, mem_space)

                linked_value = None
                if buf_sig.has_dst:
                    dst_module, dst_key = buf_sig.dst
                    if dst_module == NetworkGraphEntryCompileTarget.BufferSignature.CONTEXT and isinstance(dst_key, torch.Value):
                        linked_value = dst_key
                skip_allocate = False
                if linked_value is not None and linked_value.debugName() not in graph_output_var_names:
                    skip_allocate = _is_internal_only_value(linked_value)

                if not skip_allocate:
                    buf_sig.allocate()

        target_core_group_caps: list[int] = []
        for _, target in normalized_targets:
            target_core_group_caps.append(max(1, min(n_core_groups, target.max_n_cores // n_cores_per_group)))
            logger.debug(f"  - target: {target.op_method.__name__}, arith_intensity: {target.arith_intensity:5.2f}, max_n_cores: {target.max_n_cores}, core_group_cap: {target_core_group_caps[-1]}/{n_core_groups}")

        intensities = [max(float(target.arith_intensity), 0.0) for _, target in normalized_targets]
        min_intensity = min(intensities)
        max_intensity = max(intensities)

        if math.isclose(min_intensity, max_intensity):
            required_group_counts = [1 for _ in intensities]
        else:
            required_group_counts = []
            for intensity in intensities:
                norm = (intensity - min_intensity) / (max_intensity - min_intensity)
                required = 1 + int(round(norm * (n_core_groups - 1)))
                required_group_counts.append(max(1, min(n_core_groups, required)))

        required_group_counts = [
            min(req_groups, cap_groups)
            for req_groups, cap_groups in zip(required_group_counts, target_core_group_caps)
        ]

        compiled_entries: list[NetworkGraphCompiledEntry] = []

        def _flush_chunk(
            chunk_entries: list[NetworkGraphEntry],
            chunk_targets: list[NetworkGraphEntryCompileTarget],
            # chunk_group_counts: list[int],
        ):
            if len(chunk_targets) == 0:
                return

            # # assigned_groups: list[MCA_CoreGroup] = []
            # offset = 0

            # for req_groups in chunk_group_counts:
            #     # selected_subgroups = core_groups[offset: offset + req_groups]
            #     # merged_group = MCA_CoreGroup.merge_core_groups(selected_subgroups)
            #     # assigned_groups.append(merged_group)
            #     offset += req_groups

            compiled_entries.append(NetworkGraphCompiledEntry(
                entries=chunk_entries,
                targets=chunk_targets,
                # core_groups=assigned_groups,
                compiler_recipe=self.graph_recipe.compiler_recipe,
            ))

        chunk_entries: list[NetworkGraphEntry] = []
        chunk_targets: list[NetworkGraphEntryCompileTarget] = []
        # chunk_group_counts: list[int] = []
        used_groups = 0

        for (entry, target), req_groups in zip(normalized_targets, required_group_counts):
            if used_groups + req_groups > n_core_groups and len(chunk_targets) > 0:
                _flush_chunk(chunk_entries, chunk_targets)
                chunk_entries = []
                chunk_targets = []
                # chunk_group_counts = []
                used_groups = 0

            chunk_entries.append(entry)
            chunk_targets.append(target)
            # chunk_group_counts.append(req_groups)
            used_groups += req_groups

        _flush_chunk(chunk_entries, chunk_targets)

        return compiled_entries

    def compile(self, _allocate_mem_space: bool=True, _cached_mem_spaces: dict[str, MCA_MemorySpace]=None):
        logger.debug(f"compiling graph of module: {type(self.module).__name__}")
        
        device = self.graph_recipe.compiler_recipe.device
        
        if _allocate_mem_space:
            main_mem_space = device.create_main_mem_space(self.graph_recipe.main_data_mem_space_size_per_channel)
            l1_mem_space = device.create_l1_mem_space(self.graph_recipe.l1_data_mem_space_size_per_core, self.graph_recipe.compiler_recipe.global_core_group)
        else:
            main_mem_space = _cached_mem_spaces.get("MAIN", None)
            l1_mem_space = _cached_mem_spaces.get("L1", None)
            
            if main_mem_space is None or l1_mem_space is None:
                raise Exception("cached memory spaces must be provided for compilation when _allocate_mem_space is False")
        
        cp_targets: list[NetworkGraphEntryCompileTarget] = []
        new_entries: list[NetworkGraphEntry] = []
        
        for entry in self.graph_entries:
            if not entry.is_compilation_target:
                if len(cp_targets) > 0:
                    compiled_entries = self._compile_targets(cp_targets, main_mem_space, l1_mem_space)
                    new_entries.extend(compiled_entries)
                    cp_targets = []
                
                # FINAL: add the entry to new entries
                if entry.is_subgraph_available:
                    entry.subgraph.compile(_allocate_mem_space=False, _cached_mem_spaces={"MAIN": main_mem_space, "L1": l1_mem_space})
                new_entries.append(entry)
            else:
                # create compilation target and store it in the temporary list
                cp_targets.append(entry)
        
        if len(cp_targets) > 0:
            compiled_entries = self._compile_targets(cp_targets, main_mem_space, l1_mem_space)
            new_entries.extend(compiled_entries)
            cp_targets = []
        
        self.graph_entries = new_entries
        
        if _allocate_mem_space:
            main_mem_space.remove()
            l1_mem_space.remove()
        
        return self
    
    def get_outputs(self):
        if len(self.graph_ovars) == 1:
            return self.graph_context[self.graph_ovars[0]]
        return [self.graph_context[o] for o in self.graph_ovars]
    
    def run_graph(self, *dummy_inputs, trace_mode: bool=False, pcc_check: bool=False):
        self.graph_context[self.graph_ivars[0]] = self.module
        for idx, ivar in enumerate(self.graph_ivars[1:]):
            self.graph_context[ivar] = dummy_inputs[idx]
        
        for entry in self.graph_entries:
            try:
                self.graph_context.run_entry(entry, trace_mode=trace_mode, pcc_check=pcc_check)
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
