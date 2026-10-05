import math
import torch
from dataclasses import dataclass, field
from typing import Any

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.nn.qwen2_5_omni import *
from neuromta.system.software.utils import *


_vision_encode_cnt = 0
_text_request_cnt = 0


def vision_encode(compiler_type, context: MeshDeviceRuntimeContext, model: Qwen2_5_Omni, kv_cache: MeshKVCache, arrival_cycle: int = 0):
    global _vision_encode_cnt
    _vision_encode_cnt += 1
    _workload_id = f"encode.{_vision_encode_cnt}"
    _dependency = [f"encode.{_vision_encode_cnt - 1}"] if _vision_encode_cnt > 1 else tuple()
    
    with context.new_compiler_context(compiler_type(), arrival_cycle=arrival_cycle, workload_id=_workload_id, dependent_workload_ids=_dependency):
        video_shape = model.video_shape()
        video = MeshTensorDescriptor(shape=video_shape, tile_shape=(1, 1, 1, 32, 32), dtype=torch.bfloat16)
        visual_tokens = model.vision_encoder(video)
        if visual_tokens.shape != (1, 1, model.hidden_dim):
            raise RuntimeError(f"Unexpected visual-token shape: {visual_tokens.shape}")
        
        logits = model.prefill_vision_tokens(kv_cache, visual_tokens)
        if logits.shape != (1, visual_tokens.shape[1], model.vocab_size):
            raise RuntimeError(f"Unexpected multimodal prefill output shape: {logits.shape}")
        
        return logits
    
def text_full_request(compiler_type, context: MeshDeviceRuntimeContext, model: Qwen2_5_Omni, kv_cache: MeshKVCache, prompt_length: int, decode_tokens: int, arrival_cycle: int = 0):
    global _text_request_cnt
    _text_request_cnt += 1
    _text_prefill_workload_id = f"text.prefill.{_text_request_cnt}"
    _text_decode_workload_id = f"text.decode.{_text_request_cnt}"

    with context.new_compiler_context(compiler_type(), arrival_cycle=arrival_cycle, workload_id=_text_prefill_workload_id):
        prompt = MeshTensorDescriptor(shape=model.prompt_shape(prompt_length), tile_shape=(1, 32), dtype=torch.int64)
        logits = model.language_model(prompt, kv_cache)
        if logits.shape != (1, prompt_length, model.vocab_size):
            raise RuntimeError(f"Unexpected multimodal prefill output shape: {logits.shape}")

    with context.new_compiler_context(compiler_type(), arrival_cycle=arrival_cycle, workload_id=_text_decode_workload_id, dependent_workload_ids=(_text_prefill_workload_id,)):
        token = mesh_argmax(mesh_narrow(logits, 1, logits.shape[1] - 1, 1), dim=-1, keepdim=False)
        for _ in range(decode_tokens):
            logits = model.language_model(token, kv_cache)
            if logits.shape != (1, 1, model.vocab_size):
                raise RuntimeError(f"Unexpected decode output shape: {logits.shape}")
            token = mesh_argmax(logits, dim=-1, keepdim=False)
        
    return logits
    
def text_prefill_request(compiler_type, context: MeshDeviceRuntimeContext, model: Qwen2_5_Omni, kv_cache: MeshKVCache, prompt_length: int, arrival_cycle: int = 0):
    global _text_request_cnt
    _text_request_cnt += 1
    _text_prefill_workload_id = f"text.prefill.{_text_request_cnt}"

    with context.new_compiler_context(compiler_type(), arrival_cycle=arrival_cycle, workload_id=_text_prefill_workload_id):
        prompt = MeshTensorDescriptor(shape=model.prompt_shape(prompt_length), tile_shape=(1, 32), dtype=torch.int64)
        logits = model.language_model(prompt, kv_cache)
        if logits.shape != (1, prompt_length, model.vocab_size):
            raise RuntimeError(f"Unexpected multimodal prefill output shape: {logits.shape}")
   
    return logits

def text_decode_request(compiler_type, context: MeshDeviceRuntimeContext, model: Qwen2_5_Omni, kv_cache: MeshKVCache, decode_tokens: int, arrival_cycle: int = 0):
    global _text_request_cnt
    if not isinstance(kv_cache, MeshKVCache):
        raise TypeError("kv_cache must be a MeshKVCache.")
    if kv_cache.context is not context or get_global_mesh_device_context() is not context:
        raise ValueError("Decode must use the active runtime context that owns the KV cache.")
    if not isinstance(decode_tokens, int) or isinstance(decode_tokens, bool) or decode_tokens <= 0:
        raise ValueError("decode_tokens must be a positive integer.")
    if (kv_cache.batch_size, kv_cache.num_layers, kv_cache.num_kv_heads, kv_cache.head_dim) != (1, model.num_layers, model.num_kv_heads, model.head_dim):
        raise ValueError("The KV cache must match the model dimensions and have batch_size=1.")
    context_length = kv_cache.current_length
    if any(length != context_length for length in kv_cache.layer_lengths):
        raise ValueError("Every KV cache layer must have the same context length before decoding.")
    if context_length + decode_tokens > kv_cache.capacity:
        raise MeshKVCacheOverflowError(f"Decode context length {context_length + decode_tokens} exceeds KV capacity {kv_cache.capacity}.")
    if context_length + decode_tokens > model.max_seq_len:
        raise ValueError(f"Decode context length {context_length + decode_tokens} exceeds model max_seq_len={model.max_seq_len}.")
    
    _text_request_cnt += 1
    _text_decode_workload_id = f"text.decode.{_text_request_cnt}"

    with context.new_compiler_context(compiler_type(), arrival_cycle=arrival_cycle, workload_id=_text_decode_workload_id):
        token = MeshTensorDescriptor(shape=model.prompt_shape(1), tile_shape=(1, 32), dtype=torch.int64)
        for _ in range(decode_tokens):
            logits = model.language_model(token, kv_cache)
            if logits.shape != (1, 1, model.vocab_size):
                raise RuntimeError(f"Unexpected decode output shape: {logits.shape}")
            token = mesh_argmax(logits, dim=-1, keepdim=False)

    return logits

def prepare_kv_cache(kv_cache: MeshKVCache, context_length: int) -> MeshKVCache:
    if not isinstance(kv_cache, MeshKVCache):
        raise TypeError("kv_cache must be a MeshKVCache.")
    if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length < 0:
        raise ValueError("context_length must be a non-negative integer.")
    if context_length > kv_cache.capacity:
        raise MeshKVCacheOverflowError(f"Prepared context length {context_length} exceeds KV capacity {kv_cache.capacity}.")
    if kv_cache.context.compiler is not None:
        raise RuntimeError("Prepare the KV cache outside a compiler context.")
    kv_cache._lengths = [context_length] * kv_cache.num_layers
    return kv_cache


@dataclass
class KernelProfileEntry:
    kernel_type: MeshKernelType
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    latency: float
    core_count_average: float
    count: int = 1
    _core_count_m2: float = field(default=0.0, repr=False)

    @property
    def key(self) -> tuple:
        return self.kernel_type, self.input_shapes, self.output_shapes

    def merge(self, other: 'KernelProfileEntry'):
        if self.key != other.key:
            raise ValueError("Cannot merge kernel profile entries with different signatures.")
        total_count = self.count + other.count
        delta = other.core_count_average - self.core_count_average
        self.latency = (self.latency * self.count + other.latency * other.count) / total_count
        self.core_count_average += delta * other.count / total_count
        self._core_count_m2 += other._core_count_m2 + delta * delta * self.count * other.count / total_count
        self.count = total_count

    @property
    def core_count_stddev(self) -> float:
        return math.sqrt(self._core_count_m2 / self.count)

    @staticmethod
    def _format_shapes(shapes: tuple[tuple[int, ...], ...]) -> str:
        return "|".join("x".join(str(dim) for dim in shape) for shape in shapes)

    def to_csv_entry(self) -> str:
        return f"{self.kernel_type.value},{self._format_shapes(self.input_shapes)},{self._format_shapes(self.output_shapes)},{self.latency},{self.count},{self.core_count_average},{self.core_count_stddev}"

    @classmethod
    def csv_header(cls) -> str:
        return "kernel_type,input_shapes,output_shapes,average_latency,count,core_count_average,core_count_stddev"


class KernelProfile:
    def __init__(self):
        self.entries: dict[tuple, KernelProfileEntry] = {}

    def add(self, kernel_desc: MeshKernelDescriptor, latency: int, core_count: int):
        if core_count <= 0:
            raise ValueError("core_count must be positive.")
        entry = KernelProfileEntry(
            kernel_type=kernel_desc.kernel_type,
            input_shapes=tuple(tuple(tensor.shape) for tensor in kernel_desc.input_tensors),
            output_shapes=tuple(tuple(tensor.shape) for tensor in kernel_desc.output_tensors),
            latency=latency,
            core_count_average=float(core_count),
        )
        if entry.key in self.entries:
            self.entries[entry.key].merge(entry)
        else:
            self.entries[entry.key] = entry

    def to_csv(self) -> str:
        rows = [entry.to_csv_entry() for entry in self.entries.values()]
        return "\n".join([KernelProfileEntry.csv_header()] + rows)


@dataclass
class WorkloadProfileEntry:
    workload_id: str
    arrival_cycle: int
    start_cycle: int
    completion_cycle: int
    slo: int | None

    @property
    def response_time(self) -> int:
        return self.completion_cycle - self.arrival_cycle

    @property
    def execution_time(self) -> int:
        return self.completion_cycle - self.start_cycle

    @property
    def deadline(self) -> int | None:
        return None if self.slo is None else self.arrival_cycle + self.slo

    @property
    def slack(self) -> int | None:
        return None if self.deadline is None else self.deadline - self.completion_cycle

    @property
    def deadline_met(self) -> bool | None:
        return None if self.slack is None else self.slack >= 0

    def to_csv_entry(self) -> str:
        values = (self.workload_id, self.arrival_cycle, self.start_cycle, self.completion_cycle, self.slo, self.response_time, self.execution_time, self.slack, self.deadline_met)
        return ",".join("" if value is None else str(value) for value in values)

    @classmethod
    def csv_header(cls) -> str:
        return "workload_id,arrival_cycle,start_cycle,completion_cycle,slo,response_time,execution_time,slack,deadline_met"


class WorkloadProfile:
    def __init__(self):
        self.entries: dict[str, WorkloadProfileEntry] = {}

    def add(self, workload_id: str, arrival_cycle: int, start_cycle: int, completion_cycle: int, slo: int | None=None):
        self.entries[workload_id] = WorkloadProfileEntry(
            workload_id=workload_id,
            arrival_cycle=arrival_cycle,
            start_cycle=start_cycle,
            completion_cycle=completion_cycle,
            slo=slo,
        )

    def to_csv(self) -> str:
        rows = [entry.to_csv_entry() for entry in self.entries.values()]
        return "\n".join([WorkloadProfileEntry.csv_header()] + rows)


def save_profiles(profile: KernelProfile | WorkloadProfile, filepath: str):
    with open(filepath, "w") as f:
        f.write(profile.to_csv())
