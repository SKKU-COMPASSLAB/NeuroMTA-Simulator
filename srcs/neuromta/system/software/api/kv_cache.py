import torch

from neuromta.system.software.utils.descriptor import MeshMemoryType, MeshTensorDescriptor, MeshTensorType

from neuromta.system.software.api.common import *
from neuromta.system.software.api.utils import *
from neuromta.system.software.api.tensor import *


__all__ = [
    "MeshKVCacheOverflowError",
    "MeshKVCache",
    "mesh_kv_cache",
]


class MeshKVCacheOverflowError(RuntimeError):
    pass


class MeshKVCache:
    def __init__(self, context: MeshDeviceRuntimeContext, cache_id: str, batch_size: int, max_seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, tile_shape: tuple[int, ...], preferred_mem: MeshMemoryType, context_buckets: tuple[int, ...], overflow_policy: str):
        values = {"batch_size": batch_size, "max_seq_len": max_seq_len, "num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"KV cache dimensions must be positive integers: {values}")
        if not isinstance(preferred_mem, MeshMemoryType):
            raise TypeError(f"Expected MeshMemoryType, got {type(preferred_mem).__name__}.")
        buckets = tuple(sorted(set(int(bucket) for bucket in context_buckets)))
        if any(bucket <= 0 for bucket in buckets):
            raise ValueError(f"Invalid context buckets: {buckets}")
        if overflow_policy != "error":
            raise ValueError("The fixed-capacity KV cache currently supports only overflow_policy='error'.")
        candidates = [bucket for bucket in buckets if bucket >= max_seq_len]
        if buckets and not candidates:
            raise MeshKVCacheOverflowError(f"Requested context length {max_seq_len} exceeds the largest bucket {buckets[-1]}.")
        self.context = context
        self.cache_id = cache_id
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tile_shape = tile_shape
        self.preferred_mem = preferred_mem
        self.context_buckets = buckets
        self.capacity = min(candidates) if candidates else max_seq_len
        self.overflow_policy = overflow_policy
        self._lengths = [0] * num_layers
        storage_shape = (batch_size, num_kv_heads, self.capacity, head_dim)
        self.key_storages = tuple(MeshTensorDescriptor(storage_shape, tile_shape, dtype, tensor_type=MeshTensorType.INTERMEDIATE, reserved_shape=storage_shape).as_persistent(f"{cache_id}/layer_{layer:04d}/key") for layer in range(num_layers))
        self.value_storages = tuple(MeshTensorDescriptor(storage_shape, tile_shape, dtype, tensor_type=MeshTensorType.INTERMEDIATE, reserved_shape=storage_shape).as_persistent(f"{cache_id}/layer_{layer:04d}/value") for layer in range(num_layers))
        for descriptor in self.storage_descriptors:
            descriptor.preferred_mem = preferred_mem

    @property
    def storage_descriptors(self) -> tuple[MeshTensorDescriptor, ...]:
        return tuple(descriptor for layer in range(self.num_layers) for descriptor in (self.key_storages[layer], self.value_storages[layer]))

    @property
    def current_length(self) -> int:
        return min(self._lengths)

    @property
    def layer_lengths(self) -> tuple[int, ...]:
        return tuple(self._lengths)

    @property
    def is_reserved(self) -> bool:
        placements = self.context.runtime.persistent_state_placements
        return all(descriptor.persistent_state_id in placements for descriptor in self.storage_descriptors)

    @property
    def placements(self) -> dict:
        placements = self.context.runtime.persistent_state_placements
        return {descriptor.persistent_state_id: placements[descriptor.persistent_state_id] for descriptor in self.storage_descriptors if descriptor.persistent_state_id in placements}

    def _validate_layer(self, layer: int) -> int:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 or layer >= self.num_layers:
            raise IndexError(f"KV cache layer {layer} is outside [0, {self.num_layers}).")
        return layer

    def get_length(self, layer: int) -> int:
        return self._lengths[self._validate_layer(layer)]

    def reserve(self, capacity: int=None):
        requested_capacity = self.capacity if capacity is None else int(capacity)
        if requested_capacity <= 0:
            raise ValueError("KV cache reservation capacity must be positive.")
        if requested_capacity > self.capacity:
            raise MeshKVCacheOverflowError(f"Requested KV capacity {requested_capacity} exceeds fixed capacity {self.capacity}.")
        return self.context.reserve_state(self)

    def view(self, layer: int, start: int=0, end: int=None) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor]:
        layer = self._validate_layer(layer)
        active_end = self._lengths[layer] if end is None else int(end)
        if start < 0 or active_end <= start or active_end > self._lengths[layer]:
            raise ValueError(f"Invalid KV view range [{start}, {active_end}) for layer length {self._lengths[layer]}.")
        return mesh_narrow(self.key_storages[layer], -2, start, active_end - start), mesh_narrow(self.value_storages[layer], -2, start, active_end - start)

    def read(self, layer: int, start: int=0, end: int=None) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor]:
        return self.view(layer, start, end)

    def append(self, layer: int, key: MeshTensorDescriptor, value: MeshTensorDescriptor) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor]:
        layer = self._validate_layer(layer)
        key = _require_tensor(key, "key")
        value = _require_tensor(value, "value")
        expected_prefix = (self.batch_size, self.num_kv_heads)
        if len(key.shape) != 4 or len(value.shape) != 4 or key.shape[:2] != expected_prefix or value.shape[:2] != expected_prefix or key.shape[-1] != self.head_dim or value.shape[-1] != self.head_dim or key.shape[-2] != value.shape[-2]:
            raise ValueError(f"KV append tensors must have shape ({self.batch_size}, {self.num_kv_heads}, append_length, {self.head_dim}).")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("KV append tensors must match the cache dtype.")
        append_length = key.shape[-2]
        start = self._lengths[layer]
        end = start + append_length
        if end > self.capacity:
            larger_bucket = next((bucket for bucket in self.context_buckets if bucket >= end), None)
            suffix = f" Recompile with context bucket {larger_bucket}." if larger_bucket is not None else ""
            raise MeshKVCacheOverflowError(f"KV append end {end} exceeds fixed capacity {self.capacity}.{suffix}")
        key_dst = mesh_narrow(self.key_storages[layer], -2, start, append_length)
        value_dst = mesh_narrow(self.value_storages[layer], -2, start, append_length)
        mesh_store(key, key_dst)
        mesh_store(value, value_dst)
        self._lengths[layer] = end
        return key_dst, value_dst

    def reset(self, layer: int=None):
        if layer is None:
            self._lengths = [0] * self.num_layers
        else:
            self._lengths[self._validate_layer(layer)] = 0

    def deallocate(self) -> int:
        return self.context.deallocate_state(self)


def mesh_kv_cache(batch_size: int, max_seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype=None, tile_shape: tuple[int, ...]=None, preferred_mem: MeshMemoryType=MeshMemoryType.DEVICE_MEMORY, context_buckets: tuple[int, ...]=(), overflow_policy: str="error") -> MeshKVCache:
    context = _require_context("mesh_kv_cache")
    selected_tile_shape = context.default_tile_shape if tile_shape is None else tuple(tile_shape)
    cache_id = f"kv_cache_{context._state_counter:04d}"
    context._state_counter += 1
    cache = MeshKVCache(context, cache_id, batch_size, max_seq_len, num_layers, num_kv_heads, head_dim, context.default_dtype if dtype is None else dtype, selected_tile_shape, preferred_mem, context_buckets, overflow_policy)
    context._state_handles.append(cache)
    if context.compiler is not None:
        context._compiler_state_snapshots[cache] = cache.layer_lengths
    return cache