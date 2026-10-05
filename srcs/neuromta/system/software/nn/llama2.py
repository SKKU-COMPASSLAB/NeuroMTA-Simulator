import torch

from neuromta.system.software.api import *
from neuromta.system.software.nn.common import *


class Llama2(MeshNetworkBase):
    def __init__(self, vocab_size: int=32000, hidden_dim: int=4096, intermediate_dim: int=11008, num_layers: int=32, num_heads: int=32, num_kv_heads: int=None, max_seq_len: int=4096, rms_norm_eps: float=1e-6):
        super().__init__()
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        values = {"vocab_size": vocab_size, "hidden_dim": hidden_dim, "intermediate_dim": intermediate_dim, "num_layers": num_layers, "num_heads": num_heads, "num_kv_heads": self.num_kv_heads, "max_seq_len": max_seq_len}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"Llama2 dimensions must be positive integers: {values}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={self.num_kv_heads}.")
        if not isinstance(rms_norm_eps, (int, float)) or isinstance(rms_norm_eps, bool) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("Llama2 must be constructed within a MeshDeviceRuntimeContext.")
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = float(rms_norm_eps)
        self.tile_shape = context.default_tile_shape
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % context.default_tile_shape[-1] != 0:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by tile width {context.default_tile_shape[-1]} for zero-copy head views.")
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        self.token_embedding = parameter("token_embedding", vocab_size, hidden_dim)
        self.rope_cos = parameter("rope_cos", 1, 1, max_seq_len, self.head_dim)
        self.rope_sin = parameter("rope_sin", 1, 1, max_seq_len, self.head_dim)
        self.layers = []
        for layer_index in range(num_layers):
            self.layers.append({
                "attention_norm": parameter(f"layers/{layer_index}/attention_norm", hidden_dim),
                "query": parameter(f"layers/{layer_index}/query", hidden_dim, hidden_dim),
                "key": parameter(f"layers/{layer_index}/key", self.num_kv_heads * self.head_dim, hidden_dim),
                "value": parameter(f"layers/{layer_index}/value", self.num_kv_heads * self.head_dim, hidden_dim),
                "attention_output": parameter(f"layers/{layer_index}/attention_output", hidden_dim, hidden_dim),
                "ffn_norm": parameter(f"layers/{layer_index}/ffn_norm", hidden_dim),
                "gate": parameter(f"layers/{layer_index}/gate", intermediate_dim, hidden_dim),
                "up": parameter(f"layers/{layer_index}/up", intermediate_dim, hidden_dim),
                "down": parameter(f"layers/{layer_index}/down", hidden_dim, intermediate_dim),
            })
        self.layers = tuple(self.layers)
        self.final_norm = parameter("final_norm", hidden_dim)

    def forward(self, x: MeshTensorDescriptor, kv_cache: MeshKVCache) -> MeshTensorDescriptor:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        if not isinstance(kv_cache, MeshKVCache):
            raise TypeError(f"kv_cache must be a MeshKVCache, got {type(kv_cache).__name__}.")
        if len(x.shape) != 2:
            raise ValueError(f"Llama2 expects token indices with shape [batch, sequence], got {x.shape}.")
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Llama2 token indices must use torch.int32 or torch.int64, got {x.dtype}.")
        if kv_cache.batch_size != x.shape[0] or kv_cache.num_layers != self.num_layers or kv_cache.num_kv_heads != self.num_kv_heads or kv_cache.head_dim != self.head_dim:
            raise ValueError("The KV cache configuration does not match the Llama2 model and input batch.")
        if kv_cache.dtype != self.token_embedding.dtype:
            raise ValueError("The KV cache dtype must match the Llama2 parameter dtype.")
        sequence_length = x.shape[1]
        past_length = kv_cache.current_length
        if past_length + sequence_length > self.max_seq_len:
            raise ValueError(f"The requested context length {past_length + sequence_length} exceeds max_seq_len={self.max_seq_len}.")

        hidden = mesh_embedding(x, self.token_embedding)
        rope_cos = mesh_narrow(self.rope_cos, -2, past_length, sequence_length)
        rope_sin = mesh_narrow(self.rope_sin, -2, past_length, sequence_length)
        for layer_index, layer in enumerate(self.layers):
            normalized = mesh_rms_norm(hidden, layer["attention_norm"], eps=self.rms_norm_eps)
            query = mesh_permute(mesh_head_split(mesh_linear(normalized, layer["query"]), self.num_heads, self.head_dim), (0, 2, 1, 3))
            key = mesh_permute(mesh_head_split(mesh_linear(normalized, layer["key"]), self.num_kv_heads, self.head_dim), (0, 2, 1, 3))
            value = mesh_permute(mesh_head_split(mesh_linear(normalized, layer["value"]), self.num_kv_heads, self.head_dim), (0, 2, 1, 3))
            query, key = mesh_rope(query, key, rope_cos, rope_sin)
            kv_cache.append(layer_index, key, value)
            cached_key, cached_value = kv_cache.read(layer_index)
            attention = mesh_sdpa(query, cached_key, cached_value, is_causal=True, scale=self.head_dim ** -0.5)
            attention = mesh_head_merge(mesh_permute(attention, (0, 2, 1, 3)))
            hidden = mesh_add(hidden, mesh_linear(attention, layer["attention_output"]))
            normalized = mesh_rms_norm(hidden, layer["ffn_norm"], eps=self.rms_norm_eps)
            feed_forward = mesh_swiglu(mesh_linear(normalized, layer["gate"]), mesh_linear(normalized, layer["up"]))
            hidden = mesh_add(hidden, mesh_linear(feed_forward, layer["down"]))

        hidden = mesh_rms_norm(hidden, self.final_norm, eps=self.rms_norm_eps)
        return mesh_linear(hidden, self.token_embedding)

    def prompt_shape(self, prompt_length: int=None) -> tuple[int, int]:
        return (1, prompt_length if prompt_length is not None else self.max_seq_len)
