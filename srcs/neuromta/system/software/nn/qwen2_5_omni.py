import torch

from neuromta.system.software.api import *
from neuromta.system.software.nn.common import *


class Qwen2_5_Omni(MeshNetworkBase):
    def __init__(self, hidden_dim: int=2048, intermediate_dim: int=11008, num_layers: int=36, num_heads: int=16, num_kv_heads: int=2, vocab_size: int=151936, max_seq_len: int=32768, rms_norm_eps: float=1e-6, vision_hidden_dim: int=1280, vision_intermediate_dim: int=3420, vision_num_layers: int=32, vision_num_heads: int=16, vision_patch_size: int=14, vision_temporal_patch_size: int=2, vision_spatial_merge_size: int=2, vision_num_channels: int=3, vision_window_size: int=112, vision_full_attention_layers: tuple[int, ...]=(7, 15, 23, 31), vision_max_tokens: int=16384, vision_layer_norm_eps: float=1e-6):
        super().__init__()
        values = {"hidden_dim": hidden_dim, "intermediate_dim": intermediate_dim, "num_layers": num_layers, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "vocab_size": vocab_size, "max_seq_len": max_seq_len, "vision_hidden_dim": vision_hidden_dim, "vision_intermediate_dim": vision_intermediate_dim, "vision_num_layers": vision_num_layers, "vision_num_heads": vision_num_heads, "vision_patch_size": vision_patch_size, "vision_temporal_patch_size": vision_temporal_patch_size, "vision_spatial_merge_size": vision_spatial_merge_size, "vision_num_channels": vision_num_channels, "vision_window_size": vision_window_size, "vision_max_tokens": vision_max_tokens}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"Qwen2.5-Omni dimensions must be positive integers: {values}")
        if hidden_dim % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads, and num_heads must be divisible by num_kv_heads.")
        if vision_hidden_dim % vision_num_heads != 0 or vision_window_size % vision_patch_size != 0:
            raise ValueError("vision_hidden_dim must be divisible by vision_num_heads, and vision_window_size must be divisible by vision_patch_size.")
        if any(not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0 for eps in (rms_norm_eps, vision_layer_norm_eps)):
            raise ValueError("Normalization epsilon values must be positive.")
        full_attention_layers = tuple(int(layer) for layer in vision_full_attention_layers)
        if len(set(full_attention_layers)) != len(full_attention_layers) or any(layer < 0 or layer >= vision_num_layers for layer in full_attention_layers):
            raise ValueError(f"Invalid vision_full_attention_layers: {full_attention_layers}")
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("Qwen2_5_Omni must be constructed within a MeshDeviceRuntimeContext.")
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = float(rms_norm_eps)
        self.head_dim = hidden_dim // num_heads
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_patch_size = vision_patch_size
        self.vision_temporal_patch_size = vision_temporal_patch_size
        self.vision_spatial_merge_size = vision_spatial_merge_size
        self.vision_num_channels = vision_num_channels
        self.vision_window_size = vision_window_size
        self.vision_full_attention_layers = full_attention_layers
        self.vision_max_tokens = vision_max_tokens
        self.vision_layer_norm_eps = float(vision_layer_norm_eps)
        self.vision_head_dim = vision_hidden_dim // vision_num_heads
        self.tile_shape = context.default_tile_shape
        if self.head_dim % self.tile_shape[-1] != 0:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by tile width {self.tile_shape[-1]} for zero-copy language-model head views.")
        if self.vision_head_dim < 2:
            raise ValueError("vision_head_dim must be at least two for two-dimensional rotary sections.")
        self.mrope_sections = (max(1, self.head_dim // 8), max(1, 3 * self.head_dim // 16), max(1, 3 * self.head_dim // 16))
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        self.token_embedding = parameter("token_embedding", vocab_size, hidden_dim)
        self.language_rope_cos = parameter("language_rope_cos", 1, 1, max_seq_len, self.head_dim)
        self.language_rope_sin = parameter("language_rope_sin", 1, 1, max_seq_len, self.head_dim)
        self.language_layers = []
        for layer_index in range(num_layers):
            self.language_layers.append({
                "attention_norm": parameter(f"language_layers/{layer_index}/attention_norm", hidden_dim),
                "query": parameter(f"language_layers/{layer_index}/query", hidden_dim, hidden_dim),
                "key": parameter(f"language_layers/{layer_index}/key", num_kv_heads * self.head_dim, hidden_dim),
                "value": parameter(f"language_layers/{layer_index}/value", num_kv_heads * self.head_dim, hidden_dim),
                "attention_output": parameter(f"language_layers/{layer_index}/attention_output", hidden_dim, hidden_dim),
                "ffn_norm": parameter(f"language_layers/{layer_index}/ffn_norm", hidden_dim),
                "gate": parameter(f"language_layers/{layer_index}/gate", intermediate_dim, hidden_dim),
                "up": parameter(f"language_layers/{layer_index}/up", intermediate_dim, hidden_dim),
                "down": parameter(f"language_layers/{layer_index}/down", hidden_dim, intermediate_dim),
            })
        self.language_layers = tuple(self.language_layers)
        self.final_norm = parameter("final_norm", hidden_dim)
        self.lm_head = parameter("lm_head", vocab_size, hidden_dim)
        patch_dim = vision_temporal_patch_size * vision_patch_size * vision_patch_size * vision_num_channels
        self.vision_patch_dim = patch_dim
        self.vision_patch_projection = parameter("vision_patch_projection", vision_hidden_dim, patch_dim)
        self.vision_rope_cos = parameter("vision_rope_cos", 1, 1, vision_max_tokens, self.vision_head_dim)
        self.vision_rope_sin = parameter("vision_rope_sin", 1, 1, vision_max_tokens, self.vision_head_dim)
        self.vision_layers = []
        for layer_index in range(vision_num_layers):
            self.vision_layers.append({
                "attention_norm_weight": parameter(f"vision_layers/{layer_index}/attention_norm_weight", vision_hidden_dim),
                "attention_norm_bias": parameter(f"vision_layers/{layer_index}/attention_norm_bias", vision_hidden_dim),
                "qkv": parameter(f"vision_layers/{layer_index}/qkv", 3 * vision_hidden_dim, vision_hidden_dim),
                "attention_output": parameter(f"vision_layers/{layer_index}/attention_output", vision_hidden_dim, vision_hidden_dim),
                "ffn_norm_weight": parameter(f"vision_layers/{layer_index}/ffn_norm_weight", vision_hidden_dim),
                "ffn_norm_bias": parameter(f"vision_layers/{layer_index}/ffn_norm_bias", vision_hidden_dim),
                "mlp_up": parameter(f"vision_layers/{layer_index}/mlp_up", vision_intermediate_dim, vision_hidden_dim),
                "mlp_down": parameter(f"vision_layers/{layer_index}/mlp_down", vision_hidden_dim, vision_intermediate_dim),
            })
        self.vision_layers = tuple(self.vision_layers)
        merge_factor = vision_spatial_merge_size ** 2
        self.vision_merger_norm_weight = parameter("vision_merger_norm_weight", vision_hidden_dim)
        self.vision_merger_norm_bias = parameter("vision_merger_norm_bias", vision_hidden_dim)
        self.vision_projector = parameter("vision_projector", hidden_dim, merge_factor * vision_hidden_dim)

    def _split_heads(self, x: MeshTensorDescriptor, num_heads: int, head_dim: int) -> MeshTensorDescriptor:
        if head_dim % self.tile_shape[-1] == 0:
            return mesh_permute(mesh_head_split(x, num_heads, head_dim), (0, 2, 1, 3))
        output = mesh_tensor(x.shape[0], x.shape[1], num_heads, head_dim, tile_shape=(1, self.tile_shape[-2], 1, self.tile_shape[-1]), dtype=x.dtype)
        return mesh_permute(mesh_store(x, output), (0, 2, 1, 3))

    def _merge_heads(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        transposed = mesh_permute(x, (0, 2, 1, 3))
        if x.shape[-1] % self.tile_shape[-1] == 0:
            return mesh_head_merge(transposed)
        output = mesh_tensor(x.shape[0], x.shape[2], x.shape[1] * x.shape[3], tile_shape=(1,) + self.tile_shape, dtype=x.dtype)
        return mesh_store(transposed, output)

    def vision_encoder(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        if len(x.shape) != 5:
            raise ValueError(f"Qwen2.5-Omni Vision Encoder expects [batch, frames, height, width, channels], got {x.shape}.")
        if x.dtype != self.vision_patch_projection.dtype:
            raise ValueError("Vision input dtype must match the Qwen2.5-Omni parameter dtype.")
        batch_size, frames, height, width, channels = x.shape
        if channels != self.vision_num_channels or frames % self.vision_temporal_patch_size != 0 or height % self.vision_patch_size != 0 or width % self.vision_patch_size != 0:
            raise ValueError("Vision input channels and temporal/spatial dimensions must match the configured patch shape.")
        temporal_patches = frames // self.vision_temporal_patch_size
        patch_rows = height // self.vision_patch_size
        patch_columns = width // self.vision_patch_size
        patch_tokens = temporal_patches * patch_rows * patch_columns
        merge_factor = self.vision_spatial_merge_size ** 2
        if patch_tokens > self.vision_max_tokens or patch_rows % self.vision_spatial_merge_size != 0 or patch_columns % self.vision_spatial_merge_size != 0:
            raise ValueError("Vision patch count exceeds capacity or cannot be spatially merged without padding.")
        patches = mesh_tensor(batch_size, patch_tokens, self.vision_patch_dim, tile_shape=(1,) + self.tile_shape, dtype=x.dtype)
        hidden = mesh_linear(mesh_store(x, patches), self.vision_patch_projection)
        rope_cos = mesh_narrow(self.vision_rope_cos, -2, 0, patch_tokens)
        rope_sin = mesh_narrow(self.vision_rope_sin, -2, 0, patch_tokens)
        vision_rope_sections = (self.vision_head_dim // 2, self.vision_head_dim - self.vision_head_dim // 2)
        window_tokens = (self.vision_window_size // self.vision_patch_size) ** 2

        for layer_index, layer in enumerate(self.vision_layers):
            normalized = mesh_layer_norm(hidden, layer["attention_norm_weight"], layer["attention_norm_bias"], eps=self.vision_layer_norm_eps)
            query, key, value = mesh_chunk(mesh_linear(normalized, layer["qkv"]), 3, dim=-1)
            query = self._split_heads(query, self.vision_num_heads, self.vision_head_dim)
            key = self._split_heads(key, self.vision_num_heads, self.vision_head_dim)
            value = self._split_heads(value, self.vision_num_heads, self.vision_head_dim)
            query, key = mesh_rope(query, key, rope_cos, rope_sin, rotary_sections=vision_rope_sections)
            if layer_index in self.vision_full_attention_layers or patch_tokens <= window_tokens:
                attention = mesh_sdpa(query, key, value, scale=self.vision_head_dim ** -0.5)
            else:
                query_windows = mesh_split(query, window_tokens, dim=2)
                key_windows = mesh_split(key, window_tokens, dim=2)
                value_windows = mesh_split(value, window_tokens, dim=2)
                attention_windows = []
                for query_window, key_window, value_window in zip(query_windows, key_windows, value_windows):
                    attention_windows.append(mesh_sdpa(query_window, key_window, value_window, scale=self.vision_head_dim ** -0.5))
                attention = mesh_cat(attention_windows, dim=2)
            hidden = mesh_add(hidden, mesh_linear(self._merge_heads(attention), layer["attention_output"]))
            normalized = mesh_layer_norm(hidden, layer["ffn_norm_weight"], layer["ffn_norm_bias"], eps=self.vision_layer_norm_eps)
            hidden = mesh_add(hidden, mesh_linear(mesh_gelu(mesh_linear(normalized, layer["mlp_up"])), layer["mlp_down"]))

        hidden = mesh_layer_norm(hidden, self.vision_merger_norm_weight, self.vision_merger_norm_bias, eps=self.vision_layer_norm_eps)
        merged_tokens = temporal_patches * (patch_rows // self.vision_spatial_merge_size) * (patch_columns // self.vision_spatial_merge_size)
        merged = mesh_tensor(batch_size, merged_tokens, merge_factor * self.vision_hidden_dim, tile_shape=(1,) + self.tile_shape, dtype=x.dtype)
        return mesh_linear(mesh_store(hidden, merged), self.vision_projector)

    def prefill_vision_tokens(self, kv_cache: MeshKVCache, visual_tokens: MeshTensorDescriptor) -> None:
        return self.language_model(None, kv_cache, visual_tokens=visual_tokens)

    def language_model(self, x: MeshTensorDescriptor, kv_cache: MeshKVCache, visual_tokens: MeshTensorDescriptor=None) -> MeshTensorDescriptor:
        if not isinstance(kv_cache, MeshKVCache):
            raise TypeError(f"kv_cache must be a MeshKVCache, got {type(kv_cache).__name__}.")

        if x is not None:
            if not (isinstance(x, MeshTensorDescriptor)):
                raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
            if len(x.shape) != 2 or x.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"Language-model input must contain integer token indices with shape [batch, sequence], got {x.shape} and {x.dtype}.")
            if kv_cache.batch_size != x.shape[0] or kv_cache.num_layers != self.num_layers or kv_cache.num_kv_heads != self.num_kv_heads or kv_cache.head_dim != self.head_dim or kv_cache.dtype != self.token_embedding.dtype:
                raise ValueError("The KV cache configuration does not match the Qwen language model and input batch.")
            hidden = mesh_embedding(x, self.token_embedding)
            if visual_tokens is not None:
                if not isinstance(visual_tokens, MeshTensorDescriptor) or len(visual_tokens.shape) != 3 or visual_tokens.shape[0] != x.shape[0] or visual_tokens.shape[-1] != self.hidden_dim or visual_tokens.dtype != hidden.dtype:
                    raise ValueError("visual_tokens must have shape [batch, visual_sequence, hidden_dim] and match the language-model dtype.")
                hidden = mesh_cat((visual_tokens, hidden), dim=1)
        else:
            if visual_tokens is None:
                raise ValueError("Either x or visual_tokens must be provided.")
            if not isinstance(visual_tokens, MeshTensorDescriptor) or len(visual_tokens.shape) != 3 or visual_tokens.shape[0] != kv_cache.batch_size or visual_tokens.shape[-1] != self.hidden_dim or visual_tokens.dtype != kv_cache.dtype:
                raise ValueError("visual_tokens must have shape [batch, visual_sequence, hidden_dim] and match the KV cache dtype.")
            hidden = visual_tokens
        sequence_length = hidden.shape[1]

        past_length = kv_cache.current_length
        if past_length + sequence_length > self.max_seq_len:
            raise ValueError(f"The requested context length {past_length + sequence_length} exceeds max_seq_len={self.max_seq_len}.")
        rope_cos = mesh_narrow(self.language_rope_cos, -2, past_length, sequence_length)
        rope_sin = mesh_narrow(self.language_rope_sin, -2, past_length, sequence_length)
        for layer_index, layer in enumerate(self.language_layers):
            normalized = mesh_rms_norm(hidden, layer["attention_norm"], eps=self.rms_norm_eps)
            query = self._split_heads(mesh_linear(normalized, layer["query"]), self.num_heads, self.head_dim)
            key = self._split_heads(mesh_linear(normalized, layer["key"]), self.num_kv_heads, self.head_dim)
            value = self._split_heads(mesh_linear(normalized, layer["value"]), self.num_kv_heads, self.head_dim)
            query, key = mesh_rope(query, key, rope_cos, rope_sin, rotary_sections=self.mrope_sections)
            kv_cache.append(layer_index, key, value)
            cached_key, cached_value = kv_cache.read(layer_index)
            attention = mesh_sdpa(query, cached_key, cached_value, is_causal=True, scale=self.head_dim ** -0.5)
            hidden = mesh_add(hidden, mesh_linear(self._merge_heads(attention), layer["attention_output"]))
            normalized = mesh_rms_norm(hidden, layer["ffn_norm"], eps=self.rms_norm_eps)
            hidden = mesh_add(hidden, mesh_linear(mesh_swiglu(mesh_linear(normalized, layer["gate"]), mesh_linear(normalized, layer["up"])), layer["down"]))

        hidden = mesh_rms_norm(hidden, self.final_norm, eps=self.rms_norm_eps)
        return mesh_linear(hidden, self.lm_head)

    def video_shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.vision_temporal_patch_size, self.vision_patch_size * self.vision_spatial_merge_size, self.vision_patch_size * self.vision_spatial_merge_size, self.vision_num_channels)

    def prompt_shape(self, prompt_length: int=None) -> tuple[int, int]:
        return (1, prompt_length if prompt_length is not None else self.max_seq_len)
