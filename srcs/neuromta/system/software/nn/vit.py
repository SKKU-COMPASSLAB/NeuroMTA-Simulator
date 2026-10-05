from neuromta.system.software.api import *
from neuromta.system.software.nn.common import *


class ViT(MeshNetworkBase):
    def __init__(self, image_size: int=224, patch_size: int=32, num_layers: int=24, num_heads: int=16, hidden_dim: int=1024, mlp_dim: int=4096, num_classes: int=1000, num_channels: int=3, layer_norm_eps: float=1e-6):
        super().__init__()
        values = {"image_size": image_size, "patch_size": patch_size, "num_layers": num_layers, "num_heads": num_heads, "hidden_dim": hidden_dim, "mlp_dim": mlp_dim, "num_classes": num_classes, "num_channels": num_channels}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"ViT dimensions must be positive integers: {values}")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        if not isinstance(layer_norm_eps, (int, float)) or isinstance(layer_norm_eps, bool) or layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.layer_norm_eps = float(layer_norm_eps)
        self.num_patches = (image_size // patch_size) ** 2
        self.sequence_length = self.num_patches + 1
        self.patch_dim = patch_size * patch_size * num_channels
        self.head_dim = hidden_dim // num_heads
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("ViT must be constructed within a MeshDeviceRuntimeContext.")
        self.tile_shape = context.default_tile_shape
        if self.head_dim % self.tile_shape[-1] != 0:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by tile width {self.tile_shape[-1]} for zero-copy head views.")
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        self.patch_weight = parameter("patch_weight", self.hidden_dim, self.patch_dim)
        self.patch_bias = parameter("patch_bias", self.hidden_dim)
        self.class_token = parameter("class_token", 1, 1, self.hidden_dim)
        self.position_embedding = parameter("position_embedding", 1, self.sequence_length, self.hidden_dim)
        layers = []
        for layer_index in range(self.num_layers):
            layers.append({
                "norm1_weight": parameter(f"layers/{layer_index}/norm1_weight", self.hidden_dim),
                "norm1_bias": parameter(f"layers/{layer_index}/norm1_bias", self.hidden_dim),
                "qkv_weight": parameter(f"layers/{layer_index}/qkv_weight", 3 * self.hidden_dim, self.hidden_dim),
                "qkv_bias": parameter(f"layers/{layer_index}/qkv_bias", 3 * self.hidden_dim),
                "projection_weight": parameter(f"layers/{layer_index}/projection_weight", self.hidden_dim, self.hidden_dim),
                "projection_bias": parameter(f"layers/{layer_index}/projection_bias", self.hidden_dim),
                "norm2_weight": parameter(f"layers/{layer_index}/norm2_weight", self.hidden_dim),
                "norm2_bias": parameter(f"layers/{layer_index}/norm2_bias", self.hidden_dim),
                "mlp_weight1": parameter(f"layers/{layer_index}/mlp_weight1", self.mlp_dim, self.hidden_dim),
                "mlp_bias1": parameter(f"layers/{layer_index}/mlp_bias1", self.mlp_dim),
                "mlp_weight2": parameter(f"layers/{layer_index}/mlp_weight2", self.hidden_dim, self.mlp_dim),
                "mlp_bias2": parameter(f"layers/{layer_index}/mlp_bias2", self.hidden_dim),
            })
        self.layers = tuple(layers)
        self.final_norm_weight = parameter("final_norm_weight", self.hidden_dim)
        self.final_norm_bias = parameter("final_norm_bias", self.hidden_dim)
        self.head_weight = parameter("head_weight", self.num_classes, self.hidden_dim)
        self.head_bias = parameter("head_bias", self.num_classes)

    def forward(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        if len(x.shape) != 4:
            raise ValueError(f"ViT expects an NHWC rank-four input, got {x.shape}.")
        if x.shape[1:] != (self.image_size, self.image_size, self.num_channels):
            raise ValueError(f"ViT expects input shape [batch, {self.image_size}, {self.image_size}, {self.num_channels}], got {x.shape}.")
        if x.dtype != self.patch_weight.dtype:
            raise ValueError("ViT input dtype must match the model parameter dtype.")

        batch_size = x.shape[0]
        patches = mesh_tensor(batch_size, self.num_patches, self.patch_dim, tile_shape=(1,) + self.tile_shape)
        patches = mesh_store(x, patches)
        hidden = mesh_linear(patches, self.patch_weight, self.patch_bias)
        hidden = mesh_cat((mesh_expand(self.class_token, batch_size, -1, -1), hidden), dim=1)
        hidden = mesh_add(hidden, self.position_embedding)

        for layer in self.layers:
            normalized = mesh_layer_norm(hidden, layer["norm1_weight"], layer["norm1_bias"], eps=self.layer_norm_eps)
            qkv = mesh_linear(normalized, layer["qkv_weight"], layer["qkv_bias"])
            query, key, value = mesh_chunk(qkv, 3, dim=-1)
            query = mesh_permute(mesh_head_split(query, self.num_heads, self.head_dim), (0, 2, 1, 3))
            key = mesh_permute(mesh_head_split(key, self.num_heads, self.head_dim), (0, 2, 1, 3))
            value = mesh_permute(mesh_head_split(value, self.num_heads, self.head_dim), (0, 2, 1, 3))
            attention = mesh_sdpa(query, key, value, scale=self.head_dim ** -0.5)
            attention = mesh_head_merge(mesh_permute(attention, (0, 2, 1, 3)))
            hidden = mesh_add(hidden, mesh_linear(attention, layer["projection_weight"], layer["projection_bias"]))

            normalized = mesh_layer_norm(hidden, layer["norm2_weight"], layer["norm2_bias"], eps=self.layer_norm_eps)
            mlp = mesh_linear(normalized, layer["mlp_weight1"], layer["mlp_bias1"])
            mlp = mesh_gelu(mlp)
            mlp = mesh_linear(mlp, layer["mlp_weight2"], layer["mlp_bias2"])
            hidden = mesh_add(hidden, mlp)

        hidden = mesh_layer_norm(hidden, self.final_norm_weight, self.final_norm_bias, eps=self.layer_norm_eps)
        class_token = mesh_narrow(hidden, dim=1, start=0, length=1)
        class_token = mesh_sum(mesh_permute(class_token, (0, 2, 1)), dim=-1, keepdim=False)
        return mesh_linear(class_token, self.head_weight, self.head_bias)

    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, self.image_size, self.image_size, self.num_channels)
