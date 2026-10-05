from neuromta.system.software.api import *
from neuromta.system.software.nn.common import MeshNetworkBase


__all__ = ["MobileNetV3", "MobileNetV3Small", "MobileNetV3Large"]


def _make_divisible(value: float, divisor: int=8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class MobileNetV3(MeshNetworkBase):
    _CONFIGS = {
        "small": (
            (3, 16, 16, True, "relu", 2),
            (3, 72, 24, False, "relu", 2),
            (3, 88, 24, False, "relu", 1),
            (5, 96, 40, True, "hswish", 2),
            (5, 240, 40, True, "hswish", 1),
            (5, 240, 40, True, "hswish", 1),
            (5, 120, 48, True, "hswish", 1),
            (5, 144, 48, True, "hswish", 1),
            (5, 288, 96, True, "hswish", 2),
            (5, 576, 96, True, "hswish", 1),
            (5, 576, 96, True, "hswish", 1),
        ),
        "large": (
            (3, 16, 16, False, "relu", 1),
            (3, 64, 24, False, "relu", 2),
            (3, 72, 24, False, "relu", 1),
            (5, 72, 40, True, "relu", 2),
            (5, 120, 40, True, "relu", 1),
            (5, 120, 40, True, "relu", 1),
            (3, 240, 80, False, "hswish", 2),
            (3, 200, 80, False, "hswish", 1),
            (3, 184, 80, False, "hswish", 1),
            (3, 184, 80, False, "hswish", 1),
            (3, 480, 112, True, "hswish", 1),
            (3, 672, 112, True, "hswish", 1),
            (5, 672, 160, True, "hswish", 2),
            (5, 960, 160, True, "hswish", 1),
            (5, 960, 160, True, "hswish", 1),
        ),
    }

    def __init__(self, variant: str="small", image_size: int | tuple[int, int]=224, num_classes: int=1000, num_channels: int=3, width_multiplier: float=1.0, squeeze_excitation_ratio: float=0.25):
        super().__init__()
        self.variant = str(variant).lower()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) and not isinstance(image_size, bool) else tuple(image_size)
        values = {"num_classes": num_classes, "num_channels": num_channels}
        if self.variant not in self._CONFIGS:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")
        if len(self.image_size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 32 for value in self.image_size):
            raise ValueError("image_size must contain two positive multiples of 32.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"MobileNetV3 dimensions must be positive integers: {values}")
        if not isinstance(width_multiplier, (int, float)) or isinstance(width_multiplier, bool) or width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive.")
        if not isinstance(squeeze_excitation_ratio, (int, float)) or isinstance(squeeze_excitation_ratio, bool) or not 0 < squeeze_excitation_ratio <= 1:
            raise ValueError("squeeze_excitation_ratio must be in (0, 1].")
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("MobileNetV3 must be constructed within a MeshDeviceRuntimeContext.")
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.width_multiplier = float(width_multiplier)
        self.squeeze_excitation_ratio = float(squeeze_excitation_ratio)
        self.tile_shape = context.default_tile_shape
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        def conv(name: str, input_channels: int, output_channels: int, kernel_size: int, groups: int=1) -> dict:
            return {"weight": parameter(f"{name}/weight", kernel_size, kernel_size, output_channels, input_channels // groups), "bias": parameter(f"{name}/bias", output_channels), "groups": groups}

        input_channels = _make_divisible(16 * self.width_multiplier)
        self.stem = conv("stem", num_channels, input_channels, 3)
        blocks = []
        for index, (kernel_size, expansion_channels, output_channels, use_se, activation, stride) in enumerate(self._CONFIGS[self.variant]):
            expanded = _make_divisible(expansion_channels * self.width_multiplier)
            output = _make_divisible(output_channels * self.width_multiplier)
            squeeze = _make_divisible(expanded * self.squeeze_excitation_ratio)
            blocks.append({
                "expand": None if expanded == input_channels else conv(f"blocks/{index}/expand", input_channels, expanded, 1),
                "depthwise": conv(f"blocks/{index}/depthwise", expanded, expanded, kernel_size, expanded),
                "se_reduce": conv(f"blocks/{index}/se/reduce", expanded, squeeze, 1) if use_se else None,
                "se_expand": conv(f"blocks/{index}/se/expand", squeeze, expanded, 1) if use_se else None,
                "project": conv(f"blocks/{index}/project", expanded, output, 1),
                "activation": activation,
                "stride": stride,
                "residual": stride == 1 and input_channels == output,
            })
            input_channels = output
        self.blocks = tuple(blocks)
        final_channels = _make_divisible((576 if self.variant == "small" else 960) * self.width_multiplier)
        classifier_channels = _make_divisible((1024 if self.variant == "small" else 1280) * max(1.0, self.width_multiplier))
        self.final_conv = conv("final_conv", input_channels, final_channels, 1)
        self.classifier_conv = conv("classifier/hidden", final_channels, classifier_channels, 1)
        self.classifier_output = conv("classifier/output", classifier_channels, num_classes, 1)

    @staticmethod
    def _hard_sigmoid(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        return mesh_scale(mesh_clamp(mesh_add(x, 3.0), minimum=0.0, maximum=6.0), 1.0 / 6.0)

    @classmethod
    def _hard_swish(cls, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        return mesh_mul(x, cls._hard_sigmoid(x))

    @classmethod
    def _activate(cls, x: MeshTensorDescriptor, activation: str) -> MeshTensorDescriptor:
        return mesh_relu(x) if activation == "relu" else cls._hard_swish(x)

    @staticmethod
    def _conv(x: MeshTensorDescriptor, parameters: dict, stride: int=1) -> MeshTensorDescriptor:
        kernel_size = parameters["weight"].shape[0]
        return mesh_conv2d(x, parameters["weight"], parameters["bias"], stride=stride, padding=kernel_size // 2, groups=parameters["groups"])

    @classmethod
    def _squeeze_excitation(cls, x: MeshTensorDescriptor, reduce_parameters: dict, expand_parameters: dict) -> MeshTensorDescriptor:
        scale = mesh_avg_pool2d(x, kernel_size=(x.shape[1], x.shape[2]), stride=(x.shape[1], x.shape[2]))
        scale = mesh_relu(cls._conv(scale, reduce_parameters))
        scale = cls._hard_sigmoid(cls._conv(scale, expand_parameters))
        return mesh_mul(x, scale)

    @classmethod
    def _block(cls, x: MeshTensorDescriptor, parameters: dict) -> MeshTensorDescriptor:
        residual = x
        if parameters["expand"] is not None:
            x = cls._activate(cls._conv(x, parameters["expand"]), parameters["activation"])
        x = cls._activate(cls._conv(x, parameters["depthwise"], stride=parameters["stride"]), parameters["activation"])
        if parameters["se_reduce"] is not None:
            x = cls._squeeze_excitation(x, parameters["se_reduce"], parameters["se_expand"])
        x = cls._conv(x, parameters["project"])
        return mesh_add(residual, x) if parameters["residual"] else x

    def forward(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        expected = (self.image_size[0], self.image_size[1], self.num_channels)
        if len(x.shape) != 4 or x.shape[1:] != expected:
            raise ValueError(f"MobileNetV3 expects NHWC input shape [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {x.shape}.")
        if x.dtype != self.stem["weight"].dtype:
            raise ValueError("MobileNetV3 input dtype must match the model parameter dtype.")
        x = self._hard_swish(self._conv(x, self.stem, stride=2))
        for block in self.blocks:
            x = self._block(x, block)
        x = self._hard_swish(self._conv(x, self.final_conv))
        x = mesh_avg_pool2d(x, kernel_size=(x.shape[1], x.shape[2]), stride=(x.shape[1], x.shape[2]))
        x = self._hard_swish(self._conv(x, self.classifier_conv))
        x = self._conv(x, self.classifier_output)
        output = mesh_tensor(x.shape[0], self.num_classes, tile_shape=(1, self.tile_shape[-1]), dtype=x.dtype)
        return mesh_store(x, output)

    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, self.image_size[0], self.image_size[1], self.num_channels)


class MobileNetV3Small(MobileNetV3):
    def __init__(self, image_size: int | tuple[int, int]=224, num_classes: int=1000, num_channels: int=3, width_multiplier: float=1.0, squeeze_excitation_ratio: float=0.25):
        super().__init__("small", image_size, num_classes, num_channels, width_multiplier, squeeze_excitation_ratio)


class MobileNetV3Large(MobileNetV3):
    def __init__(self, image_size: int | tuple[int, int]=224, num_classes: int=1000, num_channels: int=3, width_multiplier: float=1.0, squeeze_excitation_ratio: float=0.25):
        super().__init__("large", image_size, num_classes, num_channels, width_multiplier, squeeze_excitation_ratio)
