from neuromta.system.software.api import *
from neuromta.system.software.nn.common import MeshNetworkBase


__all__ = ["YOLOXNano"]


class YOLOXNano(MeshNetworkBase):
    def __init__(self, image_size: int | tuple[int, int]=416, num_classes: int=80, num_channels: int=3, depth: float=0.33, width: float=0.25):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) and not isinstance(image_size, bool) else tuple(image_size)
        values = {"num_classes": num_classes, "num_channels": num_channels}
        if len(self.image_size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 32 for value in self.image_size):
            raise ValueError("image_size must contain two positive multiples of 32.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"YOLOXNano dimensions must be positive integers: {values}")
        if not isinstance(depth, (int, float)) or isinstance(depth, bool) or depth <= 0:
            raise ValueError("depth must be positive.")
        if not isinstance(width, (int, float)) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be positive.")
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("YOLOXNano must be constructed within a MeshDeviceRuntimeContext.")
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.depth = float(depth)
        self.width = float(width)
        self.tile_shape = context.default_tile_shape
        base_channels = max(8, int(64 * self.width))
        base_depth = max(round(3 * self.depth), 1)
        self.feature_channels = (base_channels * 4, base_channels * 8, base_channels * 16)
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        def conv(name: str, input_channels: int, output_channels: int, kernel_size: int, groups: int=1) -> dict:
            return {"weight": parameter(f"{name}/weight", kernel_size, kernel_size, output_channels, input_channels // groups), "bias": parameter(f"{name}/bias", output_channels), "groups": groups}

        def depthwise_separable(name: str, input_channels: int, output_channels: int, kernel_size: int=3) -> dict:
            return {"depthwise": conv(f"{name}/depthwise", input_channels, input_channels, kernel_size, input_channels), "pointwise": conv(f"{name}/pointwise", input_channels, output_channels, 1)}

        def bottleneck(name: str, channels: int, hidden_channels: int, shortcut: bool) -> dict:
            return {"conv1": conv(f"{name}/conv1", channels, hidden_channels, 1), "conv2": depthwise_separable(f"{name}/conv2", hidden_channels, channels), "shortcut": shortcut}

        def csp(name: str, input_channels: int, output_channels: int, repeats: int, shortcut: bool) -> dict:
            hidden_channels = output_channels // 2
            return {
                "main": conv(f"{name}/main", input_channels, hidden_channels, 1),
                "short": conv(f"{name}/short", input_channels, hidden_channels, 1),
                "blocks": tuple(bottleneck(f"{name}/blocks/{index}", hidden_channels, hidden_channels, shortcut) for index in range(repeats)),
                "final": conv(f"{name}/final", 2 * hidden_channels, output_channels, 1),
            }

        self.stem = conv("backbone/stem", num_channels * 4, base_channels, 3)
        self.dark2_down = depthwise_separable("backbone/dark2/down", base_channels, base_channels * 2)
        self.dark2 = csp("backbone/dark2/csp", base_channels * 2, base_channels * 2, base_depth, True)
        self.dark3_down = depthwise_separable("backbone/dark3/down", base_channels * 2, base_channels * 4)
        self.dark3 = csp("backbone/dark3/csp", base_channels * 4, base_channels * 4, base_depth * 3, True)
        self.dark4_down = depthwise_separable("backbone/dark4/down", base_channels * 4, base_channels * 8)
        self.dark4 = csp("backbone/dark4/csp", base_channels * 8, base_channels * 8, base_depth * 3, True)
        self.dark5_down = depthwise_separable("backbone/dark5/down", base_channels * 8, base_channels * 16)
        self.spp_reduce = conv("backbone/dark5/spp/reduce", base_channels * 16, base_channels * 8, 1)
        self.spp_expand = conv("backbone/dark5/spp/expand", base_channels * 8 * 4, base_channels * 16, 1)
        self.dark5 = csp("backbone/dark5/csp", base_channels * 16, base_channels * 16, base_depth, False)

        self.lateral0 = conv("neck/lateral0", base_channels * 16, base_channels * 8, 1)
        self.csp_p4 = csp("neck/csp_p4", base_channels * 16, base_channels * 8, base_depth, False)
        self.reduce1 = conv("neck/reduce1", base_channels * 8, base_channels * 4, 1)
        self.csp_p3 = csp("neck/csp_p3", base_channels * 8, base_channels * 4, base_depth, False)
        self.down_p3 = depthwise_separable("neck/down_p3", base_channels * 4, base_channels * 4)
        self.csp_n3 = csp("neck/csp_n3", base_channels * 12, base_channels * 8, base_depth, False)
        self.down_n3 = depthwise_separable("neck/down_n3", base_channels * 8, base_channels * 8)
        self.csp_n4 = csp("neck/csp_n4", base_channels * 24, base_channels * 16, base_depth, False)

        head_channels = max(8, int(256 * self.width))
        self.heads = []
        for index, input_channels in enumerate(self.feature_channels):
            self.heads.append({
                "stem": conv(f"head/{index}/stem", input_channels, head_channels, 1),
                "cls": (depthwise_separable(f"head/{index}/cls/0", head_channels, head_channels), depthwise_separable(f"head/{index}/cls/1", head_channels, head_channels)),
                "reg": (depthwise_separable(f"head/{index}/reg/0", head_channels, head_channels), depthwise_separable(f"head/{index}/reg/1", head_channels, head_channels)),
                "cls_pred": conv(f"head/{index}/cls_pred", head_channels, num_classes, 1),
                "box_pred": conv(f"head/{index}/box_pred", head_channels, 4, 1),
                "obj_pred": conv(f"head/{index}/obj_pred", head_channels, 1, 1),
            })
        self.heads = tuple(self.heads)

    @staticmethod
    def _conv(x: MeshTensorDescriptor, parameters: dict, stride: int=1, activation: bool=True) -> MeshTensorDescriptor:
        kernel_size = parameters["weight"].shape[0]
        output = mesh_conv2d(x, parameters["weight"], parameters["bias"], stride=stride, padding=kernel_size // 2, groups=parameters["groups"])
        return mesh_silu(output) if activation else output

    @classmethod
    def _depthwise_separable(cls, x: MeshTensorDescriptor, parameters: dict, stride: int=1, activation: bool=True) -> MeshTensorDescriptor:
        output = cls._conv(x, parameters["depthwise"], stride=stride)
        return cls._conv(output, parameters["pointwise"], activation=activation)

    @classmethod
    def _bottleneck(cls, x: MeshTensorDescriptor, parameters: dict) -> MeshTensorDescriptor:
        output = cls._conv(x, parameters["conv1"])
        output = cls._depthwise_separable(output, parameters["conv2"])
        return mesh_add(x, output) if parameters["shortcut"] else output

    @classmethod
    def _csp(cls, x: MeshTensorDescriptor, parameters: dict) -> MeshTensorDescriptor:
        main = cls._conv(x, parameters["main"])
        for block in parameters["blocks"]:
            main = cls._bottleneck(main, block)
        short = cls._conv(x, parameters["short"])
        return cls._conv(mesh_cat((main, short), dim=-1), parameters["final"])

    @staticmethod
    def _focus(x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        output = mesh_tensor(x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3] * 4, tile_shape=x.tile_shape, dtype=x.dtype)
        return mesh_store(x, output)

    @staticmethod
    def _resize(x: MeshTensorDescriptor, height: int, width: int) -> MeshTensorDescriptor:
        output = mesh_tensor(x.shape[0], height, width, x.shape[-1], tile_shape=x.tile_shape, dtype=x.dtype)
        return mesh_store(x, output)

    def _backbone(self, x: MeshTensorDescriptor) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor, MeshTensorDescriptor]:
        x = self._conv(self._focus(x), self.stem)
        x = self._depthwise_separable(x, self.dark2_down, stride=2)
        x = self._csp(x, self.dark2)
        x = self._depthwise_separable(x, self.dark3_down, stride=2)
        dark3 = self._csp(x, self.dark3)
        x = self._depthwise_separable(dark3, self.dark4_down, stride=2)
        dark4 = self._csp(x, self.dark4)
        x = self._depthwise_separable(dark4, self.dark5_down, stride=2)
        x = self._conv(x, self.spp_reduce)
        pooled = (x,) + tuple(mesh_max_pool2d(x, kernel_size=size, stride=1, padding=size // 2) for size in (5, 9, 13))
        x = self._conv(mesh_cat(pooled, dim=-1), self.spp_expand)
        return dark3, dark4, self._csp(x, self.dark5)

    def _neck(self, features: tuple[MeshTensorDescriptor, MeshTensorDescriptor, MeshTensorDescriptor]) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor, MeshTensorDescriptor]:
        dark3, dark4, dark5 = features
        fpn_out0 = self._conv(dark5, self.lateral0)
        p4 = self._resize(fpn_out0, dark4.shape[1], dark4.shape[2])
        f_out0 = self._csp(mesh_cat((p4, dark4), dim=-1), self.csp_p4)
        fpn_out1 = self._conv(f_out0, self.reduce1)
        p3 = self._resize(fpn_out1, dark3.shape[1], dark3.shape[2])
        pan_out2 = self._csp(mesh_cat((p3, dark3), dim=-1), self.csp_p3)
        p_out1 = self._depthwise_separable(pan_out2, self.down_p3, stride=2)
        pan_out1 = self._csp(mesh_cat((p_out1, f_out0), dim=-1), self.csp_n3)
        p_out0 = self._depthwise_separable(pan_out1, self.down_n3, stride=2)
        pan_out0 = self._csp(mesh_cat((p_out0, dark5), dim=-1), self.csp_n4)
        return pan_out2, pan_out1, pan_out0

    def _head(self, x: MeshTensorDescriptor, parameters: dict) -> MeshTensorDescriptor:
        x = self._conv(x, parameters["stem"])
        cls = x
        reg = x
        for block in parameters["cls"]:
            cls = self._depthwise_separable(cls, block)
        for block in parameters["reg"]:
            reg = self._depthwise_separable(reg, block)
        cls_output = self._conv(cls, parameters["cls_pred"], activation=False)
        box_output = self._conv(reg, parameters["box_pred"], activation=False)
        obj_output = self._conv(reg, parameters["obj_pred"], activation=False)
        return mesh_cat((box_output, obj_output, cls_output), dim=-1)

    def forward(self, x: MeshTensorDescriptor) -> tuple[MeshTensorDescriptor, MeshTensorDescriptor, MeshTensorDescriptor]:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        expected = (self.image_size[0], self.image_size[1], self.num_channels)
        if len(x.shape) != 4 or x.shape[1:] != expected:
            raise ValueError(f"YOLOXNano expects NHWC input shape [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {x.shape}.")
        if x.dtype != self.stem["weight"].dtype:
            raise ValueError("YOLOXNano input dtype must match the model parameter dtype.")
        features = self._neck(self._backbone(x))
        return tuple(self._head(feature, parameters) for feature, parameters in zip(features, self.heads))

    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, self.image_size[0], self.image_size[1], self.num_channels)
