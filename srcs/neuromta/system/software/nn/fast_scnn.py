from neuromta.system.software.api import *
from neuromta.system.software.nn.common import MeshNetworkBase


__all__ = ["FastSCNN"]


class FastSCNN(MeshNetworkBase):
    def __init__(self, image_size: tuple[int, int]=(512, 1024), num_classes: int=19, num_channels: int=3, expansion: int=6, pyramid_bins: tuple[int, ...]=(1, 2, 3, 6)):
        super().__init__()
        self.image_size = tuple(image_size)
        values = {"num_classes": num_classes, "num_channels": num_channels, "expansion": expansion}
        if len(self.image_size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value < 192 or value % 32 for value in self.image_size):
            raise ValueError("image_size must contain two multiples of 32 that are at least 192.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"FastSCNN dimensions must be positive integers: {values}")
        if not pyramid_bins or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in pyramid_bins):
            raise ValueError("pyramid_bins must contain positive integers.")
        context = get_global_mesh_device_context()
        if context is None:
            raise RuntimeError("FastSCNN must be constructed within a MeshDeviceRuntimeContext.")
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.expansion = expansion
        self.pyramid_bins = tuple(pyramid_bins)
        self.tile_shape = context.default_tile_shape
        feature_height, feature_width = self.image_size[0] // 32, self.image_size[1] // 32
        if max(self.pyramid_bins) > min(feature_height, feature_width):
            raise ValueError("The largest pyramid bin must fit the 1/32-resolution feature map.")
        parameter_namespace = f"{type(self).__module__}.{type(self).__qualname__}:{id(self):x}"

        def parameter(name: str, *shape: int) -> MeshTensorDescriptor:
            tensor = MeshTensorDescriptor(shape=shape, tile_shape=self.tile_shape[-min(len(self.tile_shape), len(shape)):], dtype=context.default_dtype, tensor_type=MeshTensorType.WEIGHT)
            tensor.parameter_id = f"{parameter_namespace}/{name}"
            return tensor

        def conv(name: str, input_channels: int, output_channels: int, kernel_size: int, groups: int=1) -> dict:
            return {"weight": parameter(f"{name}/weight", kernel_size, kernel_size, output_channels, input_channels // groups), "bias": parameter(f"{name}/bias", output_channels), "groups": groups}

        def depthwise_separable(name: str, input_channels: int, output_channels: int) -> dict:
            return {"depthwise": conv(f"{name}/depthwise", input_channels, input_channels, 3, input_channels), "pointwise": conv(f"{name}/pointwise", input_channels, output_channels, 1)}

        def bottleneck(name: str, input_channels: int, output_channels: int, stride: int) -> dict:
            expanded_channels = input_channels * expansion
            return {
                "expand": conv(f"{name}/expand", input_channels, expanded_channels, 1),
                "depthwise": conv(f"{name}/depthwise", expanded_channels, expanded_channels, 3, expanded_channels),
                "project": conv(f"{name}/project", expanded_channels, output_channels, 1),
                "stride": stride,
                "residual": stride == 1 and input_channels == output_channels,
            }

        def stage(name: str, input_channels: int, output_channels: int, repeats: int, stride: int) -> tuple[dict, ...]:
            blocks = [bottleneck(f"{name}/0", input_channels, output_channels, stride)]
            blocks.extend(bottleneck(f"{name}/{index}", output_channels, output_channels, 1) for index in range(1, repeats))
            return tuple(blocks)

        self.downsample_conv = conv("learning_to_downsample/conv", num_channels, 32, 3)
        self.downsample_dsconv1 = depthwise_separable("learning_to_downsample/dsconv1", 32, 48)
        self.downsample_dsconv2 = depthwise_separable("learning_to_downsample/dsconv2", 48, 64)
        self.stage64 = stage("global_features/stage64", 64, 64, 3, 2)
        self.stage96 = stage("global_features/stage96", 64, 96, 3, 2)
        self.stage128 = stage("global_features/stage128", 96, 128, 3, 1)
        branch_channels = 32
        self.pyramid_branches = tuple(conv(f"global_features/pyramid/{bin_size}", 128, branch_channels, 1) for bin_size in self.pyramid_bins)
        self.pyramid_fusion = conv("global_features/pyramid/fusion", 128 + branch_channels * len(self.pyramid_bins), 128, 1)
        self.low_depthwise = conv("feature_fusion/low_depthwise", 128, 128, 3, 128)
        self.low_pointwise = conv("feature_fusion/low_pointwise", 128, 128, 1)
        self.high_projection = conv("feature_fusion/high_projection", 64, 128, 1)
        self.classifier1 = depthwise_separable("classifier/dsconv1", 128, 128)
        self.classifier2 = depthwise_separable("classifier/dsconv2", 128, 128)
        self.classifier = conv("classifier/output", 128, num_classes, 1)

    @staticmethod
    def _conv(x: MeshTensorDescriptor, parameters: dict, stride: int=1, activation: bool=True, dilation: int=1) -> MeshTensorDescriptor:
        kernel_size = parameters["weight"].shape[0]
        output = mesh_conv2d(x, parameters["weight"], parameters["bias"], stride=stride, padding=dilation * (kernel_size // 2), dilation=dilation, groups=parameters["groups"])
        return mesh_relu(output) if activation else output

    @classmethod
    def _depthwise_separable(cls, x: MeshTensorDescriptor, parameters: dict, stride: int=1) -> MeshTensorDescriptor:
        output = cls._conv(x, parameters["depthwise"], stride=stride)
        return cls._conv(output, parameters["pointwise"])

    @classmethod
    def _bottleneck(cls, x: MeshTensorDescriptor, parameters: dict) -> MeshTensorDescriptor:
        output = cls._conv(x, parameters["expand"])
        output = cls._conv(output, parameters["depthwise"], stride=parameters["stride"])
        output = cls._conv(output, parameters["project"], activation=False)
        return mesh_add(x, output) if parameters["residual"] else output

    @staticmethod
    def _resize(x: MeshTensorDescriptor, height: int, width: int) -> MeshTensorDescriptor:
        output = mesh_tensor(x.shape[0], height, width, x.shape[-1], tile_shape=x.tile_shape, dtype=x.dtype)
        return mesh_store(x, output)

    @staticmethod
    def _adaptive_avg_pool(x: MeshTensorDescriptor, output_size: int) -> MeshTensorDescriptor:
        stride = (max(1, x.shape[1] // output_size), max(1, x.shape[2] // output_size))
        kernel = (x.shape[1] - (output_size - 1) * stride[0], x.shape[2] - (output_size - 1) * stride[1])
        return mesh_avg_pool2d(x, kernel_size=kernel, stride=stride)

    def _learning_to_downsample(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        x = self._conv(x, self.downsample_conv, stride=2)
        x = self._depthwise_separable(x, self.downsample_dsconv1, stride=2)
        return self._depthwise_separable(x, self.downsample_dsconv2, stride=2)

    def _global_features(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        for block in self.stage64 + self.stage96 + self.stage128:
            x = self._bottleneck(x, block)
        branches = [x]
        for bin_size, parameters in zip(self.pyramid_bins, self.pyramid_branches):
            branch = self._adaptive_avg_pool(x, bin_size)
            branch = self._conv(branch, parameters)
            branches.append(self._resize(branch, x.shape[1], x.shape[2]))
        return self._conv(mesh_cat(tuple(branches), dim=-1), self.pyramid_fusion)

    def _feature_fusion(self, high_resolution: MeshTensorDescriptor, low_resolution: MeshTensorDescriptor) -> MeshTensorDescriptor:
        low = self._conv(low_resolution, self.low_depthwise, dilation=4)
        low = self._conv(low, self.low_pointwise, activation=False)
        low = self._resize(low, high_resolution.shape[1], high_resolution.shape[2])
        high = self._conv(high_resolution, self.high_projection, activation=False)
        return mesh_relu(mesh_add(high, low))

    def forward(self, x: MeshTensorDescriptor) -> MeshTensorDescriptor:
        if not isinstance(x, MeshTensorDescriptor):
            raise TypeError(f"x must be a MeshTensorDescriptor, got {type(x).__name__}.")
        expected = (self.image_size[0], self.image_size[1], self.num_channels)
        if len(x.shape) != 4 or x.shape[1:] != expected:
            raise ValueError(f"FastSCNN expects NHWC input shape [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {x.shape}.")
        if x.dtype != self.downsample_conv["weight"].dtype:
            raise ValueError("FastSCNN input dtype must match the model parameter dtype.")
        high_resolution = self._learning_to_downsample(x)
        low_resolution = self._global_features(high_resolution)
        output = self._feature_fusion(high_resolution, low_resolution)
        output = self._depthwise_separable(output, self.classifier1)
        output = self._depthwise_separable(output, self.classifier2)
        output = self._conv(output, self.classifier, activation=False)
        return self._resize(output, self.image_size[0], self.image_size[1])

    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, self.image_size[0], self.image_size[1], self.num_channels)
