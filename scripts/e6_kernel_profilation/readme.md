# Experiment 6: Kernel Profilation

## What to do?

커널의 실행시간을 core mesh shape을 기준으로 프로파일링한다. 프로파일링의 대상의 되는 커널들은 다음과 같다.

```python
def linear_kernel_workload(M, N, K, dtype=torch.bfloat16):
    x = mesh_tensor(M, K, dtype=dtype).as_intermediate()
    w = mesh_parameter(N, K, dtype=dtype).as_weight()
    b = mesh_parameter(N, dtype=dtype).as_weight()
    return mesh_linear(x, w, b)

def conv2d_kernel_workload(N, C, H, W, K, R, S, stride, padding, dilation, dtype=torch.bfloat16):
    x = mesh_tensor(N, H, W, C, tile_shape=(1, 1, 32, 32), dtype=dtype).as_intermediate()
    w = mesh_parameter(R, S, K, C, tile_shape=(1, 1, 32, 32), dtype=dtype).as_weight()
    b = mesh_parameter(K, dtype=dtype).as_weight()
    return mesh_conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation)

def elementwise_kernel_workload(M, N, dtype=torch.bfloat16):
    x = mesh_tensor(M, N, dtype=dtype).as_intermediate()
    return mesh_relu(x)

def reduction_kernel_workload(M, N, dtype=torch.bfloat16):
    x = mesh_tensor(M, N, dtype=dtype).as_intermediate()
    return mesh_sum(x, dim=-1)

WORKLOADS: dict[str, Callable] = {
    "linear_large": lambda: linear_kernel_workload(32, 1024, 1024),
    "linear_medium": lambda: linear_kernel_workload(32, 512, 512),
    "linear_small": lambda: linear_kernel_workload(32, 256, 256),
    "conv2d_large": lambda: conv2d_kernel_workload(32, 64, 32, 32, 128, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_medium": lambda: conv2d_kernel_workload(32, 32, 16, 16, 64, 3, 3, stride=1, padding=1, dilation=1),
    "conv2d_small": lambda: conv2d_kernel_workload(32, 16, 8, 8, 32, 3, 3, stride=1, padding=1, dilation=1),
    "elementwise_large": lambda: elementwise_kernel_workload(1024, 1024),
    "elementwise_medium": lambda: elementwise_kernel_workload(512, 512),
    "elementwise_small": lambda: elementwise_kernel_workload(256, 256),
    "reduction_large": lambda: reduction_kernel_workload(1024, 1024),
    "reduction_medium": lambda: reduction_kernel_workload(512, 512),
    "reduction_small": lambda: reduction_kernel_workload(256, 256),
}
```

각 커널들에 `1x1`, `1x4`, `1x8`, `2x1`, `2x4`, `2x8`, `4x1`, `4x4`, `4x8`, `8x1`, `8x4`, `8x8` 크기의 core mesh를 할당하여 실행할 때 실행 시간을 시뮬레이션하고,
각 커널에 대해 core mesh 크기에 따라 지연시간이 어떻게 변화하는지를 시각적으로 visualize한다.

`run.py`에서는 모든 configuration에 대한 실험을 병렬적으로 멀티 프로세스 기반으로 실행한 후, 결과를 `.cache` 디렉터리에 저장한다.
(다른 실험 스크립트들을 참고한다)
`visualize.py`에서는 실험 결과를 바탕으로 각 커널들의 실행시간을 막대그래프로 시각화한다.
생성되는 이미지는 `visualize_kernel_latency_[커널 이름].png` 이다.