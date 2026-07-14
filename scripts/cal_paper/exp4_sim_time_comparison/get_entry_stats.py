from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / ".logs"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuromta_matplotlib")

if str(ROOT_DIR / "srcs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "srcs"))

from neuromta.framework import LogLevel, logger, parse_mem_cap_str
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
from neuromta.system.software.tenstorrent import MCA_CompiledNetworkGraph, MCA_NetworkRecipe


SUPPORTED_MODULES = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)


def prod(values: list[int] | tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def shape_of(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return list(item.shape)
    raise TypeError(f"Cannot extract a tensor shape from {type(value).__name__}")


def build_model_and_input(pretrained: bool, seed: int, batch_size: int, height: int, width: int) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(seed)
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights).eval()
    dummy_input = torch.randn(batch_size, 3, height, width)
    return model, dummy_input


def capture_module_records(model: nn.Module, dummy_input: torch.Tensor) -> dict[str, list[dict[str, Any]]]:
    module_paths = {module: name for name, module in model.named_modules()}
    records: dict[str, list[dict[str, Any]]] = {
        "conv": [],
        "linear": [],
        "maxpool": [],
        "avgpool": [],
    }
    handles = []

    def make_hook(module: nn.Module):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(module, nn.Conv2d):
                key = "conv"
            elif isinstance(module, nn.Linear):
                key = "linear"
            elif isinstance(module, nn.MaxPool2d):
                key = "maxpool"
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                key = "avgpool"
            else:
                return
            records[key].append({
                "module_path": module_paths[module],
                "module_type": type(module).__name__,
                "module": module,
                "input_shape": shape_of(inputs[0]),
                "output_shape": shape_of(output),
            })
        return hook

    for module in module_paths:
        if isinstance(module, SUPPORTED_MODULES):
            handles.append(module.register_forward_hook(make_hook(module)))

    with torch.inference_mode():
        model(dummy_input)

    for handle in handles:
        handle.remove()

    return records


def create_graph(model: nn.Module, dummy_input: torch.Tensor) -> MCA_CompiledNetworkGraph:
    config = TenstorrentConfig.BLACKHOLE(use_pydramsim3=False, use_pybooksim2=False)
    device = TenstorrentDevice(**config).initialize()
    device.set_command_debug_verbosity(verbose=False)
    recipe = MCA_NetworkRecipe(
        device=device,
        core_groups=device.get_npu_core_group((0, 0), (12, 14)),
        spad_space_size_per_core=parse_mem_cap_str("1MB"),
        dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    return MCA_CompiledNetworkGraph.from_trace(model, recipe, dummy_input)


def iter_graph_entries(graph: MCA_CompiledNetworkGraph) -> list[dict[str, Any]]:
    rows = []
    for group_idx, group in enumerate(graph.grouped_compiled_entries):
        for entry_idx, entry in enumerate(group):
            rows.append({
                "group_idx": group_idx,
                "entry_idx": entry_idx,
                "node_kind": entry.node.kind(),
                "op_type": entry._op_method.__name__,
                "input_nodes": "|".join(input_node.debugName() for input_node in entry.node.inputs()),
                "output_nodes": "|".join(output_node.debugName() for output_node in entry.node.outputs()),
                "context_entry_count": len(entry._ctx_entries),
            })
    return rows


def conv2d_stats(module: nn.Conv2d, input_shape: list[int], output_shape: list[int], dtype_size: int) -> dict[str, Any]:
    kernel_h, kernel_w = pair(module.kernel_size)
    stride_h, stride_w = pair(module.stride)
    channels_per_group = module.in_channels // module.groups
    out_elements = prod(output_shape)
    macs = out_elements * channels_per_group * kernel_h * kernel_w
    weight_bytes = module.weight.numel() * dtype_size
    bias_bytes = 0 if module.bias is None else module.bias.numel() * dtype_size
    input_bytes = prod(input_shape) * dtype_size
    output_bytes = prod(output_shape) * dtype_size
    scalesim_ifmap_h = (output_shape[2] - 1) * stride_h + kernel_h
    scalesim_ifmap_w = (output_shape[3] - 1) * stride_w + kernel_w
    return {
        "scalesim_kind": "conv",
        "macs": int(macs),
        "ops": int(2 * macs),
        "activation_read_bytes": int(input_bytes),
        "parameter_read_bytes": int(weight_bytes),
        "bias_read_bytes": int(bias_bytes),
        "dram_read_bytes": int(input_bytes + weight_bytes + bias_bytes),
        "dram_write_bytes": int(output_bytes),
        "dram_traffic_bytes": int(input_bytes + weight_bytes + bias_bytes + output_bytes),
        "parameter_count": int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
        "ifmap_h": int(scalesim_ifmap_h),
        "ifmap_w": int(scalesim_ifmap_w),
        "filter_h": int(kernel_h),
        "filter_w": int(kernel_w),
        "channels": int(module.in_channels),
        "num_filter": int(module.out_channels),
        "stride": int(stride_h),
    }


def linear_stats(module: nn.Linear, input_shape: list[int], output_shape: list[int], dtype_size: int) -> dict[str, Any]:
    batch = prod(input_shape[:-1]) if len(input_shape) > 1 else 1
    macs = batch * module.in_features * module.out_features
    weight_bytes = module.weight.numel() * dtype_size
    bias_bytes = 0 if module.bias is None else module.bias.numel() * dtype_size
    input_bytes = prod(input_shape) * dtype_size
    output_bytes = prod(output_shape) * dtype_size
    return {
        "scalesim_kind": "gemm",
        "macs": int(macs),
        "ops": int(2 * macs),
        "activation_read_bytes": int(input_bytes),
        "parameter_read_bytes": int(weight_bytes),
        "bias_read_bytes": int(bias_bytes),
        "dram_read_bytes": int(input_bytes + weight_bytes + bias_bytes),
        "dram_write_bytes": int(output_bytes),
        "dram_traffic_bytes": int(input_bytes + weight_bytes + bias_bytes + output_bytes),
        "parameter_count": int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
        "m": int(batch),
        "n": int(module.out_features),
        "k": int(module.in_features),
    }


def maxpool2d_stats(module: nn.MaxPool2d, input_shape: list[int], output_shape: list[int], dtype_size: int) -> dict[str, Any]:
    kernel_h, kernel_w = pair(module.kernel_size)
    input_bytes = prod(input_shape) * dtype_size
    output_bytes = prod(output_shape) * dtype_size
    ops = prod(output_shape) * max(kernel_h * kernel_w - 1, 0)
    return {
        "scalesim_kind": "unsupported",
        "macs": 0,
        "ops": int(ops),
        "activation_read_bytes": int(input_bytes),
        "parameter_read_bytes": 0,
        "bias_read_bytes": 0,
        "dram_read_bytes": int(input_bytes),
        "dram_write_bytes": int(output_bytes),
        "dram_traffic_bytes": int(input_bytes + output_bytes),
        "parameter_count": 0,
    }


def adaptive_avgpool2d_stats(input_shape: list[int], output_shape: list[int], dtype_size: int) -> dict[str, Any]:
    input_bytes = prod(input_shape) * dtype_size
    output_bytes = prod(output_shape) * dtype_size
    return {
        "scalesim_kind": "unsupported",
        "macs": 0,
        "ops": int(prod(input_shape) + prod(output_shape)),
        "activation_read_bytes": int(input_bytes),
        "parameter_read_bytes": 0,
        "bias_read_bytes": 0,
        "dram_read_bytes": int(input_bytes),
        "dram_write_bytes": int(output_bytes),
        "dram_traffic_bytes": int(input_bytes + output_bytes),
        "parameter_count": 0,
    }


def unsupported_stats(dtype_size: int) -> dict[str, Any]:
    return {
        "scalesim_kind": "unsupported",
        "macs": 0,
        "ops": 0,
        "activation_read_bytes": 0,
        "parameter_read_bytes": 0,
        "bias_read_bytes": 0,
        "dram_read_bytes": 0,
        "dram_write_bytes": 0,
        "dram_traffic_bytes": 0,
        "parameter_count": 0,
        "input_shape": [],
        "output_shape": [],
        "input_elements": 0,
        "output_elements": 0,
        "module_path": "",
        "module_type": "",
    }


def attach_stats(graph_entries: list[dict[str, Any]], module_records: dict[str, list[dict[str, Any]]], dtype_size: int) -> list[dict[str, Any]]:
    cursors = {key: 0 for key in module_records}
    entries = []
    for entry in graph_entries:
        op_type = entry["op_type"]
        if op_type == "MCA_OP_CONV2D":
            key = "conv"
        elif op_type == "MCA_OP_LINEAR":
            key = "linear"
        elif op_type == "MCA_OP_MAXPOOL2D":
            key = "maxpool"
        elif op_type == "MCA_OP_ADAPTIVE_AVGPOOL2D":
            key = "avgpool"
        else:
            key = ""

        if key and cursors[key] < len(module_records[key]):
            record = module_records[key][cursors[key]]
            cursors[key] += 1
            module = record["module"]
            input_shape = record["input_shape"]
            output_shape = record["output_shape"]
            if isinstance(module, nn.Conv2d):
                stats = conv2d_stats(module, input_shape, output_shape, dtype_size)
            elif isinstance(module, nn.Linear):
                stats = linear_stats(module, input_shape, output_shape, dtype_size)
            elif isinstance(module, nn.MaxPool2d):
                stats = maxpool2d_stats(module, input_shape, output_shape, dtype_size)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                stats = adaptive_avgpool2d_stats(input_shape, output_shape, dtype_size)
            else:
                stats = unsupported_stats(dtype_size)
            stats.update({
                "module_path": record["module_path"],
                "module_type": record["module_type"],
                "input_shape": input_shape,
                "output_shape": output_shape,
                "input_elements": prod(input_shape),
                "output_elements": prod(output_shape),
                "layer_name": record["module_path"].replace(".", "_"),
            })
        else:
            stats = unsupported_stats(dtype_size)
            stats["layer_name"] = f"group{entry['group_idx']}_entry{entry['entry_idx']}_{op_type}"

        stats["ops_tops"] = stats["ops"] / 1e12
        stats["dram_traffic_mib"] = stats["dram_traffic_bytes"] / (1024 * 1024)
        entries.append({**entry, **stats})
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-entry stats for ResNet18.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "entry_stats.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--dtype-size-bytes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    logger.set_print_options(log_level=LogLevel.INFO)
    args = parse_args()
    model, dummy_input = build_model_and_input(args.pretrained, args.seed, args.batch_size, args.height, args.width)
    module_records = capture_module_records(model, dummy_input)
    graph = create_graph(model, dummy_input)
    graph_entries = iter_graph_entries(graph)
    entries = attach_stats(graph_entries, module_records, args.dtype_size_bytes)

    total_ops = sum(entry["ops"] for entry in entries)
    total_macs = sum(entry["macs"] for entry in entries)
    total_dram_traffic = sum(entry["dram_traffic_bytes"] for entry in entries)
    payload = {
        "model": "ResNet18",
        "batch_size": args.batch_size,
        "input_shape": list(dummy_input.shape),
        "dtype_size_bytes": args.dtype_size_bytes,
        "op_count_policy": {
            "conv2d": "MAC counted as 2 ops.",
            "linear": "MAC counted as 2 ops.",
            "maxpool2d": "Each window comparison counted as 1 op.",
            "adaptive_avgpool2d": "Approximate input accumulation plus output division count.",
            "unsupported": "Unsupported/non-modeled entries are counted as 0 ops.",
        },
        "dram_traffic_policy": "Cold layer-level traffic: activation read + parameter/bias read + output write. On-chip reuse and cache effects are not subtracted.",
        "totals": {
            "macs": int(total_macs),
            "ops": int(total_ops),
            "ops_tops": total_ops / 1e12,
            "dram_traffic_bytes": int(total_dram_traffic),
            "dram_traffic_mib": total_dram_traffic / (1024 * 1024),
        },
        "entries": entries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    print(f"Wrote {args.output_json}")
    print(f"Total ops: {total_ops} ({total_ops / 1e12:.6f} TOP)")
    print(f"Total DRAM traffic: {total_dram_traffic} bytes ({total_dram_traffic / (1024 * 1024):.3f} MiB)")


if __name__ == "__main__":
    main()
