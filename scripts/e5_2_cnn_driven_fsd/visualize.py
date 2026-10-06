import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
POLICIES = ("sequential", "preemptive", "virtual", "spatial")
WORKLOAD_CLASSES = ("camera.det", "lane.seg", "driver.monitor")
COLORS = {
    "sequential": "#8C8C8C",
    "preemptive": "#F28E2B",
    "spatial": "#59A14F",
    "virtual": "#4C78A8",
}
CYCLES_PER_MS = 1_000_000


def workload_class(workload_id: str) -> str:
    for name in WORKLOAD_CLASSES:
        if workload_id.startswith(f"{name}."):
            return name
    raise ValueError(f"Unknown workload class for {workload_id}")


def read_workload_responses(policy: str) -> dict[str, int]:
    path = CACHE_DIR / f"run_{policy}" / "workload_profile.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No workload measurements in {path}")
    responses = {}
    for row in rows:
        workload_id = row["workload_id"]
        workload_class(workload_id)
        if workload_id in responses:
            raise ValueError(f"Duplicate workload {workload_id} in {path}")
        response = int(row["response_time"])
        if response <= 0:
            raise ValueError(f"Nonpositive response time for {workload_id} in {path}")
        responses[workload_id] = response
    return responses


def percentile(values: list[int], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_kernel_latency_sum(policy: str) -> tuple[float, int]:
    total_cycles = 0.0
    total_count = 0
    for workload_name in WORKLOAD_CLASSES:
        path = CACHE_DIR / f"run_{policy}" / "kernel_profile" / f"kernel_profile_{workload_name}.csv"
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"No kernel measurements in {path}")
        for row in rows:
            count = int(row["count"])
            latency = float(row["average_latency"])
            if count <= 0 or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"Invalid kernel measurements in {path}")
            total_cycles += count * latency
            total_count += count
    return total_cycles / CYCLES_PER_MS, total_count


def plot_bars(values: dict[str, float], title: str, ylabel: str, filename: str, speedup: bool) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    positions = range(len(POLICIES))
    heights = [values[policy] for policy in POLICIES]
    bars = axis.bar(
        positions,
        heights,
        width=0.65,
        color=[COLORS[policy] for policy in POLICIES],
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(positions), [policy.capitalize() for policy in POLICIES])
    axis.set_ylim(0, max(heights) * 1.2)
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    axis.set_axisbelow(True)
    if speedup:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    for bar, value in zip(bars, heights):
        label = f"{value:.2f}×" if speedup else f"{value:.3f}"
        axis.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    figure.tight_layout()
    output_path = CACHE_DIR / filename
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"generated {output_path}")


def main() -> None:
    responses = {policy: read_workload_responses(policy) for policy in POLICIES}
    baseline_ids = set(responses["sequential"])
    for policy, values in responses.items():
        if set(values) != baseline_ids:
            raise ValueError(f"Workload IDs for {policy} differ from Sequential")

    grouped = {
        policy: {
            name: [response for workload_id, response in values.items() if workload_class(workload_id) == name]
            for name in WORKLOAD_CLASSES
        }
        for policy, values in responses.items()
    }
    for name in WORKLOAD_CLASSES:
        if not grouped["sequential"][name]:
            raise ValueError(f"No {name} workload measurements")

    metrics = {
        "mean": {policy: sum(values.values()) / len(values) for policy, values in responses.items()},
        "worst": {policy: max(values.values()) for policy, values in responses.items()},
        "p95": {policy: percentile(list(values.values()), 95) for policy, values in responses.items()},
    }
    for name in WORKLOAD_CLASSES:
        metrics[name] = {policy: sum(grouped[policy][name]) / len(grouped[policy][name]) for policy in POLICIES}

    response_plots = (
        ("mean", "Mean response time speedup", "visualize_mean_response.png"),
        ("worst", "Worst response time speedup", "visualize_worst_response.png"),
        ("p95", "P95 response time speedup", "visualize_p95_response.png"),
        ("camera.det", "Camera detection mean response speedup", "visualize_camera_response.png"),
        ("lane.seg", "Lane segmentation mean response speedup", "visualize_lane_response.png"),
        ("driver.monitor", "Driver monitoring mean response speedup", "visualize_driver_response.png"),
    )
    for metric, title, filename in response_plots:
        baseline = metrics[metric]["sequential"]
        speedups = {policy: baseline / value for policy, value in metrics[metric].items()}
        plot_bars(speedups, title, "Speedup vs Sequential (×)", filename, speedup=True)

    kernel_results = {policy: read_kernel_latency_sum(policy) for policy in POLICIES}
    kernel_counts = {count for _, count in kernel_results.values()}
    if len(kernel_counts) != 1:
        raise ValueError(f"Kernel invocation counts differ across policies: {kernel_counts}")
    kernel_latency_ms = {policy: result[0] for policy, result in kernel_results.items()}
    plot_bars(kernel_latency_ms, "Sum of kernel latencies", "Kernel latency sum (ms)",
              "visualize_kernel_latency.png", speedup=False)


if __name__ == "__main__":
    main()
