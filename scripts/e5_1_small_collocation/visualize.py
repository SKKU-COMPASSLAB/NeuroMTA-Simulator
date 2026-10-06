import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
POLICIES = ("sequential", "preemptive", "virtual", "spatial")
COLORS = {
    "sequential": "#8C8C8C",
    "preemptive": "#F28E2B",
    "spatial": "#59A14F",
    "virtual": "#4C78A8",
}
CYCLES_PER_MS = 1_000_000


def read_workload_responses(policy: str) -> dict[str, int]:
    path = CACHE_DIR / f"run_{policy}" / "workload_profile.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No workload measurements in {path}")
    responses = {}
    for row in rows:
        workload_id = row["workload_id"]
        if workload_id in responses:
            raise ValueError(f"Duplicate workload {workload_id} in {path}")
        response = int(row["response_time"])
        if response <= 0:
            raise ValueError(f"Nonpositive response time for {workload_id} in {path}")
        responses[workload_id] = response
    return responses


def read_kernel_latency_sum(policy: str) -> tuple[float, int]:
    path = CACHE_DIR / f"run_{policy}" / "kernel_profile" / "kernel_profile_camera.det.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No kernel measurements in {path}")
    counts = [int(row["count"]) for row in rows]
    latencies = [float(row["average_latency"]) for row in rows]
    if any(count <= 0 for count in counts) or any(latency < 0 for latency in latencies):
        raise ValueError(f"Invalid kernel measurements in {path}")
    total_cycles = sum(count * latency for count, latency in zip(counts, latencies))
    return total_cycles / CYCLES_PER_MS, sum(counts)


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

    mean_cycles = {policy: sum(values.values()) / len(values) for policy, values in responses.items()}
    worst_cycles = {policy: max(values.values()) for policy, values in responses.items()}
    mean_speedups = {policy: mean_cycles["sequential"] / value for policy, value in mean_cycles.items()}
    worst_speedups = {policy: worst_cycles["sequential"] / value for policy, value in worst_cycles.items()}

    kernel_results = {policy: read_kernel_latency_sum(policy) for policy in POLICIES}
    kernel_counts = {count for _, count in kernel_results.values()}
    if len(kernel_counts) != 1:
        raise ValueError(f"Kernel invocation counts differ across policies: {kernel_counts}")
    kernel_latency_ms = {policy: result[0] for policy, result in kernel_results.items()}

    plot_bars(mean_speedups, "Mean response time speedup", "Speedup vs Sequential (×)",
              "visualize_mean_response.png", speedup=True)
    plot_bars(worst_speedups, "Worst response time speedup", "Speedup vs Sequential (×)",
              "visualize_worst_response.png", speedup=True)
    plot_bars(kernel_latency_ms, "Sum of kernel latencies", "Kernel latency sum (ms)",
              "visualize_kernel_latency.png", speedup=False)


if __name__ == "__main__":
    main()
