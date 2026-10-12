import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULT_PATH = CACHE_DIR / "kernel_latency.csv"


def main():
    with RESULT_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No profiling results in {RESULT_PATH}")

    workloads = {}
    for row in rows:
        name = row["workload"]
        shape = (int(row["mesh_rows"]), int(row["mesh_columns"]))
        cycles = int(row["latency_cycles"])
        if cycles <= 0 or shape in workloads.setdefault(name, {}):
            raise ValueError(f"Invalid or duplicate measurement for {name} on {shape}")
        workloads[name][shape] = cycles

    for name, measurements in workloads.items():
        shapes = sorted(measurements)
        reference = (1, 1)
        if reference not in measurements:
            raise ValueError(f"Missing 1x1 reference measurement for {name}")
        reference_latency = measurements[reference]
        # reference_cores = reference[0] * reference[1]
        positions = list(range(len(shapes)))
        labels = [f"{rows}x{columns}" for rows, columns in shapes]
        actual = [reference_latency / measurements[shape] for shape in shapes]
        # estimated = [shape[0] * shape[1] / reference_cores for shape in shapes]
        actual_colors = ["#D62728" if shape == reference else "#4C78A8" for shape in shapes]
        # estimated_colors = ["#D62728" if shape == reference else "#F28E2B" for shape in shapes]

        figure, axis = plt.subplots(figsize=(10, 4))
        axis.bar(positions, actual, width=0.65,
                 color=actual_colors, edgecolor="black", linewidth=0.6, label="Actual")
        # axis.bar([position + 0.19 for position in positions], estimated, width=0.38,
        #          color=estimated_colors, edgecolor="black", linewidth=0.6, label="Estimated (core count)")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.set_title(f"Kernel speedup: {name} (reference: 1x1)")
        axis.set_xlabel("Core mesh shape")
        axis.set_ylabel("Speedup vs 1x1 (×)")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, max(actual + [1.0]) * 1.15)
        axis.grid(axis="y", linestyle=":", alpha=0.35)
        axis.set_axisbelow(True)
        axis.legend(handles=[
            Patch(facecolor="#4C78A8", edgecolor="black", label="Actual"),
            # Patch(facecolor="#F28E2B", edgecolor="black", label="Estimated (core count)"),
            Patch(facecolor="#D62728", edgecolor="black", label="Reference (1x1)"),
        ])
        figure.tight_layout()
        output_path = CACHE_DIR / f"visualize_kernel_latency_{name}.png"
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
