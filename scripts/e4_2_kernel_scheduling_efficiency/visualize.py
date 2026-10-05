import csv
import os
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

WORKLOAD_TYPES = ("large", "medium", "small")

IMG_PATHS: dict[str, str] = {
    f"{workload1}_{workload2}": os.path.join(CACHE_DIR, f"visualize_{workload1}_{workload2}.png")
    for workload1 in WORKLOAD_TYPES
    for workload2 in WORKLOAD_TYPES
}

LOG_PATHS: dict[str, str] = {
    scheduler_type: os.path.join(CACHE_DIR, f"collocation_{scheduler_type}.csv")
    for scheduler_type in ("sequential", "virtual", "preemptive", "spatial")
}

LAYER_ABBRS: dict[str, str] = {
    "conv2d": "CV",
    "linear": "LN",
    "reduction": "RD",
    "elementwise": "EW",
}


def visualize(workload_type_1: str, workload_type_2: str):
    layers = tuple(LAYER_ABBRS)
    pairs = tuple((f"{layer1}_{workload_type_1}", f"{layer2}_{workload_type_2}") for layer1 in layers for layer2 in layers)
    scheduler_cycles = {}

    for scheduler, path in LOG_PATHS.items():
        with open(path, newline="") as file:
            rows = {(row["workload1"], row["workload2"]): int(row["collocation_cycles"]) for row in csv.DictReader(file)}
        missing = tuple(pair for pair in pairs if pair not in rows)
        if missing:
            raise ValueError(f"Missing workload pairs in {path}: {missing}")
        scheduler_cycles[scheduler] = tuple(rows[pair] for pair in pairs)

    sequential_cycles = scheduler_cycles["sequential"]
    scheduler_speedups = {scheduler: tuple(baseline / cycle for baseline, cycle in zip(sequential_cycles, cycles)) for scheduler, cycles in scheduler_cycles.items()}
    colors = {"sequential": "#8C8C8C", "virtual": "#4C78A8", "preemptive": "#F28E2B", "spatial": "#59A14F"}

    positions = tuple(range(len(pairs)))
    width = 0.2
    figure, axis = plt.subplots(figsize=(10, 4))
    for index, (scheduler, speedups) in enumerate(scheduler_speedups.items()):
        offset = (index - (len(scheduler_speedups) - 1) / 2) * width
        bar_positions = tuple(position + offset for position in positions)
        axis.bar(bar_positions, tuple(min(speedup, 4.0) for speedup in speedups), width=width, label=scheduler.capitalize(), color=colors[scheduler], edgecolor="black", linewidth=0.6)
        for position, speedup in zip(bar_positions, speedups):
            if speedup > 4.0:
                axis.text(position, 4.05, f"{speedup:.1f}×", ha="center", va="bottom", rotation=90, fontsize=7)

    axis.set_title(f"Kernel scheduling speedup: {workload_type_1} + {workload_type_2}")
    axis.set_xlabel("Kernel pair")
    axis.set_ylabel("Speedup over Sequential")
    axis.set_ylim(0, 4.6)
    axis.set_xticks(positions, tuple(f"{LAYER_ABBRS[first.split('_')[0]]}-{LAYER_ABBRS[second.split('_')[0]]}" for first, second in pairs), rotation=45, ha="right")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    axis.legend(ncol=len(scheduler_speedups))
    figure.tight_layout()

    image_path = IMG_PATHS[f"{workload_type_1}_{workload_type_2}"]
    figure.savefig(image_path, dpi=200)
    plt.close(figure)
    print(f"generated {image_path}")


def main():
    for workload_type_1 in WORKLOAD_TYPES:
        for workload_type_2 in WORKLOAD_TYPES:
            visualize(workload_type_1, workload_type_2)


if __name__ == "__main__":
    main()
