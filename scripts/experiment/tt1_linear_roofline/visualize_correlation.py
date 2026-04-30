import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SIM_LOG_DIR = os.path.join(ROOT, ".logs")
REF_LOG_DIR = os.path.join(ROOT, ".logs_tt")
SIM_LOG_PATH = os.path.join(SIM_LOG_DIR, "run_l1_feature_map.csv")
REF_LOG_PATH = os.path.join(REF_LOG_DIR, "summary_l1_feature_map.csv")


def get_data(sim_path: str, ref_path: str) -> dict[str, tuple[float, float]]:
    sim_df = pd.read_csv(sim_path)
    ref_df = pd.read_csv(ref_path)

    results: dict[str, tuple[float, float]] = {}

    for (_, sim_row), (_, ref_row) in zip(sim_df.iterrows(), ref_df.iterrows()):
        sim_bench = sim_row["Benchmark"].split("_")[3]
        ref_bench = ref_row["benchmark"].split("_")[0]
        
        if sim_bench != ref_bench:
            print(f"Warning: Benchmark mismatch between SIM and REF: {sim_bench} vs {ref_bench}")
            continue
        
        if int(sim_bench.split("x")[0]) <= 64:
            continue
        
        sim_cycle = sim_row["Timestamp (cycles)"]
        ref_cycle = ref_row["timestamp"]
        
        total_ops = sim_row["Total OPs"]
        
        # results[sim_bench] = (total_ops / sim_cycle, total_ops / ref_cycle)
        results[sim_bench] = (sim_cycle, ref_cycle)

    return results

def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize correlation between SIM and HW performance")
    parser.add_argument(
        "--sim-log",
        default=SIM_LOG_PATH,
        help="Path to the simulation log CSV file",
    )
    parser.add_argument(
        "--ref-log",
        default=REF_LOG_PATH,
        help="Path to the hardware reference log CSV file",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join(ROOT, "correlation_sim_vs_hw.png"),
        help="Path to save the output figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    return parser.parse_args()


def plot_correlation(data: dict[str, tuple[float, float]], save_path: str, show_plot: bool = True) -> None:
    if not data:
        raise ValueError("No matched benchmark results were found.")

    bench_names = list(data.keys())
    sim_perfs = np.array([data[name][0] for name in bench_names], dtype=float)
    ref_perfs = np.array([data[name][1] for name in bench_names], dtype=float)

    pearson_corr = np.corrcoef(sim_perfs, ref_perfs)[0, 1] if len(sim_perfs) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 7))
    color_map = plt.get_cmap("tab10")
    colors = [color_map(i % color_map.N) for i in range(len(bench_names))]

    for idx, (name, x_val, y_val) in enumerate(zip(bench_names, sim_perfs, ref_perfs)):
        ax.scatter(
            x_val,
            y_val,
            s=90,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.6,
            color=colors[idx],
            label=name,
        )

    axis_min = min(sim_perfs.min(), ref_perfs.min())
    axis_max = max(sim_perfs.max(), ref_perfs.max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="gray", label="y = x")

    ax.set_title("Simulation vs Hardware Performance Correlation", fontsize=14, pad=10)
    ax.set_xlabel("Simulation Performance (Ops/Cycle)", fontsize=12)
    ax.set_ylabel("Hardware Performance (Ops/Cycle)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    
    fig.text(
        0.14,
        0.93,
        f"Pearson Correlation = {pearson_corr:.6f}",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved correlation plot: {save_path}")
    print(f"Pearson correlation: {pearson_corr:.6f}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = _build_args()
    results = get_data(sim_path=args.sim_log, ref_path=args.ref_log)
    plot_correlation(results, save_path=args.save_path, show_plot=not args.no_show)