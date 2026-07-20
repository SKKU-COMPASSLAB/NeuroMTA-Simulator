import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SIM_LOG_DIR = os.path.join(ROOT, ".logs")
REF_LOG_DIR = os.path.join(ROOT, ".logs_tt")
SIM_LOG_PATH = os.path.join(SIM_LOG_DIR, "run.csv")
CONV_REF_LOG_PATH = os.path.join(REF_LOG_DIR, "conv_summary_l1_feature_map.csv")
LINEAR_REF_LOG_PATH = os.path.join(REF_LOG_DIR, "linear_summary_l1_feature_map.csv")
CORE_COLOR_SCHEME = [
    "#E63946",
    "#1D4ED8",
    "#0F766E",
    "#FF7F11",
    "#7C3AED",
    "#16A34A",
    "#F72585",
    "#4B5563",
]
BENCHMARK_MARKERS = {"LN": "o", "CV": "^", "OTHER": "s"}


def get_data(sim_path: str, linear_ref_path: str, conv_ref_path: str) -> dict[str, tuple[float, float, int]]:
    sim_df = pd.read_csv(sim_path)
    linear_ref_df = pd.read_csv(linear_ref_path)
    conv_ref_df = pd.read_csv(conv_ref_path)

    results: dict[str, tuple[float, float, int]] = {}
    
    for _, sim_row in sim_df.iterrows():
        sim_bench = sim_row["Benchmark"]
        sim_cycle = sim_row["Timestamp (cycles)"]
        # total_ops = sim_row["Total OPs"]
        n_cores = sim_row["Number of Cores"]

        if sim_bench.startswith("CV"):
            ref_row = conv_ref_df[conv_ref_df["benchmark"] == sim_bench]
        elif sim_bench.startswith("LN"):
            ref_row = linear_ref_df[linear_ref_df["benchmark"] == sim_bench]
        else:
            print(f"Warning: Unrecognized benchmark type for SIM benchmark: {sim_bench}")
            continue
        
        if ref_row.empty:
            print(f"Warning: No matching reference result found for SIM benchmark: {sim_bench}")
            continue
        
        ref_cycle = ref_row.iloc[0]["timestamp"]
        
        results[sim_bench] = (sim_cycle, ref_cycle, n_cores)

    return results

def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize correlation between SIM and HW performance")
    parser.add_argument(
        "--sim-log",
        default=SIM_LOG_PATH,
        help="Path to the simulation log CSV file",
    )
    parser.add_argument(
        "--linear-ref-log",
        default=LINEAR_REF_LOG_PATH,
        help="Path to the hardware reference log CSV file",
    )
    parser.add_argument(
        "--conv-ref-log",
        default=CONV_REF_LOG_PATH,
        help="Path to the convolution hardware reference log CSV file",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join(SIM_LOG_DIR, "exp1_2_validation_correlation.pdf"),
        help="Path to save the output figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    return parser.parse_args()


def plot_correlation(data: dict[str, tuple[float, float, int]], save_path: str, show_plot: bool = True) -> None:
    if not data:
        raise ValueError("No matched benchmark results were found.")

    bench_names = list(data.keys())
    sim_perfs = np.array([data[name][0] for name in bench_names], dtype=float)
    ref_perfs = np.array([data[name][1] for name in bench_names], dtype=float)
    core_counts = np.array([data[name][2] for name in bench_names], dtype=int)

    pearson_corr = np.corrcoef(sim_perfs, ref_perfs)[0, 1] if len(sim_perfs) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(4, 4))
    
    unique_core_counts = sorted(set(core_counts))
    core_colors = {
        cores: CORE_COLOR_SCHEME[i % len(CORE_COLOR_SCHEME)]
        for i, cores in enumerate(unique_core_counts)
    }

    present_types = []
    for benchmark_type in ("LN", "CV", "OTHER"):
        points = []
        for name in bench_names:
            current_type = "LN" if name.startswith("LN") else "CV" if name.startswith("CV") else "OTHER"
            if current_type == benchmark_type:
                points.append((data[name][0], data[name][1], data[name][2]))
        if not points:
            continue
        present_types.append(benchmark_type)
        for x, y, cores in points:
            ax.scatter(
                x,
                y,
                s=90,
                marker=BENCHMARK_MARKERS[benchmark_type],
                color=core_colors[cores],
                edgecolors="black",
                linewidths=0.6,
            )

    type_handles = [
        Line2D(
            [0], [0], marker=BENCHMARK_MARKERS[benchmark_type], color="none",
            markerfacecolor="gray", markeredgecolor="black", markersize=9,
            label=benchmark_type,
        )
        for benchmark_type in present_types
    ]
    type_legend = ax.legend(
        handles=type_handles, loc="lower right", fontsize=13,
        title="Layer Type", framealpha=0.9, 
    )
    ax.add_artist(type_legend)

    core_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=core_colors[cores],
            markeredgecolor="black", markersize=10, label=f"{cores} Tiles",
        )
        for cores in unique_core_counts
    ]
    ax.legend(
        handles=core_handles, loc="upper left", fontsize=13,
        title="Number of Tiles", framealpha=0.9,
    )

    axis_min = min(sim_perfs.min(), ref_perfs.min())
    axis_max = max(sim_perfs.max(), ref_perfs.max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ax.set_title("Simulation vs Hardware Correlation", fontsize=14, pad=10)
    ax.set_xlabel("Simulation Performance (Ops/Cycle)", fontsize=13)
    ax.set_ylabel("Hardware Performance (Ops/Cycle)", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    fig.text(
        0.2, 0.26,
        f"Pearson Correlation\n= {pearson_corr:.6f}",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        backgroundcolor='white',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    )

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=200)
    print(f"Saved correlation plot: {save_path}")
    print(f"Pearson correlation: {pearson_corr:.6f}")
    
    # show errors for each benchmark
    print("\nDetailed Results:")
    for name in bench_names:
        sim_cycle, ref_cycle, n_cores = data[name]
        mse = (sim_cycle - ref_cycle) / ref_cycle * 100
        t = f"{name:<40s}: SIM={sim_cycle:<7.0f} | REF={ref_cycle:<7.0f} | MSE={mse:>7.2f}% | Tiles={n_cores}"
        if abs(mse) > 30:
            print(f"\033[91m{t}\033[0m") 
        else:
            print(t)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = _build_args()
    results = get_data(sim_path=args.sim_log, linear_ref_path=args.linear_ref_log, conv_ref_path=args.conv_ref_log)
    plot_correlation(results, save_path=args.save_path, show_plot=not args.no_show)
