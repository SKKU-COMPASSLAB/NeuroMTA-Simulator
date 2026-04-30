import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SIM_LOG_DIR = os.path.join(ROOT, ".logs")
REF_LOG_DIR = os.path.join(ROOT, ".logs_tt")
SIM_LOG_PATH = os.path.join(SIM_LOG_DIR, "run.csv")
CONV_REF_LOG_PATH = os.path.join(REF_LOG_DIR, "conv_summary_l1_feature_map.csv")
LINEAR_REF_LOG_PATH = os.path.join(REF_LOG_DIR, "linear_summary_l1_feature_map.csv")


def get_data(sim_path: str, linear_ref_path: str, conv_ref_path: str) -> dict[str, tuple[float, float]]:
    sim_df = pd.read_csv(sim_path)
    linear_ref_df = pd.read_csv(linear_ref_path)
    conv_ref_df = pd.read_csv(conv_ref_path)

    results: dict[str, tuple[float, float]] = {}
    
    for _, sim_row in sim_df.iterrows():
        sim_bench = sim_row["Benchmark"]
        sim_cycle = sim_row["Timestamp (cycles)"]
        # total_ops = sim_row["Total OPs"]

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
    group_colors = {
        "LN": "tab:blue",
        "CV": "tab:orange",
        "OTHER": "tab:gray",
    }

    ln_x = []
    ln_y = []
    cv_x = []
    cv_y = []
    other_x = []
    other_y = []

    for name, x_val, y_val in zip(bench_names, sim_perfs, ref_perfs):
        if name.startswith("LN"):
            ln_x.append(x_val)
            ln_y.append(y_val)
        elif name.startswith("CV"):
            cv_x.append(x_val)
            cv_y.append(y_val)
        else:
            other_x.append(x_val)
            other_y.append(y_val)

    # Plot groups: LN and CV have legend entries, others are gray without legend
    if ln_x:
        ax.scatter(ln_x, ln_y, s=90, alpha=0.9, edgecolors="black", linewidths=0.6, color=group_colors["LN"], label="LN")
    if cv_x:
        ax.scatter(cv_x, cv_y, s=90, alpha=0.9, edgecolors="black", linewidths=0.6, color=group_colors["CV"], label="CV")
    if other_x:
        ax.scatter(other_x, other_y, s=90, alpha=0.9, edgecolors="black", linewidths=0.6, color=group_colors["OTHER"], label=None)

    axis_min = min(sim_perfs.min(), ref_perfs.min())
    axis_max = max(sim_perfs.max(), ref_perfs.max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")

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
    results = get_data(sim_path=args.sim_log, linear_ref_path=args.linear_ref_log, conv_ref_path=args.conv_ref_log)
    plot_correlation(results, save_path=args.save_path, show_plot=not args.no_show)