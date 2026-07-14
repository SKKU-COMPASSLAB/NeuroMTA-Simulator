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
        default=os.path.join(SIM_LOG_DIR, "exp1_2_validation_correlation.png"),
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
    
    # 코어 수의 min/max 계산
    min_cores = core_counts.min()
    max_cores = core_counts.max()
    core_range = max_cores - min_cores if max_cores > min_cores else 1
    
    # 색상 정의 (RGB)
    base_colors = {
        "LN": (0.0, 0.0, 1.0),      # Blue
        "CV": (1.0, 0.647, 0.0),    # Orange
        "OTHER": (0.5, 0.5, 0.5),   # Gray
    }
    
    def get_alpha_by_cores(core_count):
        """코어 개수에 따라 투명도 계산 (0.4 ~ 0.95)"""
        normalized = (core_count - min_cores) / core_range
        return 0.4 + normalized * 0.55
    
    # 벤치마크별로 그룹화
    ln_data = [(name, data[name][0], data[name][1], data[name][2]) 
               for name in bench_names if name.startswith("LN")]
    cv_data = [(name, data[name][0], data[name][1], data[name][2]) 
               for name in bench_names if name.startswith("CV")]
    other_data = [(name, data[name][0], data[name][1], data[name][2]) 
                  for name in bench_names if not (name.startswith("LN") or name.startswith("CV"))]
    
    # 각 그룹별 색상별로 표시
    unique_core_counts = sorted(set(core_counts))
    
    # LN 그룹: 코어 수별로 다른 투명도로 표시
    if ln_data:
        first_ln = True
        for cores in unique_core_counts:
            ln_subset = [(x, y) for name, x, y, c in ln_data if c == cores]
            if ln_subset:
                ln_x = [item[0] for item in ln_subset]
                ln_y = [item[1] for item in ln_subset]
                alpha = get_alpha_by_cores(cores)
                label = "LN" if first_ln else None
                ax.scatter(ln_x, ln_y, s=90, alpha=alpha, edgecolors="black", linewidths=0.6, 
                          color=base_colors["LN"], label=label)
                first_ln = False
    
    # CV 그룹: 코어 수별로 다른 투명도로 표시
    if cv_data:
        first_cv = True
        for cores in unique_core_counts:
            cv_subset = [(x, y) for name, x, y, c in cv_data if c == cores]
            if cv_subset:
                cv_x = [item[0] for item in cv_subset]
                cv_y = [item[1] for item in cv_subset]
                alpha = get_alpha_by_cores(cores)
                label = "CV" if first_cv else None
                ax.scatter(cv_x, cv_y, s=90, alpha=alpha, edgecolors="black", linewidths=0.6, 
                          color=base_colors["CV"], label=label)
                first_cv = False
    
    # 나머지 그룹
    if other_data:
        for cores in unique_core_counts:
            other_subset = [(x, y) for name, x, y, c in other_data if c == cores]
            if other_subset:
                other_x = [item[0] for item in other_subset]
                other_y = [item[1] for item in other_subset]
                alpha = get_alpha_by_cores(cores)
                ax.scatter(other_x, other_y, s=90, alpha=alpha, edgecolors="black", linewidths=0.6, 
                          color=base_colors["OTHER"], label=None)
    
    # 첫 번째 legend: LN/CV 색상 구분
    legend1 = ax.legend(loc="lower right", fontsize=11)
    ax.add_artist(legend1)
    
    # 두 번째 legend: 코어 개수별 투명도
    core_legend_handles = []
    for cores in unique_core_counts:
        alpha = get_alpha_by_cores(cores)
        handle = ax.scatter([], [], s=90, alpha=alpha, edgecolors="black", linewidths=0.6, 
                           color="gray", label=f"{cores} Cores")
        core_legend_handles.append(handle)
    
    legend2 = ax.legend(handles=core_legend_handles, loc="upper left", fontsize=10, title="Number of Cores", framealpha=0.9)

    axis_min = min(sim_perfs.min(), ref_perfs.min())
    axis_max = max(sim_perfs.max(), ref_perfs.max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ax.set_title("Simulation vs Hardware Correlation", fontsize=14, pad=10)
    ax.set_xlabel("Simulation Performance (Ops/Cycle)", fontsize=12)
    ax.set_ylabel("Hardware Performance (Ops/Cycle)", fontsize=12)
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
        t = f"{name:<40s}: SIM={sim_cycle:<7.0f} | REF={ref_cycle:<7.0f} | MSE={mse:>7.2f}% | Cores={n_cores}"
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