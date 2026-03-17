import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from neuromta.framework import logger


def normalize_bool(value: object, column_name: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported {column_name} value: {value}")


def build_plot_for_use_l1_cache(ax, grouped: pd.DataFrame, use_l1_cache: bool) -> None:
    sub = grouped[grouped["use_l1_cache_bool"] == use_l1_cache].copy()
    title = f"use_l1_cache={use_l1_cache}: Thread Utilization by Core Count"
    ax.set_title(title)
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    threads = ["LD", "EX", "ST"]
    core_counts = sorted(sub["n_cores"].unique().tolist())
    x_positions = list(range(len(core_counts)))
    bar_width = 0.24
    center = (len(threads) - 1) / 2.0

    pivot = sub.pivot(index="n_cores", columns="thread", values="utilization")
    pivot = pivot.reindex(index=core_counts).reindex(columns=threads)

    for idx, thread in enumerate(threads):
        values = pivot[thread].fillna(0.0).to_numpy()
        pos = [x + (idx - center) * bar_width for x in x_positions]
        ax.bar(pos, values, width=bar_width, label=thread)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in core_counts])
    ax.margins(x=0.01)
    ax.legend(loc="upper right")


def build_ex_cycle_plot_for_use_l1_cache(ax, grouped: pd.DataFrame, use_l1_cache: bool) -> None:
    sub = grouped[
        (grouped["use_l1_cache_bool"] == use_l1_cache)
        & (grouped["thread"] == "EX")
    ].copy()
    title = f"use_l1_cache={use_l1_cache}: EX Total/Active Cycles by Core Count"
    ax.set_title(title)
    ax.set_ylabel("Cycles")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    core_counts = sorted(sub["n_cores"].unique().tolist())
    x_positions = list(range(len(core_counts)))
    bar_width = 0.35

    pivot_total = sub.set_index("n_cores")["mean_total_time"].reindex(core_counts)
    pivot_active = sub.set_index("n_cores")["mean_active_time"].reindex(core_counts)

    pos_total = [x - bar_width / 2 for x in x_positions]
    pos_active = [x + bar_width / 2 for x in x_positions]

    ax.bar(pos_total, pivot_total.fillna(0.0).to_numpy(), width=bar_width, label="EX Total Cycles")
    ax.bar(pos_active, pivot_active.fillna(0.0).to_numpy(), width=bar_width, label="EX Active Cycles")

    ax.margins(x=0.01)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in core_counts])
    ax.legend(loc="upper right")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / ".logs"
    csv_path = logs_dir / "summarized_results.csv"
    output_png = logs_dir / "visualized_core_num_to_thread_utilization.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "n_cores",
        "use_l1_cache",
        "thread",
        "active_time",
        "total_time",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["use_l1_cache_bool"] = df["use_l1_cache"].apply(lambda v: normalize_bool(v, "use_l1_cache"))
    df["n_cores"] = pd.to_numeric(df["n_cores"], errors="coerce")
    df["active_time"] = pd.to_numeric(df["active_time"], errors="coerce")
    df["total_time"] = pd.to_numeric(df["total_time"], errors="coerce")
    df = df.dropna(subset=["n_cores", "active_time", "total_time"])

    # Same config is defined by (n_cores, use_l1_cache), so average cycles first.
    grouped = (
        df.groupby(["n_cores", "use_l1_cache_bool", "thread"], as_index=False)
        .agg(
            mean_active_time=("active_time", "mean"),
            mean_total_time=("total_time", "mean"),
        )
    )
    grouped["utilization"] = grouped["mean_active_time"] / grouped["mean_total_time"] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False, sharey=False)

    build_plot_for_use_l1_cache(axes[0, 0], grouped, use_l1_cache=False)
    build_plot_for_use_l1_cache(axes[0, 1], grouped, use_l1_cache=True)
    build_ex_cycle_plot_for_use_l1_cache(axes[1, 0], grouped, use_l1_cache=False)
    build_ex_cycle_plot_for_use_l1_cache(axes[1, 1], grouped, use_l1_cache=True)

    for ax in axes.flat:
        ax.set_xlabel("Number of Cores")
    fig.tight_layout()

    os.makedirs(output_png.parent, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    logger.info(f"Saved plot to '{output_png}'")


if __name__ == "__main__":
    main()
