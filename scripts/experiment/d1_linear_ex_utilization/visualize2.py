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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / ".logs"
    input_csv = logs_dir / "summarized_results.csv"
    output_png = logs_dir / "visualized_l1_buffer_to_ex_total_vs_active_cycles.png"

    df = pd.read_csv(input_csv)

    required_columns = {
        "l1_buffer_size",
        "use_l1_cache",
        "use_bcast",
        "core_id",
        "thread",
        "active_time",
        "total_time",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["use_l1_cache_bool"] = df["use_l1_cache"].apply(lambda v: normalize_bool(v, "use_l1_cache"))
    df["use_bcast_bool"] = df["use_bcast"].apply(lambda v: normalize_bool(v, "use_bcast"))

    ex_df = df[df["thread"] == "EX"].copy()
    grouped = (
        ex_df.groupby(["use_l1_cache_bool", "use_bcast_bool", "l1_buffer_size"], as_index=False)
        .agg(total_cycles=("total_time", "mean"), active_cycles=("active_time", "mean"))
        .sort_values(["use_l1_cache_bool", "use_bcast_bool", "l1_buffer_size"])
    )

    l1_sizes = sorted(grouped["l1_buffer_size"].unique())
    x_positions = list(range(len(l1_sizes)))
    bar_width = 0.38
    global_cycle_max = max(float(grouped["total_cycles"].max()), float(grouped["active_cycles"].max()), 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)

    plot_specs = [
        (False, False, "No Bcast, From Main"),
        (False, True, "With Bcast, From Main"),
        (True, False, "No Bcast, From L1"),
        (True, True, "With Bcast, From L1"),
    ]

    for ax, (use_l1_cache, use_bcast, title) in zip(axes.ravel(), plot_specs):
        sub_df = grouped[
            (grouped["use_l1_cache_bool"] == use_l1_cache)
            & (grouped["use_bcast_bool"] == use_bcast)
        ]
        sub_df = sub_df.set_index("l1_buffer_size").reindex(l1_sizes)

        total_values = sub_df["total_cycles"].fillna(0.0).to_numpy()
        active_values = sub_df["active_cycles"].fillna(0.0).to_numpy()

        series = [
            ("EX Total Cycles", total_values),
            ("EX Active Cycles", active_values),
        ]
        n_series = len(series)
        group_center_offset = (n_series - 1) / 2.0

        for idx, (label, values) in enumerate(series):
            positions = [x + (idx - group_center_offset) * bar_width for x in x_positions]
            ax.bar(positions, values, width=bar_width, label=label)

        local_max = max(float(total_values.max()), float(active_values.max()), 1.0)
        y_offset = local_max * 0.02
        for idx, _ in enumerate(l1_sizes):
            total_at_idx = float(total_values[idx])
            active_at_idx = float(active_values[idx])
            if total_at_idx <= 0:
                continue

            utilization_ratio = active_at_idx / total_at_idx
            ax.text(
                x_positions[idx],
                total_at_idx + y_offset,
                f"{utilization_ratio * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                rotation=50,
                
            )

        ax.set_title(title)
        ax.set_ylabel("Cycles")
        ax.set_ylim(0, global_cycle_max * 1.12)
        ax.margins(x=0.01)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True, shadow=False)

    for ax in axes.ravel():
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(v) for v in l1_sizes], rotation=45, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[-1, :]:
        ax.set_xlabel("L1 Buffer LD/ST Space Size (KB)")

    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    logger.info(f"Saved plot to '{output_png}'")


if __name__ == "__main__":
    main()
