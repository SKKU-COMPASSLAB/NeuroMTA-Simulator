from pathlib import Path
from neuromta.framework import logger

import matplotlib.pyplot as plt
import pandas as pd


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
    output_png = logs_dir / "visualized_l1_buffer_to_ld_st_ex_ratio.png"

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

    # Convert activated cycles to ratio for each core/thread sample.
    df["active_ratio"] = df["active_time"] / df["total_time"]
    df["use_l1_cache_bool"] = df["use_l1_cache"].apply(lambda v: normalize_bool(v, "use_l1_cache"))
    df["use_bcast_bool"] = df["use_bcast"].apply(lambda v: normalize_bool(v, "use_bcast"))

    grouped = (
        df.groupby(["use_l1_cache_bool", "use_bcast_bool", "l1_buffer_size", "thread"], as_index=False)["active_ratio"]
        .mean()
        .sort_values(["use_l1_cache_bool", "use_bcast_bool", "l1_buffer_size", "thread"])
    )

    bar_width = 0.24
    threads = ["LD", "EX", "ST"]
    l1_sizes = sorted(grouped["l1_buffer_size"].unique())
    x_positions = list(range(len(l1_sizes)))
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
        pivot = sub_df.pivot(index="l1_buffer_size", columns="thread", values="active_ratio")
        pivot = pivot.reindex(index=l1_sizes)
        pivot = pivot.reindex(columns=threads)

        n_threads = len(threads)
        group_center_offset = (n_threads - 1) / 2.0

        for idx, thread in enumerate(threads):
            values = pivot[thread].fillna(0.0).to_numpy() * 100.0
            positions = [pos + (idx - group_center_offset) * bar_width for pos in x_positions]
            ax.bar(positions, values, width=bar_width, label=thread)

        ax.set_ylabel("Mean Active Cycle Ratio (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.margins(x=0.01)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=True, shadow=False)

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
