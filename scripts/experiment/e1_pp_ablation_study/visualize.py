import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from neuromta.framework import logger


def _to_bool_series(series):
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _prepare_curve(df, prefix, use_l1_data_space):
    curve = df[df["prefix"] == prefix].copy()
    if use_l1_data_space is not None:
        curve = curve[curve["use_l1_data_space"] == use_l1_data_space]
    return curve.sort_values("l1_buf_size")


def main():
    parser = argparse.ArgumentParser(description="Visualize PP ablation performance by L1 buffer size")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".logs", "summary.csv"),
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".logs", "visualize_e1_pp_ablation.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    required_cols = {"prefix", "l1_buf_size", "use_l1_data_space", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["l1_buf_size"] = pd.to_numeric(df["l1_buf_size"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["use_l1_data_space"] = _to_bool_series(df["use_l1_data_space"])
    df = df.dropna(subset=["l1_buf_size", "timestamp", "use_l1_data_space"])

    with_pp = _prepare_curve(df, prefix="run_with_pp", use_l1_data_space=None)
    wo_pp_dram = _prepare_curve(df, prefix="run_wo_pp", use_l1_data_space=False)
    wo_pp_l1 = _prepare_curve(df, prefix="run_wo_pp", use_l1_data_space=True)

    plt.figure(figsize=(10, 4))

    if not wo_pp_dram.empty:
        plt.plot(
            wo_pp_dram["l1_buf_size"] / 1024.0,
            wo_pp_dram["timestamp"],
            marker="s",
            linewidth=2,
            label="Without PP + No L1 Data Cache",
        )

    if not wo_pp_l1.empty:
        plt.plot(
            wo_pp_l1["l1_buf_size"] / 1024.0,
            wo_pp_l1["timestamp"],
            marker="^",
            linewidth=2,
            label="Without PP + L1 Data Cache",
        )
  
    if not with_pp.empty:
        plt.plot(
            with_pp["l1_buf_size"] / 1024.0,
            with_pp["timestamp"],
            marker="o",
            linewidth=2,
            label="With PP",
        )

    plt.xlabel("L1 LD/ST Space Size per Core (KB) (L1 cache not included)")
    plt.ylabel("Timestamp")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    logger.info(f"Saved plot to: '{args.output}'")


if __name__ == "__main__":
    main()

