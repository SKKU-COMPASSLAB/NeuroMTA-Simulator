from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuromta_matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / ".logs"
FIGSIZE = (10, 2)
BAR_WIDTH = 0.36
FONT_SIZE_TICK = 12
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 12
IMAGE_DPI = 500


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def entry_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["group_idx"]), int(row["entry_idx"])


def load_timings(path: Path, time_field: str) -> dict[tuple[int, int], dict[str, Any]]:
    timings: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = entry_key(row)
            timings[key] = {
                "status": row.get("status", ""),
                "node_kind": row.get("node_kind", ""),
                "op_type": row.get("op_type", ""),
                "time_sec": parse_float(row.get(time_field)),
                "time_field": time_field,
                "error": row.get("error", ""),
            }
    return timings


def load_entry_stats(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain an entries list")
    return sorted(entries, key=lambda item: (int(item["group_idx"]), int(item["entry_idx"])))


def build_records(
    stats_entries: list[dict[str, Any]],
    neuromta_timings: dict[tuple[int, int], dict[str, Any]],
    scalesim_timings: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for entry in stats_entries:
        key = (int(entry["group_idx"]), int(entry["entry_idx"]))
        neuromta = neuromta_timings.get(key, {})
        scalesim = scalesim_timings.get(key, {})
        neuromta_time = neuromta.get("time_sec")
        scalesim_time = scalesim.get("time_sec")
        ops_tops = float(entry.get("ops_tops", 0.0))

        records.append(
            {
                "key": key,
                "label": "AvgPool" if entry["layer_name"] == "AdaptiveAvgPool" else entry["layer_name"],
                "layer_name": entry["layer_name"],
                "op_type": entry["op_type"],
                "ops_tops": ops_tops,
                "neuromta_time_sec": neuromta_time,
                "scalesim_time_sec": scalesim_time,
                "neuromta_tops_per_sec": None if not neuromta_time else ops_tops / neuromta_time,
                "scalesim_tops_per_sec": None if not scalesim_time else ops_tops / scalesim_time,
                "neuromta_status": neuromta.get("status", "missing"),
                "scalesim_status": scalesim.get("status", "missing"),
            }
        )
    return records


def valid_bar_values(records: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.array([np.nan if record[field] is None else float(record[field]) for record in records], dtype=float)


def annotate_missing(ax: plt.Axes, xs: np.ndarray, records: list[dict[str, Any]], field: str, label: str, y: float) -> None:
    for x, record in zip(xs, records):
        if record[field] is None:
            ax.text(x, y, label, ha="center", va="bottom", fontsize=FONT_SIZE_TICK, rotation=90, color="red")


def finite_bar_values(*arrays: np.ndarray) -> np.ndarray:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return values[values > 0]


def missing_label_y(values: np.ndarray, log_y: bool) -> float:
    if len(values) == 0:
        return 1e-6
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if log_y:
        return min_value * 0.6
    return min_value + max((max_value - min_value) * 0.08, min_value * 0.15, 1e-12)


def set_padded_ylim(ax: plt.Axes, values: np.ndarray, log_y: bool) -> None:
    if len(values) == 0:
        return
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if log_y:
        ax.set_ylim(max(min_value * 0.45, 1e-15), max_value * 8.0)
    else:
        ax.set_ylim(0, max_value * 1.45)


def finish_bar_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    records: list[dict[str, Any]],
    ylabel: str,
    output_path: Path,
    log_y: bool,
    y_values: np.ndarray,
) -> None:
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_xticks(np.arange(len(records)))
    ax.set_xticklabels([str(idx) for idx in range(len(records))], fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Layer Index", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.legend(
        frameon=False,
        ncols=2,
        fontsize=FONT_SIZE_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        borderaxespad=0.0,
        handlelength=1.2,
        columnspacing=1.0,
    )
    ax.margins(x=0.01)
    if log_y:
        ax.set_yscale("log")
    set_padded_ylim(ax, y_values, log_y)
    fig.tight_layout(pad=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(fig)


def plot_latency(records: list[dict[str, Any]], output_path: Path, log_y: bool) -> None:
    x = np.arange(len(records))
    width = BAR_WIDTH
    neuromta = valid_bar_values(records, "neuromta_time_sec")
    scalesim = valid_bar_values(records, "scalesim_time_sec")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - width / 2, scalesim, width, label="SCALE-Sim", color="#f37a5c", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, neuromta, width, label="NeuroMTA", color="#69A3EB", edgecolor="black", linewidth=0.6)

    finite_values = finite_bar_values(neuromta, scalesim)
    # missing_y = missing_label_y(finite_values, log_y)
    # annotate_missing(ax, x + width / 2, records, "neuromta_time_sec", "missing", missing_y)
    # annotate_missing(ax, x - width / 2, records, "scalesim_time_sec", "skip", missing_y)

    finish_bar_plot(
        fig,
        ax,
        records,
        ylabel="Wall Time (sec)",
        output_path=output_path,
        log_y=log_y,
        y_values=finite_values,
    )


def plot_tops_per_sec(records: list[dict[str, Any]], output_path: Path, log_y: bool) -> None:
    x = np.arange(len(records))
    width = BAR_WIDTH
    neuromta = valid_bar_values(records, "neuromta_tops_per_sec")
    scalesim = valid_bar_values(records, "scalesim_tops_per_sec")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - width / 2, scalesim, width, label="SCALE-Sim", color="#f37a5c", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, neuromta, width, label="NeuroMTA", color="#69A3EB", edgecolor="black", linewidth=0.6)
    
    finite_values = finite_bar_values(neuromta, scalesim)
    # missing_y = missing_label_y(finite_values, log_y)
    # annotate_missing(ax, x + width / 2, records, "neuromta_tops_per_sec", "missing", missing_y)
    # annotate_missing(ax, x - width / 2, records, "scalesim_tops_per_sec", "skip", missing_y)

    finish_bar_plot(
        fig,
        ax,
        records,
        ylabel="Throughput (TOPS/sec)",
        output_path=output_path,
        log_y=log_y,
        y_values=finite_values,
    )


def print_neuromta_average_ops_per_sec(records: list[dict[str, Any]]) -> None:
    values = [
        float(record["neuromta_tops_per_sec"]) * 1e12
        for record in records
        if record["neuromta_tops_per_sec"] is not None
    ]
    if not values:
        print("NeuroMTA average ops/sec: n/a")
        return
    print(f"NeuroMTA average ops/sec: {sum(values) / len(values):,.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NeuroMTA and SCALE-Sim per-entry simulation time.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--neuromta-csv", type=Path, default=None)
    parser.add_argument("--scalesim-csv", type=Path, default=None)
    parser.add_argument("--entry-stats-json", type=Path, default=None)
    parser.add_argument("--time-field", default="wall_time_sec", help="CSV time column used for latency and TOPS/sec.")
    parser.add_argument("--linear-y", action="store_true", help="Use linear y-axis instead of log scale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    neuromta_csv = args.neuromta_csv or results_dir / "neuromta_entry_timings.csv"
    scalesim_csv = args.scalesim_csv or results_dir / "scalesim_entry_timings.csv"
    entry_stats_json = args.entry_stats_json or results_dir / "entry_stats.json"

    stats_entries = load_entry_stats(entry_stats_json)
    neuromta_timings = load_timings(neuromta_csv, args.time_field)
    scalesim_timings = load_timings(scalesim_csv, args.time_field)
    records = build_records(stats_entries, neuromta_timings, scalesim_timings)

    plot_latency(records, results_dir / "exp4_sim_time_comparison_latency.png", log_y=not args.linear_y)
    plot_tops_per_sec(records, results_dir / "exp4_sim_time_comparison_tops_per_sec.png", log_y=not args.linear_y)
    print_neuromta_average_ops_per_sec(records)

    print(f"Wrote {results_dir / 'exp4_sim_time_comparison_latency.png'}")
    print(f"Wrote {results_dir / 'exp4_sim_time_comparison_tops_per_sec.png'}")


if __name__ == "__main__":
    main()
