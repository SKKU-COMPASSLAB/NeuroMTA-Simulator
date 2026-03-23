import os
import argparse
import math
import matplotlib.pyplot as plt

from neuromta.framework import logger
from neuromta_monitor.profiler import ProfilerFileLoaderHub


def has_plot_content(x, y):
    return len(x) > 0 and any(v != 0 for v in y)


def visualize_stat(ax: plt.Axes, x, y, meta):
    metric_name = meta['metric_name'].capitalize()
    metric_unit = meta['metric_unit']
    metric_name_lower = str(meta['metric_name']).lower()
    is_utilization = "utilization" in metric_name_lower

    if is_utilization:
        y = [v * 100 for v in y]
        metric_unit = "%"

    realtime_label = "utilization" if is_utilization else "bandwidth"

    if len(y) == 0:
        y_max = 115.0 if is_utilization else 1.15
        y_mean = 0.0
    else:
        y_max_data = max(y)
        if is_utilization:
            y_max = 115.0
        elif "bandwidth" in metric_name_lower:
            y_max = y_max_data * 1.15
        else:
            y_max = y_max_data * 1.15
        y_mean = sum(y) / len(y)

    y_max = max(y_max, 1e-9)

    ax.plot(x, y, color="#6eadf0", label=realtime_label, linewidth=0.7)
    ax.axhline(y_mean, color="red", linestyle="--", linewidth=1.2, label="mean")

    x_annot = 0 if len(x) == 0 else (x[0] + (x[-1] - x[0]) * 0.01)
    y_offset = y_max * 0.02
    unit_suffix = f" {metric_unit}" if metric_unit else ""
    ax.annotate(f"{y_mean:.2f}{unit_suffix}", xy=(x_annot, y_mean + y_offset), color="red", ha="left")

    ax.set_xlabel("Timestamp (cycle)")
    ax.set_ylabel(f"{metric_name} ({metric_unit})")
    ax.set_ylim(bottom=0, top=y_max)
    ax.set_xlim(left=0)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=True, shadow=False)


def save_individual_visualizations(
    loader_hub: ProfilerFileLoaderHub,
    output_dir: str,
    fig_width: float = 10.0,
    fig_height: float = 5.0,
):
    for i in range(loader_hub.n_profilers):
        stat = loader_hub.create_stat(i)
        meta = loader_hub.metadata[i]

        if not has_plot_content(stat.x, stat.y):
            metric_name = meta['metric_name'].replace(" ", "_").lower()
            logger.info(f"Skipped empty visualization for profiler {i} ({metric_name})")
            continue

        metric_name = meta['metric_name'].replace(" ", "_").lower()
        img_path = os.path.join(output_dir, f"profiler_{i}_{metric_name}.png")

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        visualize_stat(ax, stat.x, stat.y, meta)
        fig.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

        logger.info(f"Saved visualization for profiler {i} ({metric_name}) at '{img_path}'")


def save_bundle_visualization(
    loader_hub: ProfilerFileLoaderHub,
    output_dir: str,
    n_cols: int = 2,
    subplot_fig_width: float = 6.0,
    subplot_fig_height: float = 4.0,
):
    n_stats = loader_hub.n_profilers
    n_cols = max(1, n_cols)

    stats = [loader_hub.create_stat(i) for i in range(n_stats)]
    metas = [loader_hub.metadata[i] for i in range(n_stats)]

    x_max = 0
    has_x_data = any(len(stat.x) > 0 for stat in stats)
    if has_x_data:
        x_max = max(int(max(stat.x)) for stat in stats if len(stat.x) > 0)

    shared_x = list(range(0, x_max + 1))

    drawable_items = []
    for i in range(n_stats):
        stat = stats[i]
        meta = metas[i]
        stat_y_by_x = {int(x): y for x, y in zip(stat.x, stat.y)}
        padded_y = [stat_y_by_x.get(x, 0) for x in shared_x]

        if not has_plot_content(shared_x, padded_y):
            metric_name = meta['metric_name'].replace(" ", "_").lower()
            logger.info(f"Skipped empty subplot for profiler {i} ({metric_name})")
            continue

        drawable_items.append((i, meta, padded_y))

    if len(drawable_items) == 0:
        logger.info("Skipped bundled visualization because all profilers were empty")
        return

    n_rows = math.ceil(len(drawable_items) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_fig_width * n_cols, subplot_fig_height * n_rows),
        squeeze=False,
    )

    for plot_idx, (i, meta, padded_y) in enumerate(drawable_items):
        row_idx, col_idx = divmod(plot_idx, n_cols)
        ax = axes[row_idx][col_idx]

        visualize_stat(ax, shared_x, padded_y, meta)
        ax.set_xlim(0, x_max)

        metric_name = meta['metric_name'].replace(" ", "_").lower()
        ax.set_title(f"Profiler {i}: {metric_name}")

    for i in range(len(drawable_items), n_rows * n_cols):
        row_idx, col_idx = divmod(i, n_cols)
        fig.delaxes(axes[row_idx][col_idx])

    img_path = os.path.join(output_dir, "profilers_bundle.png")
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

    logger.info(f"Saved bundled visualization at '{img_path}'")


def visualize_monitoring_data(
    profile_dir: str,
    output_dir: str,
    save_individual: bool = True,
    save_bundle: bool = True,
    bundle_cols: int = 2,
    fig_width: float = 6,
    fig_height: float = 4,
):
    os.makedirs(output_dir, exist_ok=True)

    loader_hub = ProfilerFileLoaderHub(profile_dir)

    if save_individual:
        save_individual_visualizations(
            loader_hub,
            output_dir,
            fig_width=fig_width,
            fig_height=fig_height,
        )

    if save_bundle:
        save_bundle_visualization(
            loader_hub,
            output_dir,
            n_cols=bundle_cols,
            subplot_fig_width=fig_width,
            subplot_fig_height=fig_height,
        )

def main():
    FILEROOT = os.path.dirname(os.path.abspath(__file__))
    LOGDIR = os.path.join(FILEROOT, ".logs")
    
    parser = argparse.ArgumentParser(description="Visualize monitoring data from profiler files.")
    parser.add_argument("--test", type=str, help="Name of the test to visualize")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["individual", "bundle", "both"],
        default="both",
        help="Output mode: separate images, bundled subplots, or both",
    )
    parser.add_argument(
        "--bundle-cols",
        type=int,
        default=2,
        help="Number of columns for bundled subplot image",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6,
        help="Figure width. In bundle mode, width of each subplot.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=4,
        help="Figure height. In bundle mode, height of each subplot.",
    )
    args = parser.parse_args()

    profile_dir = os.path.join(LOGDIR, args.test, "profiles")
    output_dir = os.path.join(LOGDIR, args.test, "visualizations")

    save_individual = args.mode in ["individual", "both"]
    save_bundle = args.mode in ["bundle", "both"]

    visualize_monitoring_data(
        profile_dir,
        output_dir,
        save_individual=save_individual,
        save_bundle=save_bundle,
        bundle_cols=args.bundle_cols,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

if __name__ == "__main__":
    main()