import os
import pandas as pd
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT_DIR, ".logs", "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def build_speedup_data(src_path: str):
    # Define category mappings and ordered axes
    temporal_map = {
        'IGNORE': 'IG',
        'SINGLE_L1': 'SL',
        'ALL_L1': 'AL',
        'SINGLE_MAIN': 'SM',
        'ALL_MAIN': 'AM',
    }

    # Map multiple spatial types into three categories: SIG, SSL, SSM
    spatial_map = {
        'IGNORE': 'IG',
        'SINGLE_L1': 'SL',
        'ALL_L1': 'AL',
        'SINGLE_MAIN': 'SM',
        'ALL_MAIN': 'AM',
    }

    spatial_categories = ['IG', 'SL', 'SM']
    temporal_categories = ['IG', 'SL', 'SM']

    # Read CSV and collect timestamps per (temporal, spatial) category
    df = pd.read_csv(src_path)
    timestamps = {}
    for _, row in df.iterrows():
        t_raw = row['Temporal Reuse']
        s_raw = row['Spatial Reuse']
        # Skip rows with unexpected labels
        if t_raw not in temporal_map or s_raw not in spatial_map:
            continue
        t_cat = temporal_map[t_raw]
        s_cat = spatial_map[s_raw]
        timestamps[(t_cat, s_cat)] = float(row['Timestamp (cycles)'])

    baseline_key = ('IG', 'IG')
    if baseline_key not in timestamps:
        raise RuntimeError("Baseline data for IG/IG not found in input CSV")
    baseline_ts = timestamps[baseline_key]

    # Build data matrix: speedup = baseline / timestamp
    data = np.full((len(temporal_categories), len(spatial_categories)), np.nan, dtype=np.float32)
    for i, t in enumerate(temporal_categories):
        for j, s in enumerate(spatial_categories):
            key = (t, s)
            if key in timestamps and timestamps[key] > 0:
                data[i, j] = baseline_ts / timestamps[key]

    return data, spatial_categories, temporal_categories


def draw_heatmap_on_axis(ax, data, spatial_categories, temporal_categories):
    # Mask missing values for nicer plotting
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='lightgray')

    # Use data as-is so x=spatial (cols), y=temporal (rows). Data shape: (temporal, spatial)
    masked_no_swap = np.ma.masked_invalid(data)

    im = ax.imshow(masked_no_swap, interpolation='nearest', cmap=cmap, aspect='auto')

    # Set ticks and labels (x: spatial, y: temporal)
    ax.set_xticks(np.arange(len(spatial_categories)))
    ax.set_xticklabels(spatial_categories, fontsize=11)
    ax.set_yticks(np.arange(len(temporal_categories)))
    ax.set_yticklabels(temporal_categories, fontsize=11)
    ax.set_xlabel('Spatial Pattern', fontsize=12)
    ax.set_ylabel('Temporal Pattern', fontsize=12)

    # Prepare normalization bounds for colormap -> determine readable text color per cell
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except ValueError:
        vmin = np.nan
        vmax = np.nan

    def _normalized_value(x):
        if np.isnan(x) or np.isnan(vmin) or np.isnan(vmax) or vmax == vmin:
            return 0.5
        return (x - vmin) / (vmax - vmin)

    # Annotate each cell with numeric value (or '-') using luminance-based text color
    # Loop over data matrix indices: rows -> temporal (y), cols -> spatial (x)
    for y in range(len(temporal_categories)):
        for x in range(len(spatial_categories)):
            val = data[y, x]
            if np.isnan(val):
                txt = '-'
                color = 'black'
            else:
                txt = f"{val:.2f}x"
                # Determine text color from colormap luminance
                norm_val = _normalized_value(val)
                r, g, b, _ = cmap(norm_val)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                color = 'white' if luminance < 0.5 else 'black'
            ax.text(x, y, txt, ha='center', va='center', color=color, fontsize=10)

    return im


def draw(src_path: str, img_path: str):
    data, spatial_categories, temporal_categories = build_speedup_data(src_path)
    fig, ax = plt.subplots(figsize=(3, 2.4))
    im = draw_heatmap_on_axis(ax, data, spatial_categories, temporal_categories)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # plt.title('Speedup Heatmap (baseline: IG/IG)', fontsize=12)
    fig.tight_layout(pad=0.1)
    fig.savefig(img_path, dpi=500)
    plt.close(fig)
    print(f"Heatmap saved to '{img_path}'")


def draw_category_table_on_axis(ax):
    patterns = pd.DataFrame({
        "Pattern": ["IG", "SL", "AL", "SM", "AM"],
        "Description": [
            "Ignore",
            "Single L1 Buffer",
            "All L1 Buffers",
            "Single Main Buffer",
            "All Main Buffers",
        ]
    })

    ax.axis('off')
    t = ax.table(
        cellText=patterns.values,
        colLabels=patterns.columns,
        cellLoc='left',
        colLoc='left',
        loc='center',
        colWidths=[0.4, 0.7],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.5)

    for (row, col), cell in t.get_celld().items():
        txt = cell.get_text()
        txt.set_color('black')
        if row == 0:
            txt.set_fontweight('bold')

    return t

def draw_category_table(img_path: str):
    # reuse_types = pd.DataFrame({
    #     "Type": ["T", "S"],
    #     "Description": [
    #         "Temporal Reuse",
    #         "Spatial Reuse",
    #     ]
    # })
    
    patterns = pd.DataFrame({
        "Pattern": ["IG", "SL", "AL", "SM", "AM"],
        "Description": [
            "Ignore",
            "Single L1 Buffer",
            "All L1 Buffers",
            "Single Main Buffer",
            "All Main Buffers",
        ]
    })
    
    fig = plt.figure(figsize=(2.6, 2.4))
    widths = [0.4, 0.7]

    # Turn off axes and render tables centered in each subplot
    # for ax in axes:
    #     ax.axis('off')

    # t1 = axes[0].table(
    #     cellText=reuse_types.values,
    #     colLabels=reuse_types.columns,
    #     cellLoc='left',
    #     colLoc='left',
    #     loc='center',
    #     colWidths=widths,
    # )
    # t1.auto_set_font_size(False)
    # t1.set_fontsize(10)
    # t1.scale(1, 1.5)

    t2 = fig.add_subplot(111)
    draw_category_table_on_axis(t2)

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(pad=0.4)
    plt.savefig(img_path, dpi=500)
    print(f"Table saved to '{img_path}'")


def draw_combined(src_path: str, img_path: str):
    data, spatial_categories, temporal_categories = build_speedup_data(src_path)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(5.8, 2.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    im = draw_heatmap_on_axis(axes[0], data, spatial_categories, temporal_categories)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    draw_category_table_on_axis(axes[1])

    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(img_path, dpi=500)
    plt.close(fig)
    print(f"Combined figure saved to '{img_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heatmap Visualization for Data Reuse Patterns")
    parser.add_argument("-t", "--test-name", type=str, default="linear_all_main", help="Name of the test", dest="test_name")
    args = parser.parse_args()

    log_dir = os.path.join(ROOT_DIR, ".logs")
    src_path = os.path.join(log_dir, f"{args.test_name}.csv")
    img_path = os.path.join(log_dir, "exp2_2_conv2d_dse_reuse_pattern_combined.png")

    draw_combined(src_path, img_path)
