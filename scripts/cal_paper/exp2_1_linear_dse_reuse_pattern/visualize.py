import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


pattern_alias = {
    "IGNORE": "IG",
    "ALL_MAIN": "AM",
    "ALL_L1": "AL",
    "SINGLE_MAIN": "SM",
    "SINGLE_L1": "SL",
}


def draw(src_path: str, img_path: str):
    workloads = {}
    df = pd.read_csv(src_path)
    for _, row in df.iterrows():
        timestamp:    int = row['Timestamp (cycles)']
        temporal_reuse: int = pattern_alias[row['Temporal Reuse']]
        spatial_reuse: int = pattern_alias[row['Spatial Reuse']]
        name = f"T{temporal_reuse}/S{spatial_reuse}"

        workloads[name] = timestamp
        
    baseline_name = "TIG/SIG"
    baseline_timestamp = workloads[baseline_name]

    plt.figure(figsize=(5, 2.4))

    x_labels = list(sorted(workloads.keys(), key=lambda n: workloads[n], reverse=True))
    x_values = np.arange(len(x_labels))
    y_values = baseline_timestamp / np.array([workloads[name] for name in x_labels], dtype=np.float32)

    # Color bars relative to baseline: below baseline -> light pink, baseline -> red, above baseline -> skyblue
    colors = []
    hatches = []
    light_pink = '#FFC0CB'
    baseline_red = "#EF3E5C"
    above_blue = "#69A3EB"
    for name, val in zip(x_labels, y_values):
        if name == baseline_name:
            colors.append(baseline_red)
            hatches.append('////')  # Add hatching for bars below baseline
        elif val < 1.0:
            colors.append(light_pink)
            hatches.append('')  # No hatching for baseline
        else:
            colors.append(above_blue)
            hatches.append('')  # No hatching for bars above baseline

    plt.bar(x_values, y_values, color=colors, edgecolor='black', width=0.6, hatch=hatches)
    plt.xticks(x_values, x_labels, rotation=45, ha='right', fontsize=10)
    plt.hlines(1.0, -0.5, len(x_labels)-0.5, colors='red', linestyles='dashed', linewidth=0.8, label='Baseline (TIG/SIG)')
    plt.annotate('Baseline (TIG/SIG)', xy=(-0.3, 1.2), xytext=(-0.3, 1.2), fontsize=10, color='red')
    
    label_descs = [
        "T : Temporal    | S : Spatial",
        "AM: All Main    | AL: All L1",
        "SM: Single Main | SL: Single L1",
        "IG: Ignore (No Reuse)",
    ]
    label_y_distance = 0.5
    label_y_st_position = 3.7
    
    for i, desc in enumerate(label_descs):
        plt.text(
            -0.3, label_y_st_position - i * label_y_distance, 
            desc, 
            fontsize=10, 
            color='black',
            fontfamily='monospace'
        )
    
    plt.xlabel('Data Reuse Pattern', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.margins(x=0.00)
    plt.ylim(0, y_values.max() * 1.2)

    plt.tight_layout(pad=0.1)
    plt.savefig(img_path, dpi=500)
    
    print(f"Roofline graph saved to '{img_path}'")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
    
    parser = argparse.ArgumentParser(description="Roofline Analysis Visualization")
    parser.add_argument("-t", "--test-name", type=str, default=f"linear_all_main", help="Name of the test", dest="test_name")
    args = parser.parse_args()

    log_dir  = os.path.join(ROOT_DIR, ".logs")
    src_path = os.path.join(log_dir, f"{args.test_name}.csv")
    img_path = os.path.join(log_dir, f"{args.test_name}.png")
    
    draw(src_path, img_path)