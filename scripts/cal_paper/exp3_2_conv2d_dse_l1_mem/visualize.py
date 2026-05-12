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
        ld_ex_buffer_size: int = row['LD/EX Buffer Size (KB)']
        ex_st_buffer_size: int = row['EX/ST Buffer Size (KB)']
        bcast_buffer_size: int = row['Broadcast Buffer Size (KB)']
        cache_buffer_size: int = row['Cache Buffer Size (KB)']
        
        name = f"{cache_buffer_size}"
        
        workloads[name] = min(workloads.get(name, float('inf')), timestamp)
        
    baseline_name = max(workloads.keys(), key=lambda n: workloads[n])  # Find the configuration with the lowest timestamp as baseline
    baseline_timestamp = workloads[baseline_name]

    plt.figure(figsize=(5, 2))

    x_labels = list(sorted(workloads.keys(), key=lambda n: int(n)))
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
    plt.xticks(x_values, x_labels, rotation=0, ha='center', fontsize=10)
    plt.hlines(1.0, -0.5, len(x_labels)-0.5, colors='red', linestyles='dashed', linewidth=0.8, label=f'Baseline ({baseline_name})')
    
    plt.xlabel('Cache Region [KB] (Total 1MB / PP: 42% / IC: 68%)', fontsize=11)
    plt.ylabel('Speedup', fontsize=12)
    plt.margins(x=0.00)
    # plt.ylim(0, 2)

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