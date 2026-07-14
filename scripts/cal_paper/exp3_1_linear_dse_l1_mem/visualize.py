import os
import pandas as pd
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuromta_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

def draw(src_path: str, img_path: str):
    workloads = {}
    df = pd.read_csv(src_path)
    for _, row in df.iterrows():
        timestamp:    int = row['Timestamp (cycles)']
        cache_buffer_size: int = row['Cache Buffer Size (KB)']
        name = f"{cache_buffer_size}"
        
        workloads[name] = min(workloads.get(name, float('inf')), timestamp)

    plt.figure(figsize=(5, 2))

    x_labels = list(sorted(workloads.keys(), key=lambda n: int(n)))
    x_values = np.arange(len(x_labels))
    baseline_timestamp = max(workloads.values())
    y_values = baseline_timestamp / np.array([workloads[name] for name in x_labels], dtype=np.float32)

    plt.bar(x_values, y_values, color="#69A3EB", edgecolor='black', width=0.6)
    plt.xticks(x_values, x_labels, rotation=0, ha='center', fontsize=10)
    plt.hlines(1.0, -0.5, len(x_labels)-0.5, colors='red', linestyles='dashed', linewidth=0.8)
    
    plt.xlabel('Cache Region Size (KB)', fontsize=11)
    plt.ylabel('Speedup', fontsize=12)
    plt.margins(x=0.00)

    plt.tight_layout(pad=0.1)
    plt.savefig(img_path, dpi=500)
    
    print(f"Speedup graph saved to '{img_path}'")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
    
    parser = argparse.ArgumentParser(description="Cache Region Sweep Visualization")
    parser.add_argument("-t", "--test-name", type=str, default=f"linear_all_main", help="Name of the test", dest="test_name")
    args = parser.parse_args()

    log_dir  = os.path.join(ROOT_DIR, ".logs")
    src_path = os.path.join(log_dir, f"{args.test_name}.csv")
    img_path = os.path.join(log_dir, f"{args.test_name}.png")
    
    draw(src_path, img_path)
