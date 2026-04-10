import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

PEAK_PERFORMANCE = (12 * 14) * 4096   
PEAK_BANDWIDTH   = 387.88  # theoretical peak bandwidth in GB/cycle


def draw(peak_perf: int, peak_mem_bw: int, peak_noc_bw: int, src_path: str, img_path: str, img_title: str):
    ai_x = np.logspace(0, 5, 500)

    mem_bw_limit = ai_x * peak_mem_bw               # Bandwidth-bound: P = AI * PEAK_BANDWIDTH
    noc_bw_limit = ai_x * peak_noc_bw               # NoC Bandwidth-bound: P = AI * PEAK_NOC_BANDWIDTH
    compute_limit = np.full_like(ai_x, peak_perf)   # Compute-bound:   P = PEAK_COMPUTE
    mem_roofline = np.minimum(mem_bw_limit, compute_limit)   # Roofline is determined by the lower of the two limits.
    noc_roofline = np.minimum(noc_bw_limit, compute_limit)   # Roofline is determined by the lower of the two limits.

    workloads = {}
    df = pd.read_csv(src_path)
    for idx, row in df.iterrows():
        name:         str = row['Benchmark']
        ops:          int = row['Total OPs']
        timestamp:    int = row['Timestamp (cycles)']
        main_traffic: int = row['Main Memory Traffic (Bytes)']
        l1_traffic:   int = row['L1 Memory Traffic (Bytes)']
        
        perf = ops / timestamp  # OPs/cycle
        ai = ops / (main_traffic + l1_traffic)  # OPs/Byte
        
        workloads[name] = {'AI': ai, 'PERF': perf}

    plt.figure(figsize=(8, 5.5))
    plt.loglog(ai_x, mem_roofline, color='red', linewidth=0.8, label='Memory Roofline', linestyle='-')
    plt.loglog(ai_x, noc_roofline, color='red', linewidth=0.8, label='NoC Roofline', linestyle='--')
    
    mem_bound_marker = 'o'
    comp_bound_marker = '^'
    # colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow']
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
        '#ffffff', '#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    ]
    
    # mem_bound_cnt = 0
    # comp_bound_cnt = 0
    
    mem_ai_balance = peak_perf / peak_mem_bw
    noc_ai_balance = peak_perf / peak_noc_bw

    for i, (name, data) in enumerate(workloads.items()):
        if data['AI'] < mem_ai_balance or data['AI'] < noc_ai_balance:
            marker = mem_bound_marker
            color = colors[i % len(colors)]
        else:
            marker = comp_bound_marker
            color = colors[i % len(colors)]
        
        # Plot each workload point
        plt.loglog(
            data['AI'], 
            data['PERF'], 
            marker=marker, 
            color=color, 
            markersize=8, 
            mec='black',
            linestyle='', 
            label=name
        )

    # Annotate the balance point
    plt.annotate(
        f'Memory Balance: ({mem_ai_balance:.2f} OPs/Byte, {peak_perf:.2f} OPs/Cycle)',
        xy=(mem_ai_balance, peak_perf),
        xytext=(mem_ai_balance * 0.005, peak_perf * 2),
        fontsize=10,
        horizontalalignment='left',
        verticalalignment='top'
    )
    
    # Draw dashed lines to indicate the balance point
    plt.vlines(mem_ai_balance, mem_roofline.min() * 0.1, peak_perf, color='black', linestyle=':', alpha=1)

    # Final plot adjustments
    plt.title(img_title, fontsize=11)
    plt.xlabel('Arithmetic Intensity (OPs/Byte) - $\\log$ scale', fontsize=10)
    plt.ylabel('Performance (OPs/Cycle or GFLOP/s) - $\\log$ scale', fontsize=10)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.xlim(ai_x.min(), ai_x.max()) 
    plt.ylim(mem_roofline.min() * 0.1, peak_perf * 2.5)

    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout(pad=0.8)
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
    
    img_title = f"Tenstorrent Roofline Analysis - {args.test_name.replace('_', ' ').title()}"
    if "linear" in args.test_name:
        img_title += " (M x N x K Dimensions)"
    
    draw(PEAK_PERFORMANCE, PEAK_BANDWIDTH, src_path, img_path, img_title)