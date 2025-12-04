import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

PEAK_PERFORMANCE   = 2 * (8 * 2) * (32 * 32)  # 4x4 Core Grid | 32x32 MXU | MAC = 2 OPs 
PEAK_BANDWIDTH = 387.88  # theoretical peak bandwidth in GB/cycle


def draw(peak_perf: int, peak_bw: int, src_path: str, img_path: str, img_title: str):
    ai_x = np.logspace(-3, 3, 500) 

    bandwidth_limit = ai_x * peak_bw                # Bandwidth-bound: P = AI * PEAK_BANDWIDTH
    compute_limit = np.full_like(ai_x, peak_perf)   # Compute-bound:   P = PEAK_COMPUTE
    roofline = np.minimum(bandwidth_limit, compute_limit)   # Roofline is determined by the lower of the two limits.

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
        
        name = name.split('_')[0] # extract only MNK dimensions
        
        workloads[name] = {'AI': ai, 'PERF': perf}

    plt.figure(figsize=(8, 5.5))
    plt.loglog(ai_x, roofline, color='red', linewidth=1, label='Theoretical Peak')
    
    mem_bound_marker = 'o'
    comp_bound_marker = '^'
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow']
    
    mem_bound_cnt = 0
    comp_bound_cnt = 0
    
    AI_balance = peak_perf / peak_bw

    for i, (name, data) in enumerate(workloads.items()):
        if data['AI'] < AI_balance:
            index = mem_bound_cnt
            mem_bound_cnt += 1
            
            marker = mem_bound_marker
            color = colors[index % len(colors)]
        else:
            index = comp_bound_cnt
            comp_bound_cnt += 1
            
            marker = comp_bound_marker
            color = colors[index % len(colors)]
        
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
    
    # Plot the machine balance point
    plt.loglog(
        AI_balance, 
        peak_perf, 
        marker='*', 
        color='red', 
        markersize=13,
        mec='black',
        linestyle='', 
        label=f'Machine Balance'
    )
    
    # Annotate the balance point
    plt.annotate(
        f'Balance Point: ({AI_balance:.2f} OPs/Byte, {peak_perf:.2f} OPs/Cycle)',
        xy=(AI_balance, peak_perf),
        xytext=(AI_balance * 0.005, peak_perf * 2),
        fontsize=10,
        horizontalalignment='left',
        verticalalignment='top'
    )
    
    # Draw dashed lines to indicate the balance point
    plt.vlines(AI_balance, roofline.min() * 0.1, peak_perf, color='black', linestyle=':', alpha=1)

    # Final plot adjustments
    plt.title(img_title, fontsize=11)
    plt.xlabel('Arithmetic Intensity (OPs/Byte) - $\\log$ scale', fontsize=10)
    plt.ylabel('Performance (OPs/Cycle or GFLOP/s) - $\\log$ scale', fontsize=10)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.xlim(ai_x.min(), ai_x.max()) 
    plt.ylim(roofline.min() * 0.1, peak_perf * 2.5)

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