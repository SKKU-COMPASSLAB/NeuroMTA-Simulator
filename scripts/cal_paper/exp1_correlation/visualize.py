import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PEAK_PERFORMANCE   = 2 * (4 * 4) * (32 * 32)  # 4x4 Core Grid | 32x32 MXU | MAC = 2 OPs 
PEAK_BANDWIDTH = 387.88  # theoretical peak bandwidth in GB/cycle
PEAK_NOC_BANDWIDTH = PEAK_BANDWIDTH
CORE_COLOR_SCHEME = [
    '#E63946',  # vivid red
    '#1D4ED8',  # strong blue
    '#0F766E',  # deep teal
    '#FF7F11',  # vivid orange
    '#7C3AED',  # strong violet
    '#16A34A',  # bright green
    '#F72585',  # magenta
    '#4B5563',  # neutral gray
]
BENCHMARK_MARKERS = {'LN': 'o', 'CV': '^', 'OTHER': 's'}


def rename_benchmark(name: str) -> str:
    name = name.replace("n_cores_", 'NC')
    name = name.replace("bfloat", "BF").replace("int", "INT").replace("float", "FP")
    return name


def draw(peak_perf_per_core: float, peak_mem_bw: float, peak_noc_bw: float, src_path: str, img_path: str, img_title: str):
    ai_x = np.logspace(1, 5, 500)

    workloads = {}
    df = pd.read_csv(src_path)
    for _, row in df.iterrows():
        name:         str = row['Benchmark']
        ops:          int = row['Total OPs']
        timestamp:    int = row['Timestamp (cycles)']
        main_traffic: int = row['Main Memory Traffic (Bytes)']
        l1_traffic:   int = row['L1 Memory Traffic (Bytes)']
        n_cores:      int = row['Number of Cores']
        
        perf = ops / timestamp  # OPs/cycle
        ai = ops / (main_traffic + l1_traffic)  # OPs/Byte
        
        workloads[name] = {'AI': ai, 'PERF': perf, 'N_CORES': n_cores}

    n_cores_list = sorted({int(v) for v in df['Number of Cores'].unique()})
    # if len(n_cores_list) > len(peak_perfs):
    #     raise ValueError("number of unique cores in csv is larger than number of peak_perfs")
    peak_perfs = [peak_perf_per_core * n_cores for n_cores in n_cores_list]

    if len(n_cores_list) == len(peak_perfs):
        n_cores_to_peak_perf = dict(zip(n_cores_list, peak_perfs))
    else:
        peak_per_core = max(peak_perfs) / max(n_cores_list)
        n_cores_to_peak_perf = {
            n_cores: min(peak_perfs, key=lambda peak: abs(peak - n_cores * peak_per_core))
            for n_cores in n_cores_list
        }

    plt.figure(figsize=(6, 3))

    n_cores_to_color = {
        n_cores: CORE_COLOR_SCHEME[i % len(CORE_COLOR_SCHEME)]
        for i, n_cores in enumerate(n_cores_list)
    }

    y_min_candidates = []
    y_max_candidates = []
    for n_cores in n_cores_list:
        peak_perf = n_cores_to_peak_perf[n_cores]
        mem_bw_limit = ai_x * peak_mem_bw
        noc_bw_limit = ai_x * peak_noc_bw
        compute_limit = np.full_like(ai_x, peak_perf)
        mem_roofline = np.minimum(mem_bw_limit, compute_limit)
        noc_roofline = np.minimum(noc_bw_limit, compute_limit)

        color = n_cores_to_color[n_cores]
        plt.loglog(
            ai_x,
            mem_roofline,
            color=color,
            linewidth=0.9,
            linestyle='-',
            label=f'{n_cores} Tiles Mem Roofline'
        )
        plt.loglog(
            ai_x,
            noc_roofline,
            color=color,
            linewidth=0.9,
            linestyle='--',
            label=f'{n_cores} Tiles NoC Roofline'
        )

        mem_ai_balance = peak_perf / peak_mem_bw
        plt.vlines(mem_ai_balance, mem_roofline.min() * 0.1, peak_perf, color=color, linestyle=':', alpha=0.8)

        y_min_candidates.append(mem_roofline.min())
        y_max_candidates.append(peak_perf)
    
    present_types = set()
    for name, data in workloads.items():
        n_cores = data['N_CORES']
        benchmark_type = 'LN' if name.startswith('LN') else 'CV' if name.startswith('CV') else 'OTHER'
        present_types.add(benchmark_type)
        marker = BENCHMARK_MARKERS[benchmark_type]
        color = n_cores_to_color[n_cores]
        
        # Plot each workload point
        plt.loglog(
            data['AI'], 
            data['PERF'], 
            marker=marker, 
            color=color, 
            markersize=10, 
            mec='black',
            linestyle='')
        

    # Final plot adjustments
    # plt.title(img_title, fontsize=11)
    plt.xlabel('Arithmetic Intensity (OPs/Byte)', fontsize=12)
    plt.ylabel('Performance (OPs/Cycle)', fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.xlim(ai_x.min(), ai_x.max())
    plt.ylim(min(y_min_candidates) * 0.1, max(y_max_candidates) * 2.5)

    core_handles = []
    for n_cores in n_cores_list:
        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=n_cores_to_color[n_cores], markeredgecolor='black', markersize=9, linestyle='', label=f'{n_cores} Tiles')
        core_handles.append(handle)

    ax = plt.gca()
    core_legend = ax.legend(
        handles=core_handles, loc='lower right', fontsize=10,
        title='Number of Tiles', framealpha=0.9,
    )
    ax.add_artist(core_legend)

    type_handles = [
        Line2D(
            [0], [0], marker=BENCHMARK_MARKERS[benchmark_type], color='none',
            markerfacecolor='gray', markeredgecolor='black', markersize=9,
            label=benchmark_type,
        )
        for benchmark_type in ('LN', 'CV', 'OTHER')
        if benchmark_type in present_types
    ]
    type_legend = ax.legend(
        handles=type_handles, loc='upper left', fontsize=10,
        title='Layer Type', framealpha=0.9,
    )
    ax.add_artist(type_legend)

    roofline_handles = [
        Line2D([0], [0], color='gray', linewidth=1.2, linestyle='-', label='Memory Roofline'),
        Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--', label='NoC Roofline'),
    ]
    ax.legend(
        handles=roofline_handles, loc='upper right', fontsize=10,
        title='Roofline', framealpha=0.9,
    )
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
    
    img_title = f"Tenstorrent Roofline Analysis - {args.test_name.replace('_', ' ').title()}"
    if "linear" in args.test_name:
        img_title += " (M x N x K Dimensions)"
    
    draw(PEAK_PERFORMANCE, PEAK_BANDWIDTH, PEAK_NOC_BANDWIDTH, src_path, img_path, img_title)
