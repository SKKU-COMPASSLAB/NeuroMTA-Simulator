import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig


class ColorMap:
    MAIN = "#1f77b4"
    MEAN = "#ff7f0e"


def create_bandwidth_utilization_plot(ax: plt.Axes, arr: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Axes:
    ax.plot(arr, label="Bandwidth Utilization", color=ColorMap.MAIN)
    ax.axhline(y=np.mean(arr), color=ColorMap.MEAN, linestyle="-", linewidth=2, label="Average Bandwidth")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xmargin(0)
    ax.set_ylim(0, None)
    
    ax.grid(True)
    ax.legend()
    
    return ax

def visualize_bandwidth_utilization_graph(
    PROFILE_DIR: str,
    ICNT_CORE_TRACE_FNAME: str,
    MAIN_MEM_CORE_TRACE_FNAME: str,
    IMG_SAVE_FNAME: str,
):
    results = []
    timestamp = 0
    
    for filename in os.listdir(PROFILE_DIR):
        filepath = os.path.join(PROFILE_DIR, filename)
        
        df = pd.read_csv(filepath)
        
        df['command_id'] = df['command_id'].str.strip()
        mxu_row = df[df['command_id'] == 'mxu_tiled_gemm']
        
        timestamp = max(timestamp, df['last_commit_time'].iloc[0])
        
        result = mxu_row['duration'].iloc[0] / mxu_row['last_commit_time'].iloc[0]
        results.append(result)

    logger.info(f"Average MXU Utilization: {sum(results)/len(results)*100:.2f}%")
    
    icnt_analysis_available = os.path.isfile(ICNT_CORE_TRACE_FNAME)        
    main_mem_analysis_available = os.path.isfile(MAIN_MEM_CORE_TRACE_FNAME)
    
    n_fig = int(icnt_analysis_available) + int(main_mem_analysis_available)
    fig_cursor = 0
    fig, axs = plt.subplots(n_fig, 1, figsize=(12, 8))
    
    if n_fig == 1:
        axs = [axs]
    
    if main_mem_analysis_available:
        main_mem_core_analyzer = MainMemCoreAnalyzer()
        main_mem_core_analyzer.load_traces(MAIN_MEM_CORE_TRACE_FNAME)
        main_mem_bandwidth_arr = np.array(main_mem_core_analyzer.dump_bandwidth_analysis(bin_size=1), dtype=np.float32)
        
        if len(main_mem_bandwidth_arr) == 0:
            main_mem_bandwidth_arr = np.zeros((timestamp,), dtype=np.float32)
        
        logger.info(f"Main Memory Bandwidth: AVG {np.mean(main_mem_bandwidth_arr):.2f} / MAX {np.max(main_mem_bandwidth_arr):.2f} (B/cycle)")
        create_bandwidth_utilization_plot(axs[fig_cursor], main_mem_bandwidth_arr, "Main Memory Bandwidth Utilization", "Time (cycle)", "Bandwidth (B/cycle)")
        fig_cursor += 1
    
    if icnt_analysis_available:
        hw_config = TenstorrentConfig.BLACKHOLE()
        icnt_config: IcntConfig = hw_config["icnt_config"]
        flit_size = icnt_config.flit_size
        
        icnt_core_analyzer = IcntCoreAnalyzer()
        icnt_core_analyzer.load_traces(ICNT_CORE_TRACE_FNAME)
        icnt_bandwidth_arr = np.array(icnt_core_analyzer.dump_bandwidth_analysis(bin_size=1), dtype=np.float32) * flit_size
        
        if len(icnt_bandwidth_arr) == 0:
            icnt_bandwidth_arr = np.zeros((timestamp,), dtype=np.float32)
        
        logger.info(f"ICNT Bandwidth: AVG {np.mean(icnt_bandwidth_arr):.2f} / MAX {np.max(icnt_bandwidth_arr):.2f} (B/cycle)")

        create_bandwidth_utilization_plot(axs[fig_cursor], icnt_bandwidth_arr, "ICNT Bandwidth Utilization", "Time (cycle)", "Bandwidth (B/cycle)")
        fig_cursor += 1

    plt.tight_layout()
    plt.savefig(IMG_SAVE_FNAME, dpi=1000)
    
    logger.info(f"Bandwidth utilization plot saved to \"{IMG_SAVE_FNAME}\".")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize test results from CSV files.")
    parser.add_argument("-t", "--test", type=str, help="Directory containing traces and analysis result", required=True)
    args = parser.parse_args()

    test_name = args.test

    TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", test_name))
    PROFILE_DIR = os.path.join(TEST_ROOT, ".profiles")
    ANALYSIS_DIR = os.path.join(TEST_ROOT, ".analysis")
    ICNT_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv")
    MAIN_MEM_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "main_mem_core_trace.csv")
    IMG_SAVE_FNAME = os.path.join(ANALYSIS_DIR, "bandwidth_utilization.png")

    visualize_bandwidth_utilization_graph(
        PROFILE_DIR,
        ICNT_CORE_TRACE_FNAME,
        MAIN_MEM_CORE_TRACE_FNAME,
        IMG_SAVE_FNAME
    )