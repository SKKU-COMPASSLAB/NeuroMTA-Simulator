import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.hardware.analyzer.icnt_core_analyzer import IcntCoreAnalyzer
from neuromta.hardware.analyzer.main_mem_core_analyzer import MainMemCoreAnalyzer
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig


PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), ".analysis")
ICNT_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "icnt_core_trace.csv")
MAIN_MEM_CORE_TRACE_FNAME = os.path.join(ANALYSIS_DIR, "main_mem_core_trace.csv")
IMG_SAVE_FNAME = os.path.join(ANALYSIS_DIR, "bandwidth_utilization.png")


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

if __name__ == "__main__":
    results = []
    
    for filename in os.listdir(PROFILE_DIR):
        filepath = os.path.join(PROFILE_DIR, filename)
        
        df = pd.read_csv(filepath)
        
        df['command_id'] = df['command_id'].str.strip()
        mxu_row = df[df['command_id'] == 'mxu_tiled_gemm']
        
        result = mxu_row['duration'].iloc[0] / mxu_row['last_commit_time'].iloc[0]
        results.append(result)

    print(f"Average MXU Utilization: {sum(results)/len(results)*100:.2f}%")
    
    hw_config = TenstorrentConfig.BLACKHOLE()
    icnt_config: IcntConfig = hw_config["icnt_config"]
    flit_size = icnt_config.flit_size
    
    icnt_core_analyzer = IcntCoreAnalyzer()
    icnt_core_analyzer.load_traces(ICNT_CORE_TRACE_FNAME)
    
    icnt_bandwidth_arr = np.array(icnt_core_analyzer.dump_bandwidth_analysis(bin_size=1), dtype=np.float32) * flit_size
    
    print(f"ICNT Bandwidth: mean={np.mean(icnt_bandwidth_arr):.2f} B/cycle, max={np.max(icnt_bandwidth_arr):.2f} B/cycle")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    create_bandwidth_utilization_plot(ax1, icnt_bandwidth_arr, "ICNT Bandwidth Utilization", "Time (cycle)", "Bandwidth (B/cycle)")

    plt.tight_layout()
    plt.savefig(IMG_SAVE_FNAME, dpi=1000)
    
    logger.info(f"Bandwidth utilization plot saved to \"{IMG_SAVE_FNAME}\".")