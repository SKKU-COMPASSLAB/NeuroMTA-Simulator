import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.mta.tenstorrent import TenstorrentConfig


class ColorMap:
    light_blue = "#297994"
    blue       = "#1D1DB0"
    light_red  = "#FF7F7F"
    red        = "#BF2E2E"
    light_green= "#7FCB7F"
    green      = "#187818"


def create_bandwidth_graph(ax: plt.Axes, trace_path: str, label: str, color: str, rolling_avg_window: int=10):
    df = pd.read_csv(trace_path)
    
    if 'timestamp' not in df.columns or 'bandwidth[bytes/cycle]' not in df.columns:
        print(f"Trace file {trace_path} does not contain required columns.")
        return

    df['bandwidth[bytes/cycle]'] = df['bandwidth[bytes/cycle]'].rolling(window=rolling_avg_window).mean()

    ax.plot(df['timestamp'], df['bandwidth[bytes/cycle]'], label=label, color=color, linewidth=0.5)
    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Bandwidth (bytes/cycle)')
    ax.legend()
    ax.grid(True)


def visualize_booksim2_trace(trace_dir: str, output_dir: str, rolling_avg_window: int=10):
    os.makedirs(output_dir, exist_ok=True)
    
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    
    router_ids = set()
    for f in trace_files:
        router_ids.add(int(f.split('_')[0]))

    for i in router_ids:
        rx_trace_file = f"{i}_rx.csv"
        tx_trace_file = f"{i}_tx.csv"
        output_path = os.path.join(output_dir, f"router_{i}_bandwidth.png")
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
        create_bandwidth_graph(axs, os.path.join(trace_dir, rx_trace_file), label=f"RX Bandwidth", color=ColorMap.blue, rolling_avg_window=rolling_avg_window)
        create_bandwidth_graph(axs, os.path.join(trace_dir, tx_trace_file), label=f"TX Bandwidth", color=ColorMap.red,  rolling_avg_window=rolling_avg_window)

        plt.title(f"Router {i} Bandwidth")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"BookSim2) Saved bandwidth graph for router {i} to \"{output_path}\"")
        
        
def visualize_dramsim3_trace(trace_dir: str, output_dir: str, rolling_avg_window: int=10):
    os.makedirs(output_dir, exist_ok=True)
    
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]

    channels = set()
    for f in trace_files:
        channels.add(int(f.split('_')[0]))

    for i in channels:
        read_trace_file = f"{i}_read.csv"
        write_trace_file = f"{i}_write.csv"
        output_path = os.path.join(output_dir, f"channel_{i}_bandwidth.png")
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
        create_bandwidth_graph(axs, os.path.join(trace_dir, read_trace_file),  label=f"Read Bandwidth",  color=ColorMap.blue, rolling_avg_window=rolling_avg_window)
        create_bandwidth_graph(axs, os.path.join(trace_dir, write_trace_file), label=f"Write Bandwidth", color=ColorMap.red,  rolling_avg_window=rolling_avg_window)

        plt.title(f"DRAM Channel {i} Bandwidth")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"DRAMSim3) Saved bandwidth graph for channel {i} to \"{output_path}\"")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str, required=True, dest="test_dir", help='Directory containing BookSim2 trace CSV files')
    parser.add_argument('--output-dir', type=str, default=None, dest="output_dir", help='Directory to save the visualizations')
    parser.add_argument('--booksim2-rolling', type=int, default=10, dest="booksim2_rolling_avg_window", help='Rolling average window size for BookSim2 bandwidth graph')
    parser.add_argument('--dramsim3-rolling', type=int, default=10, dest="dramsim3_rolling_avg_window", help='Rolling average window size for DRAMSim3 bandwidth graph')
    args = parser.parse_args()

    test_dir = args.test_dir
    output_dir = args.output_dir if args.output_dir is not None else test_dir
    booksim2_rolling_avg_window = args.booksim2_rolling_avg_window
    dramsim3_rolling_avg_window = args.dramsim3_rolling_avg_window
    
    booksim2_trace_dir = os.path.join(test_dir, ".analysis", "booksim2")
    dramsim3_trace_dir = os.path.join(test_dir, ".analysis", "dramsim3")
    booksim2_output_dir = os.path.join(output_dir, ".visualize", "booksim2")
    dramsim3_output_dir = os.path.join(output_dir, ".visualize", "dramsim3")
    
    os.makedirs(booksim2_output_dir, exist_ok=True)
    os.makedirs(dramsim3_output_dir, exist_ok=True)

    visualize_booksim2_trace(booksim2_trace_dir, booksim2_output_dir, rolling_avg_window=booksim2_rolling_avg_window)
    visualize_dramsim3_trace(dramsim3_trace_dir, dramsim3_output_dir, rolling_avg_window=dramsim3_rolling_avg_window)