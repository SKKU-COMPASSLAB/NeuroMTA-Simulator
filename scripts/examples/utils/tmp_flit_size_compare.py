import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
TARGET_TESTS = ["test_rt_op_tt_sharded_main_linear", "test_rt_op_tt_sharded_main_conv2d"]
FLIT_SIZES  = [16, 32, 64, 128, 256, 512]

class ColorMap:
    light_blue = "#297994"
    blue       = "#1D1DB0"
    light_red  = "#FF7F7F"
    red        = "#BF2E2E"
    light_green= "#7FCB7F"
    green      = "#187818"
    
COLORS = [ColorMap.light_blue, ColorMap.light_red, ColorMap.light_green, ColorMap.blue, ColorMap.red, ColorMap.green]
    
def create_execution_time_graph(ax: plt.Axes, target_tests: list[str], flit_sizes: list[int]):
    x = np.arange(len(flit_sizes))
    
    ax.set_title("Speedup over Different NoC Flit Sizes")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    
    for i, target_test in enumerate(target_tests):
        summary_files = [os.path.join(EXAMPLES_DIR, target_test, f"flit{flit_size}B", ".profiles", "summary.json") for flit_size in flit_sizes]
        exec_times = []
        
        for summary_file in summary_files:
            if not os.path.exists(summary_file):
                print(f"Warning: Summary file not found: {summary_file}")
                exec_times.append(np.nan)
                continue
            
            content = json.load(open(summary_file, "r"))
            exec_time = content["simulation_summary"]["total_cycles"]
            
            exec_times.append(exec_time)
            
        label = target_test.split("_")[-1]
        
        exec_times = np.array(exec_times)
        exec_times = exec_times[0] / exec_times  # Normalize to the first flit size
        
        _width = 0.7 / len(target_tests)
        _x = x + (i - len(target_tests) / 2) * _width + _width / 2

        ax.bar(_x, exec_times, width=_width, label=label, color=COLORS[i], edgecolor="black", linewidth=1)

    ax.set_xticks(x, [str(size) + "B" for size in flit_sizes])
    ax.set_xlabel("NoC Flit Size (Bandwidth = Flit Size x 2 x Frequency)")
    ax.set_ylabel("Speedup (Normalized to 32B)")
    ax.legend()
    
if __name__ == "__main__":
    image_path = os.path.join(EXAMPLES_DIR, "utils", "tmp_flit_size_compare_execution_time.png")
    
    fig, ax = plt.subplots(figsize=(8, 3.4))
    
    create_execution_time_graph(ax, TARGET_TESTS, FLIT_SIZES)
    
    plt.tight_layout()
    plt.savefig(image_path, dpi=500)
    plt.show()

    print(f"figure saved to {image_path}")