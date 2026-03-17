import os
import json

from neuromta.framework import logger


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.dirname(__file__))
    test_prefixes = ["run_with_pp", "run_wo_pp"]
    test_configs: dict[str, list[str]] = {t_prefix: [] for t_prefix in test_prefixes}
    
    for t_prefix in test_prefixes:
        test_dirname = os.path.join(ROOT, ".logs", t_prefix)
        for t_config in os.listdir(test_dirname):
            test_configs[t_prefix].append(t_config)
            
    results = ["prefix,config,l1_buf_size,use_l1_data_space,timestamp"]
            
    for t_prefix, t_configs in test_configs.items():
        for t_config in t_configs:
            summary_file_path = os.path.join(ROOT, ".logs", t_prefix, t_config, "simulation_summary.json")
            with open(summary_file_path, 'rt') as file:
                data = json.load(file)
                results.append(f"{t_prefix},{t_config},{data['l1_buf_size']},{data['use_l1_data_space']},{data['timestamp']}")
              
    result_file_path = os.path.join(ROOT, ".logs", "summary.csv")  
    with open(result_file_path, "wt") as file:
        file.write("\n".join(results))
    logger.info(f"Summary saved to '{result_file_path}'")