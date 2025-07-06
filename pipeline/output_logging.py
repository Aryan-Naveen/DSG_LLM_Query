import os
from datetime import datetime
import shutil

def save_experiment_results(output_cfg, experiment_dataframe, condensed_results_keys, experiment_name=None):
    # Create unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"experiment_{timestamp}"
    log_path = os.path.join(output_cfg["log_dir"], exp_name)
    os.makedirs(log_path, exist_ok=True)

    # Save results dataframe
    raw_results_path = os.path.join(log_path, "raw_experiment_results.csv")
    experiment_dataframe.to_csv(raw_results_path, index=False)
    
    condensed_results_path = os.path.join(log_path, "condensed_experiment_results.csv")
    experiment_dataframe[condensed_results_keys].to_csv(condensed_results_path, index=False)

    print(f"[INFO] Logs saved to {log_path}")
    return log_path


def save_config_to_results(config_paths, results_dir):
    result_config_path = f"{results_dir}/configs"
    os.makedirs(result_config_path, exist_ok=True)
    
    for config_path in config_paths:
        filename = os.path.basename(config_path)
        dest_file = "base_config.yaml" if filename == "base_eval_config.yaml" else "experiment_config.yaml"

        destination = os.path.join(result_config_path, dest_file)
        shutil.copy(config_path, destination)