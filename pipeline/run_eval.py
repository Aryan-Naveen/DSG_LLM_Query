import argparse
import yaml
import copy
from tqdm import tqdm
import pandas as pd
import time


from prompt_builder import (
    load_dataset,
    serialize_dataset,
    build_prompt,
)

from task_dataset import load_task_dataset
from llm.interface import LLMClient
from evaluator import evaluate_summary 
from output_logging import (
    save_experiment_results, 
    save_config_to_results
)

from visualize import (
    vizualize_fns
)





def recursive_merge(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_merge(base[k], v)
        else:
            base[k] = v
    return base


def parse_args():
    parser = argparse.ArgumentParser(description="Load config file")
    parser.add_argument('--base_config', type=str, default='configs/base_eval_config.yaml', help="Path to the config file")
    parser.add_argument('--experiment_config', type=str, default='configs/experiment_multi.yaml', help="Path to the experiment config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config):
    dataset_cfg = config['dataset']
    prompt_cfg = config['prompt']
    serialization_cfg = prompt_cfg['serialization']
    llm_cfg = config['llm']
    eval_cfg = config['evaluation']


    dsg_dataset = load_dataset(dataset_cfg)
    scene_reprs = serialize_dataset(dsg_dataset, serialization_cfg)
    task_dataset = load_task_dataset(prompt_cfg)
    
    llmclient = LLMClient(llm_cfg)
    
    predicted_answers = {}
    for task in tqdm(task_dataset):
        prompt = build_prompt(scene_reprs[task_dataset[task]['scene_id']], task_dataset[task]['query'], prompt_cfg)
        start = time.perf_counter()
        pred_answer = llmclient.query(prompt)
        duration = time.perf_counter() - start - llm_cfg['delay']
        
        predicted_answers[task] = {
            'answer': pred_answer,
            'elapsed_time': duration
        }
    
    return evaluate_summary(predicted_answers, task_dataset, eval_cfg, serialization_cfg)
    



if __name__ == "__main__":
    args = parse_args()
    base_config = load_config(args.base_config)
    
    with open(args.experiment_config, "r") as f:
        experiments_config = yaml.safe_load(f)
    
    viz_config = experiments_config['visualization']
    condensed_results_keys = experiments_config['condensed_results_keys']
    
    experiments_df = pd.DataFrame()
    for experiment in experiments_config["experiments"]:
        print(f"EXPERIMENT: {experiment['name']}")
        config_copy = copy.deepcopy(base_config)
        config = recursive_merge(config_copy, experiment["overrides"])        

        experiments_df = pd.concat([experiments_df, run_experiment(config)], ignore_index=True)

    results_path = save_experiment_results(base_config['output'], experiments_df, condensed_results_keys)

    save_config_to_results([args.experiment_config, args.base_config], results_path)
    
    vizualize_fns[viz_config['type']](experiments_df, results_path, viz_config['args'])
    