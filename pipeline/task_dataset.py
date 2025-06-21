import pandas as pd
import os

FILES = {
    'count': ['object-count.csv'],
    'room': ['room-attributes.csv'],
    'spatial': ['spatial-reasoning.csv'],
    'all': ['object-count.csv', 'spatial-reasoning.csv', 'room-attributes.csv']
}

def load_task_dataset(cfg):
    files = FILES[cfg['task']]
    directory = cfg['task_path']

    all_dfs = []
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        task_category = file_name[:-4]  # Strip .csv
        df['task_category'] = task_category

        # Create new unique ID: e.g., 'count_0'
        df['unique_id'] = df['task_category'] + '_' + df['id'].astype(str)
        all_dfs.append(df)

    task_df = pd.concat(all_dfs, ignore_index=True)

    # Convert to dictionary indexed by 'unique_id'
    data_dict = task_df.set_index('unique_id').to_dict(orient='index')

    return data_dict


if __name__ == '__main__':
    from run_eval import load_config
    
    cfg = load_config('/home/anaveen/Documents/mit_research_ws/01_dsg_prompting/dsg_llm_eval/configs/eval_config.yaml')
    
    print(load_task_dataset(cfg['prompt']))