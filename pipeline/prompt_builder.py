import os
import spark_dsg as dsg
from typing import List, Dict
from models.serialization import serialization_functions
from pathlib import Path


def load_dataset(dataset_cfg: dict) -> Dict[str, dsg.DynamicSceneGraph]:
    directory = Path(dataset_cfg['scene_dir'])
    return {
        file.name: dsg.DynamicSceneGraph.load(file)
        for file in directory.glob("*.json")
    }   
    
    
def serialize_dataset(
    scene_graphs: Dict[str, dsg.DynamicSceneGraph],
    serialization_cfg: dict
) -> Dict[str, str]:
    
    serialization = serialization_cfg['type']
    detail_keys = serialization_cfg['detail_keys']

    if serialization not in serialization_functions:
        raise ValueError(f"Unknown serialization type: {serialization}")
    
    serialize_fn = serialization_functions[serialization]
    
    dsg_serialized = {
        name: serialize_fn(scene_graph, detail_keys)
        for name, scene_graph in scene_graphs.items()
    }
    
    if serialization_cfg['verbose']:
        for name in dsg_serialized:
            print(dsg_serialized[name])
    return dsg_serialized

def build_prompt(scene_repr: str, query: str, prompt_cfg: dict) -> str:
    with open(prompt_cfg['template_path'], "r", encoding="utf-8") as f:
        prompt_template = f.read()    

    prompt = prompt_template.replace("{{scene_repr}}", scene_repr)
    prompt = prompt.replace("{{query}}", query)
    return prompt


if __name__ == '__main__':
    dataset = load_dataset('data/scenes/scene_graphs')
    serialized_dataset = serialize_dataset(dataset, 'indented-summary', [])
    
    print(list(serialized_dataset.values())[0])