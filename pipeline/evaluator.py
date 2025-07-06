from llm.interface import LLMClient
import yaml
from tqdm import tqdm
import pandas as pd

def evaluate_summary(predicted_answers: dict, ground_truth_answers: dict, cfg: dict, serialization_cfg: dict, debug: bool = False) -> dict:
    evaluator = LLMClient(cfg['llm'])

    with open(cfg['expected_template'], "r", encoding="utf-8") as f:
        template = f.read()    

    rows = []
    
    for qid in tqdm(ground_truth_answers):        
        eval_prompt = template.replace("{{question}}", ground_truth_answers[qid]["query"])
        eval_prompt = eval_prompt.replace("{{ground_truth}}", ground_truth_answers[qid]["answer"])
        eval_prompt = eval_prompt.replace("{{predicted}}", predicted_answers[qid]['answer'])

        if debug: 
            # Find the index of the last period
            last_sentence = eval_prompt.rfind('.')

            # Find the index of the second-to-last period (searching up to just before the last one)
            second_last_sentence = eval_prompt.rfind('.', 0, last_sentence)

            # Get the substring up to (but not including) everything after the second-to-last dot
            result = eval_prompt[:second_last_sentence]
            eval_prompt += " Explain why.\n"

        result = evaluator.query(eval_prompt).lower()
        
        if debug:
            print(eval_prompt)
            print(result)
            return
        
        try:
            score = float(result)
        except ValueError:
            score = -1  # or handle differently if your LLM might output text instead

        if "NA" in serialization_cfg['detail_keys']: serialization_cfg['detail_keys'] = []
        
        row = {
            "question_id": qid.split('_')[1],
            "question_type": qid.split('_')[0],
            "serialization": '-'.join(serialization_cfg['type']),
            "num_attributes": len(serialization_cfg['detail_keys']),
            "question": ground_truth_answers[qid]["query"],
            "ground_truth_answer": ground_truth_answers[qid]["answer"],
            "predicted_answer": predicted_answers[qid]['answer'],
            'llm_elapsed_time': predicted_answers[qid]['elapsed_time'],
            "score": score
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
    

if __name__ == '__main__':
    predicted_answers = {
        0: "To determine which object most frequently appears directly next to swivel chairs or chairs in Room 3, we first need to identify the positions of the swivel chairs and chairs in that room.\n\nIn Room 3, the objects are as follows:\n- **Chairs**:\n  - Chair (id = 273) at position [-12.93421085, 16.49990415, 1.82354166]\n  - Chair (id = 266) at position [-8.59625987, 16.88161511, 1.78747596]\n  \n- **Swivel**:\n  - Swivel (id = 276) at position [-12.91257052, 16.49470501, 1.45444641]\n  - Swivel (id = 276) at position [-12.91257052, 16.49470501, 1.45444641]\n\nNext, we need to check the objects that are adjacent to these chairs and swivel chairs. \n\nHowever, the scene information does not provide explicit adjacency data, such as which objects are next to each other. Therefore, we can only infer adjacency based on their positions.\n\nGiven the positions of the chairs and swivel chairs, we can check for other objects that are close to these positions. \n\nThe objects in Room 3 are:\n- 2 cabinets\n- 3 chairs\n- 10 desks\n- 5 lights\n- 2 sofas\n- 1 swivel\n- 1 wardrobe\n\nSince we do not have the exact positions of the other objects in Room 3, we cannot definitively determine which object appears most frequently next to the chairs or swivel chairs based on the provided data.\n\nIf we had the positions of all objects in Room 3, we could calculate distances to find which object is most frequently adjacent to the chairs and swivel chairs. \n\nIn conclusion, without specific positional data for all objects in Room 3, we cannot determine which object most frequently appears next to the swivel chairs or chairs.",
    }
    
    groundtruth_answers = {
        0: {'query': 'What object most frequently appears directly next to swivel chairs or chairs in room 3?', 'answer': 'A desk.'},
    }
    
    with open('/home/anaveen/Documents/mit_research_ws/01_dsg_prompting/dsg_llm_eval/configs/eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    eval_cfg = config['evaluation']
    
    
    print(evaluate_summary(predicted_answers, groundtruth_answers, eval_cfg, debug=True))    