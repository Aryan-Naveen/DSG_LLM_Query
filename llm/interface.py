import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import json
from typing import Optional, Union, Dict
import yaml
import time

class LLMClient:
    def __init__(self, config):
        """
        Initialize the LLM client.
        - model_name: OpenAI model like "gpt-4-0613"
        - api_key: your OpenAI API key
        - function_defs: list of function definitions for function-calling
        """
        self.model_name = config['model_name']
        self.cfg = config        


    def query(self, prompt: str) -> Union[str, Dict]:
        """
        Query the model with the prompt.
        mode:
          - "text": returns plain text reply
          - "json": tries to parse and return JSON from the reply
          - "function_call": expects the model to respond with a function call structure and returns the function call info as dict
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Prepare kwargs for function calls if needed
        response = client.chat.completions.create(model=self.model_name,
                messages=messages,
                temperature=self.cfg['temperature'],
                max_tokens=self.cfg['max_tokens']
            )

        message = response.choices[0].message
        time.sleep(self.cfg['delay'])
        

        if self.cfg['mode'] == "text":
            # Return plain text
            return message.content.strip()

        elif self.cfg['mode'] == "json":
            # Try to parse JSON out of the message content
            text = message.get("content", "")
            try:
                parsed = json.loads(text)
                return parsed
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text with an error key
                return {"error": "Failed to parse JSON", "raw": text}

        else:
            raise ValueError(f"Unsupported mode: {self.cfg['mode']}")

if __name__ == '__main__':
    with open('/home/anaveen/Documents/mit_research_ws/01_dsg_prompting/dsg_llm_eval/configs/eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    llm_config = config['llm']
    llmclient = LLMClient(llm_config)

    print(llmclient.query("how are you?"))