# DSG LLM Evaluation

This project evaluates different serialization methods for scene graphs to determine their effectiveness in enabling LLMs to reason about and answer questions about 3D environments.

## Features

- Multiple serialization formats for scene graphs:
  - Indented text
  - JSON
  - Triplets
  - Natural language
- Evaluation of LLM performance across different question types:
  - Object counting
  - Room attributes
  - Spatial reasoning
- Configurable experiment setup
- Detailed result analysis and visualization

## Prerequisites

- Python 3.8+
- pip
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dsg_llm_eval
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Running Evaluation

1. Configure your experiment in `configs/experiment_config.yaml`
2. Run the evaluation:
   ```bash
   python pipeline/run_eval.py \
     --base_config configs/base_eval_config.yaml \
     --experiment_config configs/experiment_config.yaml
   ```

### Configuration

- `base_eval_config.yaml`: Contains base configuration including model parameters, dataset paths, and output settings
- `experiment_config.yaml`: Defines experiment variations (serialization methods, attributes, etc.)

## Project Structure

```
dsg_llm_eval/
├── configs/                 # Configuration files
├── data/                    # Dataset and prompt templates
├── llm/                     # LLM interface code
├── pipeline/                # Main evaluation pipeline
│   ├── models/             # Serialization implementations
│   ├── __init__.py
│   ├── evaluator.py        # Evaluation logic
│   ├── prompt_builder.py   # Prompt construction
│   ├── run_eval.py         # Main script
│   └── visualize.py        # Result visualization
└── results/                # Output directory for experiment results
```

## Results

Experiment results are saved in the `results/` directory with timestamps. Each run includes:
- Raw and processed results in CSV format
- Configuration files used
- Visualization plots

## License

[Your License Here]

## Acknowledgements

[Add any dependencies or references here]

## Contributing

[Your contribution guidelines here]
