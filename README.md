# Comparison of RAG and Fine-Tuning for Mammogram Report Analysis

This repository contains the code for a project that compares two methodologies for Natural Language Processing (NLP) tasks on mammogram reports:

1.  **Fine-tuning** traditional transformer models (e.g., BERT) on a labeled dataset for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)**.
2.  **Few-shot prompting** with Large Language Models (LLMs) like GPT via a Retrieval-Augmented Generation (RAG) pipeline for NER.

The primary objective is to evaluate how the performance of these two approaches scales with the amount of available labeled data, from very few examples (e.g., 5-10 reports) to a larger corpus.

-----

## Key Features

  - **Dual-Task Support**: Implements distinct, fine-tunable models for both **NER** and **Relation Extraction (RE)**, allowing for comprehensive information extraction.
  - **Modular Model Framework**: Allows for easy substitution between locally fine-tuned models and API-based RAG models through a unified interface.
  - **Configuration-Driven Experiments**: Ensures reproducibility and simplifies experiment management for both training and evaluation using YAML configuration files.
  - **Reproducible Data Sampling**: Includes a script to automatically generate multiple, distinct, and stratified data samples for various training set sizes, ensuring balanced label distribution.
  - **Standardized Evaluation**: Calculates and reports key metrics for NER (entity-level $F\_1$-score, Precision, Recall).
  - **Automated Testing**: Integrated with GitHub Actions for continuous integration, running a full suite of unit tests with `pytest` on every push and pull request to the main branch.

-----

## Project Structure

The repository is organized to maintain a clear separation between configuration, source code, data, and results.

```
.
├── .github/workflows/        # CI workflows for automated testing
│   └── python-tests.yml
├── configs/                  # Experiment configuration files
│   ├── data_preparation_config.yaml
│   ├── rag_ner_config.yaml
│   ├── rag_re_config.yaml
│   ├── training_ner_config.yaml
│   ├── training_re_config.yaml
│   ├── inference_ner_config.yaml
│   └── inference_re_config.yaml
├── data/                     # (Git-ignored) Raw and processed data
│   ├── raw/                  # Place raw all.jsonl here
│   └── processed/
│       ├── test.jsonl
│       └── train-5/
│           ├── sample-1/
│           │   └── train.jsonl
│           └── ...
├── output/                   # (Git-ignored) Models, logs, evaluation results, etc.
├── prompts/                  # Prompt templates for the RAG pipeline (Spanish templates available)
│   ├── rag_ner_prompt_spanish.txt
│   ├── rag_ner_prompt_spanish_strict.txt
│   ├── rag_re_prompt_spanish.txt
│   └── rag_re_prompt_spanish_strict.txt
├── scripts/                  # High-level scripts to run experiments
│   ├── data/
│   │   ├── generate_partitions.py
│   │   └── build_vector_db.py
│   ├── training/
│   │   ├── run_ner_training.py
│   │   └── run_re_training.py
│   └── evaluation/
│       ├── generate_finetuned_predictions.py
│       ├── generate_rag_predictions.py
│       └── calculate_final_metrics.py
├── src/                      # Source code for the project
│   ├── data_loader/          # NER and RE dataloaders
│   ├── evaluation/           # Prediction and evaluation logic
│   ├── llm_services/         # Interface for LLM providers (e.g., OpenAI)
│   ├── models/               # NER and RE model definitions
│   ├── training/             # Reusable training loop
│   ├── utils/                # Utility classes (e.g., CostTracker)
│   └── vector_db/            # Vector database management for RAG
├── tests/                    # Unit and integration tests
│   ├── unit/
│   └── integration/
├── .gitignore
├── requirements.txt
└── README.md
```

-----

## Setup and Installation

Follow these steps to configure the project environment.

### Prerequisites

  - Python 3.11+
  - A virtual environment manager, such as `venv` or `conda`.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Joacodef/Mammo_RAG_and_Fine-tune.git
    cd Mammo_RAG_and_Fine-tune
    ```
2.  **Create and activate a virtual environment:**
    For example, using `conda`:
    ```bash
    conda create --name mammo-nlp python=3.11
    conda activate mammo-nlp
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    If you plan to use API-based models (e.g., OpenAI), copy the example environment file.
    ```bash
    cp .env_example .env
    ```
    Open the newly created `.env` file and add your secret keys (e.g., `OPENAI_API_KEY=...`).

-----

## Running the Experiments

The entire experimental workflow is executed using command-line scripts.

### 1\. Data Preparation

First, prepare the datasets for both fine-tuning and RAG.

  - **1.1. Generate Data Partitions**
    Generate the stratified training and test sets from your raw data file. Ensure your data is located at `data/raw/all.jsonl` and configure the partitions in `configs/data_preparation_config.yaml`.
    ```bash
    python scripts/data/generate_partitions.py --config-path configs/data_preparation_config.yaml
    ```
  - **1.2. Build Vector Database (for RAG)**
    This step creates the FAISS index from the full training set, which is required for the RAG pipeline.
    Use the task-specific RAG config (NER or RE). Example for NER:
    ```bash
    python scripts/data/build_vector_db.py --config-path configs/rag_ner_config.yaml
    ```
    For RE, use `configs/rag_re_config.yaml`.

### 2\. Model Training and Prediction

  - **2.1. Train Fine-Tuned Models**
    Train either NER or RE models across the generated data partitions.
      - **NER training:**
        ```bash
        python scripts/training/run_ner_training.py \
          --config-path configs/training_ner_config.yaml \
          --partition-dir data/processed/train-50
        ```
      - **RE training:**
        ```bash
        python scripts/training/run_re_training.py \
          --config-path configs/training_re_config.yaml \
          --partition-dir data/processed/train-50
        ```
  - **2.2. Generate Predictions with RAG**
  Run the RAG pipeline on the test set to generate predictions. Use the appropriate RAG config for your task.
  ```bash
  # NER example
  python scripts/evaluation/generate_rag_predictions.py --config-path configs/rag_ner_config.yaml

  # RE example
  python scripts/evaluation/generate_rag_predictions.py --config-path configs/rag_re_config.yaml
  ```

### 3\. Evaluation

The evaluation is a two-step process: generate raw prediction files, then calculate metrics.

  - **3.1. Generate Predictions for Fine-Tuned Models**
  Run inference on the test set for all trained model samples in a directory. Use the `inference_*.yaml` configs.
  ```bash
  python scripts/evaluation/generate_finetuned_predictions.py --config-path configs/inference_ner_config.yaml --model-dir output/models/ner/<run_folder>
  ```
  - **3.2. Calculate Final Metrics**
    Calculate metrics from the generated prediction files.
      - **For a fine-tuned model:**
        ```bash
        python scripts/evaluation/calculate_final_metrics.py \
          --prediction-path <path_to_finetuned_predictions.jsonl> \
          --type ner \
          --config-path configs/inference_ner_config.yaml \
          --output-path <path_to_save_final_metrics.json>
        ```
      - **For RAG predictions:**
        ```bash
        python scripts/evaluation/calculate_final_metrics.py \
          --prediction-path <path_to_rag_predictions.jsonl> \
          --type ner \
          --config-path configs/rag_ner_config.yaml \
          --output-path <path_to_save_final_metrics.json>
        ```
        (For RE replace `--type ner` and the config with the RE equivalents.)

-----

## Testing

The repository includes a suite of unit and integration tests to ensure code quality and correctness. To run the tests locally, execute the following command from the root directory:

```bash
python -m pytest
```

Tests are also run automatically via GitHub Actions on every push or pull request to the `main` branch.
