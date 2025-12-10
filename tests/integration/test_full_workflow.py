import pytest
import yaml
from pathlib import Path
import json
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path for script imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all main functions from the scripts that will be tested
from scripts.data.generate_partitions import generate_partitions
from scripts.data.build_vector_db import main as build_vector_db_main
from scripts.training.run_ner_training import run_batch_training
from scripts.evaluation.generate_rag_predictions import main as generate_rag_predictions_main
from scripts.evaluation.generate_finetuned_predictions import run_prediction_and_save
from scripts.evaluation.calculate_final_metrics import main as calculate_metrics

# --- Fixtures for the Full Workflow Test ---

@pytest.fixture(scope="module")
def workflow_configs(tmp_path_factory):
    """
    Provides a centralized dictionary of configuration objects for the entire workflow.
    This fixture runs only once per module, creating all necessary configs in a shared temporary directory.
    """
    tmp_path = tmp_path_factory.mktemp("workflow_data")
    
    # --- Data Preparation Config ---
    data_prep_config = {
        'data': {
            'base_seed': 42,
            'n_samples': 1,
            'test_split_ratio': 0.5, # 50% for test to ensure we have data in small sample
            'raw_input_file': str(tmp_path / 'raw' / 'all.jsonl'),
            'partitions_dir': str(tmp_path / 'processed'),
            'holdout_test_set_path': str(tmp_path / 'processed' / 'test.jsonl'),
            'partition_sizes': [2] # Create one small partition of size 2
        }
    }
    
    # --- Training Config ---
    training_config = {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'entity_labels': ["FIND", "REG"]
        },
        'paths': {
            'output_dir': str(tmp_path / "models")
        }
    }

    # --- RAG Config ---
    rag_config = {
        'task': 'ner',
        'llm': {
            'provider': 'openai',
            'openai': {'model': 'mock-gpt-model', 'temperature': 0.0}
        },
        'vector_db': {
            'index_path': str(tmp_path / "vector_db/index.bin"),
            'source_data_path': str(tmp_path / "processed/train-2/sample-1/train.jsonl"),
            'embedding_model': 'prajjwal1/bert-tiny'
        },
        'rag_prompt': {
            'prompt_template_path': str(tmp_path / "prompts/ner_rag_prompt.txt"),
            'n_examples': 1,
            'entity_labels': [{'name': 'FIND', 'description': 'A finding.'}]
        },
        'test_file': str(tmp_path / 'processed' / 'test.jsonl'),
        'output_dir': str(tmp_path / "output/rag_results")
    }

    return {
        "tmp_path": tmp_path,
        "data_prep": data_prep_config,
        "training": training_config,
        "rag": rag_config
    }
    

@pytest.fixture(scope="module")
def setup_workflow_environment(workflow_configs):
    """
    Sets up the file system with raw data and prompt templates needed for the test.
    This fixture depends on `workflow_configs` to know where to create the files.
    """
    tmp_path = workflow_configs["tmp_path"]
    
    # Create raw data
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_data = [
        {"text": "Report one has a finding.", "entities": [{"label": "FIND", "start_offset": 19, "end_offset": 26}]},
        {"text": "Report two has a region.", "entities": [{"label": "REG", "start_offset": 17, "end_offset": 23}]},
        {"text": "Report three has another finding.", "entities": [{"label": "FIND", "start_offset": 26, "end_offset": 33}]},
        {"text": "Report four has another region.", "entities": [{"label": "REG", "start_offset": 25, "end_offset": 31}]}
    ]
    with open(workflow_configs["data_prep"]["data"]["raw_input_file"], 'w') as f:
        for record in raw_data:
            f.write(json.dumps(record) + '\n')

    # Create prompt template
    prompt_template_path = Path(workflow_configs["rag"]["rag_prompt"]["prompt_template_path"])
    prompt_template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_template_path, 'w') as f:
        f.write("Entities: {entity_definitions}\nExamples: {examples}\nText: {new_report_text}")
        
    return workflow_configs


# --- Main Workflow Test Function ---

@patch('src.utils.cost_tracker.CostTracker.save_log')
@patch('src.llm_services.OpenAIClient')
def test_full_experiment_workflow(mock_openai_client, mock_save_log, setup_workflow_environment):
    """
    Tests the full, end-to-end experimental workflow, from data preparation
    to the final metric calculation for both RAG and fine-tuned pipelines.
    """
    # --- 0. Setup Mocks and Get Configs ---
    # Mock the LLM API call to return a predictable, simple entity list
    mock_api_response = [{"text": "a finding", "label": "FIND"}]
    mock_client_instance = MagicMock()
    mock_client_instance.get_ner_prediction.return_value = mock_api_response
    mock_openai_client.return_value = mock_client_instance

    # Retrieve the configuration dictionaries from the setup fixture
    configs = setup_workflow_environment
    tmp_path = configs["tmp_path"]
    
    # --- 1. Data Partitioning ---
    print("\n--- (1/5) Running Data Partitioning ---")
    generate_partitions(configs["data_prep"])
    
    # Assert that the data was partitioned correctly
    assert Path(configs["data_prep"]["data"]["holdout_test_set_path"]).exists()
    assert Path(configs["rag"]["vector_db"]["source_data_path"]).exists()
    print("--- Data Partitioning Successful ---")

    # --- 2. Build Vector DB for RAG ---
    print("\n--- (2/5) Building Vector Database ---")
    # We need to pass the config as a path to a temporary file
    rag_config_path = tmp_path / "rag_config.yaml"
    with open(rag_config_path, 'w') as f:
        yaml.dump(configs["rag"], f)
    build_vector_db_main(config_path=str(rag_config_path), force_rebuild=True)
    
    assert Path(configs["rag"]["vector_db"]["index_path"]).exists()
    print("--- Vector Database Build Successful ---")

        # --- 3. Train Fine-Tuned Model ---
    print("\n--- (3/5) Running Fine-Tuned NER Training ---")
    training_config_path = tmp_path / "training_config.yaml"
    with open(training_config_path, 'w') as f:
        yaml.dump(configs["training"], f)
    
    partition_dir = str(Path(configs["data_prep"]["data"]["partitions_dir"]) / "train-2")
    run_batch_training(config_path=str(training_config_path), partition_dir=partition_dir)

    # Dynamically find the timestamped output directory for the trained model
    base_model_output_dir = Path(configs["training"]["paths"]["output_dir"]) / "ner"
    
    # Find the single timestamped directory created by the training run which starts with 'train-2'
    timestamp_dirs = [d for d in base_model_output_dir.iterdir() if d.is_dir() and d.name.startswith("train-2")]

    assert len(timestamp_dirs) == 1, "Expected a single timestamped training output directory."
    trained_model_dir = timestamp_dirs[0] / "sample-1"
    assert trained_model_dir.exists(), "Trained model directory was not created."
    print("--- Fine-Tuned Training Successful ---")
    
    # --- 4. Generate Predictions for Both Pipelines ---
    print("\n--- (4/5) Generating Predictions ---")
    
    # RAG predictions
    generate_rag_predictions_main(config_path=str(rag_config_path))
    rag_output_dir = Path(configs["rag"]["output_dir"])
    
    # Correctly navigate into the 'ner' and 'n-shot' subdirectories
    n_examples = configs["rag"]["rag_prompt"]["n_examples"]
    shot_dir_name = f"{n_examples}-shot"
    rag_base_output_dir = rag_output_dir / "ner" / shot_dir_name

    # Find the single timestamped run directory created by the script
    rag_run_dirs = [d for d in rag_base_output_dir.iterdir() if d.is_dir()]
    assert len(rag_run_dirs) == 1, "Expected a single RAG prediction output directory."
    
    # The new filename is predictions.jsonl
    rag_predictions_path = rag_run_dirs[0] / "predictions.jsonl"
    assert rag_predictions_path.exists(), "RAG predictions file was not created."
    print("RAG predictions generated.")

    # Fine-tuned predictions (logic remains the same, but uses the corrected `trained_model_dir` path)
    finetuned_eval_output_dir = tmp_path / "output" / "finetuned_eval_results"
    evaluation_config = {
        'task': 'ner',
        'model_path': str(trained_model_dir),
        'test_file': configs["data_prep"]["data"]["holdout_test_set_path"],
        'model': {'entity_labels': configs["training"]["model"]["entity_labels"]},
        'output_dir': str(finetuned_eval_output_dir),
        'batch_size': 1
    }
    run_prediction_and_save(evaluation_config)
    finetuned_predictions_path = finetuned_eval_output_dir / f"predictions_{trained_model_dir.name}.jsonl"
    assert finetuned_predictions_path.exists(), "Fine-tuned predictions file was not created."
    print("Fine-tuned predictions generated.")

    # --- 5. Calculate Final Metrics for Both ---
    print("\n--- (5/5) Calculating Final Metrics ---")
    
    # RAG metrics
    rag_metrics_path = rag_run_dirs[0] / "final_metrics.json"
    calculate_metrics(
        prediction_path=str(rag_predictions_path),
        prediction_dir=None,
        eval_type='ner', # Changed from 'rag' to 'ner' as per calculate_final_metrics.py
        output_path=str(rag_metrics_path)
    )
    assert rag_metrics_path.exists(), "RAG metrics file was not created."
    print("RAG metrics calculated.")

    # Fine-tuned metrics
    finetuned_metrics_path = finetuned_eval_output_dir / "final_metrics.json"
    calculate_metrics(
        prediction_path=str(finetuned_predictions_path),
        prediction_dir=None,
        eval_type='ner',
        output_path=str(finetuned_metrics_path)
    )
    assert finetuned_metrics_path.exists(), "Fine-tuned metrics file was not created."
    print("Fine-tuned metrics calculated.")
    
    print("\n--- Full Experiment Workflow Test Successful ---")