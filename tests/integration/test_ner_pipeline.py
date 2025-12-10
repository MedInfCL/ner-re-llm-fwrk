# tests/integration/test_ner_pipeline.py
import pytest
import yaml
from pathlib import Path
import json
import sys

# Add the project root to the Python path to allow for absolute imports
# This is crucial for the integration test to find the src and scripts modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.run_ner_training import run_batch_training
from scripts.evaluation.generate_finetuned_predictions import run_prediction_and_save
from scripts.evaluation.calculate_final_metrics import main as calculate_metrics

# --- Fixtures for Test Data and Configuration ---

@pytest.fixture
def ner_integration_config():
    """
    Provides a minimal, fast configuration for the NER integration test.
    This uses a very small model to speed up download and training time.
    """
    return {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu" # Force CPU to ensure test runs on any machine
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny', # A very small model for fast testing
            'entity_labels': ["FIND", "REG"]
        },
        'paths': {
            'output_dir': "output/models"
        }
    }

@pytest.fixture
def ner_training_data():
    """Provides a few records for the training set."""
    return [
        {"text": "Report one has a finding in the REG region.", "entities": [{"label": "FIND", "start_offset": 19, "end_offset": 26}, {"label": "REG", "start_offset": 34, "end_offset": 44}]},
        {"text": "Report two has a REG.", "entities": [{"label": "REG", "start_offset": 17, "end_offset": 20}]}
    ]

@pytest.fixture
def ner_test_data():
    """Provides a few records for the test/evaluation set."""
    return [
        {"text": "This test has a FIND.", "entities": [{"label": "FIND", "start_offset": 17, "end_offset": 21}]},
        {"text": "And this one has a REG.", "entities": [{"label": "REG", "start_offset": 20, "end_offset": 23}]}
    ]

# --- Integration Test ---

def test_ner_training_and_evaluation_pipeline(tmp_path, ner_integration_config, ner_training_data, ner_test_data):
    """
    Tests the complete NER pipeline from training to evaluation.
    This test performs the following steps:
    1. Sets up a temporary directory structure with test data and configs.
    2. Runs the main training script on a minimal dataset.
    3. Asserts that the expected model artifacts were created.
    4. Runs the main evaluation script using the newly trained model.
    5. Asserts that the evaluation metrics file was created.
    """
    # --- 1. Setup temporary directory and files ---
    
    # Create training data partition directory
    partition_dir = tmp_path / "processed" / "train-2" / "sample-1"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Write training data
    train_file = partition_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for record in ner_training_data:
            f.write(json.dumps(record) + '\n')
            
    # Write test data
    test_file_path = tmp_path / "processed" / "test.jsonl"
    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, 'w') as f:
        for record in ner_test_data:
            f.write(json.dumps(record) + '\n')

    # Write training config
    training_config_path = tmp_path / "training_ner_config.yaml"
    ner_integration_config['paths']['output_dir'] = str(tmp_path / "output" / "models")
    with open(training_config_path, 'w') as f:
        yaml.dump(ner_integration_config, f)

    # --- 2. Run Training ---
    print("\n--- Running NER Training ---")
    run_batch_training(
        config_path=str(training_config_path),
        partition_dir=str(tmp_path / "processed" / "train-2")
    )

    # --- 3. Assert Training Outputs ---
    # The output path now includes a timestamp, so we must find it dynamically.
    base_output_dir = Path(ner_integration_config['paths']['output_dir']) / "ner"

    # Find the single timestamped directory created by the training run
    # It will start with the partition name, e.g., "train-2"
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("train-2")]
    assert len(run_dirs) == 1, "Expected a single timestamped training output directory starting with 'train-2'."
    trained_run_dir = run_dirs[0]

    # Define the final path to the specific sample model
    expected_model_dir = trained_run_dir / "sample-1"

    assert expected_model_dir.exists(), f"Model output directory was not created at {expected_model_dir}."
    
    # Check for either the standard PyTorch weights file or the SafeTensors equivalent
    weights_file_exists = (
        (expected_model_dir / "pytorch_model.bin").exists() or
        (expected_model_dir / "model.safetensors").exists()
    )
    assert weights_file_exists, "Model weights file (pytorch_model.bin or model.safetensors) is missing."
    
    assert (expected_model_dir / "config.json").exists(), "Model config file is missing."
    assert (expected_model_dir / "tokenizer_config.json").exists(), "Tokenizer config file is missing."
    print("--- Training Successful and Artifacts Verified ---")

    # --- 4. Run Evaluation ---
    print("\n--- Running NER Evaluation ---")
    
    # The evaluation function is called directly on the single model trained in this test.
    # We must create a config dictionary with the expected 'model_path' key.
    evaluation_config = {
        'task': "ner",
        'model_path': str(expected_model_dir), # Use model_path for the specific sample
        'test_file': str(test_file_path),
        'model': { # Nest the entity labels as required
            'entity_labels': ner_integration_config['model']['entity_labels']
        },
        'output_dir': str(tmp_path / "output" / "evaluation_results_ner"),
        'batch_size': 1
    }
    
    # Call the evaluation function with the correct config for a single run
    run_prediction_and_save(evaluation_config)

    # --- 5. Assert Prediction Outputs ---
    # The function saves directly to the output_dir, it does not create a timestamped folder.
    output_dir = Path(evaluation_config['output_dir'])
    
    # The filename was changed from 'raw_predictions' to 'predictions' for the NER task.
    expected_prediction_file = output_dir / f"predictions_{expected_model_dir.name}.jsonl"
    assert expected_prediction_file.exists(), "Prediction file was not created."
    print("--- Prediction Generation Successful and Artifacts Verified ---")

    # --- 6. Run Metrics Calculation ---
    print("\n--- Running NER Metrics Calculation ---")
    metrics_output_dir = tmp_path / "output" / "final_metrics_ner"
    final_metrics_path = metrics_output_dir / "final_metrics.json"

    # Call the unified metrics calculation script with the correct eval_type
    calculate_metrics(
        prediction_path=str(expected_prediction_file),
        prediction_dir=None,
        eval_type='ner',
        output_path=str(final_metrics_path)
    )

    # --- 7. Assert Final Metrics Outputs ---
    assert final_metrics_path.exists(), "Final metrics report file was not created."
    
    # Optionally, check the content of the final report
    with open(final_metrics_path, 'r') as f:
        report = json.load(f)

    assert "FIND" in report, "FIND entity not found in final metrics report."
    assert "REG" in report, "REG entity not found in final metrics report."
    assert "weighted avg" in report, "'weighted avg' not found in final metrics report."
    print("--- Metrics Calculation Successful and Report Verified ---")