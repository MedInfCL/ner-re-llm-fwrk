# tests/integration/test_re_pipeline.py
import pytest
import yaml
from pathlib import Path
import json
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.run_re_training import run_batch_re_training
from scripts.evaluation.generate_finetuned_predictions import run_prediction_and_save
from scripts.evaluation.calculate_final_metrics import main as calculate_metrics

# --- Fixtures for RE Test Data and Configuration ---

@pytest.fixture
def re_integration_config():
    """
    Provides a minimal configuration for the RE integration test.
    """
    return {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 2, # RE can often use slightly larger batches
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'relation_labels': ["describir", "ubicar", "No_Relation"]
        },
        'paths': {
            'output_dir': "output/models_re"
        }
    }

@pytest.fixture
def re_training_data():
    """
    Provides a few records for the RE training set, including entities and relations.
    """
    return [
        {
            "text": "Nódulo periareolar derecho bien delimitado.",
            "entities": [
                {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 2, "label": "REG", "start_offset": 7, "end_offset": 26}
            ],
            "relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"}
            ]
        },
        {
            "text": "Microcalcificaciones agrupadas.",
            "entities": [
                {"id": 3, "label": "HALL", "start_offset": 0, "end_offset": 20},
                {"id": 4, "label": "CARACT", "start_offset": 21, "end_offset": 31}
            ],
            "relations": [
                {"from_id": 3, "to_id": 4, "type": "describir"}
            ]
        }
    ]

@pytest.fixture
def re_test_data():
    """Provides a few records for the RE test/evaluation set."""
    return [
        {
            "text": "Nódulo en mama izquierda.",
            "entities": [
                {"id": 5, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 6, "label": "REG", "start_offset": 10, "end_offset": 24}
            ],
            "relations": [
                {"from_id": 5, "to_id": 6, "type": "ubicar"}
            ]
        },
        {
            "text": "Microcalcificaciones agrupadas.",
            "entities": [
                {"id": 7, "label": "HALL", "start_offset": 0, "end_offset": 20},
                {"id": 8, "label": "CARACT", "start_offset": 21, "end_offset": 31}
            ],
            "relations": [
                {"from_id": 7, "to_id": 8, "type": "describir"}
            ]
        }
    ]

# --- RE Integration Test ---

def test_re_training_and_evaluation_pipeline(tmp_path, re_integration_config, re_training_data, re_test_data):
    """
    Tests the complete RE pipeline from training to evaluation.
    """
    # --- 1. Setup temporary directory and files ---
    partition_dir = tmp_path / "processed_re" / "train-2" / "sample-1"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = partition_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for record in re_training_data:
            f.write(json.dumps(record) + '\n')
            
    test_file_path = tmp_path / "processed_re" / "test.jsonl"
    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, 'w') as f:
        for record in re_test_data:
            f.write(json.dumps(record) + '\n')

    training_config_path = tmp_path / "training_re_config.yaml"
    re_integration_config['paths']['output_dir'] = str(tmp_path / "output" / "models_re")
    with open(training_config_path, 'w') as f:
        yaml.dump(re_integration_config, f)

    # --- 2. Run RE Training ---
    print("\n--- Running RE Training ---")
    run_batch_re_training(
        config_path=str(training_config_path),
        partition_dir=str(tmp_path / "processed_re" / "train-2")
    )

    # The output path now includes a timestamp, so we must find it dynamically.
    base_output_dir = Path(re_integration_config['paths']['output_dir']) / "re"

    # Find the single timestamped directory created by the training run
    # It will start with the partition name, e.g., "train-2"
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("train-2")]
    assert len(run_dirs) == 1, "Expected a single timestamped training output directory starting with 'train-2'."
    trained_run_dir = run_dirs[0]
    
    # Define the final path to the specific sample model
    expected_model_dir = trained_run_dir / "sample-1"
    
    assert expected_model_dir.exists(), "RE Model output directory was not created."
    weights_file_exists = (
        (expected_model_dir / "pytorch_model.bin").exists() or
        (expected_model_dir / "model.safetensors").exists()
    )
    assert weights_file_exists, "RE Model weights file is missing."
    assert (expected_model_dir / "config.json").exists(), "RE Model config file is missing."
    print("--- RE Training Successful and Artifacts Verified ---")

    # --- 4. Run RE Prediction Generation ---
    print("\n--- Running RE Evaluation ---")
    evaluation_config = {
        'task': "re",
        'model_path': str(expected_model_dir),
        'test_file': str(test_file_path),
        'model': { 'relation_labels': re_integration_config['model']['relation_labels'] },
        'output_dir': str(tmp_path / "output" / "evaluation_results_re"),
        'batch_size': 1
    }
    
    # Call the evaluation function directly with the config dictionary
    run_prediction_and_save(evaluation_config)

    # --- 5. Assert Prediction Outputs ---
    output_dir = Path(evaluation_config['output_dir'])
    expected_prediction_file = output_dir / f"predictions_{expected_model_dir.name}.jsonl"
    assert expected_prediction_file.exists(), "Decoded RE prediction file was not created."
    print("--- RE Prediction Generation Successful and Artifacts Verified ---")

    # --- 6. Run Metrics Calculation ---
    print("\n--- Running RE Metrics Calculation ---")
    metrics_output_dir = tmp_path / "output" / "final_metrics_re"
    final_metrics_path = metrics_output_dir / "final_metrics.json"

    # The main script's function signature requires the test_file path,
    # even if it's not used by the RE metrics calculator.
    calculate_metrics(
        prediction_path=str(expected_prediction_file),
        prediction_dir=None,
        eval_type='re',
        output_path=str(final_metrics_path)
    )

    # --- 7. Assert Final Metrics Outputs ---
    assert final_metrics_path.exists(), "Final RE metrics report file was not created."
    
    # Verify the content of the generated metrics report
    with open(final_metrics_path, 'r') as f:
        report = json.load(f)
    
    assert "describir" in report, "'describir' relation not found in final metrics report."
    assert "ubicar" in report, "'ubicar' relation not found in final metrics report."
    assert "weighted avg" in report, "'weighted avg' not found in final metrics report."
    print("--- RE Metrics Calculation Successful and Report Verified ---")