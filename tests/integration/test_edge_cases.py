# tests/integration/test_edge_cases.py
import pytest
import yaml
from pathlib import Path
import sys

# Add the project root to the Python path for script imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.run_ner_training import run_batch_training

# --- Fixtures for Edge Case Tests ---

@pytest.fixture
def ner_edge_case_config(tmp_path):
    """
    Provides a standard NER training configuration for edge case testing.
    """
    config = {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'entity_labels': ["FIND", "REG"]
        },
        'paths': {
            'output_dir': str(tmp_path / "output" / "models")
        }
    }
    # Create and save the config file to the temporary directory
    config_path = tmp_path / "training_ner_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

# --- Edge Case Test Functions ---

def test_training_skips_empty_data_file(tmp_path, ner_edge_case_config):
    """
    Tests that the training script gracefully skips a sample directory
    that contains an empty train.jsonl file and does not create an output directory.
    """
    # --- 1. Setup: Create a partition with an empty training file ---
    partition_dir = tmp_path / "processed" / "train-empty"
    sample_dir = partition_dir / "sample-1"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an empty train.jsonl file
    (sample_dir / "train.jsonl").touch()

    # --- 2. Act: Run the batch training script ---
    print("\n--- Running Training on Partition with Empty Sample ---")
    run_batch_training(
        config_path=str(ner_edge_case_config),
        partition_dir=str(partition_dir)
    )

    # --- 3. Assert: Verify that no model directory was created ---
    with open(ner_edge_case_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Correct the base directory path to search for the timestamped folder
    base_output_dir = Path(config['paths']['output_dir']) / "ner"
    
    # Find the timestamped directory created by the run. Its name will start with the partition name.
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("train-empty")]
    
    # Assert that a single run directory was created
    assert len(run_dirs) == 1, "Expected a single timestamped output directory for the run."
    run_output_dir = run_dirs[0]

    # Assert that the timestamped run directory contains NO sample subdirectories
    sample_outputs = [d for d in run_output_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')]
    assert len(sample_outputs) == 0, "A model directory was created for an empty sample, which is incorrect."

    print("--- Test Successful: Script correctly skipped the empty sample. ---")



def test_training_fails_with_missing_config_keys(tmp_path):
    """
    Tests that the training script raises a KeyError if a critical key
    is missing from the configuration file.
    """
    # --- 1. Setup: Create a deliberately malformed config file ---
    # This configuration is missing the entire 'model' block.
    invalid_config = {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'device': "cpu"
        },
        'paths': {
            'output_dir': str(tmp_path / "output" / "models")
        }
    }
    invalid_config_path = tmp_path / "invalid_config.yaml"
    with open(invalid_config_path, 'w') as f:
        yaml.dump(invalid_config, f)

    # Setup a dummy partition directory, which is required by the script signature
    partition_dir = tmp_path / "processed" / "train-valid"
    sample_dir = partition_dir / "sample-1" # Define sample_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal, non-empty train.jsonl to bypass the file existence check.
    # This ensures the script proceeds to use the invalid config.
    dummy_train_file = sample_dir / "train.jsonl"
    with open(dummy_train_file, 'w') as f:
        f.write('{"text": "dummy text", "entities": []}\n')

    # --- 2. Act & Assert: Run the script and expect an OSError ---
    print("\n--- Running Training with Invalid Config ---")
    
    # Assert that an OSError is raised when the tokenizer receives `None` as a model path.
    with pytest.raises(OSError):
        run_batch_training(
            config_path=str(invalid_config_path),
            partition_dir=str(partition_dir)
        )

    print("--- Test Successful: Script correctly raised OSError for invalid config. ---")