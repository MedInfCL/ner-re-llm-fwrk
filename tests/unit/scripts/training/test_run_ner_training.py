# tests/unit/scripts/training/test_run_ner_training.py
import pytest
import yaml
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.training.run_ner_training import run_batch_training

# --- Fixtures for Testing ---

@pytest.fixture
def mock_ner_config(tmp_path):
    """Provides a mock NER training configuration dictionary."""
    return {
        'seed': 42,
        'model': {'base_model': 'mock-bert', 'entity_labels': ['FIND']},
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'paths': {'output_dir': str(tmp_path / 'output' / 'models')}
    }

@pytest.fixture
def setup_partition_dir(tmp_path):
    """Creates a temporary partition directory with sample subdirectories."""
    partition_dir = tmp_path / "processed" / "train-2"
    
    # Valid sample with a training file
    sample_1_dir = partition_dir / "sample-1"
    sample_1_dir.mkdir(parents=True, exist_ok=True)
    (sample_1_dir / "train.jsonl").touch()

    # Another valid sample
    sample_2_dir = partition_dir / "sample-2"
    sample_2_dir.mkdir(exist_ok=True)
    (sample_2_dir / "train.jsonl").touch()

    # A directory that is not a valid sample
    (partition_dir / "not_a_sample").mkdir(exist_ok=True)
    
    # A file that should be ignored
    (partition_dir / "some_file.txt").touch()

    return partition_dir

# --- Mocks for Core Dependencies ---

@patch('scripts.training.run_ner_training.shutil.copy')
@patch('scripts.training.run_ner_training.Trainer')
@patch('scripts.training.run_ner_training.BertNerModel')
@patch('scripts.training.run_ner_training.NERDataModule')
def test_successful_training_run(
    mock_NERDataModule, mock_BertNerModel, mock_Trainer, mock_shutil_copy,
    mock_ner_config, setup_partition_dir, tmp_path
):
    """
    Tests the ideal 'happy path' where valid samples are found and processed correctly.
    """
    # --- 1. Setup: Create the config file on disk ---
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_ner_config, f)

    # Mock NERDataModule to return a non-empty dataset
    mock_datamodule_instance = MagicMock()
    mock_datamodule_instance.train_dataset.__len__.return_value = 10 # Non-empty
    mock_datamodule_instance.label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    mock_NERDataModule.return_value = mock_datamodule_instance
    
    # Mock Trainer
    mock_trainer_instance = MagicMock()
    mock_Trainer.return_value = mock_trainer_instance

    # --- 2. Act ---
    run_batch_training(str(config_path), str(setup_partition_dir))

    # --- 3. Assert ---
    # Should be called twice, once for each valid sample
    assert mock_NERDataModule.call_count == 2
    assert mock_BertNerModel.call_count == 2
    assert mock_Trainer.call_count == 2
    
    # Verify that the core training and saving methods were called for each sample
    assert mock_trainer_instance.train.call_count == 2
    assert mock_trainer_instance.save_model.call_count == 2
    mock_shutil_copy.assert_called_once()


@patch('scripts.training.run_ner_training.Trainer')
@patch('scripts.training.run_ner_training.NERDataModule')
@patch('scripts.training.run_ner_training.BertNerModel')
def test_skips_directory_without_train_file(
    mock_BertNerModel, mock_NERDataModule, mock_Trainer,
    mock_ner_config, setup_partition_dir, tmp_path
):
    """
    Tests that the script correctly skips a sample directory if it is missing 'train.jsonl'.
    """
    # --- 1. Setup: Create the config file and modify the partition ---
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_ner_config, f)

    # CORRECTED: Mock NERDataModule to return a non-empty dataset for the valid sample
    mock_datamodule_instance = MagicMock()
    mock_datamodule_instance.train_dataset.__len__.return_value = 10 # Non-empty
    mock_datamodule_instance.label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    mock_NERDataModule.return_value = mock_datamodule_instance

    # Remove the train.jsonl file from the second sample
    (setup_partition_dir / "sample-2" / "train.jsonl").unlink()

    # --- 2. Act ---
    run_batch_training(str(config_path), str(setup_partition_dir))

    # --- 3. Assert ---
    # The training components should only be instantiated for the one valid sample ('sample-1')
    mock_NERDataModule.assert_called_once()
    mock_BertNerModel.assert_called_once()
    mock_Trainer.assert_called_once()


@patch('scripts.training.run_ner_training.Trainer')
@patch('scripts.training.run_ner_training.NERDataModule')
@patch('scripts.training.run_ner_training.BertNerModel')
def test_skips_sample_with_empty_dataset(
    mock_BertNerModel, mock_NERDataModule, mock_Trainer,
    mock_ner_config, setup_partition_dir, tmp_path
):
    """
    Tests that the script skips a sample if its training dataset is empty.
    """
    # --- 1. Setup: Create config and mock the DataModule's behavior ---
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_ner_config, f)

    # Configure the mock NERDataModule to return datasets of different lengths
    mock_datamodule_instance_non_empty = MagicMock()
    mock_datamodule_instance_non_empty.train_dataset.__len__.return_value = 10
    mock_datamodule_instance_non_empty.label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    
    mock_datamodule_instance_empty = MagicMock()
    mock_datamodule_instance_empty.train_dataset.__len__.return_value = 0
    mock_datamodule_instance_empty.label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}

    mock_NERDataModule.side_effect = [
        mock_datamodule_instance_non_empty,
        mock_datamodule_instance_empty
    ]
    
    # --- 2. Act ---
    run_batch_training(str(config_path), str(setup_partition_dir))

    # --- 3. Assert ---
    # NERDataModule should be called for both samples...
    assert mock_NERDataModule.call_count == 2
    # ...but the Model and Trainer should only be called once for the non-empty one.
    mock_BertNerModel.assert_called_once()
    mock_Trainer.assert_called_once()


@patch('shutil.copy')
def test_handles_no_valid_sample_directories(mock_shutil_copy, tmp_path, mock_ner_config):
    """
    Tests that the script exits gracefully if no 'sample-*' directories are found.
    """
    # --- 1. Setup ---
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_ner_config, f)
        
    empty_partition_dir = tmp_path / "empty_partition"
    empty_partition_dir.mkdir()

    # --- 2. Act ---
    run_batch_training(str(config_path), str(empty_partition_dir))

    # --- 3. Assert ---
    # If no samples are found, the script should exit before creating an output
    # directory, so shutil.copy should not have been called.
    mock_shutil_copy.assert_not_called()

def test_raises_file_not_found_for_bad_config(tmp_path):
    """
    Tests that a FileNotFoundError is raised for an invalid config path.
    """
    # --- 1. Setup ---
    non_existent_config = str(tmp_path / "non_existent_config.yaml")
    dummy_partition_dir = tmp_path / "dummy_partition"
    dummy_partition_dir.mkdir()
    
    # --- 2. Act & Assert ---
    with pytest.raises(FileNotFoundError):
        run_batch_training(non_existent_config, str(dummy_partition_dir))