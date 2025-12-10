# tests/unit/scripts/training/test_run_re_training.py
import pytest
import yaml
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.training.run_re_training import run_batch_re_training

# --- Fixtures for Testing ---

@pytest.fixture
def mock_re_config(tmp_path):
    """Provides a mock RE training configuration dictionary."""
    return {
        'seed': 42,
        'model': {
            'base_model': 'mock-bert-re',
            'relation_labels': ['ubicar', 'describir', 'No_Relation']
        },
        'trainer': {
            'n_epochs': 1,
            'batch_size': 2,
            'device': "cpu"
        },
        'paths': {'output_dir': str(tmp_path / 'output' / 'models_re')}
    }

@pytest.fixture
def setup_partition_dir(tmp_path):
    """Creates a temporary partition directory with sample subdirectories."""
    partition_dir = tmp_path / "processed" / "train-10"
    
    # Valid sample with a training file
    (partition_dir / "sample-1").mkdir(parents=True, exist_ok=True)
    (partition_dir / "sample-1" / "train.jsonl").touch()

    # Another valid sample
    (partition_dir / "sample-2").mkdir(exist_ok=True)
    (partition_dir / "sample-2" / "train.jsonl").touch()

    return partition_dir

# --- Test Cases ---

@patch('scripts.training.run_re_training.shutil.copy')
@patch('scripts.training.run_re_training.Trainer')
@patch('scripts.training.run_re_training.REModel')
@patch('scripts.training.run_re_training.REDataModule')
def test_successful_re_training_run(
    mock_REDataModule, mock_REModel, mock_Trainer, mock_shutil_copy,
    mock_re_config, setup_partition_dir, tmp_path
):
    """
    Tests the ideal 'happy path' for the RE training script.
    """
    # --- 1. Setup ---
    config_path = tmp_path / "re_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_re_config, f)

    # Mock the DataModule and Trainer instances
    mock_datamodule_instance = MagicMock()
    mock_datamodule_instance.relation_map = {'ubicar': 0, 'describir': 1, 'No_Relation': 2}
    mock_REDataModule.return_value = mock_datamodule_instance

    mock_trainer_instance = MagicMock()
    mock_Trainer.return_value = mock_trainer_instance

    # --- 2. Act ---
    run_batch_re_training(str(config_path), str(setup_partition_dir))

    # --- 3. Assert ---
    # Assert that all components were called twice (for sample-1 and sample-2)
    assert mock_REDataModule.call_count == 2
    assert mock_REModel.call_count == 2
    assert mock_Trainer.call_count == 2
    
    # Assert that the core training and saving methods were called for each sample
    assert mock_trainer_instance.train.call_count == 2
    assert mock_trainer_instance.save_model.call_count == 2

    # Assert that the configuration was copied to the output directory
    mock_shutil_copy.assert_called_once()

@patch('scripts.training.run_re_training.Trainer')
@patch('scripts.training.run_re_training.REModel')
@patch('scripts.training.run_re_training.REDataModule')
def test_skips_directory_without_train_file(
    mock_REDataModule, mock_REModel, mock_Trainer,
    mock_re_config, setup_partition_dir, tmp_path
):
    """
    Tests that the script correctly skips a sample if 'train.jsonl' is missing.
    """
    # --- 1. Setup ---
    config_path = tmp_path / "re_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_re_config, f)

    # Remove the training file from one of the samples
    (setup_partition_dir / "sample-2" / "train.jsonl").unlink()

    # --- 2. Act ---
    run_batch_re_training(str(config_path), str(setup_partition_dir))

    # --- 3. Assert ---
    # The training components should only have been instantiated once for the valid sample
    mock_REDataModule.assert_called_once()
    mock_REModel.assert_called_once()
    mock_Trainer.assert_called_once()

@patch('shutil.copy')
def test_handles_no_valid_sample_directories(mock_shutil_copy, tmp_path, mock_re_config):
    """
    Tests that the script exits gracefully if no valid 'sample-*' directories are found.
    """
    # --- 1. Setup ---
    config_path = tmp_path / "re_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_re_config, f)
        
    empty_partition_dir = tmp_path / "empty_partition"
    empty_partition_dir.mkdir()

    # --- 2. Act ---
    run_batch_re_training(str(config_path), str(empty_partition_dir))

    # --- 3. Assert ---
    # If no samples are found, no output directory should be created,
    # and therefore shutil.copy should not be called.
    mock_shutil_copy.assert_not_called()

def test_raises_file_not_found_for_bad_config(tmp_path):
    """
    Tests that a FileNotFoundError is raised if the config path is invalid.
    """
    # --- 1. Setup ---
    non_existent_config = tmp_path / "non_existent_config.yaml"
    dummy_partition_dir = tmp_path / "dummy_partition"
    dummy_partition_dir.mkdir()

    # --- 2. Act & Assert ---
    with pytest.raises(FileNotFoundError):
        run_batch_re_training(str(non_existent_config), str(dummy_partition_dir))