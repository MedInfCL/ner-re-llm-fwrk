import pytest
import yaml
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.data.build_vector_db import main as build_vector_db_main

@pytest.fixture
def mock_rag_config():
    """Provides a mock RAG configuration dictionary."""
    return {
        'vector_db': {
            'embedding_model': 'mock-embedding-model',
            'source_data_path': 'mock/data/path.jsonl',
            'index_path': 'mock/index/path.bin'
        }
    }

@pytest.fixture
def mock_config_file(tmp_path, mock_rag_config):
    """Creates a temporary YAML config file for testing."""
    config_path = tmp_path / "rag_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_rag_config, f)
    return str(config_path)

# Patch the components that the script depends on
@patch('scripts.data.build_vector_db.DatabaseManager')
@patch('scripts.data.build_vector_db.SentenceEmbedder')
def test_main_script_flow(mock_embedder, mock_db_manager, mock_config_file, mock_rag_config):
    """
    Tests the main function of the build_vector_db script.

    It verifies that the script correctly:
    1. Loads the configuration.
    2. Initializes the SentenceEmbedder and DatabaseManager with the correct parameters.
    3. Calls the build_index method on the database manager.
    """
    # --- Act ---
    # Run the main function from the script
    build_vector_db_main(config_path=mock_config_file, force_rebuild=False)

    # --- Assertions ---

    # 1. Verify that the SentenceEmbedder was initialized correctly
    mock_embedder.assert_called_once_with(
        model_name=mock_rag_config['vector_db']['embedding_model']
    )

    # 2. Verify that the DatabaseManager was initialized correctly
    mock_db_manager.assert_called_once_with(
        embedder=mock_embedder.return_value,
        source_data_path=mock_rag_config['vector_db']['source_data_path'],
        index_path=mock_rag_config['vector_db']['index_path']
    )

    # 3. Verify that the build_index method was called with the correct argument
    mock_db_manager.return_value.build_index.assert_called_once_with(force_rebuild=False)

@patch('scripts.data.build_vector_db.DatabaseManager')
@patch('scripts.data.build_vector_db.SentenceEmbedder')
def test_main_script_with_force_rebuild(mock_embedder, mock_db_manager, mock_config_file):
    """
    Tests that the 'force_rebuild' flag is correctly passed to the build_index method.
    """
    build_vector_db_main(config_path=mock_config_file, force_rebuild=True)
    
    # Assert that the method was called with force_rebuild=True
    mock_db_manager.return_value.build_index.assert_called_once_with(force_rebuild=True)

def test_main_script_with_missing_config_file(capsys):
    """
    Tests that the script prints an error and exits gracefully if the config file is not found.
    """
    build_vector_db_main(config_path="non_existent_file.yaml", force_rebuild=False)
    
    # Capture the printed output
    captured = capsys.readouterr()
    assert "Error: Configuration file not found" in captured.out

@patch('scripts.data.build_vector_db.DatabaseManager')
@patch('scripts.data.build_vector_db.SentenceEmbedder')
def test_main_script_with_incomplete_config(mock_embedder, mock_db_manager, tmp_path, capsys):
    """
    Tests that the script prints an error if the config file is missing required keys.
    """
    # Create a config file with a missing key
    incomplete_config = {
        'vector_db': {
            'embedding_model': 'mock-model'
            # 'source_data_path' and 'index_path' are missing
        }
    }
    config_path = tmp_path / "incomplete_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(incomplete_config, f)
        
    build_vector_db_main(config_path=str(config_path), force_rebuild=False)
    
    captured = capsys.readouterr()
    assert "Error: Missing required keys" in captured.out
    
    # Verify that the core components were not initialized
    mock_embedder.assert_not_called()
    mock_db_manager.assert_not_called()