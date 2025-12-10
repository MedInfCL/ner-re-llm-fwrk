import pytest
import yaml
from pathlib import Path
import json
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data.build_vector_db import main as build_vector_db_main
from scripts.evaluation.generate_rag_predictions import main as generate_rag_predictions_main

# --- Fixtures for RAG Test Data and Configuration ---

@pytest.fixture
def rag_integration_config(tmp_path):
    """Provides a minimal configuration for the RAG integration test."""
    config = {
        'task': 'ner',
        'llm': {
            'provider': 'openai',
            'openai': {
                'model': 'mock-gpt-model',
                'temperature': 0.0
            }
        },
        'vector_db': {
            'index_path': str(tmp_path / "vector_db/test_index.bin"),
            'source_data_path': str(tmp_path / "data/source_data.jsonl"),
            'embedding_model': 'prajjwal1/bert-tiny' # Use a fast, small model
        },
        'rag_prompt': {
            'prompt_template_path': str(tmp_path / "prompts/test_prompt.txt"),
            'n_examples': 1,
            'entity_labels': [
                {'name': 'FIND', 'description': 'A finding.'},
                {'name': 'REG', 'description': 'A region.'}
            ]
        },
        'test_file': str(tmp_path / "data/test_data.jsonl"),
        'output_dir': str(tmp_path / "output/rag_results")
    }
    # Create the config file
    config_path = tmp_path / "rag_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)

@pytest.fixture
def setup_rag_test_environment(tmp_path, rag_integration_config):
    """Sets up all necessary files for the RAG pipeline test."""
    config_path = Path(rag_integration_config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create source data for the vector DB
    source_data = [
        {"text": "A finding in the left region.", "entities": [{"start_offset": 2, "end_offset": 9, "label": "FIND"}]}
    ]
    source_data_path = Path(config['vector_db']['source_data_path'])
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_data_path, 'w') as f:
        for record in source_data:
            f.write(json.dumps(record) + '\n')

    # Create test data to be evaluated
    test_data = [
        {"text": "This is a test report with a finding.", "entities": []}
    ]
    test_data_path = Path(config['test_file'])
    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_data_path, 'w') as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')

    # Create a prompt template file
    prompt_template_path = Path(config['rag_prompt']['prompt_template_path'])
    prompt_template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_template_path, 'w') as f:
        f.write("Test prompt: {new_report_text}")

    return config_path

# --- RAG Integration Test ---

@patch('scripts.evaluation.generate_rag_predictions.Langfuse')
@patch('src.llm_services.OpenAIClient')
def test_rag_prediction_pipeline(mock_openai_client, mock_langfuse, setup_rag_test_environment, monkeypatch):
    """
    Tests the complete RAG prediction pipeline.
    This test mocks the external API call to OpenAI.
    """
    # --- 1. Setup Mocks ---
    mock_api_response_entities = [{"text": "a finding", "label": "FIND", "start_offset": 0}]
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")

    mock_client_instance = MagicMock()
    mock_client_instance.get_ner_prediction.return_value = mock_api_response_entities
    mock_openai_client.return_value = mock_client_instance
    
    # Mock the Langfuse instance to prevent network calls
    mock_langfuse_instance = MagicMock()
    mock_langfuse.return_value = mock_langfuse_instance

    config_path = setup_rag_test_environment

    # --- 2. Build the Vector Database ---
    print("\n--- Building RAG Vector Database for Test ---")
    build_vector_db_main(config_path=config_path, force_rebuild=True)

    # --- 3. Run RAG Prediction Generation ---
    print("\n--- Running RAG Prediction Generation ---")
    generate_rag_predictions_main(config_path=config_path)

    # --- 4. Assert Outputs ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # The script creates a nested directory: output/{task}/{n-shot}/{run_name}/
    output_dir = Path(config['output_dir'])
    task = config.get('task', 'ner')
    n_examples = str(config.get('rag_prompt', {}).get('n_examples', 0)) + "-shot"
    
    # This is the directory where the timestamped run folder is created
    run_base_dir = output_dir / task / n_examples
    
    # Find the single timestamped directory created by the script
    run_dirs = [d for d in run_base_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "Expected a single timestamped RAG output directory."
    
    # Check for the correctly named prediction file inside the new directory
    prediction_file = run_dirs[0] / "predictions.jsonl"
    assert prediction_file.exists(), f"RAG prediction file was not created at {prediction_file}"

    # Assert the content of the prediction file
    with open(prediction_file, 'r') as f:
        predictions = [json.loads(line) for line in f]
    assert len(predictions) == 1
    # Define the expected entity after post-processing has corrected its offsets
    expected_postprocessed_entity = [{
        "text": "a finding",
        "label": "FIND",
        "start_offset": 27,
        "end_offset": 36
    }]
    assert predictions[0]['predicted_entities'] == expected_postprocessed_entity

    # Assert that the langfuse client's flush method was called once
    mock_langfuse_instance.flush.assert_called_once()

    print("--- RAG Pipeline Test Successful ---")


@patch('scripts.evaluation.generate_rag_predictions.Langfuse')
@patch('src.llm_services.OpenAIClient')
def test_rag_prediction_resume_functionality(mock_openai_client, mock_langfuse, rag_integration_config, tmp_path, monkeypatch):
    """
    Tests that the RAG prediction script correctly resumes from a partially completed run.
    """
    # --- 1. Setup: Create a more extensive test environment ---
    config_path = Path(rag_integration_config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create a test file with multiple records
    test_data = [
        {"text": "First test report.", "entities": []},
        {"text": "Second test report.", "entities": []},
        {"text": "Third test report.", "entities": []}
    ]
    test_data_path = Path(config['test_file'])
    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_data_path, 'w') as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')
            
    # Create the prompt template file that the config expects
    prompt_template_path = Path(config['rag_prompt']['prompt_template_path'])
    prompt_template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_template_path, 'w') as f:
        f.write("Test prompt: {new_report_text}")
        
    # --- 2. Simulate an Interrupted Run ---
    # Create the output directory that would have been made by the first run
    run_dir = Path(config['output_dir']) / "ner" / "20250829_120000_resume_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a partial predictions file, simulating that the first record was completed
    partial_predictions_file = run_dir / "predictions.jsonl"
    completed_record = {
        "source_text": "First test report.",
        "true_entities": [],
        "predicted_entities": [{"text": "first finding", "label": "FIND", "start_offset": 0}],
        "prompt_used": "..."
    }
    with open(partial_predictions_file, 'w') as f:
        f.write(json.dumps(completed_record) + '\n')

    # --- 3. Setup Mocks ---
    mock_api_response = [{"text": "some finding", "label": "FIND", "start_offset": 0}]
    mock_client_instance = MagicMock()
    mock_client_instance.get_ner_prediction.return_value = mock_api_response
    mock_openai_client.return_value = mock_client_instance
    mock_langfuse.return_value = MagicMock() # Mock langfuse to prevent network calls

    # --- 4. Execute the Resume Run ---
    print("\n--- Running RAG Prediction Generation with --resume-dir ---")
    generate_rag_predictions_main(config_path=str(config_path), resume_dir=str(run_dir))

    # --- 5. Assert Correct Behavior ---
    # Assert that the API was only called for the remaining, unprocessed records
    assert mock_client_instance.get_ner_prediction.call_count == 2, \
        "Expected the LLM client to be called only for the two remaining records."

    # Assert that the final prediction file contains all results (the original + the new ones)
    with open(partial_predictions_file, 'r') as f:
        final_predictions = [json.loads(line) for line in f]
    
    assert len(final_predictions) == 3, "Expected the final predictions file to contain all three records."
    
    # Check that the first record is the one we manually created
    assert final_predictions[0]["source_text"] == "First test report."
    # Check that the other records were processed
    processed_texts = {p["source_text"] for p in final_predictions}
    assert "Second test report." in processed_texts
    assert "Third test report." in processed_texts

    print("--- RAG Resume Functionality Test Successful ---")