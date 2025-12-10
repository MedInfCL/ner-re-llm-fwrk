# tests/integration/test_rag_re_pipeline.py
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
from scripts.evaluation.calculate_final_metrics import main as calculate_metrics

# --- Fixtures for RE RAG Test Data and Configuration ---

@pytest.fixture
def rag_re_integration_config(tmp_path):
    """Provides a minimal configuration for the RE RAG integration test."""
    config = {
        'task': 're',
        'llm': {
            'provider': 'openai',
            'openai': {'model': 'mock-gpt-model', 'temperature': 0.0}
        },
        'vector_db': {
            'index_path': str(tmp_path / "vector_db/re_test_index.bin"),
            'source_data_path': str(tmp_path / "data/re_source_data.jsonl"),
            'embedding_model': 'prajjwal1/bert-tiny'
        },
        'rag_prompt': {
            'prompt_template_path': str(tmp_path / "prompts/re_test_prompt.txt"),
            'n_examples': 1,
            'relation_labels': [
                {'name': 'ubicar', 'description': 'Location relation.'}
            ]
        },
        'test_file': str(tmp_path / "data/re_test_data.jsonl"),
        'output_dir': str(tmp_path / "output/rag_re_results")
    }
    config_path = tmp_path / "rag_re_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)

@pytest.fixture
def setup_rag_re_test_environment(tmp_path, rag_re_integration_config):
    """Sets up all necessary files for the RE RAG pipeline test."""
    config_path = Path(rag_re_integration_config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create source data with relations for the vector DB
    source_data = [{
        "text": "Nódulo en mama izquierda.",
        "entities": [
            {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6}, 
            {"id": 2, "label": "REG", "start_offset": 10, "end_offset": 24}
        ],
        "relations": [{"from_id": 1, "to_id": 2, "type": "ubicar"}]
    }]
    source_data_path = Path(config['vector_db']['source_data_path'])
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_data_path, 'w') as f:
        f.write(json.dumps(source_data[0]) + '\n')

    # Create test data to be evaluated
    test_data = [{
        "text": "Un hallazgo en la mama derecha.",
        "entities": [
            {"id": 3, "label": "HALL", "start_offset": 3, "end_offset": 11}, 
            {"id": 4, "label": "REG", "start_offset": 18, "end_offset": 30}
        ],
        "relations": [{"from_id": 3, "to_id": 4, "type": "ubicar"}]
    }]
    test_data_path = Path(config['test_file'])
    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_data_path, 'w') as f:
        f.write(json.dumps(test_data[0]) + '\n')

    # Create a prompt template file
    prompt_template_path = Path(config['rag_prompt']['prompt_template_path'])
    prompt_template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_template_path, 'w') as f:
        f.write("Test RE prompt: {new_report_text}")

    return config_path
# --- RE RAG Integration Test ---

@patch('scripts.evaluation.generate_rag_predictions.Langfuse')
@patch('src.llm_services.OpenAIClient')
def test_rag_re_prediction_pipeline(mock_openai_client, mock_langfuse, setup_rag_re_test_environment):
    """
    Tests the complete RE RAG prediction and evaluation pipeline.
    """
    # --- 1. Setup Mocks ---
    mock_api_response_relations = [{"from_id": 3, "to_id": 4, "type": "ubicar"}]
    mock_client_instance = MagicMock()
    mock_client_instance.get_re_prediction.return_value = mock_api_response_relations
    mock_openai_client.return_value = mock_client_instance
    mock_langfuse.return_value = MagicMock()

    config_path = setup_rag_re_test_environment

    # --- 2. Build Vector DB ---
    build_vector_db_main(config_path=config_path, force_rebuild=True)

    # --- 3. Run RE RAG Prediction ---
    generate_rag_predictions_main(config_path=config_path)

    # --- 4. Assert Prediction Outputs ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # The script creates a nested directory: output/{task}/{n-shot}/{run_name}/
    output_dir = Path(config['output_dir'])
    task = config.get('task', 're')
    n_examples = str(config.get('rag_prompt', {}).get('n_examples', 0)) + "-shot"
    
    # This is the directory where the timestamped run folder is created
    run_base_dir = output_dir / task / n_examples
    
    # Find the single timestamped directory created by the script
    run_dirs = [d for d in run_base_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "Expected a single timestamped RAG output directory."
    
    prediction_file = run_dirs[0] / "predictions.jsonl"
    assert prediction_file.exists(), f"RAG prediction file was not created at {prediction_file}"

    with open(prediction_file, 'r') as f:
        predictions = [json.loads(line) for line in f]
    assert len(predictions) == 1
    assert predictions[0]['predicted_relations'] == mock_api_response_relations

    # --- 5. Run Metrics Calculation ---
    metrics_path = run_dirs[0] / "final_metrics.json"
    calculate_metrics(
        prediction_path=str(prediction_file),
        prediction_dir=None,
        eval_type='re',
        output_path=str(metrics_path)
    )
    assert metrics_path.exists()