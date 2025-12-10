# tests/unit/scripts/evaluation/test_generate_finetuned_predictions.py
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.generate_finetuned_predictions import run_prediction_and_save, convert_numpy_types

# --- Fixtures for Testing ---

@pytest.fixture
def ner_config(tmp_path):
    """Provides a minimal configuration for a NER prediction task."""
    return {
        'task': 'ner',
        'model_path': str(tmp_path / 'models' / 'ner_model'),
        'test_file': str(tmp_path / 'data' / 'test.jsonl'),
        'output_dir': str(tmp_path / 'output' / 'ner_results'),
        'model': { # Needed for NERDataModule initialization
            'entity_labels': ["FIND"]
        }
    }

@pytest.fixture
def re_config(tmp_path):
    """Provides a minimal configuration for a RE prediction task."""
    return {
        'task': 're',
        'model_path': str(tmp_path / 'models' / 're_model'),
        'test_file': str(tmp_path / 'data' / 'test.jsonl'),
        'output_dir': str(tmp_path / 'output' / 're_results'),
        'model': { # REDataModule requires relation_labels in config
            'relation_labels': ['ubicar', 'describir']
        }
    }

# --- Mocks for Core Dependencies ---
# We patch the specific modules where they are looked up in the script under test.
@patch('scripts.evaluation.generate_finetuned_predictions.decode_entities_from_tokens')
@patch('scripts.evaluation.generate_finetuned_predictions.REDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.NERDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.REModel')
@patch('scripts.evaluation.generate_finetuned_predictions.BertNerModel')
@patch('scripts.evaluation.generate_finetuned_predictions.Predictor')
@patch('builtins.open', new_callable=mock_open, read_data='{"text": "Sample text.", "entities": [{"start_offset": 0, "end_offset": 6, "label": "FIND"}]}')
@patch('pathlib.Path.mkdir')
def test_ner_workflow(
    mock_mkdir,
    mock_file_open,
    mock_predictor,
    mock_bert_ner_model,
    mock_re_model,
    mock_ner_datamodule,
    mock_re_datamodule,
    mock_decode_entities,
    ner_config
):
    """
    Tests the end-to-end prediction generation workflow for a NER task,
    verifying that it produces the new, unified entity schema.
    """
    # --- Setup Mocks ---
    # The predictor still returns raw integer labels
    mock_predictor_instance = mock_predictor.return_value
    mock_predictor_instance.predict.return_value = (
        [[1, 0]], # Mock predictions
        [[1, 0]], # Mock true labels
        np.array([])
    )

    # Mock the new decoder function to return a predictable entity list
    mock_decoded_entities = [{"text": "Sample", "label": "FIND"}]
    mock_decode_entities.return_value = mock_decoded_entities

    # --- Act ---
    run_prediction_and_save(ner_config)

    # --- Assertions ---
    # 1. Verify correct modules were initialized for NER
    mock_ner_datamodule.assert_called_once()
    mock_bert_ner_model.assert_called_once_with(base_model=ner_config['model_path'])
    
    # 2. Verify Predictor was used and decoder was called
    mock_predictor_instance.predict.assert_called_once()
    mock_decode_entities.assert_called_once()

    # 3. Verify file I/O and output content
    handle = mock_file_open()
    written_data = handle.write.call_args[0][0]
    
    expected_output = {
        "source_text": "Sample text.",
        "true_entities": [{
            "text": "Sample", 
            "label": "FIND", 
            "start_offset": 0, 
            "end_offset": 6
        }],
        "predicted_entities": mock_decoded_entities
    }
    assert json.loads(written_data) == expected_output

@patch('scripts.evaluation.generate_finetuned_predictions.decode_relations_from_ids')
@patch('scripts.evaluation.generate_finetuned_predictions.REDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.NERDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.REModel')
@patch('scripts.evaluation.generate_finetuned_predictions.BertNerModel')
@patch('scripts.evaluation.generate_finetuned_predictions.Predictor')
@patch('builtins.open', new_callable=mock_open, read_data='{"text": "Sample RE text.", "entities": [{"id": 1, "start_offset": 0, "end_offset": 6}, {"id": 2, "start_offset": 7, "end_offset": 12}]}')
@patch('pathlib.Path.mkdir')
def test_re_workflow(
    mock_mkdir,
    mock_file_open,
    mock_predictor,
    mock_bert_ner_model,
    mock_re_model,
    mock_ner_datamodule,
    mock_re_datamodule,
    mock_decode_relations, # Renamed for clarity
    re_config
):
    """
    Tests the end-to-end prediction generation workflow for a RE task.
    """
    # --- Setup Mocks ---
    mock_predictor_instance = mock_predictor.return_value
    mock_predictor_instance.predict.return_value = ([0, 1], [1, 2], np.array([]))

    # Mock the new decoder to return a predictable relation list
    mock_decoded_relations = [{"from_id": 1, "to_id": 2, "type": "ubicar"}]
    mock_decode_relations.return_value = mock_decoded_relations

    # --- Act ---
    run_prediction_and_save(re_config)

    # --- Assertions ---
    # 1. Verify correct modules were initialized for RE
    mock_re_datamodule.assert_called_once_with(config=re_config, test_file=re_config['test_file'])
    mock_re_model.assert_called_once()

    # 2. Verify Predictor was used and the new decoder was called
    mock_predictor_instance.predict.assert_called_once()
    assert mock_decode_relations.call_count == 2 # Called for true and predicted labels

    # 3. Verify output content matches the new unified format
    handle = mock_file_open()
    written_data = handle.write.call_args[0][0]
    expected_output = {
        "source_text": "Sample RE text.",
        "true_relations": mock_decoded_relations,
        "predicted_relations": mock_decoded_relations
    }
    assert json.loads(written_data) == expected_output

def test_convert_numpy_types():
    """
    Tests the convert_numpy_types helper function to ensure it correctly
    converts numpy types to native Python types for JSON serialization.
    """
    numpy_data = {
        "integer": np.int64(10),
        "float": np.float32(3.14),
        "nested": { "array": np.array([1, 2, 3]) }
    }
    converted_data = convert_numpy_types(numpy_data)
    assert isinstance(converted_data["integer"], int)
    assert isinstance(converted_data["float"], float)
    assert isinstance(converted_data["nested"]["array"], list)
    assert converted_data["nested"]["array"] == [1, 2, 3]

def test_invalid_task_raises_error(ner_config):
    """
    Tests that the function raises a ValueError if the task in the config
    is not 'ner' or 're'.
    """
    invalid_config = ner_config.copy()
    invalid_config['task'] = 'unknown_task'
    
    with pytest.raises(ValueError, match="Configuration file must specify a 'task': 'ner' or 're'"):
        run_prediction_and_save(invalid_config)