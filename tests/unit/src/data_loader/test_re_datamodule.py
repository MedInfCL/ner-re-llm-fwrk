# tests/unit/src/data_loader/test_re_datamodule.py
import pytest
import torch
import json
from unittest.mock import MagicMock, patch
import warnings

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.data_loader.re_datamodule import REDataModule, REDataset

# --- Fixtures for RE Testing ---

@pytest.fixture
def mock_re_config():
    """Provides a mock configuration dictionary for the REDataModule."""
    return {
        'model': {
            'base_model': 'mock-re-model',
            'relation_labels': ["describir", "ubicar", "No_Relation"]
        },
        'trainer': {
            'batch_size': 2
        },
        'batch_size': 2
    }

@pytest.fixture
def mock_re_tokenizer():
    """Creates a mock tokenizer for RE tests."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones((1, 512), dtype=torch.long),
        "attention_mask": torch.ones((1, 512), dtype=torch.long)
    }
    tokenizer.add_special_tokens = MagicMock()
    
    # Patch from_pretrained to return this mock
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
        yield tokenizer

@pytest.fixture
def re_jsonl_file(tmp_path):
    """Creates a temporary .jsonl file with sample RE data."""
    data = [
        {
            "text": "Nódulo periareolar derecho bien delimitado.",
            "entities": [
                {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 2, "label": "REG", "start_offset": 7, "end_offset": 26},
                {"id": 3, "label": "CARACT", "start_offset": 27, "end_offset": 45}
            ],
            "relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"},
                {"from_id": 1, "to_id": 3, "type": "describir"}
            ]
        },
        {
            "text": "Mamas densas.",
            "entities": [
                {"id": 10, "label": "MAMAS", "start_offset": 0, "end_offset": 5},
                {"id": 11, "label": "DENS", "start_offset": 6, "end_offset": 12}
            ],
            "relations": [
                 # This relation type is not in the mock_re_config, so it should be ignored
                {"from_id": 10, "to_id": 11, "type": "ignored_relation"}
            ]
        }
    ]
    file_path = tmp_path / "re_test_data.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

# --- Fixtures for Robustness Testing ---

@pytest.fixture
def invalid_entities_file_re(tmp_path):
    """Creates a .jsonl file where an entity is not a dictionary."""
    data = [
        {"text": "This record has an invalid entity.", "entities": ["not_a_dictionary"]}
    ]
    file_path = tmp_path / "invalid_entity.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data[0]) + '\n')
    return file_path

@pytest.fixture
def invalid_relations_file_re(tmp_path):
    """Creates a .jsonl file where a relation is not a dictionary."""
    data = [
        {
            "text": "This record has an invalid relation.",
            "entities": [{"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6}],
            "relations": ["not_a_dictionary"]
        }
    ]
    file_path = tmp_path / "invalid_relation.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data[0]) + '\n')
    return file_path

# --- Test Cases for REDataset ---

def test_redataset_instance_creation(re_jsonl_file, mock_re_tokenizer, mock_re_config):
    """
    Tests that REDataset creates the correct number of instances (entity pairs).
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    dataset = REDataset(file_path=str(re_jsonl_file), tokenizer=mock_re_tokenizer, relation_map=relation_map)
    
    # Record 1 has 3 entities -> 3 * 2 = 6 instances (all relations are in map).
    # Record 2 has 2 entities -> 2 * 1 = 2 pairs.
    # The pair (10, 11) has type "ignored_relation", which is not in the map, so it's skipped.
    # The pair (11, 10) gets "No_Relation", which is in the map, so it's created.
    # Total expected instances = 6 + 1 = 7.
    assert len(dataset.instances) == 7

def test_redataset_relation_assignment(re_jsonl_file, mock_re_tokenizer, mock_re_config):
    """
    Tests that relations are correctly assigned, including "No_Relation",
    and that pairs with unmapped relation types are filtered out.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    dataset = REDataset(file_path=str(re_jsonl_file), tokenizer=mock_re_tokenizer, relation_map=relation_map)

    # Find the instance for the (1, 2) pair from the data
    instance_ubicar = next(inst for inst in dataset.instances if inst['head']['id'] == 1 and inst['tail']['id'] == 2)
    assert instance_ubicar['relation'] == "ubicar"
    
    # Find the instance for the (3, 1) pair, which has no defined relation
    instance_no_relation = next(inst for inst in dataset.instances if inst['head']['id'] == 3 and inst['tail']['id'] == 1)
    assert instance_no_relation['relation'] == "No_Relation"
    
    # Verify that the pair (10, 11) with the unmapped relation type ("ignored_relation") was NOT created.
    instance_ignored_exists = any(inst for inst in dataset.instances if inst['head']['id'] == 10 and inst['tail']['id'] == 11)
    assert not instance_ignored_exists

def test_redataset_getitem_structure_and_types(re_jsonl_file, mock_re_tokenizer, mock_re_config):
    """
    Tests that __getitem__ returns the correct structure and tensor types.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    dataset = REDataset(file_path=str(re_jsonl_file), tokenizer=mock_re_tokenizer, relation_map=relation_map)
    
    item = dataset[0]
    
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert item['input_ids'].shape == (512,)
    assert isinstance(item['label'], torch.Tensor)


def test_redataset_entity_marker_insertion(mock_re_tokenizer, mock_re_config):
    """
    Tests that entity markers are inserted correctly into the text.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    # We don't need a file for this, we can manually create the dataset with one instance
    dataset = REDataset.__new__(REDataset)
    dataset.tokenizer = mock_re_tokenizer
    dataset.relation_map = relation_map

    # Case 1: Head entity appears first
    dataset.instances = [{
        "text": "Nódulo periareolar derecho bien delimitado.",
        "head": {"start_offset": 0, "end_offset": 6},   # "Nódulo"
        "tail": {"start_offset": 7, "end_offset": 26},  # "periareolar derecho"
        "relation": "ubicar"
    }]
    
    _ = dataset[0] # Trigger __getitem__
    
    call_args = dataset.tokenizer.call_args[0]
    marked_text = call_args[0]
    expected_text = "[E1_START]Nódulo[E1_END] [E2_START]periareolar derecho[E2_END] bien delimitado."
    assert marked_text == expected_text

    # Case 2: Tail entity appears first
    dataset.instances = [{
        "text": "Nódulo periareolar derecho bien delimitado.",
        "head": {"start_offset": 7, "end_offset": 26},
        "tail": {"start_offset": 0, "end_offset": 6},
        "relation": "ubicar"
    }]
    
    _ = dataset[0]
    
    call_args = dataset.tokenizer.call_args[0]
    marked_text = call_args[0]
    expected_text_reversed = "[E2_START]Nódulo[E2_END] [E1_START]periareolar derecho[E1_END] bien delimitado."
    assert marked_text == expected_text_reversed


# --- Test Cases for REDataModule ---

def test_redatamodule_initialization_and_special_tokens(mock_re_config, mock_re_tokenizer):
    """
    Tests that the REDataModule initializes correctly and adds special tokens.
    """
    datamodule = REDataModule(config=mock_re_config)
    
    assert datamodule.tokenizer is mock_re_tokenizer
    mock_re_tokenizer.add_special_tokens.assert_called_once()
    
    # Check if the relation map was created correctly
    assert "describir" in datamodule.relation_map
    assert "No_Relation" in datamodule.relation_map
    assert datamodule.relation_map['ubicar'] == 1


def test_redatamodule_setup_and_dataloader_creation(mock_re_config, mock_re_tokenizer, re_jsonl_file):
    """
    Tests that the setup method creates datasets and dataloaders correctly.
    """
    datamodule = REDataModule(config=mock_re_config, train_file=str(re_jsonl_file), test_file=str(re_jsonl_file))
    datamodule.setup()
    
    assert isinstance(datamodule.train_dataset, REDataset)
    assert len(datamodule.train_dataset) > 0

    train_dl = datamodule.train_dataloader()
    test_dl = datamodule.test_dataloader()

    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert train_dl.batch_size == mock_re_config['trainer']['batch_size']
    assert next(iter(train_dl)) is not None

# --- Tests for Input Data Robustness ---

def test_redataset_with_invalid_entities_type(invalid_entities_file_re, mock_re_tokenizer, mock_re_config):
    """
    Tests that a TypeError is raised if an entity is not a dictionary.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    with pytest.raises(TypeError, match="Entities must be dictionaries"):
        _ = REDataset(file_path=str(invalid_entities_file_re), tokenizer=mock_re_tokenizer, relation_map=relation_map)

def test_redataset_with_invalid_relations_type(invalid_relations_file_re, mock_re_tokenizer, mock_re_config):
    """
    Tests that a TypeError is raised if a relation is not a dictionary.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}
    with pytest.raises(TypeError, match="Relations must be dictionaries"):
        _ = REDataset(file_path=str(invalid_relations_file_re), tokenizer=mock_re_tokenizer, relation_map=relation_map)


def test_redataset_warns_for_unmapped_relation(re_jsonl_file, mock_re_tokenizer, mock_re_config):
    """
    Tests that a warning is issued for relation types not found in the config.
    """
    relation_map = {label: i for i, label in enumerate(mock_re_config['model']['relation_labels'])}

    # The re_jsonl_file contains a record with "ignored_relation", which is not in the map.
    # We expect the dataset initialization to raise a UserWarning.
    with pytest.warns(UserWarning, match="Relation type 'ignored_relation' not in config and will be ignored."):
        _ = REDataset(file_path=str(re_jsonl_file), tokenizer=mock_re_tokenizer, relation_map=relation_map)