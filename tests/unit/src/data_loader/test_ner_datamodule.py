import pytest
import torch
import json
import warnings
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule, NERDataset

# Mock configuration for testing
@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for the NERDataModule."""
    return {
        'model': {
            'base_model': 'mock-model',
            'entity_labels': ["FIND", "REG"]
        },
        'trainer': {
            'batch_size': 2
        },
        'batch_size': 2 # For test_dataloader
    }

# Fixture to create a temporary JSONL file for testing
@pytest.fixture
def jsonl_file(tmp_path):
    """
    Creates a temporary .jsonl file with sample data for tests.
    """
    data = [
        {"text": "Report one with a finding.", "entities": [{"label": "FIND", "start_offset": 18, "end_offset": 25}]},
        {"text": "Report two, nothing to see here.", "entities": []},
        {"text": "Report three with an unknown entity.", "entities": [{"label": "UNKNOWN", "start_offset": 19, "end_offset": 33}]}
    ]
    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

# --- Fixture for Real Tokenizer ---

@pytest.fixture(scope="module")
def real_tokenizer():
    """Provides a real tokenizer instance for more accurate offset testing."""
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

# --- Tests for _align_labels ---

def test_align_labels_multi_token_entity(real_tokenizer):
    """
    Tests correct BIO labeling for an entity that spans multiple whole tokens.
    Example: "left breast" -> B-LOC, I-LOC
    """
    text = "Nodule in the left breast."
    entities = [{"label": "LOC", "start_offset": 15, "end_offset": 27}] # "left breast"
    label_map = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    
    tokenization = real_tokenizer(text, return_offsets_mapping=True)
    offset_mapping = torch.tensor(tokenization['offset_mapping'])
    
    dataset = NERDataset.__new__(NERDataset)
    dataset.label_map = label_map
    dataset.warned_entities = set()
    
    labels = dataset._align_labels(offset_mapping, entities)
    
    # Robustly find token indices using character offsets
    entity_token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        if max(start, 15) < min(end, 27): # Check for overlap with "left breast"
            entity_token_indices.append(i)
            
    assert len(entity_token_indices) >= 2, "Expected 'left breast' to be at least two tokens"
    assert labels[entity_token_indices[0]] == label_map["B-LOC"]
    assert labels[entity_token_indices[1]] == label_map["I-LOC"]

def test_align_labels_subword_entity(real_tokenizer):
    """
    Tests correct BIO labeling for a word that is broken into subword tokens.
    All sub-tokens should be correctly labeled B-FIND, I-FIND, ...
    """
    text = "Review for microcalcifications."
    entities = [{"label": "FIND", "start_offset": 11, "end_offset": 30}] # "microcalcifications"
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}

    tokenization = real_tokenizer(text, return_offsets_mapping=True)
    offset_mapping = torch.tensor(tokenization['offset_mapping'])
    
    dataset = NERDataset.__new__(NERDataset)
    dataset.label_map = label_map
    dataset.warned_entities = set()
    
    labels = dataset._align_labels(offset_mapping, entities)
    
    # Find all tokens that fall within the entity's character span
    entity_token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        if max(start, 11) < min(end, 30):
            entity_token_indices.append(i)

    assert len(entity_token_indices) > 1, "Expected 'microcalcifications' to be split into subwords"
    assert labels[entity_token_indices[0]] == label_map["B-FIND"]
    for idx in entity_token_indices[1:]:
        assert labels[idx] == label_map["I-FIND"]

def test_align_labels_consecutive_entities(real_tokenizer):
    """
    Tests correct labeling for two distinct entities, handling potential subword tokenization.
    """
    text = "A nodule in the left breast."
    entities = [
        {"label": "FIND", "start_offset": 2, "end_offset": 8},   # "nodule"
        {"label": "LOC", "start_offset": 16, "end_offset": 28} # "left breast."
    ]
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2, "B-LOC": 3, "I-LOC": 4}

    tokenization = real_tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    dataset = NERDataset.__new__(NERDataset)
    dataset.label_map = label_map
    dataset.warned_entities = set()
    
    labels = dataset._align_labels(torch.tensor(offset_mapping), entities)

    # --- Robustly find token indices using character offsets ---
    nodule_char_start, nodule_char_end = 2, 8
    loc_char_start, loc_char_end = 16, 28

    nodule_indices = []
    left_breast_indices = []

    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0: # Skip special tokens
            continue
        if max(start, nodule_char_start) < min(end, nodule_char_end):
            nodule_indices.append(i)
        if max(start, loc_char_start) < min(end, loc_char_end):
            left_breast_indices.append(i)

    # --- Assertions ---
    # Assert that the entity was found and check the BIO scheme
    assert len(nodule_indices) > 0, "Expected 'nodule' to map to at least one token"
    assert labels[nodule_indices[0]] == label_map["B-FIND"]
    for idx in nodule_indices[1:]:
        assert labels[idx] == label_map["I-FIND"]

    # Assert that the second entity was found and check the BIO scheme
    assert len(left_breast_indices) > 0, "Expected 'left breast' to map to at least one token"
    assert labels[left_breast_indices[0]] == label_map["B-LOC"]
    for idx in left_breast_indices[1:]:
        assert labels[idx] == label_map["I-LOC"]


@pytest.fixture
def mock_tokenizer():
    """
    Creates a mock tokenizer that simulates returning tokenization output
    including the character-to-token offset mapping with consistent tensor shapes.
    """
    tokenizer = MagicMock()
    
    # Add the required token ID attributes to the mock object
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.pad_token_id = 0

    # The tokenizer is called once in `__getitem__` and returns a dictionary.
    # We now create a realistic input_ids tensor.
    tokenization_result = {
        # --- Start of Change ---
        "input_ids": torch.tensor([[
            tokenizer.cls_token_id, # [CLS]
            500, 600, 700, 800, 900, 1000, # Mock word IDs
            tokenizer.sep_token_id  # [SEP]
        ]]),
        # --- End of Change ---
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
        "offset_mapping": torch.tensor([
            (0, 0),      # [CLS]
            (0, 6),      # "Report"
            (7, 10),     # "one"
            (11, 15),    # "with"
            (16, 17),    # "a"
            (18, 25),    # "finding"
            (25, 26),    # "."
            (0, 0)       # [SEP]
        ]).unsqueeze(0) # Add batch dimension
    }
    
    tokenizer.return_value = tokenization_result
    
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
        yield tokenizer




# --- Test Cases for NERDataset ---

def test_dataset_loading(jsonl_file, mock_tokenizer):
    """
    Tests that the NERDataset correctly loads data from a .jsonl file
    and reports the correct length.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    assert len(dataset) == 3
    assert dataset.data[0]['text'] == "Report one with a finding."

def test_getitem_returns_correct_structure(jsonl_file, mock_tokenizer):
    """
    Tests that the __getitem__ method returns a dictionary with the correct keys
    and that the output tensor shapes are consistent.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    # Get a single processed item from the dataset.
    item = dataset[0]
    
    # --- Assertions ---
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    
    # The shape of the labels tensor must match the shape of the input_ids tensor.
    # This makes the test robust to changes in the mock tokenizer's output size.
    assert item['labels'].shape == item['input_ids'].shape


def test_align_labels_correctly(jsonl_file, mock_tokenizer):
    """
    Tests that entity labels are correctly aligned with token indices
    using the offset_mapping.
    """
    # Define the mapping from labels to IDs for this test.
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2} 
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    # The first record in the test file has one entity:
    # {"label": "FIND", "start_offset": 18, "end_offset": 25} -> "finding"
    entities = dataset.data[0]["entities"]
    
    # Get the mock offset mapping from the tokenizer fixture.
    # The .squeeze() removes the batch dimension for testing.
    offset_mapping = mock_tokenizer().get("offset_mapping").squeeze()

    # Call the alignment method directly to test its logic.
    labels = dataset._align_labels(offset_mapping, entities)

    # --- Assertions ---
    # The expected labels tensor based on the mock offset_mapping:
    # Token 5 corresponds to "finding" (offsets 18-25).
    # [CLS] and [SEP] tokens should be ignored (-100, which becomes 0).
    #
    # Offsets:  (0,0) (0,6) (7,10) (11,15) (16,17) (18,25) (25,26) (0,0)
    # Tokens:   [CLS] Rprt  one    with    a       finding  .      [SEP]
    # Labels:   O     O     O      O       O       B-FIND   O      O
    # IDs:      0     0     0      0       0       1        0      0
    # The special tokens ([CLS] and [SEP]) at the beginning and end should
    # now be labeled -100 to be ignored by the loss function.
    expected_labels = torch.tensor([-100, 0, 0, 0, 0, 1, 0, -100], dtype=torch.long)

    assert torch.equal(labels, expected_labels)


def test_align_labels_with_unknown_entity(jsonl_file, mock_tokenizer):
    """
    Tests that an entity with a label not in the label_map is ignored and
    a warning is issued.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Access the item with the unknown entity to trigger the warning
        _ = dataset[2] 
        assert len(w) == 1
        assert "Entity label 'UNKNOWN' not found" in str(w[0].message)

def test_align_labels_with_no_entities(jsonl_file, mock_tokenizer):
    """
    Tests that a record with no entities results in only 'O' labels.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)

    # Get the item with no entities
    item = dataset[1]
    labels = item['labels']

    # All "real" token labels should be 0 ('O').
    # Special tokens should be -100. We mask the special tokens for the check.
    special_token_mask = (item['input_ids'] == mock_tokenizer.cls_token_id) | \
                         (item['input_ids'] == mock_tokenizer.sep_token_id) | \
                         (item['input_ids'] == mock_tokenizer.pad_token_id)
    
    assert torch.all(labels[~special_token_mask] == 0)


# --- Test Cases for NERDataModule ---

def test_datamodule_initialization(mock_config, mock_tokenizer):
    """
    Tests that NERDataModule initializes the tokenizer and creates the label_map correctly.
    """
    datamodule = NERDataModule(config=mock_config)
    
    assert datamodule.tokenizer is not None
    assert "B-FIND" in datamodule.label_map
    assert "I-REG" in datamodule.label_map
    assert datamodule.label_map["O"] == 0

def test_datamodule_setup(mock_config, mock_tokenizer, jsonl_file):
    """
    Tests that the setup method correctly creates the train and test datasets.
    """
    datamodule = NERDataModule(config=mock_config, train_file=str(jsonl_file), test_file=str(jsonl_file))
    datamodule.setup()
    
    assert isinstance(datamodule.train_dataset, NERDataset)
    assert isinstance(datamodule.test_dataset, NERDataset)
    assert len(datamodule.train_dataset) == 3

def test_datamodule_dataloaders(mock_config, mock_tokenizer, jsonl_file):
    """
    Tests that the train_dataloader and test_dataloader methods return DataLoader instances.
    """
    datamodule = NERDataModule(config=mock_config, train_file=str(jsonl_file), test_file=str(jsonl_file))
    datamodule.setup()
    
    train_dl = datamodule.train_dataloader()
    test_dl = datamodule.test_dataloader()

    assert train_dl is not None
    assert test_dl is not None
    assert train_dl.batch_size == mock_config['trainer']['batch_size']
    assert next(iter(train_dl)) is not None


# --- Fixtures for Robustness Testing ---

@pytest.fixture
def empty_jsonl_file(tmp_path):
    """Creates an empty .jsonl file."""
    file_path = tmp_path / "empty_data.jsonl"
    file_path.touch()
    return file_path

@pytest.fixture
def missing_key_jsonl_file(tmp_path):
    """Creates a .jsonl file where a record is missing the 'text' key."""
    # A .jsonl file has one JSON object per line.
    record = {"entities": []} 
    file_path = tmp_path / "missing_key.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    return file_path

@pytest.fixture
def malformed_jsonl_file(tmp_path):
    """Creates a file containing a line that is not valid JSON."""
    file_path = tmp_path / "malformed.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('{"text": "valid"}\n')
        f.write('this is not json\n')
    return file_path

# --- Tests for Input Data Robustness ---

def test_dataset_with_empty_file(empty_jsonl_file, mock_tokenizer):
    """
    Tests that the dataset handles an empty file gracefully.
    """
    dataset = NERDataset(file_path=str(empty_jsonl_file), tokenizer=mock_tokenizer, label_map={})
    
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]

def test_dataset_with_missing_key(missing_key_jsonl_file, mock_tokenizer):
    """
    Tests that a KeyError is raised if a record is missing the 'text' key.
    """
    dataset = NERDataset(file_path=str(missing_key_jsonl_file), tokenizer=mock_tokenizer, label_map={})
    
    # Accessing the item should trigger the error when it tries to get record["text"]
    with pytest.raises(KeyError):
        _ = dataset[0]

def test_dataset_with_malformed_json(malformed_jsonl_file, mock_tokenizer):
    """
    Tests that a json.JSONDecodeError is raised if the file contains invalid JSON.
    """
    # The error should be raised during the initialization of the dataset
    with pytest.raises(json.JSONDecodeError):
        _ = NERDataset(file_path=str(malformed_jsonl_file), tokenizer=mock_tokenizer, label_map={})


@pytest.fixture
def invalid_entities_file(tmp_path):
    """Creates a .jsonl file with an entity that is not a dictionary."""
    data = [
        {"text": "Report one with a finding.", "entities": ["not_a_dictionary"]}
    ]
    file_path = tmp_path / "invalid_entities.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

@pytest.fixture
def missing_entity_key_file(tmp_path):
    """Creates a .jsonl file with an entity missing a required key."""
    data = [
        {"text": "Report one with a finding.", "entities": [{"start_offset": 18, "end_offset": 25}]}
    ]
    file_path = tmp_path / "missing_entity_key.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

@pytest.fixture
def invalid_offset_type_file(tmp_path):
    """Creates a .jsonl file with a non-integer offset."""
    data = [
        {"text": "Report one with a finding.", "entities": [{"label": "FIND", "start_offset": "18", "end_offset": 25}]}
    ]
    file_path = tmp_path / "invalid_offset_type.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

@pytest.fixture
def inverted_offset_file(tmp_path):
    """Creates a .jsonl file with start_offset > end_offset."""
    data = [
        {"text": "Report one with a finding.", "entities": [{"label": "FIND", "start_offset": 25, "end_offset": 18}]}
    ]
    file_path = tmp_path / "inverted_offset.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path


def test_dataset_with_invalid_entities_type(invalid_entities_file, mock_tokenizer):
    """
    Tests that a TypeError is raised if an entity is not a dictionary.
    """
    dataset = NERDataset(file_path=str(invalid_entities_file), tokenizer=mock_tokenizer, label_map={})

    with pytest.raises(TypeError):
        _ = dataset[0]

def test_dataset_with_missing_entity_key(missing_entity_key_file, mock_tokenizer):
    """
    Tests that a KeyError is raised if an entity is missing a required key.
    """
    dataset = NERDataset(file_path=str(missing_entity_key_file), tokenizer=mock_tokenizer, label_map={})

    with pytest.raises(KeyError):
        _ = dataset[0]

def test_dataset_with_invalid_offset_type(invalid_offset_type_file, mock_tokenizer):
    """
    Tests that a TypeError is raised if an offset is not an integer.
    """
    # Provide a label_map that includes the entity from the test file
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(invalid_offset_type_file), tokenizer=mock_tokenizer, label_map=label_map)

    with pytest.raises(TypeError):
        _ = dataset[0]

def test_dataset_with_inverted_offsets(inverted_offset_file, mock_tokenizer):
    """
    Tests that no error is raised but the label is not applied if start_offset > end_offset.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(inverted_offset_file), tokenizer=mock_tokenizer, label_map=label_map)

    item = dataset[0]
    labels = item['labels']

    # All "real" token labels should be 0 ('O') since the inverted offsets are ignored.
    # Special tokens should be -100.
    special_token_mask = (item['input_ids'] == mock_tokenizer.cls_token_id) | \
                         (item['input_ids'] == mock_tokenizer.sep_token_id) | \
                         (item['input_ids'] == mock_tokenizer.pad_token_id)

    assert torch.all(labels[~special_token_mask] == 0)