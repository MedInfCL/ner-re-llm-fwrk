# tests/unit/scripts/evaluation/test_decoder.py
import pytest
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.generate_finetuned_predictions import decode_entities_from_tokens

# --- Fixtures ---

@pytest.fixture(scope="module")
def tokenizer():
    """Provides a real, lightweight tokenizer for all tests in this module."""
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

@pytest.fixture
def label_maps():
    """Provides label-to-ID and ID-to-label mappings."""
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2, "B-LOC": 3, "I-LOC": 4}
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

# --- Helper Function to find token indices robustly ---

def find_token_indices_for_span(offset_mapping, char_start, char_end):
    """Finds token indices that overlap with a given character span."""
    token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0: # Skip special tokens
            continue
        if max(start, char_start) < min(end, char_end):
            token_indices.append(i)
    return token_indices

# --- Test Cases for Decoder ---

def test_decode_simple_entity(tokenizer, label_maps):
    """Tests decoding a single entity."""
    text = "There is a nodule."
    label_map, inv_label_map = label_maps
    
    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    # Find token indices for "nodule" (char span 11-17)
    nodule_indices = find_token_indices_for_span(offset_mapping, 11, 17)
    
    predictions = [0] * len(offset_mapping)
    if nodule_indices:
        predictions[nodule_indices[0]] = label_map["B-FIND"]
        for i in nodule_indices[1:]:
            predictions[i] = label_map["I-FIND"]
    
    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert result == [{"text": "nodule", "label": "FIND", "start_offset": 11, "end_offset": 17}]

def test_decode_no_entities(tokenizer, label_maps):
    """Tests that a sequence of all 'O' labels returns an empty list."""
    text = "There is nothing here."
    _, inv_label_map = label_maps
    predictions = [0, 0, 0, 0, 0]
    
    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert result == []

def test_decode_multi_token_entity(tokenizer, label_maps):
    """Tests decoding an entity that spans multiple full tokens."""
    text = "Nodule in the left breast."
    label_map, inv_label_map = label_maps

    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    loc_indices = find_token_indices_for_span(offset_mapping, 14, 25) # "left breast"
    
    predictions = [0] * len(offset_mapping)
    if loc_indices:
        predictions[loc_indices[0]] = label_map["B-LOC"]
        for i in loc_indices[1:]:
            predictions[i] = label_map["I-LOC"]
    
    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert result == [{"text": "left breast", "label": "LOC", "start_offset": 14, "end_offset": 25}]

def test_decode_subword_entity(tokenizer, label_maps):
    """Tests decoding an entity that is split into subword tokens."""
    text = "Review for microcalcifications."
    label_map, inv_label_map = label_maps

    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    find_indices = find_token_indices_for_span(offset_mapping, 11, 30) # "microcalcifications"
        
    predictions = [0] * len(offset_mapping)
    if find_indices:
        predictions[find_indices[0]] = label_map["B-FIND"]
        for i in find_indices[1:]:
            predictions[i] = label_map["I-FIND"]
        
    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert result == [{"text": "microcalcifications", "label": "FIND", "start_offset": 11, "end_offset": 30}]

def test_decode_multiple_disjoint_entities(tokenizer, label_maps):
    """Tests decoding multiple separate entities in the same text."""
    text = "A nodule and a mass were found."
    label_map, inv_label_map = label_maps
    
    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    nodule_indices = find_token_indices_for_span(offset_mapping, 2, 8)
    mass_indices = find_token_indices_for_span(offset_mapping, 15, 19)
    
    predictions = [0] * len(offset_mapping)
    if nodule_indices:
        predictions[nodule_indices[0]] = label_map["B-FIND"]
        for i in nodule_indices[1:]:
            predictions[i] = label_map["I-FIND"]
    if mass_indices:
        predictions[mass_indices[0]] = label_map["B-FIND"]
        for i in mass_indices[1:]:
            predictions[i] = label_map["I-FIND"]

    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert len(result) == 2
    assert {"text": "nodule", "label": "FIND", "start_offset": 2, "end_offset": 8} in result
    assert {"text": "mass", "label": "FIND", "start_offset": 15, "end_offset": 19} in result

def test_decode_consecutive_entities(tokenizer, label_maps):
    """Tests decoding two different entities that are right next to each other."""
    text = "A dense finding was seen."
    label_map, inv_label_map = label_maps
    
    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']

    dense_indices = find_token_indices_for_span(offset_mapping, 2, 7) # dense
    finding_indices = find_token_indices_for_span(offset_mapping, 8, 15) # finding

    predictions = [0] * len(offset_mapping)
    if dense_indices:
        predictions[dense_indices[0]] = label_map["B-LOC"]
    if finding_indices:
        predictions[finding_indices[0]] = label_map["B-FIND"]

    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert len(result) == 2
    assert {"text": "dense", "label": "LOC", "start_offset": 2, "end_offset": 7} in result
    assert {"text": "finding", "label": "FIND", "start_offset": 8, "end_offset": 15} in result

def test_decode_handles_invalid_i_tag(tokenizer, label_maps):
    """Tests that an I-tag not following a B-tag or I-tag of the same type is ignored."""
    text = "The left breast is dense."
    label_map, inv_label_map = label_maps
    
    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    left_indices = find_token_indices_for_span(offset_mapping, 4, 8) # "left"
    breast_indices = find_token_indices_for_span(offset_mapping, 9, 15) # "breast"
    
    predictions = [0] * len(offset_mapping)
    # Correct B-LOC, but invalid I-FIND
    if left_indices:
        predictions[left_indices[0]] = label_map["B-LOC"]
    if breast_indices:
        predictions[breast_indices[0]] = label_map["I-FIND"] # Invalid I-tag

    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)

    # Only the valid 'left' entity should be decoded
    assert result == [{"text": "left", "label": "LOC", "start_offset": 4, "end_offset": 8}]

def test_decode_entity_at_end_of_text(tokenizer, label_maps):
    """Tests that an entity right at the end of the input is correctly captured."""
    text = "We saw a nodule"
    label_map, inv_label_map = label_maps
    
    tokenization = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = tokenization['offset_mapping']
    
    nodule_indices = find_token_indices_for_span(offset_mapping, 9, 15) # "nodule"
    
    predictions = [0] * len(offset_mapping)
    if nodule_indices:
        predictions[nodule_indices[0]] = label_map["B-FIND"]
        for i in nodule_indices[1:]:
            predictions[i] = label_map["I-FIND"]
    
    result = decode_entities_from_tokens(text, predictions, inv_label_map, tokenizer)
    
    assert result == [{"text": "nodule", "label": "FIND", "start_offset": 9, "end_offset": 15}]