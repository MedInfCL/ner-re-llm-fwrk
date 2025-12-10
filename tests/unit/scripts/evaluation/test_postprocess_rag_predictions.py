# tests/unit/scripts/evaluation/test_postprocess_rag_predictions.py
import pytest
import json
from unittest.mock import patch, mock_open

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.postprocess_rag_predictions import find_nearest_match, postprocess_predictions, resolve_overlaps

# --- Fixtures for Testing ---

@pytest.fixture
def sample_record_needs_correction():
    """Provides a sample record where the entity offset is incorrect."""
    return {
        "source_text": "Ambas mamas son densas. Se observa un nódulo en la mama derecha.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 27, # Incorrect, points into "observa un"
            "end_offset": 37,
            "text": "nódulo"
        }]
    }

@pytest.fixture
def sample_record_perfect_match():
    """Provides a sample record where the entity offset is already correct."""
    return {
        "source_text": "Ambas mamas son densas. Se observa un nódulo en la mama derecha.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 38, # Correct
            "end_offset": 44,
            "text": "nódulo"
        }]
    }

@pytest.fixture
def sample_record_no_match():
    """Provides a sample record where the entity text is not in the source text."""
    return {
        "source_text": "Ambas mamas son densas.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 10,
            "end_offset": 18,
            "text": "hallazgo" # This word is not in the source_text
        }]
    }

@pytest.fixture
def sample_record_multiple_occurrences():
    """Provides a sample record with multiple occurrences of the same entity text."""
    return {
        "source_text": "Nódulo en mama izquierda. Se revisa el nódulo contralateral.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 5, # Hint points to the first one
            "end_offset": 11,
            "text": "Nódulo"
        }, {
            "label": "FIND",
            "start_offset": 40, # Hint points to the second one
            "end_offset": 46,
            "text": "nódulo"
        }]
    }

# --- Tests for find_nearest_match helper function ---

def test_find_nearest_match_perfect_match():
    """Tests that the correct offsets are returned when the hint is accurate."""
    text = "Se observa un nódulo."
    substring = "nódulo"
    start_hint = 14
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 14
    assert end == 20

def test_find_nearest_match_correction_needed():
    """Tests that the function corrects a slightly inaccurate offset hint."""
    text = "Se observa un nódulo."
    substring = "nódulo"
    start_hint = 10 # Hint is a bit early
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 14
    assert end == 20

def test_find_nearest_match_no_match_found():
    """Tests that None is returned when the substring does not exist in the text."""
    text = "Se observa una asimetría."
    substring = "nódulo"
    start_hint = 14
    start, end = find_nearest_match(substring, text, start_hint)
    assert start is None
    assert end is None

def test_find_nearest_match_selects_closest_occurrence():
    """Tests that the function selects the occurrence nearest to the hint."""
    text = "Un nódulo aquí, y otro nódulo por allá."
    substring = "nódulo"
    
    # Hint is closer to the second occurrence
    start_hint = 20
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 23 # "nódulo por allá"
    assert end == 29

    # Hint is closer to the first occurrence
    start_hint = 5
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 3 # "nódulo aquí"
    assert end == 9

def test_find_nearest_match_with_punctuation():
    """Tests that a match is found even if the substring is adjacent to punctuation."""
    text = "A dense finding, which was expected, was seen."
    substring = "finding"
    start_hint = 9
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 8
    assert end == 15

def test_find_nearest_match_disambiguates_by_hint():
    """Tests that the hint correctly disambiguates between two identical substrings."""
    text = "A finding was seen, and another finding was also noted."
    substring = "finding"

    # Hint is closer to the first "finding"
    hint_1 = 5
    start_1, end_1 = find_nearest_match(substring, text, hint_1)
    assert start_1 == 2
    assert end_1 == 9

    # Hint is closer to the second "finding"
    hint_2 = 30
    start_2, end_2 = find_nearest_match(substring, text, hint_2)
    assert start_2 == 32
    assert end_2 == 39

# --- Tests for postprocess_predictions main function ---

@patch("builtins.open")
def test_postprocess_predictions_corrects_offsets(mock_open_func, sample_record_needs_correction):
    """
    Tests that the main function reads a file, corrects the offsets, and writes the result.
    """
    # Create a mock file handle for reading, pre-filled with the test data.
    read_data = json.dumps(sample_record_needs_correction)
    mock_read_handle = mock_open(read_data=read_data).return_value

    # Create a separate, empty mock file handle for writing.
    mock_write_handle = mock_open().return_value

    # Configure the main 'open' mock to return the read handle on the first call,
    # and the write handle on the second call.
    mock_open_func.side_effect = [mock_read_handle, mock_write_handle]

    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl", allow_entity_overlap=True)

    # Verify the content written to the dedicated write handle
    written_data = mock_write_handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)

    corrected_entity = corrected_record["predicted_entities"][0]
    assert corrected_entity["start_offset"] == 38
    assert corrected_entity["end_offset"] == 44
    assert corrected_entity["text"] == "nódulo"

@patch("builtins.open")
def test_postprocess_predictions_drops_unmatched_entities(mock_open_func, sample_record_no_match):
    """
    Tests that entities with no match in the source text are dropped from the output.
    """
    # Create mock handles for reading and writing
    read_data = json.dumps(sample_record_no_match)
    mock_read_handle = mock_open(read_data=read_data).return_value
    mock_write_handle = mock_open().return_value

    # Configure the 'open' mock to return the correct handle for each call
    mock_open_func.side_effect = [mock_read_handle, mock_write_handle]

    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl", allow_entity_overlap=True)

    # Verify the content written to the dedicated write handle
    written_data = mock_write_handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)
    
    # The list of predicted entities should now be empty
    assert len(corrected_record["predicted_entities"]) == 0

@patch("builtins.open")
def test_postprocess_predictions_handles_malformed_entity(mock_open_func):
    """
    Tests that the function gracefully skips entities with missing or invalid keys.
    """
    malformed_record = {
        "source_text": "Some text.",
        "predicted_entities": [
            {"label": "FIND"}, # Missing text and start_offset
            {"text": "Some text", "start_offset": "invalid"} # Invalid offset type
        ]
    }
    
    # Create mock handles for reading and writing
    read_data = json.dumps(malformed_record)
    mock_read_handle = mock_open(read_data=read_data).return_value
    mock_write_handle = mock_open().return_value

    # Configure the 'open' mock to return the correct handle for each call
    mock_open_func.side_effect = [mock_read_handle, mock_write_handle]

    # The function should run without raising an error
    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl", allow_entity_overlap=True)
    
    # Verify the content written to the dedicated write handle
    written_data = mock_write_handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)
    
    # Both malformed entities should have been dropped
    assert len(corrected_record["predicted_entities"]) == 0


@pytest.fixture
def sample_record_with_overlaps():
    """Provides a sample record with overlapping entities."""
    return {
        "source_text": "Se observa un nódulo espiculado en la mama.",
        "predicted_entities": [
            {
                "label": "FIND",
                "start_offset": 14,
                "end_offset": 32,
                "text": "nódulo espiculado"  # The longer entity
            },
            {
                "label": "FIND",
                "start_offset": 14,
                "end_offset": 20,
                "text": "nódulo"  # The shorter, overlapping entity
            }
        ]
    }

# --- Tests for resolve_overlaps helper function ---

def test_resolve_overlaps_fully_contained():
    """Tests that the shorter, fully contained entity is removed."""
    entities = [
        {'start_offset': 10, 'end_offset': 30, 'text': 'a long entity text'},
        {'start_offset': 12, 'end_offset': 28, 'text': 'a short entity'}
    ]
    resolved = resolve_overlaps(entities)
    assert len(resolved) == 1
    assert resolved[0]['text'] == 'a long entity text'

def test_resolve_overlaps_partial_overlap():
    """Tests that the shorter entity is removed in a partial overlap."""
    entities = [
        {'start_offset': 10, 'end_offset': 25, 'text': 'the first long one'},
        {'start_offset': 20, 'end_offset': 30, 'text': 'the second one'}
    ]
    resolved = resolve_overlaps(entities)
    assert len(resolved) == 1
    assert resolved[0]['text'] == 'the first long one'

def test_resolve_overlaps_no_overlap():
    """Tests that no entities are removed when there is no overlap."""
    entities = [
        {'start_offset': 10, 'end_offset': 20, 'text': 'first'},
        {'start_offset': 21, 'end_offset': 30, 'text': 'second'}
    ]
    resolved = resolve_overlaps(entities)
    assert len(resolved) == 2

def test_resolve_overlaps_mixed_scenario():
    """Tests a mix of overlapping and non-overlapping entities."""
    entities = [
        {'start_offset': 50, 'end_offset': 60, 'text': 'entity C'},
        {'start_offset': 10, 'end_offset': 30, 'text': 'entity A long'},
        {'start_offset': 15, 'end_offset': 25, 'text': 'entity B short'},
    ]
    resolved = resolve_overlaps(entities)
    assert len(resolved) == 2
    assert {"start_offset": 10, "end_offset": 30, "text": "entity A long"} in resolved
    assert {"start_offset": 50, "end_offset": 60, "text": "entity C"} in resolved


@patch("builtins.open")
def test_postprocess_predictions_resolves_overlaps(mock_open_func, sample_record_with_overlaps):
    """
    Tests that the main function correctly resolves overlaps when allow_entity_overlap is False.
    """
    # Create mock handles for reading and writing
    read_data = json.dumps(sample_record_with_overlaps)
    mock_read_handle = mock_open(read_data=read_data).return_value
    mock_write_handle = mock_open().return_value

    # Configure the 'open' mock to return the correct handle for each call
    mock_open_func.side_effect = [mock_read_handle, mock_write_handle]

    # Run postprocessing with overlap resolution enabled
    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl", allow_entity_overlap=False)

    # Verify the content written to the dedicated write handle
    written_data = mock_write_handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)

    # Only the longest, non-overlapping entity should remain
    predicted_entities = corrected_record["predicted_entities"]
    assert len(predicted_entities) == 1
    assert predicted_entities[0]["text"] == "nódulo espiculado"


@pytest.fixture
def sample_record_case_mismatch():
    """Provides a sample record where the entity text has a different case."""
    return {
        "source_text": "Se observa un nódulo en la mama derecha.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 38,
            "end_offset": 44,
            "text": "Nódulo"  # Capitalized, but lowercase in source
        }]
    }

@patch("builtins.open")
def test_postprocess_predictions_handles_case_mismatch(mock_open_func, sample_record_case_mismatch):
    """
    Tests that an entity with a case mismatch is dropped because the default search is case-sensitive.
    """
    # Create mock handles for reading and writing
    read_data = json.dumps(sample_record_case_mismatch)
    mock_read_handle = mock_open(read_data=read_data).return_value
    mock_write_handle = mock_open().return_value

    mock_open_func.side_effect = [mock_read_handle, mock_write_handle]

    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl", allow_entity_overlap=True)

    written_data = mock_write_handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)

    # Since the default .find() is case-sensitive, "Nódulo" will not be found in "nódulo".
    # The entity should be dropped. This test confirms the current behavior.
    assert len(corrected_record["predicted_entities"]) == 0