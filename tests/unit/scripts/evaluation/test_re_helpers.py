# tests/unit/scripts/evaluation/test_re_helpers.py
import pytest
import itertools

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.calculate_final_metrics import calculate_re_metrics
from scripts.evaluation.generate_finetuned_predictions import decode_relations_from_ids

# --- Fixtures for RE Helper Testing ---

@pytest.fixture
def mock_re_predictions():
    """Provides mock prediction data in the unified RE format."""
    return [
        {
            "true_relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"},
                {"from_id": 3, "to_id": 4, "type": "describir"}
            ],
            "predicted_relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"},    # True Positive (ubicar)
                {"from_id": 5, "to_id": 6, "type": "describir"} # False Positive (describir)
            ]
            # This also implies one FN for the (3,4,"describir") relation.
        }
    ]

# --- Tests for calculate_re_metrics ---

def test_calculate_re_metrics_logic(mock_re_predictions):
    """
    Tests the unified metric calculation for RE outputs.
    """
    report = calculate_re_metrics(mock_re_predictions)

    assert "ubicar" in report
    assert "describir" in report
    assert "weighted avg" in report

    # ubicar: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0, Support=1
    assert report["ubicar"]["precision"] == 1.0
    assert report["ubicar"]["recall"] == 1.0
    assert report["ubicar"]["f1-score"] == 1.0
    
    # describir: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0, Support=1
    assert report["describir"]["precision"] == 0.0
    assert report["describir"]["recall"] == 0.0
    assert report["describir"]["f1-score"] == 0.0

    # Weighted Avg F1 = ((1.0 * 1) + (0.0 * 1)) / 2 = 0.5
    assert report["weighted avg"]["f1-score"] == 0.5

# --- Tests for decode_relations_from_ids ---

@pytest.fixture
def re_decoder_setup():
    """Provides a sample record and relation maps for the decoder test."""
    record = {
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
    }
    relation_map = {"ubicar": 0, "describir": 1, "No_Relation": 2}
    inv_relation_map = {v: k for k, v in relation_map.items()}
    return record, relation_map, inv_relation_map

def test_decode_relations_from_ids(re_decoder_setup):
    """
    Tests that the decoder correctly reconstructs relation dictionaries from integer IDs.
    """
    record, relation_map, inv_relation_map = re_decoder_setup
    
    # This flat list corresponds to the 6 permutations of the 3 entities.
    # (1,2)->ubicar, (1,3)->describir, (2,1)->No_Rel, (2,3)->No_Rel, etc.
    predictions = [
        relation_map["ubicar"], 
        relation_map["describir"], 
        relation_map["No_Relation"], 
        relation_map["No_Relation"],
        relation_map["No_Relation"],
        relation_map["No_Relation"]
    ]

    decoded = decode_relations_from_ids(
        record=record,
        predictions=predictions,
        inv_relation_map=inv_relation_map,
        datamodule_relation_map=relation_map
    )

    assert len(decoded) == 2
    assert {"from_id": 1, "to_id": 2, "type": "ubicar"} in decoded
    assert {"from_id": 1, "to_id": 3, "type": "describir"} in decoded