# tests/unit/scripts/evaluation/test_calculate_final_metrics.py
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.calculate_final_metrics import calculate_ner_metrics, calculate_re_metrics, _calculate_iou, calculate_ner_metrics_relaxed, aggregate_metrics


# --- Fixtures for Testing ---

@pytest.fixture
def mock_unified_predictions():
    """
    Provides mock prediction data in the new, unified format.
    This format is used by both RAG and fine-tuned pipelines.
    """
    return [
        {
            "source_text": "Nódulo en mama derecha.",
            "true_entities": [
                {"text": "Nódulo", "label": "FIND", "start_offset": 0, "end_offset": 6},
                {"text": "mama derecha", "label": "REG", "start_offset": 10, "end_offset": 22}
            ],
            "predicted_entities": [
                {"text": "Nódulo", "label": "FIND", "start_offset": 0, "end_offset": 6},       # True Positive (FIND)
                {"text": "mama izquierda", "label": "REG", "start_offset": 10, "end_offset": 23}  # False Positive (REG) - different boundaries
            ]
            # This setup also implies one FN for "mama derecha" (REG)
        },
        {
            "source_text": "Sin hallazgos de jerarquía.",
            "true_entities": [], # No true entities in this record
            "predicted_entities": [
                {"text": "hallazgos de jerarquía", "label": "FIND", "start_offset": 4, "end_offset": 26} # False Positive (FIND)
            ]
        }
    ]

# --- Test Cases ---

def test_calculate_ner_metrics(mock_unified_predictions):
    """
    Tests the unified metric calculation for NER outputs.
    It verifies that the set-based comparison correctly calculates TP, FP, and FN
    to produce the final precision, recall, and F1-score.
    """
    # --- Act ---
    report = calculate_ner_metrics(mock_unified_predictions)

    # --- Assertions ---
    assert "FIND" in report
    assert "REG" in report
    assert "micro avg" in report
    assert "weighted avg" in report

    # Based on the mock data:
    # FIND: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3, Support=1
    assert report["FIND"]["precision"] == 0.5
    assert report["FIND"]["recall"] == 1.0
    assert report["FIND"]["f1-score"] == pytest.approx(2/3)
    assert report["FIND"]["support"] == 1
    
    # REG: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0, Support=1
    assert report["REG"]["precision"] == 0.0
    assert report["REG"]["recall"] == 0.0
    assert report["REG"]["f1-score"] == 0.0
    assert report["REG"]["support"] == 1

    # Micro average: Total TP=1, Total FP=2, Total FN=1 -> P=1/3, R=0.5, F1=0.4, Support=2
    assert report["micro avg"]["precision"] == pytest.approx(1/3)
    assert report["micro avg"]["recall"] == 0.5
    assert report["micro avg"]["f1-score"] == 0.4
    assert report["micro avg"]["support"] == 2
    
    # Weighted average F1 = (F1_FIND * Sup_FIND + F1_REG * Sup_REG) / Total_Sup
    #                    = ((2/3 * 1) + (0.0 * 1)) / 2 = 1/3
    assert report["weighted avg"]["f1-score"] == pytest.approx(1/3)

def test_calculate_ner_metrics_empty_predictions():
    """
    Tests that the function returns an empty report when given an empty list of predictions.
    """
    report = calculate_ner_metrics([])
    assert report == {}

def test_calculate_ner_metrics_perfect_match():
    """
    Tests the metrics calculation for a perfect match scenario.
    """
    predictions = [
        {
            "true_entities": [{"text": "nodule", "label": "FIND", "start_offset": 0, "end_offset": 6}],
            "predicted_entities": [{"text": "nodule", "label": "FIND", "start_offset": 0, "end_offset": 6}]
        }
    ]
    report = calculate_ner_metrics(predictions)
    assert report["FIND"]["precision"] == 1.0
    assert report["FIND"]["recall"] == 1.0
    assert report["FIND"]["f1-score"] == 1.0
    assert report["weighted avg"]["f1-score"] == 1.0

def test_calculate_ner_metrics_no_match():
    """
    Tests the metrics calculation when there are no matches.
    """
    predictions = [
        {
            "true_entities": [{"text": "nodule", "label": "FIND", "start_offset": 0, "end_offset": 6}],
            "predicted_entities": [{"text": "nodule", "label": "FIND", "start_offset": 10, "end_offset": 16}] # Different offsets
        }
    ]
    report = calculate_ner_metrics(predictions)
    assert report["FIND"]["precision"] == 0.0
    assert report["FIND"]["recall"] == 0.0
    assert report["FIND"]["f1-score"] == 0.0
    assert report["weighted avg"]["f1-score"] == 0.0

# --- Fixtures for RE Testing ---

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

def test_calculate_re_metrics(mock_re_predictions):
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



# --- Tests for _calculate_iou ---

@pytest.mark.parametrize("box1, box2, expected_iou", [
    ((0, 10), (0, 10), 1.0),       # Perfect overlap
    ((0, 10), (5, 15), 1/3),       # Partial overlap (IoU is 5 / 15 = 1/3)
    ((0, 5), (5, 10), 0.0),        # Touching, but no overlap
    ((0, 5), (6, 10), 0.0),        # No overlap
    ((0, 10), (2, 8), 0.6),        # Contained within
])
def test_calculate_iou(box1, box2, expected_iou):
    """Tests the IoU calculation with various overlap scenarios."""
    iou = _calculate_iou(box1, box2)
    assert iou == pytest.approx(expected_iou, rel=1e-4)

# --- Fixtures for Relaxed NER Testing ---

@pytest.fixture
def mock_relaxed_ner_predictions():
    """Provides mock data for relaxed NER metric calculation."""
    return [
        {
            # Scenario 1: Partial overlap, should be a TP with low threshold
            "true_entities": [{"label": "FIND", "start_offset": 10, "end_offset": 20}], # "a finding"
            "predicted_entities": [{"label": "FIND", "start_offset": 12, "end_offset": 22}]
        },
        {
            # Scenario 2: Label grouping, should be a TP
            "true_entities": [{"label": "HALL_presente", "start_offset": 5, "end_offset": 15}],
            "predicted_entities": [{"label": "HALL", "start_offset": 5, "end_offset": 15}]
        },
        {
            # Scenario 3: Not enough overlap, should be FP/FN
            "true_entities": [{"label": "LOC", "start_offset": 0, "end_offset": 10}],
            "predicted_entities": [{"label": "LOC", "start_offset": 8, "end_offset": 18}] # IoU = 2/18 = 0.111
        }
    ]

# --- Tests for calculate_ner_metrics_relaxed ---

def test_calculate_ner_metrics_relaxed_iou_threshold(mock_relaxed_ner_predictions):
    """
    Tests that the IoU threshold correctly determines matches in relaxed evaluation.
    """
    # Low threshold: The partial overlap in scenario 1 (IoU=0.66) should be a TP.
    # The low overlap in scenario 3 (IoU=0.11) is still a miss.
    report_low_thresh = calculate_ner_metrics_relaxed(
        predictions=mock_relaxed_ner_predictions,
        label_groups={},
        overlap_threshold=0.5
    )
    assert report_low_thresh["FIND"]["f1-score"] == 1.0
    assert report_low_thresh["LOC"]["f1-score"] == 0.0

    # High threshold: The partial overlap in scenario 1 (IoU=0.66) is now not enough.
    report_high_thresh = calculate_ner_metrics_relaxed(
        predictions=mock_relaxed_ner_predictions,
        label_groups={},
        overlap_threshold=0.7
    )
    assert report_high_thresh["FIND"]["f1-score"] == 0.0
    assert report_high_thresh["LOC"]["f1-score"] == 0.0


def test_calculate_ner_metrics_relaxed_label_grouping(mock_relaxed_ner_predictions):
    """
    Tests that label grouping correctly merges synonymous labels.
    """
    label_groups = {
        "FINDING_GROUP": ["FIND", "HALL_presente", "HALL"]
    }
    report = calculate_ner_metrics_relaxed(
        predictions=mock_relaxed_ner_predictions,
        label_groups=label_groups,
        overlap_threshold=0.5
    )

    # Both "FIND" (from scenario 1) and "HALL_presente"/"HALL" (from scenario 2)
    # should be counted as True Positives for the "FINDING_GROUP".
    assert "FINDING_GROUP" in report
    assert report["FINDING_GROUP"]["f1-score"] == 1.0
    assert report["FINDING_GROUP"]["support"] == 2 # 2 true entities belong to this group


# --- Fixture for Aggregation Testing ---

@pytest.fixture
def mock_metric_reports():
    """Provides a list of mock classification reports for aggregation testing."""
    return [
        {
            "weighted avg": {
                "precision": 0.8,
                "recall": 0.9,
                "f1-score": 0.85
            }
        },
        {
            "weighted avg": {
                "precision": 0.9,
                "recall": 0.95,
                "f1-score": 0.92
            }
        },
        {
            "weighted avg": {
                "precision": 0.7,
                "recall": 0.85,
                "f1-score": 0.77
            }
        }
    ]

# --- Test for aggregate_metrics ---

def test_aggregate_metrics(mock_metric_reports):
    """
    Tests that the aggregate_metrics function correctly calculates the mean and
    standard deviation from a list of reports.
    """
    summary = aggregate_metrics(mock_metric_reports)

    # Expected Means
    # Precision: (0.8 + 0.9 + 0.7) / 3 = 0.8
    # Recall: (0.9 + 0.95 + 0.85) / 3 = 0.9
    # F1-score: (0.85 + 0.92 + 0.77) / 3 = 0.8466...
    assert summary["precision"]["mean"] == pytest.approx(0.8)
    assert summary["recall"]["mean"] == pytest.approx(0.9)
    assert summary["f1-score"]["mean"] == pytest.approx(0.846666)

    # Expected Standard Deviations (using numpy.std)
    # np.std([0.8, 0.9, 0.7]) = 0.0816...
    # np.std([0.9, 0.95, 0.85]) = 0.0408...
    # np.std([0.85, 0.92, 0.77]) = 0.0612...
    assert summary["precision"]["std"] == pytest.approx(0.081649, rel=1e-4)
    assert summary["recall"]["std"] == pytest.approx(0.040824, rel=1e-4)
    assert summary["f1-score"]["std"] == pytest.approx(0.061282, rel=1e-4)


def test_calculate_ner_metrics_with_valid_labels(mock_unified_predictions):
    """
    Tests that the strict NER metrics calculation correctly filters entities
    based on the valid_labels set.
    """
    # Calculate metrics for the "FIND" label only.
    report = calculate_ner_metrics(mock_unified_predictions, valid_labels={"FIND"})

    # Assert that "REG" is NOT in the report.
    assert "REG" not in report
    
    # Assert that "FIND" is present and its metrics are calculated as before.
    # The calculations should ignore all "REG" entities.
    # FIND: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3, Support=1
    assert report["FIND"]["precision"] == 0.5
    assert report["FIND"]["recall"] == 1.0
    assert report["FIND"]["f1-score"] == pytest.approx(2/3)

    # The weighted avg should now only reflect the metrics for "FIND".
    assert report["weighted avg"]["f1-score"] == pytest.approx(2/3)

    
def test_calculate_re_metrics_with_valid_labels(mock_re_predictions):
    """
    Tests that the RE metrics calculation correctly filters relations
    based on the valid_labels set.
    """
    # Calculate metrics for the "ubicar" relation type only.
    report = calculate_re_metrics(mock_re_predictions, valid_labels={"ubicar"})

    # Assert that "describir" is NOT in the report.
    assert "describir" not in report
    
    # Assert that "ubicar" is present and its metrics are correct.
    # ubicar: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    assert report["ubicar"]["precision"] == 1.0
    assert report["ubicar"]["recall"] == 1.0
    assert report["ubicar"]["f1-score"] == 1.0

    # The weighted average should now only reflect the metrics for "ubicar".
    assert report["weighted avg"]["f1-score"] == 1.0


def test_calculate_ner_metrics_relaxed_with_valid_labels(mock_relaxed_ner_predictions):
    """
    Tests that the relaxed NER metrics calculation correctly filters entities
    based on the valid_labels set, even when label groups are present.
    """
    label_groups = {
        "FINDING_GROUP": ["FIND", "HALL_presente", "HALL"]
    }
    
    # Calculate metrics, but only validate the "LOC" label.
    # The "FINDING_GROUP" should be excluded from the final report.
    report = calculate_ner_metrics_relaxed(
        predictions=mock_relaxed_ner_predictions,
        label_groups=label_groups,
        overlap_threshold=0.5,
        valid_labels={"LOC"}
    )

    # Assert that the finding group is NOT in the report.
    assert "FINDING_GROUP" not in report
    
    # Assert that "LOC" is present, and its metrics are correct (0 matches).
    assert "LOC" in report
    assert report["LOC"]["f1-score"] == 0.0
    assert report["LOC"]["support"] == 1