import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.cost_tracker import CostTracker

# --- Fixtures for Testing CostTracker ---

@pytest.fixture
def mock_datetime():
    """Mocks datetime.now() to return a fixed timestamp."""
    fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0)
    with patch('src.utils.cost_tracker.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_timestamp
        yield mock_dt

# --- Test Cases for CostTracker ---

def test_cost_tracker_initialization(tmp_path):
    """
    Tests that the CostTracker initializes with an empty log and creates the log directory.
    """
    log_dir = tmp_path / "test_logs"
    tracker = CostTracker(log_dir=str(log_dir))
    
    assert tracker.requests_log == []
    assert log_dir.exists()

def test_log_request_with_known_model(mock_datetime):
    """
    Tests that log_request correctly calculates the cost for a model in its price list.
    """
    tracker = CostTracker()
    tracker.log_request(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
    
    assert len(tracker.requests_log) == 1
    log_entry = tracker.requests_log[0]
    
    # Expected cost calculation:
    # Input: (1000 / 1,000,000) * $5.00 = $0.005
    # Output: (500 / 1,000,000) * $15.00 = $0.0075
    # Total = $0.0125
    expected_cost = 0.0125
    
    assert log_entry["model"] == "gpt-4o"
    assert log_entry["total_tokens"] == 1500
    assert log_entry["estimated_cost_usd"] == pytest.approx(expected_cost)
    assert log_entry["timestamp"] == mock_datetime.now.return_value.isoformat()

def test_log_request_with_unknown_model():
    """
    Tests that log_request assigns a cost of 0 for a model not in its price list.
    """
    tracker = CostTracker()
    tracker.log_request(model="unknown-model", prompt_tokens=1000, completion_tokens=500)
    
    assert len(tracker.requests_log) == 1
    assert tracker.requests_log[0]["estimated_cost_usd"] == 0.0

def test_get_summary_with_data():
    """
    Tests that get_summary returns the correct aggregated totals.
    """
    tracker = CostTracker()
    tracker.log_request(model="gpt-4o", prompt_tokens=1000, completion_tokens=500) # Cost: 0.0125
    tracker.log_request(model="gpt-3.5-turbo", prompt_tokens=2000, completion_tokens=1000) # Cost: 0.0025
    
    summary = tracker.get_summary()
    
    assert summary["total_requests"] == 2
    assert summary["total_tokens"] == 4500
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.015)

def test_get_summary_when_empty():
    """
    Tests that get_summary returns all zeros when no requests have been logged.
    """
    tracker = CostTracker()
    summary = tracker.get_summary()
    
    assert summary["total_requests"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_estimated_cost_usd"] == 0.0

@patch('pandas.DataFrame.to_csv')
def test_save_log_with_data(mock_to_csv, tmp_path):
    """
    Tests that save_log calls the to_csv method when there are requests in the log.
    """
    log_dir = tmp_path / "test_logs"
    tracker = CostTracker(log_dir=str(log_dir))
    tracker.log_request(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
    
    tracker.save_log()
    
    mock_to_csv.assert_called_once()

@patch('pandas.DataFrame.to_csv')
def test_save_log_when_empty(mock_to_csv, tmp_path):
    """
    Tests that save_log does not call the to_csv method when the log is empty.
    """
    log_dir = tmp_path / "test_logs"
    tracker = CostTracker(log_dir=str(log_dir))
    
    tracker.save_log()
    
    mock_to_csv.assert_not_called()