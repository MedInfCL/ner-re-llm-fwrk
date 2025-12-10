import pytest
import torch
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.evaluation.predictor import Predictor

# --- Fixtures for Predictor Testing ---

@pytest.fixture
def mock_device():
    """Provides a mock device object."""
    return torch.device('cpu')

@pytest.fixture
def mock_ner_model():
    """Creates a mock model that simulates a NER model's output."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = None
    
    # NER model output: (batch_size, seq_len, num_labels)
    # The logits should lead to predictable argmax results.
    logits = torch.zeros(2, 5, 4) # batch_size=2, seq_len=5, num_labels=4
    logits[0, 0, 1] = 1 # Sample 1, token 0 -> label 1
    logits[0, 1, 2] = 1 # Sample 1, token 1 -> label 2
    logits[1, 0, 3] = 1 # Sample 2, token 0 -> label 3
    
    model.return_value = MagicMock(logits=logits)
    return model

@pytest.fixture
def mock_re_model():
    """Creates a mock model that simulates a RE model's output."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = None
    
    # RE model output: (batch_size, num_labels)
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]]) # batch_size=2, num_labels=2
    model.return_value = MagicMock(logits=logits)
    return model

@pytest.fixture
def mock_ner_dataloader():
    """Creates a mock DataLoader yielding batches for a NER task."""
    # Batch with padded labels (-100)
    batch = {
        'input_ids': torch.ones((2, 5), dtype=torch.long),
        'attention_mask': torch.ones((2, 5), dtype=torch.long),
        'labels': torch.tensor([
            [1, 2, -100, -100, -100], # True labels for sample 1 (2 are valid)
            [3, 0, 1, -100, -100]    # True labels for sample 2 (3 are valid)
        ], dtype=torch.long),
    }
    loader = MagicMock()
    loader.__iter__.return_value = iter([batch])
    return loader

@pytest.fixture
def mock_re_dataloader():
    """Creates a mock DataLoader yielding batches for a RE task."""
    batch = {
        'input_ids': torch.ones((2, 10), dtype=torch.long),
        'attention_mask': torch.ones((2, 10), dtype=torch.long),
        'label': torch.tensor([1, 0], dtype=torch.long), # Note singular 'label'
    }
    loader = MagicMock()
    loader.__iter__.return_value = iter([batch])
    return loader

@pytest.fixture
def mock_empty_dataloader():
    """Creates a mock DataLoader that is empty."""
    loader = MagicMock()
    loader.__iter__.return_value = iter([])
    return loader


# --- Test Cases for Predictor ---

def test_predictor_initialization(mock_ner_model, mock_device):
    """
    Tests if the Predictor initializes correctly, moves the model to the
    device, and sets it to evaluation mode.
    """
    predictor = Predictor(model=mock_ner_model, device=mock_device)
    
    assert predictor.model is mock_ner_model
    assert predictor.device is mock_device
    mock_ner_model.to.assert_called_once_with(mock_device)
    mock_ner_model.eval.assert_called_once()

@patch('src.evaluation.predictor.tqdm', side_effect=lambda iterable, **kwargs: iterable)
def test_predict_ner_task(mock_tqdm, mock_ner_model, mock_ner_dataloader, mock_device):
    """
    Tests the predict method for a NER task, ensuring correct logic for
    token-level, padded predictions.
    """
    predictor = Predictor(model=mock_ner_model, device=mock_device)
    predictions, true_labels, _ = predictor.predict(mock_ner_dataloader, 'ner')

    # Expected predictions based on mock_ner_model logits
    expected_preds = [
        [1, 2, 0, 0, 0],  # Full, un-filtered prediction for sample 1
        [3, 0, 0, 0, 0]   # Full, un-filtered prediction for sample 2
    ]
    
    # Expected true labels based on mock_ner_dataloader, with -100 filtered out
    expected_true = [
        [1, 2],       # True labels for the 2 valid tokens in sample 1
        [3, 0, 1]     # True labels for the 3 valid tokens in sample 2
    ]

    assert len(predictions) == 2
    assert len(true_labels) == 2
    predictions_as_list = [p.tolist() for p in predictions]
    assert predictions_as_list == expected_preds
    assert true_labels == expected_true

@patch('src.evaluation.predictor.tqdm', side_effect=lambda iterable, **kwargs: iterable)
def test_predict_re_task(mock_tqdm, mock_re_model, mock_re_dataloader, mock_device):
    """
    Tests the predict method for a RE task, ensuring correct logic for
    sequence-level predictions.
    """
    predictor = Predictor(model=mock_re_model, device=mock_device)
    predictions, true_labels, _ = predictor.predict(mock_re_dataloader, 're')

    # Expected predictions based on mock_re_model logits argmax
    expected_preds = [1, 0]
    # Expected true labels from the dataloader
    expected_true = [1, 0]
    
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert predictions == expected_preds
    assert true_labels == expected_true

@patch('src.evaluation.predictor.tqdm', side_effect=lambda iterable, **kwargs: iterable)
def test_predict_with_empty_dataloader(mock_tqdm, mock_ner_model, mock_empty_dataloader, mock_device):
    """
    Tests that the predict method handles an empty dataloader gracefully.
    """
    predictor = Predictor(model=mock_ner_model, device=mock_device)
    predictions, true_labels, logits = predictor.predict(mock_empty_dataloader, 'ner')

    assert predictions == []
    assert true_labels == []
    assert logits.size == 0