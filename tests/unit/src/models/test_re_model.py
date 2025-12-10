# tests/unit/src/models/test_re_model.py
import pytest
import torch
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.models.re_model import REModel

# Mock configuration for the RE model
@pytest.fixture
def mock_re_config():
    """Provides a mock configuration for the REModel."""
    return {
        'base_model': 'bert-base-cased',
        'n_labels': 3  # e.g., 'describir', 'ubicar', 'No_Relation'
    }

# Mock tokenizer with an expanded vocabulary size
@pytest.fixture
def mock_tokenizer_re():
    """Creates a mock tokenizer with added special tokens for RE."""
    tokenizer = MagicMock()
    # Simulate a tokenizer with special tokens added, increasing its length
    tokenizer.__len__.return_value = 30526  # Standard bert-base-cased is 28996
    return tokenizer

def test_re_model_initialization(mock_re_config, mock_tokenizer_re):
    """
    Tests that the REModel initializes correctly, loads the base model,
    and resizes token embeddings for the new special tokens.
    """
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_from_pretrained:
        # This is the mock for the inner Hugging Face model instance
        inner_model_mock = MagicMock()
        mock_from_pretrained.return_value = inner_model_mock
        
        # Instantiate our wrapper model
        model = REModel(
            base_model=mock_re_config['base_model'],
            n_labels=mock_re_config['n_labels'],
            tokenizer=mock_tokenizer_re
        )

        # --- Assertions ---

        # 1. Verify that the base model was loaded with the correct number of labels
        mock_from_pretrained.assert_called_once_with(
            mock_re_config['base_model'],
            num_labels=mock_re_config['n_labels']
        )

        # 2. Verify that the model resized its token embeddings to fit the tokenizer
        inner_model_mock.resize_token_embeddings.assert_called_once_with(len(mock_tokenizer_re))
        
        # 3. Verify that model attributes are set correctly
        assert model.n_labels == mock_re_config['n_labels']
        assert model.base_model_name == mock_re_config['base_model']

def test_re_model_forward_pass(mock_re_config, mock_tokenizer_re):
    """
    Tests the forward pass of the REModel to ensure it passes inputs
    correctly to the underlying sequence classification model.
    """
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_from_pretrained:
        # Mock the inner model and its forward pass (`__call__`)
        inner_model_mock = MagicMock()
        mock_from_pretrained.return_value = inner_model_mock

        # Define the expected output from the forward pass
        expected_logits = torch.randn(2, mock_re_config['n_labels'])
        mock_output = MagicMock()
        mock_output.logits = expected_logits
        mock_output.loss = torch.tensor(0.8)
        inner_model_mock.return_value = mock_output

        # Instantiate our wrapper
        model = REModel(
            base_model=mock_re_config['base_model'],
            n_labels=mock_re_config['n_labels'],
            tokenizer=mock_tokenizer_re
        )

        # Prepare mock input tensors
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        labels = torch.randint(0, mock_re_config['n_labels'], (batch_size,))

        # --- Act ---
        outputs = model.forward(input_ids, attention_mask, labels=labels)

        # --- Assertions ---

        # 1. Verify that the inner model's forward pass was called correctly
        inner_model_mock.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 2. Verify that the output has the correct structure
        assert torch.equal(outputs.logits, expected_logits)
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, mock_re_config['n_labels'])