# tests/unit/src/models/test_ner_bert.py
import pytest
import torch
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.models.ner_bert import BertNerModel

# Mock configuration for the NER model
@pytest.fixture
def mock_model_config():
    """Provides a mock configuration for the BertNerModel."""
    return {
        'base_model': 'bert-base-uncased',
        'n_labels': 5  # e.g., O, B-PERS, I-PERS, B-LOC, I-LOC
    }

# Mock the Hugging Face AutoModelForTokenClassification.from_pretrained method
@pytest.fixture
def mock_automodel_from_pretrained():
    """
    Mocks the `AutoModelForTokenClassification.from_pretrained` method,
    yielding the patch object so that calls to it can be asserted.
    """
    with patch('transformers.AutoModelForTokenClassification.from_pretrained') as mock_from_pretrained:
        # Configure the patch to return a mock model instance
        mock_model_instance = MagicMock()
        mock_from_pretrained.return_value = mock_model_instance
        yield mock_from_pretrained

def test_bert_ner_model_initialization(mock_model_config, mock_automodel_from_pretrained):
    """
    Tests that the BertNerModel initializes correctly, loading the base model
    and its configuration with the correct number of labels.
    """
    # Patch the AutoConfig loader within this test's scope
    with patch('transformers.AutoConfig.from_pretrained') as mock_config_loader:
        # Set up the return value for the config loader
        mock_config_obj = MagicMock()
        mock_config_loader.return_value = mock_config_obj

        # Instantiate the model, which will trigger the mocked loaders
        model = BertNerModel(
            base_model=mock_model_config['base_model'],
            n_labels=mock_model_config['n_labels']
        )

        # --- Assertions ---

        # 1. Verify that the configuration was loaded with the correct parameters
        mock_config_loader.assert_called_once_with(
            mock_model_config['base_model'],
            num_labels=mock_model_config['n_labels']
        )
        
        # 2. Verify that the model was loaded with the correct parameters
        #    The `mock_automodel_from_pretrained` fixture yields the patcher for this method.
        mock_automodel_from_pretrained.assert_called_once_with(
            mock_model_config['base_model'],
            config=mock_config_obj
        )

        # 3. Verify that the model attributes were set correctly
        assert model.n_labels == mock_model_config['n_labels']
        assert model.base_model_name == mock_model_config['base_model']

def test_bert_ner_model_forward_pass(mock_model_config):
    """
    Tests the forward pass of the BertNerModel to ensure it correctly
    invokes the underlying model and returns the expected output structure.
    """
    # Patch the from_pretrained method directly for this test
    with patch('transformers.AutoModelForTokenClassification.from_pretrained') as mock_from_pretrained:
        # This is the mock for the inner Hugging Face model instance
        inner_model_mock = MagicMock()
        mock_from_pretrained.return_value = inner_model_mock

        # Define the expected output structure from the inner model's forward pass
        batch_size, seq_len = 2, 16
        expected_logits = torch.randn(batch_size, seq_len, mock_model_config['n_labels'])
        mock_output = MagicMock()
        mock_output.logits = expected_logits
        mock_output.loss = torch.tensor(1.2)
        
        # Configure the mock model's `__call__` to return the mock output
        # This simulates the forward pass: `self.model(...)`
        inner_model_mock.return_value = mock_output

        # Instantiate our wrapper model
        model = BertNerModel(
            base_model=mock_model_config['base_model'],
            n_labels=mock_model_config['n_labels']
        )

        # Prepare mock input tensors
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        labels = torch.randint(0, mock_model_config['n_labels'], (batch_size, seq_len))
        
        # --- Act ---
        outputs = model.forward(input_ids, attention_mask, labels)
        
        # --- Assertions ---

        # 1. Verify that the inner model's forward pass (`__call__`) was executed once
        inner_model_mock.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 2. Verify that the returned output has the correct structure and values
        assert torch.equal(outputs.logits, expected_logits)
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, seq_len, mock_model_config['n_labels'])