import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.vector_db.sentence_embedder import SentenceEmbedder

# --- Fixtures for Testing SentenceEmbedder ---

@pytest.fixture
def mock_sentence_transformer():
    """
    Mocks the SentenceTransformer class to avoid loading a real model.
    Yields the patch object so that calls to it can be asserted.
    """
    # Patches the class in the namespace where it is imported and used.
    with patch('src.vector_db.sentence_embedder.SentenceTransformer') as mock_class:
        # This is the mock for the model instance that the class constructor returns
        mock_model_instance = MagicMock()
        
        # Configure the mock instance's methods
        embedding_dim = 384
        mock_model_instance.get_sentence_embedding_dimension.return_value = embedding_dim
        
        # The encode method should return a numpy array of the correct shape
        mock_embeddings = np.random.rand(2, embedding_dim).astype('float32')
        mock_model_instance.encode.return_value = mock_embeddings
        
        # The class constructor will return our configured mock instance
        mock_class.return_value = mock_model_instance
        
        yield mock_class

# --- Test Cases for SentenceEmbedder ---

@patch('torch.cuda.is_available', return_value=True)
def test_initialization_on_gpu(mock_cuda_available, mock_sentence_transformer):
    """
    Tests that the embedder initializes on 'cuda' when CUDA is available.
    """
    model_name = "test-model"
    embedder = SentenceEmbedder(model_name=model_name)
    
    assert embedder.device == "cuda"
    # Verify the SentenceTransformer was instantiated with the correct device
    mock_sentence_transformer.assert_called_once_with(model_name, device="cuda")

@patch('torch.cuda.is_available', return_value=False)
def test_initialization_on_cpu(mock_cuda_available, mock_sentence_transformer):
    """
    Tests that the embedder falls back to 'cpu' when CUDA is not available.
    """
    model_name = "test-model"
    embedder = SentenceEmbedder(model_name=model_name)
    
    assert embedder.device == "cpu"
    mock_sentence_transformer.assert_called_once_with(model_name, device="cpu")

def test_initialization_with_explicit_device(mock_sentence_transformer):
    """
    Tests that the embedder uses the explicitly provided device.
    """
    model_name = "test-model"
    embedder = SentenceEmbedder(model_name=model_name, device="cpu")
    
    assert embedder.device == "cpu"
    mock_sentence_transformer.assert_called_once_with(model_name, device="cpu")

def test_embed_successful(mock_sentence_transformer):
    """
    Tests the embed method with a non-empty list of texts.
    """
    embedder = SentenceEmbedder(model_name="test-model")
    texts_to_embed = ["This is a test.", "This is another test."]
    
    embeddings = embedder.embed(texts_to_embed)

    # Get the mock model instance to check method calls on it
    mock_model_instance = mock_sentence_transformer.return_value
    
    # Assert that the underlying model's encode method was called correctly
    mock_model_instance.encode.assert_called_once_with(
        texts_to_embed,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Assert that the returned embeddings match the mock's return value
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384) # (num_texts, embedding_dim)
    assert np.array_equal(embeddings, mock_model_instance.encode.return_value)

def test_embed_with_empty_list(mock_sentence_transformer):
    """
    Tests that the embed method handles an empty list correctly.
    """
    embedder = SentenceEmbedder(model_name="test-model")
    
    embeddings = embedder.embed([])
    
    mock_model_instance = mock_sentence_transformer.return_value
    
    # Assert that the encode method was NOT called for an empty list
    mock_model_instance.encode.assert_not_called()
    
    # Assert that the returned array is empty but has the correct shape
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 384) # (0, embedding_dim)