import pytest
import numpy as np
import json
from unittest.mock import MagicMock, patch, ANY
import faiss

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.vector_db.database_manager import DatabaseManager
from src.vector_db.sentence_embedder import SentenceEmbedder

# --- Fixtures for Testing DatabaseManager ---

@pytest.fixture
def mock_embedder():
    """
    Creates a mock SentenceEmbedder that returns predictable embeddings.
    """
    embedder = MagicMock(spec=SentenceEmbedder)
    
    # Configure the mock to return embeddings of a specific dimension (e.g., 10)
    embedding_dim = 10
    
    # Define different embeddings for different texts to make search testable
    mock_embeddings = {
        "First test report.": np.random.rand(1, embedding_dim).astype('float32'),
        "Second test report.": np.random.rand(1, embedding_dim).astype('float32'),
        "Query text for search.": np.random.rand(1, embedding_dim).astype('float32')
    }

    # The embed method will look up the text in the dictionary
    def embed_side_effect(texts: list):
        # Stack embeddings for all texts in the input list
        return np.vstack([mock_embeddings[text] for text in texts])

    embedder.embed.side_effect = embed_side_effect
    return embedder

@pytest.fixture
def source_jsonl_file(tmp_path):
    """
    Creates a temporary .jsonl source data file for the vector database.
    """
    data = [
        {"text": "First test report.", "entities": []},
        {"text": "Second test report.", "entities": []}
    ]
    file_path = tmp_path / "source_data.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return str(file_path)

@pytest.fixture
def empty_source_jsonl_file(tmp_path):
    """Creates an empty temporary .jsonl file."""
    file_path = tmp_path / "empty_source_data.jsonl"
    file_path.touch()
    return str(file_path)


# --- Test Cases for DatabaseManager ---

def test_database_manager_initialization(mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests that the DatabaseManager initializes correctly, loads source data,
    and sets up paths as expected.
    """
    index_path = str(tmp_path / "test.index")
    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=index_path
    )

    assert manager.embedder is mock_embedder
    assert len(manager.source_data) == 2
    assert manager.source_data[0]["text"] == "First test report."
    assert manager.index is None
    assert Path(index_path).parent.exists()

@patch('faiss.write_index')
def test_build_index_creates_new_index(mock_write_index, mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests that `build_index` correctly creates a new FAISS index when one
    does not already exist.
    """
    index_path = str(tmp_path / "test.index")
    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=index_path
    )

    manager.build_index()

    # Assert that the embedder was called with the correct texts
    texts_to_embed = [record['text'] for record in manager.source_data]
    mock_embedder.embed.assert_called_once_with(texts_to_embed)

    # Assert that the index was created and has the correct number of vectors
    assert manager.index is not None
    assert isinstance(manager.index, faiss.IndexFlatL2)
    assert manager.index.ntotal == 2

    # Assert that the index was saved
    mock_write_index.assert_called_once()

@patch('faiss.read_index')
def test_build_index_loads_existing_index(mock_read_index, mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests that `build_index` loads an existing index file instead of rebuilding it.
    """
    index_path = tmp_path / "test.index"
    
    # Create a dummy index file to simulate its existence
    dummy_index = faiss.IndexFlatL2(10)
    faiss.write_index(dummy_index, str(index_path))

    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=str(index_path)
    )
    
    # Configure the mock to return the dummy index when called
    mock_read_index.return_value = dummy_index
    # Set the ntotal attribute to match the source data size
    mock_read_index.return_value.ntotal = 2


    manager.build_index()

    # Assert that read_index was called and embed was not
    mock_read_index.assert_called_once_with(str(index_path))
    mock_embedder.embed.assert_not_called()
    assert manager.index is dummy_index

@patch('faiss.read_index')
@patch('faiss.write_index')
def test_build_index_force_rebuild(mock_write_index, mock_read_index, mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests that `build_index` with `force_rebuild=True` rebuilds the index
    even if an index file already exists.
    """
    index_path = str(tmp_path / "test.index")
    Path(index_path).touch() # Create an empty file to simulate existence

    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=index_path
    )

    manager.build_index(force_rebuild=True)

    # Assert that the load function was NOT called
    mock_read_index.assert_not_called()
    
    # Assert that the embedding and writing functions WERE called
    assert mock_embedder.embed.called
    assert mock_write_index.called

def test_search_functionality(mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests the search method to ensure it returns the correct documents.
    """
    index_path = str(tmp_path / "test.index")
    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=index_path
    )
    manager.build_index()

    # Mock the search result from the FAISS index
    # Let's say the query is most similar to the second document (index 1)
    mock_distances = np.array([[0.1]], dtype='float32')
    mock_indices = np.array([[1]], dtype='int64') # Closest is index 1
    manager.index.search = MagicMock(return_value=(mock_distances, mock_indices))

    query = "Query text for search."
    results = manager.search(query, top_k=1)

    # Assert embed was called for the query
    mock_embedder.embed.assert_called_with([query])

    # Assert search was called on the index with the correct top_k value
    manager.index.search.assert_called_once_with(ANY, 1)
    
    # Assert the correct document was returned
    assert len(results) == 1
    assert results[0]["text"] == "Second test report."

def test_search_with_no_index(mock_embedder, source_jsonl_file, tmp_path):
    """
    Tests that calling `search` before building an index returns an empty list.
    """
    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=source_jsonl_file,
        index_path=str(tmp_path / "test.index")
    )
    
    results = manager.search("any query", top_k=1)
    
    assert results == []
    mock_embedder.embed.assert_not_called()

def test_handling_missing_source_data(mock_embedder, tmp_path):
    """
    Tests that the manager handles a missing source data file gracefully.
    """
    # Path to a file that does not exist
    non_existent_path = str(tmp_path / "non_existent.jsonl")
    
    manager = DatabaseManager(
        embedder=mock_embedder,
        source_data_path=non_existent_path,
        index_path=str(tmp_path / "test.index")
    )

    assert manager.source_data == []

    # Attempt to build the index; it should not raise an error
    manager.build_index()

    # The index should be None as there's no data to build from
    assert manager.index is None
    mock_embedder.embed.assert_not_called()