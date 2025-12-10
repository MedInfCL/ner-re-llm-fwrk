import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vector_db.sentence_embedder import SentenceEmbedder

class DatabaseManager:
    """
    Manages the creation, storage, and querying of a FAISS vector database.

    This class orchestrates the process of converting text data into embeddings,
    building a searchable index, and retrieving relevant documents for RAG.
    """

    def __init__(self, embedder: SentenceEmbedder, source_data_path: str, index_path: str):
        """
        Initializes the DatabaseManager.

        Args:
            embedder (SentenceEmbedder): An instance of the SentenceEmbedder class.
            source_data_path (str): The path to the .jsonl file containing the
                                    source documents for the RAG examples.
            index_path (str): The path where the FAISS index file is or will be saved.
        """
        self.embedder = embedder
        self.source_data_path = Path(source_data_path)
        self.index_path = Path(index_path)
        
        self.index = None
        self.source_data = self._load_source_data()
        
        # Ensure the directory for the index exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_source_data(self) -> List[Dict[str, Any]]:
        """
        Loads the source documents from the specified .jsonl file.
        Each line in the file is expected to be a JSON object.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a record from the source file.
        """
        if not self.source_data_path.exists():
            print(f"Warning: Source data file not found at {self.source_data_path}. "
                  "The database will be empty until the file is available and the index is built.")
            return []
            
        with open(self.source_data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def build_index(self, force_rebuild: bool = False):
        """
        Builds or loads the FAISS index.

        If an index file already exists at `index_path`, it loads it.
        Otherwise, it creates embeddings for the source data, builds a new
        FAISS index, and saves it.

        Args:
            force_rebuild (bool): If True, it will rebuild the index even if an
                                  existing index file is found.
        """
        if self.index_path.exists() and not force_rebuild:
            print(f"Loading existing FAISS index from: {self.index_path}")
            self.load_index()
            # Verify that the loaded index corresponds to the current source data
            if self.index and self.index.ntotal != len(self.source_data):
                print("Warning: Index size does not match source data size. Rebuilding index.")
                self._create_and_save_index()
        else:
            if not self.source_data:
                print("Error: Cannot build index because source data is empty.")
                return
            print("Building new FAISS index...")
            self._create_and_save_index()
            
    def _create_and_save_index(self):
        """
        Handles the process of embedding data, creating an index, and saving it.
        """
        # We use the 'text' field from our .jsonl records for embedding
        texts_to_embed = [record['text'] for record in self.source_data]
        
        embeddings = self.embedder.embed(texts_to_embed)
        
        if embeddings.size == 0:
            print("Error: No embeddings were generated. Cannot build index.")
            return

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"Successfully built index with {self.index.ntotal} vectors.")
        self.save_index()

    def save_index(self):
        """Saves the current FAISS index to the specified file."""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
            print(f"Index saved to: {self.index_path}")

    def load_index(self):
        """Loads a FAISS index from the specified file."""
        self.index = faiss.read_index(str(self.index_path))
        print(f"Index loaded successfully with {self.index.ntotal} vectors.")

    def search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Searches the index for the most similar documents to a query text.

        Args:
            query_text (str): The text to find similar documents for.
            top_k (int): The number of similar documents to return.

        Returns:
            List[Dict[str, Any]]: A list of the top_k most similar source data records.
                                  Returns an empty list if the index is not built.
        """
        if not self.index:
            print("Error: Index is not built or loaded. Cannot perform search.")
            return []

        query_embedding = self.embedder.embed([query_text])
        
        # The search returns distances and indices of the nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the original data records using the returned indices
        results = [self.source_data[i] for i in indices[0]]
        
        return results