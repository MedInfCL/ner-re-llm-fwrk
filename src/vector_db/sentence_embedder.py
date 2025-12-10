import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceEmbedder:
    """
    A wrapper class for loading and using a sentence-transformer model to
    create text embeddings.
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Initializes the SentenceEmbedder.

        Args:
            model_name (str): The name of the sentence-transformer model to load
                              from the Hugging Face Hub (e.g., 'all-MiniLM-L6-v2').
            device (str, optional): The device to run the model on ('cuda' or 'cpu').
                                    If None, it will auto-detect CUDA availability.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing SentenceEmbedder on device: '{self.device}'")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Successfully loaded embedding model: '{model_name}'")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of text strings into a numpy array of embeddings.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            np.ndarray: A 2D numpy array where each row is the vector
                        embedding for the corresponding text.
        """
        if not texts:
            # Return an empty array with the correct shape if input is empty
            return np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
            
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        print("Embedding generation complete.")
        return embeddings