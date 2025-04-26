"""
Embedding module that wraps OpenAI embeddings with e5 as fallback.
"""
from typing import List, Dict, Any, Union, Optional
import os
import numpy as np
from functools import lru_cache

# Try to import necessary libraries
try:
    import openai
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Please install required packages: pip install openai sentence-transformers torch")

class EmbeddingService:
    """Service to generate embeddings using OpenAI with e5 as fallback."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "text-embedding-ada-002",
                 e5_model_name: str = "intfloat/e5-large-v2"):
        """
        Initialize the embedding service.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: OpenAI embedding model name
            e5_model_name: e5 model name to use as fallback
        """
        self.model_name = model_name
        self.e5_model_name = e5_model_name
        
        # Setup OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.openai_api_key:
            import openai
            self.client = openai.OpenAI(api_key=self.openai_api_key)

        # Load e5 model (lazy loaded on first use)
        self._e5_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    @lru_cache()
    def e5_model(self):
        """Lazy load the e5 model on first use."""
        if self._e5_model is None:
            self._e5_model = SentenceTransformer(self.e5_model_name, device=self._device)
        return self._e5_model

    def get_openai_embedding(self, text: str) -> List[float]:
        """
        Get embeddings from OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding floats
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")

        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    def get_e5_embedding(self, text: str) -> List[float]:
        """
        Get embeddings from e5 model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding floats
        """
        # e5 works best with a prefix for different types of embeddings
        text = f"passage: {text}"
        embedding = self.e5_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_embedding(self, text: str, use_fallback: bool = True) -> List[float]:
        """
        Get embeddings, trying OpenAI first and falling back to e5 if needed.
        
        Args:
            text: Text to embed
            use_fallback: Whether to use e5 as fallback if OpenAI fails
            
        Returns:
            List of embedding floats
        """
        if not text or not text.strip():
            # Return zeros for empty text
            return [0.0] * (1536 if self.model_name == "text-embedding-ada-002" else 768)
            
        try:
            return self.get_openai_embedding(text)
        except Exception as e:
            if use_fallback:
                print(f"OpenAI embedding failed, using e5 fallback. Error: {e}")
                return self.get_e5_embedding(text)
            else:
                raise

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks.
        
        Args:
            documents: List of document dictionaries containing "text" and other fields
            
        Returns:
            List of documents with added "embedding" field
        """
        results = []
        for doc in documents:
            doc_with_embedding = doc.copy()
            doc_with_embedding["embedding"] = self.get_embedding(doc["text"])
            results.append(doc_with_embedding)
        return results

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query text.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list of floats
        """
        # For e5, queries should be prefixed differently than documents
        if not self.client:
            query = f"query: {query}"
            
        return self.get_embedding(query)

# Example usage
if __name__ == "__main__":
    embedding_service = EmbeddingService()
    # Example: embedding = embedding_service.embed_query("What is machine learning?")