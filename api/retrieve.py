"""
Retrieval module implementing hybrid search (vector + keyword) for document QA.
"""
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from collections import Counter

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    print("Please install required packages: pip install scikit-learn nltk")

# Local imports
from embed import EmbeddingService

class HybridRetriever:
    """
    A hybrid retrieval system combining vector and keyword search.
    """
    
    def __init__(self, 
                 embedding_service: Optional[EmbeddingService] = None,
                 vector_weight: float = 0.7):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_weight: Weight for vector search results (0-1)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_weight = max(0.0, min(1.0, vector_weight))  # Clamp between 0-1
        self.keyword_weight = 1.0 - self.vector_weight
        self._stopwords = set(stopwords.words('english'))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text by removing stopwords and taking most common terms.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self._stopwords]
        
        # Count occurrences and get the most common ones
        counter = Counter(filtered_tokens)
        return [word for word, _ in counter.most_common(max_keywords)]
    
    def _keyword_search(self, query_keywords: List[str], documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform keyword-based search.
        
        Args:
            query_keywords: Keywords from the query
            documents: List of documents to search
            
        Returns:
            List of (document, score) tuples
        """
        results = []
        
        for doc in documents:
            doc_keywords = self._extract_keywords(doc["text"])
            
            # Calculate score based on keyword overlap
            matches = sum(1 for keyword in query_keywords if keyword in doc_keywords)
            if not query_keywords:
                score = 0.0
            else:
                score = matches / len(query_keywords)
                
            results.append((doc, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _vector_search(self, query_embedding: List[float], documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform embedding-based vector search.
        
        Args:
            query_embedding: Embedding of the query
            documents: List of documents to search
            
        Returns:
            List of (document, score) tuples
        """
        results = []
        
        for doc in documents:
            # If document doesn't have an embedding yet, generate one
            doc_embedding = doc.get("embedding")
            if doc_embedding is None:
                doc_embedding = self.embedding_service.get_embedding(doc["text"])
                
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append((doc, similarity))
            
        return sorted(results, key=lambda x: x[1], reverse=True)
            
    def search(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword approaches.
        
        Args:
            query: Search query
            documents: List of documents to search through
            top_k: Number of results to return
            
        Returns:
            List of top_k document dictionaries with added score field
        """
        # Get query embedding and keywords
        query_embedding = self.embedding_service.embed_query(query)
        query_keywords = self._extract_keywords(query)
        
        # Get vector and keyword search results
        vector_results = self._vector_search(query_embedding, documents)
        keyword_results = self._keyword_search(query_keywords, documents)
        
        # Combine results with weighted scoring
        doc_scores = {}
        for doc, score in vector_results:
            doc_id = doc["chunk_id"]
            doc_scores[doc_id] = {"doc": doc, "score": score * self.vector_weight}
            
        for doc, score in keyword_results:
            doc_id = doc["chunk_id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += score * self.keyword_weight
            else:
                doc_scores[doc_id] = {"doc": doc, "score": score * self.keyword_weight}
                
        # Sort by combined score and return top_k results
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        
        # Format results
        results = []
        for item in sorted_results:
            doc = item["doc"].copy()
            doc["score"] = item["score"]
            results.append(doc)
            
        return results

# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        {"chunk_id": "1", "text": "Machine learning is a subset of artificial intelligence"},
        {"chunk_id": "2", "text": "Deep learning uses neural networks with many layers"},
    ]
    
    retriever = HybridRetriever()
    # Example: results = retriever.search("What is machine learning?", documents, top_k=1)