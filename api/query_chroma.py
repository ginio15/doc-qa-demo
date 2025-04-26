"""
Demonstration script for querying documents using ChromaDB.
"""
import argparse
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def query_chroma(query: str, top_k: int = 5, chromadb_path: str = ".chromadb", collection_name: str = "docs"):
    """
    Query documents from ChromaDB.
    
    Args:
        query: Query text
        top_k: Number of results to return
        chromadb_path: Path to ChromaDB storage
        collection_name: Name of the ChromaDB collection
        
    Returns:
        List of top_k document chunks
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(collection_name)
    
    # Initialize embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    
    # Format results
    formatted_results = []
    for i, (doc_id, text, metadata) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    )):
        formatted_results.append({
            "chunk_id": doc_id,
            "text": text,
            "metadata": metadata,
            "score": results["distances"][0][i] if "distances" in results else None
        })
    
    return formatted_results

def print_results(results: List[Dict[str, Any]]):
    """Print formatted results."""
    print(f"Found {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results):
        score = result.get("score")
        score_str = f"Score: {1 - score:.4f}" if score is not None else "Score: N/A"
        
        print(f"Result {i+1} [{score_str}]")
        print(f"Source: {result['metadata'].get('source_file', 'Unknown')}")
        print(f"Page: {result['metadata'].get('page_no', 'N/A')}")
        print("-" * 40)
        print(result["text"][:300] + ("..." if len(result["text"]) > 300 else ""))
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query documents from ChromaDB")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--chromadb-path", default=".chromadb", help="Path to ChromaDB storage")
    parser.add_argument("--collection", default="docs", help="ChromaDB collection name")
    args = parser.parse_args()
    
    results = query_chroma(
        query=args.query, 
        top_k=args.top_k,
        chromadb_path=args.chromadb_path,
        collection_name=args.collection
    )
    
    print_results(results)