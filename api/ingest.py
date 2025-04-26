"""
Document ingestion module for parsing and chunking documents.
"""
from typing import List, Dict, Any, Optional, Tuple, Iterator, Generator
import os
import re
from pathlib import Path
import hashlib
import uuid
import argparse
import json

# Import document loaders based on type
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    import tiktoken
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install required packages: pip install langchain langchain-community tiktoken chromadb sentence-transformers")

# Document types and their respective loaders
DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader, 
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    # Add more document types as needed
}

# Default embedding model
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class DocumentIngestor:
    """Handles document parsing and chunking for the QA system."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 chromadb_path: str = ".chromadb",
                 collection_name: str = "docs",
                 embedding_model: str = DEFAULT_EMB_MODEL):
        """
        Initialize the document ingestor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            chromadb_path: Path for persistent ChromaDB storage
            collection_name: Name of the ChromaDB collection to use
            embedding_model: Name of the sentence-transformer model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize tokenizer for token-based operations
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except:
            self.tokenizer = None
            
        # Initialize ChromaDB client and collection
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        try:
            self.chromadb_client = chromadb.PersistentClient(path=chromadb_path)
            self.collection = self.chromadb_client.get_or_create_collection(collection_name)
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}")
            self.chromadb_client = None
            self.collection = None
            
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_model.encode(texts).tolist()
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document using the appropriate loader based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in DOCUMENT_LOADERS:
            raise ValueError(f"Unsupported document type: {file_ext}")
        
        loader_cls = DOCUMENT_LOADERS[file_ext]
        loader = loader_cls(file_path)
        return loader.load()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        return self.text_splitter.split_documents(documents)
    
    def extract_text_with_positions(self, file_path: str) -> List[Tuple[int, int, int, str]]:
        """
        Extract text with position information from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of tuples (page_no, char_start, char_end, text)
        """
        documents = self.load_document(file_path)
        result = []
        
        for doc in documents:
            page_no = doc.metadata.get('page', 0)
            text = doc.page_content
            
            # For PDF documents processed with PyPDFLoader, we can track character positions
            # For other document types, we use approximations
            if 'source' in doc.metadata and doc.metadata['source'].lower().endswith('.pdf'):
                # PDFs typically don't provide char_start and char_end in metadata
                # We'll use the cumulative length as an approximation
                prev_lengths = sum(len(d.page_content) for d in documents 
                                  if d.metadata.get('page', 0) < page_no)
                char_start = prev_lengths
                char_end = prev_lengths + len(text)
            else:
                # For non-PDF documents, use page-based positioning
                char_start = 0
                char_end = len(text)
            
            result.append((page_no, char_start, char_end, text))
        
        # Sort by page_no, then by char_start
        result.sort(key=lambda x: (x[0], x[1]))
        return result
    
    def split_text_lines_into_token_chunks(self, 
                                          text_lines: List[str], 
                                          chunk_size: int = 350, 
                                          chunk_overlap: int = 50) -> List[str]:
        """
        Split a list of text lines into chunks of approximately chunk_size tokens with overlap.
        
        Args:
            text_lines: List of text lines to split
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if not text_lines:
            return []
            
        if self.tokenizer is None:
            # Fallback if tiktoken is not available - rough approximation
            avg_tokens_per_char = 0.25  # Rough estimate: 4 chars ≈ 1 token
            char_chunk_size = int(chunk_size / avg_tokens_per_char)
            char_chunk_overlap = int(chunk_overlap / avg_tokens_per_char)
            
            # Join lines and split by character count
            text = "\n".join(text_lines)
            chunks = []
            
            for i in range(0, len(text), char_chunk_size - char_chunk_overlap):
                chunk = text[i:i + char_chunk_size]
                chunks.append(chunk)
                
            return chunks
        
        # Use tiktoken for more precise token counting
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for line in text_lines:
            line_tokens = self.tokenizer.encode(line)
            line_token_count = len(line_tokens)
            
            # If adding this line exceeds the chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_chunk_tokens + line_token_count > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                
                # Start a new chunk with overlap
                # Find lines to include for overlap
                overlap_tokens = 0
                overlap_lines = []
                
                for overlap_line in reversed(current_chunk):
                    overlap_line_tokens = len(self.tokenizer.encode(overlap_line))
                    if overlap_tokens + overlap_line_tokens <= chunk_overlap:
                        overlap_lines.insert(0, overlap_line)
                        overlap_tokens += overlap_line_tokens
                    else:
                        break
                
                current_chunk = overlap_lines
                current_chunk_tokens = overlap_tokens
            
            # Add the current line to the chunk
            current_chunk.append(line)
            current_chunk_tokens += line_token_count
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document by loading and chunking it.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunks as dictionaries with text, metadata, and chunk_id
        """
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for each chunk
            chunk_id = hashlib.md5(f"{file_path}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
            
            # Extract position information
            page_no = chunk.metadata.get('page', 0)
            
            # Calculate char positions based on document type and available metadata
            if 'source' in chunk.metadata and chunk.metadata['source'].lower().endswith('.pdf'):
                # For PDFs, use an approximation based on chunk index
                char_start = i * (self.chunk_size - self.chunk_overlap)
                char_end = char_start + len(chunk.page_content)
            else:
                # For other documents, use simpler positioning
                char_start = i * (self.chunk_size - self.chunk_overlap)
                char_end = char_start + len(chunk.page_content)
            
            # Prepare the chunk with its metadata
            processed_chunk = {
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "position": {
                    "page_no": page_no,
                    "char_start": char_start,
                    "char_end": char_end
                },
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "source_file": file_path,
                }
            }
            processed_chunks.append(processed_chunk)
            
        return processed_chunks
        
    def parse(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Parse a document into rows for further processing.
        This is a wrapper around the document loading process to match the interface in the example code.
        
        Args:
            file_path: Path to the document
            
        Yields:
            Rows of document data
        """
        documents = self.load_document(file_path)
        for doc in documents:
            yield {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            
    def chunk(self, rows: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """
        Chunk rows of document data.
        
        Args:
            rows: Rows of document data
            
        Yields:
            Chunk data
        """
        # Convert rows to documents for text splitter
        documents = [
            Document(page_content=row["text"], metadata=row["metadata"])
            for row in rows
        ]
        
        # Chunk documents
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Yield chunks
        for i, doc in enumerate(chunked_docs):
            yield {
                "text": doc.page_content,
                "page_no": doc.metadata.get("page", 0),
                "char_start": i * (self.chunk_size - self.chunk_overlap),  # Approximate
                "char_end": (i * (self.chunk_size - self.chunk_overlap)) + len(doc.page_content),
                "chunk_index": i,
            }
            
    def ingest_file(self, file_path: str) -> str:
        """
        Ingest a file into ChromaDB.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document ID
        """
        if self.collection is None:
            raise ValueError("ChromaDB collection not initialized")
        
        path = Path(file_path)
        doc_id = str(uuid.uuid4())
        
        # Parse and chunk the document
        rows = list(self.parse(str(path)))
        chunks = list(self.chunk(rows))
        
        # Generate embeddings
        embeds = self.embed([c["text"] for c in chunks])
        
        # Add to ChromaDB
        self.collection.add(
            ids=[f"{doc_id}_{i}" for i, _ in enumerate(chunks)],
            embeddings=embeds,
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": doc_id, **c} for c in chunks]
        )
        
        print(f"✅ {path.name}: {len(chunks)} chunks")
        return doc_id
        
    def ingest_files(self, file_paths: List[str]) -> List[str]:
        """
        Ingest multiple files into ChromaDB.
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for file_path in file_paths:
            try:
                doc_id = self.ingest_file(file_path)
                doc_ids.append(doc_id)
            except Exception as e:
                print(f"❌ Error ingesting {file_path}: {e}")
        
        return doc_ids
        
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the QA system")
    parser.add_argument("files", nargs="+", help="Files to ingest")
    parser.add_argument("--chromadb-path", default=".chromadb", help="Path to ChromaDB storage")
    parser.add_argument("--collection", default="docs", help="ChromaDB collection name")
    args = parser.parse_args()
    
    ingestor = DocumentIngestor(
        chromadb_path=args.chromadb_path,
        collection_name=args.collection
    )
    
    doc_ids = ingestor.ingest_files(args.files)
    print("Documents ingested:", doc_ids)