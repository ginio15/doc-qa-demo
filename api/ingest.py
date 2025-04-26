"""
Document ingestion module for parsing and chunking documents.
"""
from typing import List, Dict, Any, Optional
import os
import re
from pathlib import Path
import hashlib

# Import document loaders based on type
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    print("Please install langchain packages: pip install langchain langchain-community")

# Document types and their respective loaders
DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    # Add more document types as needed
}

class DocumentIngestor:
    """Handles document parsing and chunking for the QA system."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize the document ingestor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
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
            
            # Prepare the chunk with its metadata
            processed_chunk = {
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "source_file": file_path,
                }
            }
            processed_chunks.append(processed_chunk)
            
        return processed_chunks

# Example usage
if __name__ == "__main__":
    ingestor = DocumentIngestor()
    # Example: process_document("path/to/your/document.pdf")