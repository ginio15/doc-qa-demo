"""
Main FastAPI application with document QA endpoints.
"""
from typing import List, Dict, Any, Optional 
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import tempfile
from pathlib import Path
import uuid
import shutil

# Import local modules
from ingest import DocumentIngestor
from embed import EmbeddingService
from retrieve import HybridRetriever
from query_chroma import query_chroma

# Initialize FastAPI app
app = FastAPI(title="Document QA API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define data directory for storing documents and embeddings
DATA_DIR = os.getenv("DATA_DIR", "./data")
DOCUMENTS_DIR = f"{DATA_DIR}/documents"
CHROMADB_PATH = os.getenv("CHROMADB_PATH", ".chromadb")
COLLECTION_NAME = "docs"

# Initialize services
embedding_service = EmbeddingService()
ingestor = DocumentIngestor(chromadb_path=CHROMADB_PATH, collection_name=COLLECTION_NAME)
retriever = HybridRetriever(embedding_service=embedding_service)

# Create data directories if they don't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Data models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str

# Background processing functions
def process_document_background(file_path: str, original_filename: str):
    """Process a document in the background using ChromaDB."""
    try:
        # Use the ingest_file method to add the document to ChromaDB
        doc_id = ingestor.ingest_file(file_path)
        
        # Update metadata with original filename
        if ingestor.collection is not None:
            # Get all chunks for this document
            results = ingestor.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results and results["ids"]:
                # Update metadata for each chunk
                for i, chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if "metadatas" in results else {}
                    metadata["original_filename"] = original_filename
                    
                    # Update the metadata in ChromaDB
                    ingestor.collection.update(
                        ids=[chunk_id],
                        metadatas=[metadata]
                    )
        
        print(f"✅ Ingested {original_filename} with ID {doc_id}")
    except Exception as e:
        print(f"Error processing document: {e}")

# API endpoints
@app.get("/")
def read_root():
    """Root endpoint to verify API is running."""
    return {"status": "ok", "message": "Document QA API is running"}

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a document for processing."""
    # Generate a unique ID for the document
    doc_id = str(uuid.uuid4())
    
    # Create a filename for storage
    file_extension = Path(file.filename).suffix
    stored_filename = f"{doc_id}{file_extension}"
    file_path = os.path.join(DOCUMENTS_DIR, stored_filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the document in the background
    background_tasks.add_task(
        process_document_background, 
        file_path=file_path, 
        original_filename=file.filename
    )
    
    return {
        "id": doc_id,
        "filename": file.filename,
        "status": "processing"
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Answer a question based on the document corpus."""
    if ingestor.collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    # Get the number of documents in the collection
    collection_info = ingestor.collection.count()
    if collection_info == 0:
        raise HTTPException(status_code=404, detail="No documents found in the corpus")
    
    # Use query_chroma to get relevant documents
    results = query_chroma(
        query=request.query,
        top_k=request.top_k,
        chromadb_path=CHROMADB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Format results for response
    formatted_results = []
    for result in results:
        # Remove embedding if present
        if "embedding" in result:
            del result["embedding"]
        formatted_results.append(result)
    
    return {
        "query": request.query,
        "results": formatted_results
    }

@app.get("/documents", response_model=List[str])
async def list_documents():
    """List all documents that have been uploaded."""
    if ingestor.collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    # Get all documents from ChromaDB
    try:
        results = ingestor.collection.get()
        
        # Extract unique filenames from metadata
        filenames = set()
        if results and "metadatas" in results:
            for metadata in results["metadatas"]:
                if metadata and "original_filename" in metadata:
                    filenames.add(metadata["original_filename"])
                elif metadata and "source_file" in metadata:
                    # Use the source_file as fallback
                    source_file = metadata["source_file"]
                    if source_file:
                        filenames.add(os.path.basename(source_file))
                        
        return sorted(list(filenames))
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)