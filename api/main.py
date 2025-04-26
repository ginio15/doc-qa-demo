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
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.json"

# Initialize services
embedding_service = EmbeddingService()
ingestor = DocumentIngestor()
retriever = HybridRetriever(embedding_service=embedding_service)

# Create data directories if they don't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)

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

# Helper functions
def save_embeddings(embeddings: List[Dict[str, Any]]):
    """Save embeddings to file."""
    # Convert embeddings to serializable format
    serializable_embeddings = []
    for doc in embeddings:
        doc_copy = doc.copy()
        if "embedding" in doc_copy:
            doc_copy["embedding"] = doc_copy["embedding"]  # already serializable
        serializable_embeddings.append(doc_copy)
    
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(serializable_embeddings, f)

def load_embeddings() -> List[Dict[str, Any]]:
    """Load embeddings from file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    
    with open(EMBEDDINGS_FILE, 'r') as f:
        return json.load(f)

# Background processing functions
def process_document_background(file_path: str, original_filename: str):
    """Process a document in the background."""
    try:
        # Load and chunk the document
        chunks = ingestor.process_document(file_path)
        
        # Add original filename to metadata
        for chunk in chunks:
            chunk["metadata"]["original_filename"] = original_filename
        
        # Generate embeddings
        chunks_with_embeddings = embedding_service.embed_documents(chunks)
        
        # Load existing embeddings
        existing_embeddings = load_embeddings()
        
        # Add new embeddings
        existing_embeddings.extend(chunks_with_embeddings)
        
        # Save updated embeddings
        save_embeddings(existing_embeddings)
        
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
    # Load embeddings
    documents = load_embeddings()
    
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found in the corpus")
    
    # Retrieve relevant documents
    results = retriever.search(request.query, documents, top_k=request.top_k)
    
    # Clean up results (remove embeddings from response)
    for result in results:
        if "embedding" in result:
            del result["embedding"]
    
    return {
        "query": request.query,
        "results": results
    }

@app.get("/documents", response_model=List[str])
async def list_documents():
    """List all documents that have been uploaded."""
    documents = load_embeddings()
    
    # Extract unique filenames
    filenames = set()
    for doc in documents:
        filename = doc["metadata"].get("original_filename")
        if filename:
            filenames.add(filename)
            
    return sorted(list(filenames))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)