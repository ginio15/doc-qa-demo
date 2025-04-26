# Document Q&A Demo

A document question answering system with FastAPI backend and React frontend that allows you to upload documents (PDF, DOCX, TXT) and ask natural language questions about them.

## Features

- Document upload and processing (PDF, DOCX, TXT)
- Document chunking and embedding generation
- Hybrid search (vector + keyword) for accurate information retrieval
- OpenAI embeddings with e5 model fallback
- React UI for document upload and question asking

## Project Structure

```
doc-qa-demo/
│
├─ api/               # FastAPI code
│   ├─ ingest.py      # parses & chunks docs
│   ├─ embed.py       # wraps OpenAI + fallback e5
│   ├─ retrieve.py    # hybrid search
│   └─ main.py        # /ask endpoint
└─ web/               # React UI
```

## Setup Instructions

### Backend (FastAPI)

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages from requirements.txt to avoid dependency conflicts:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

   > **Note:** If you're using TensorFlow, there may be compatibility issues with protobuf versions.
   > The requirements.txt file specifies protobuf<5.0.0 to ensure compatibility with TensorFlow.

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key-here  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will run on http://localhost:8000

### Frontend (React)

1. Install Node.js dependencies:
   ```bash
   cd web
   npm install
   ```

2. Run the React development server:
   ```bash
   npm start
   ```
   The UI will run on http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. Upload one or more documents using the document uploader
3. Ask questions about the uploaded documents
4. View the results showing the most relevant document chunks

## API Endpoints

- `GET /`: Root endpoint to check if API is running
- `POST /documents/upload`: Upload a document for processing
- `POST /ask`: Ask a question about the documents
- `GET /documents`: List all uploaded documents

## Requirements

- Python 3.8+
- Node.js 14+
- OpenAI API key (optional, e5 model will be used as fallback)

## Troubleshooting

### Dependency Conflicts

If you encounter dependency conflicts with protobuf and TensorFlow, you can fix them by installing specific versions:

```bash
pip uninstall protobuf -y
pip install protobuf>=3.20.3,<5.0.0
```

For other dependency issues, try installing the requirements.txt file with the --no-deps flag and then manually installing the conflicting packages:

```bash
pip install -r requirements.txt --no-deps
pip install package-name==specific-version
```