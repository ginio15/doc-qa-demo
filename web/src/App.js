import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import DocumentUploader from './components/DocumentUploader';
import QuestionInput from './components/QuestionInput';
import ResultsList from './components/ResultsList';

function App() {
  const [documents, setDocuments] = useState([]);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  // Load the list of documents when the component mounts
  useEffect(() => {
    fetchDocumentsList();
  }, []);

  // Fetch the list of documents from the API
  const fetchDocumentsList = async () => {
    try {
      const response = await axios.get('/documents');
      setDocuments(response.data);
    } catch (err) {
      console.error('Error fetching documents:', err);
      // Don't set error state here to avoid showing error on first load
    }
  };

  // Handle document upload
  const handleDocumentUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setUploadSuccess(false);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      await axios.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setUploadSuccess(true);
      fetchDocumentsList(); // Refresh documents list
    } catch (err) {
      console.error('Error uploading document:', err);
      setError('Failed to upload document. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle asking a question
  const handleAskQuestion = async (query) => {
    setIsLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const response = await axios.post('/ask', { query, top_k: 5 });
      setResults(response.data.results);
    } catch (err) {
      console.error('Error asking question:', err);
      setError('Failed to get answer. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Document Q&A Demo</h1>
        <p>Upload documents and ask questions about them</p>
      </header>
      
      <main className="App-main">
        <section className="upload-section">
          <h2>Upload Documents</h2>
          <DocumentUploader onUpload={handleDocumentUpload} isLoading={isLoading} />
          {uploadSuccess && <p className="success-message">Document uploaded successfully!</p>}
          
          <div className="documents-list">
            <h3>Uploaded Documents ({documents.length})</h3>
            {documents.length > 0 ? (
              <ul>
                {documents.map((doc, index) => (
                  <li key={index}>{doc}</li>
                ))}
              </ul>
            ) : (
              <p>No documents uploaded yet.</p>
            )}
          </div>
        </section>
        
        <section className="question-section">
          <h2>Ask a Question</h2>
          <QuestionInput onSubmit={handleAskQuestion} isLoading={isLoading} isDisabled={documents.length === 0} />
          
          {error && <p className="error-message">{error}</p>}
          
          <div className="results-container">
            {results.length > 0 && (
              <ResultsList results={results} />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;