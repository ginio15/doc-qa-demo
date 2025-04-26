import React from 'react';
import ReactMarkdown from 'react-markdown';

const ResultsList = ({ results }) => {
  return (
    <div className="results-list">
      <h3>Results</h3>
      {results.length === 0 ? (
        <p>No results found. Try a different question.</p>
      ) : (
        <div>
          {results.map((result, index) => (
            <div 
              key={result.chunk_id || index} 
              className="result-item"
              style={{
                padding: '15px',
                marginBottom: '15px',
                borderRadius: '4px',
                border: '1px solid #eee',
                backgroundColor: '#f9f9f9',
              }}
            >
              <div className="result-header" style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '14px', color: '#666' }}>
                  Source: {result.metadata?.original_filename || 'Unknown'}
                  {result.metadata?.page && `, Page ${result.metadata.page}`}
                </span>
                <span style={{ fontSize: '14px', color: '#3498db' }}>
                  Score: {(result.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="result-content">
                <ReactMarkdown>{result.text}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ResultsList;