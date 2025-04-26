import React, { useState } from 'react';

const QuestionInput = ({ onSubmit, isLoading, isDisabled }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="question-form">
      <div className="input-group" style={{ display: 'flex' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={isDisabled ? "Upload documents first to ask questions" : "Ask a question about your documents..."}
          disabled={isLoading || isDisabled}
          style={{
            flex: 1,
            padding: '10px',
            borderRadius: '4px 0 0 4px',
            border: '1px solid #ccc',
            fontSize: '16px',
          }}
        />
        <button
          type="submit"
          disabled={isLoading || isDisabled || !query.trim()}
          style={{
            padding: '10px 15px',
            backgroundColor: isLoading || isDisabled || !query.trim() ? '#cccccc' : '#3498db',
            color: 'white',
            border: 'none',
            borderRadius: '0 4px 4px 0',
            cursor: isLoading || isDisabled || !query.trim() ? 'not-allowed' : 'pointer',
            fontSize: '16px',
          }}
        >
          {isLoading ? 'Loading...' : 'Ask'}
        </button>
      </div>
    </form>
  );
};

export default QuestionInput;