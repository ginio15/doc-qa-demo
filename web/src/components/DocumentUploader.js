import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const DocumentUploader = ({ onUpload, isLoading }) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0]);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
    disabled: isLoading,
  });

  return (
    <div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? 'active' : ''} ${isLoading ? 'disabled' : ''}`}
      style={{
        border: '2px dashed #cccccc',
        borderRadius: '4px',
        padding: '20px',
        textAlign: 'center',
        cursor: isLoading ? 'not-allowed' : 'pointer',
        backgroundColor: isDragActive ? '#f0f8ff' : '#fafafa',
        color: '#666',
      }}
    >
      <input {...getInputProps()} />
      {isLoading ? (
        <p>Uploading document...</p>
      ) : isDragActive ? (
        <p>Drop the document here...</p>
      ) : (
        <p>
          Drag & drop a document here, or click to select<br />
          <small>Supported formats: PDF, TXT, DOCX</small>
        </p>
      )}
    </div>
  );
};

export default DocumentUploader;