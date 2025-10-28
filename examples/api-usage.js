/**
 * Example usage of the PG Vector Search API (Fastify version)
 * 
 * This file demonstrates how to use the various API endpoints
 * for data ingestion and search operations.
 */

const axios = require('axios');

// Configuration
const API_BASE_URL = 'http://localhost:3000/api';

// Helper function to make API calls
async function apiCall(method, endpoint, data = null) {
  try {
    const config = {
      method,
      url: `${API_BASE_URL}${endpoint}`,
      headers: {
        'Content-Type': 'application/json'
      }
    };
    
    if (data) {
      config.data = data;
    }
    
    const response = await axios(config);
    return response.data;
  } catch (error) {
    console.error(`API Error (${method} ${endpoint}):`, error.response?.data || error.message);
    throw error;
  }
}

// Helper function for file uploads
async function uploadFile(endpoint, filePath, fields = {}) {
  try {
    const FormData = require('form-data');
    const fs = require('fs');
    
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    Object.entries(fields).forEach(([key, value]) => {
      form.append(key, value);
    });
    
    const response = await axios.post(`${API_BASE_URL}${endpoint}`, form, {
      headers: {
        ...form.getHeaders()
      }
    });
    
    return response.data;
  } catch (error) {
    console.error(`File upload error (${endpoint}):`, error.response?.data || error.message);
    throw error;
  }
}

// Example 1: Create a document
async function createDocument() {
  console.log('\n=== Creating a Document ===');
  
  const document = {
    title: 'Introduction to Vector Databases',
    content: `Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for applications like semantic search, recommendation systems, and machine learning.

Unlike traditional databases that store structured data, vector databases are optimized for similarity search operations. They use various indexing algorithms like HNSW, IVFFlat, and LSH to enable fast approximate nearest neighbor (ANN) search.

Popular vector databases include Pinecone, Weaviate, Chroma, and PostgreSQL with the pgvector extension. Each has its own strengths in terms of performance, scalability, and feature set.`,
    contentType: 'text',
    metadata: {
      category: 'Database',
      difficulty: 'intermediate',
      tags: ['vector database', 'similarity search', 'machine learning'],
      author: 'API Example'
    }
  };
  
  const result = await apiCall('POST', '/documents', document);
  console.log('Document created:', result);
  return result.id;
}

// Example 2: Semantic Search
async function semanticSearch() {
  console.log('\n=== Semantic Search ===');
  
  const searchQuery = {
    query: 'machine learning databases',
    limit: 5,
    similarityThreshold: 0.6
  };
  
  const results = await apiCall('POST', '/search/semantic', searchQuery);
  console.log('Semantic search results:', results);
  return results;
}

// Example 3: Full-Text Search
async function fullTextSearch() {
  console.log('\n=== Full-Text Search ===');
  
  const searchQuery = {
    query: 'vector database',
    limit: 5
  };
  
  const results = await apiCall('POST', '/search/text', searchQuery);
  console.log('Full-text search results:', results);
  return results;
}

// Example 4: Hybrid Search
async function hybridSearch() {
  console.log('\n=== Hybrid Search ===');
  
  const searchQuery = {
    query: 'similarity search algorithms',
    limit: 5,
    semanticWeight: 0.7,
    textWeight: 0.3
  };
  
  const results = await apiCall('POST', '/search/hybrid', searchQuery);
  console.log('Hybrid search results:', results);
  return results;
}

// Example 5: Metadata Search
async function metadataSearch() {
  console.log('\n=== Metadata Search ===');
  
  const searchQuery = {
    metadata: {
      category: 'Database',
      difficulty: 'intermediate'
    },
    limit: 5
  };
  
  const results = await apiCall('POST', '/search/metadata', searchQuery);
  console.log('Metadata search results:', results);
  return results;
}

// Example 6: Get Similar Documents
async function getSimilarDocuments(documentId) {
  console.log('\n=== Similar Documents ===');
  
  const results = await apiCall('GET', `/search/similar/${documentId}?limit=3`);
  console.log('Similar documents:', results);
  return results;
}

// Example 7: Get All Documents
async function getAllDocuments() {
  console.log('\n=== All Documents ===');
  
  const results = await apiCall('GET', '/documents?limit=10');
  console.log('All documents:', results);
  return results;
}

// Example 8: Update Document
async function updateDocument(documentId) {
  console.log('\n=== Update Document ===');
  
  const updates = {
    title: 'Updated: Introduction to Vector Databases',
    metadata: {
      category: 'Database',
      difficulty: 'advanced',
      tags: ['vector database', 'similarity search', 'machine learning', 'updated'],
      author: 'API Example',
      lastUpdated: new Date().toISOString()
    }
  };
  
  const result = await apiCall('PUT', `/documents/${documentId}`, updates);
  console.log('Document updated:', result);
  return result;
}

// Example 9: Bulk Ingest
async function bulkIngest() {
  console.log('\n=== Bulk Ingest ===');
  
  const documents = [
    {
      title: 'Neural Networks Fundamentals',
      content: 'Neural networks are computing systems inspired by biological neural networks...',
      contentType: 'text',
      metadata: {
        category: 'AI/ML',
        difficulty: 'beginner',
        tags: ['neural networks', 'deep learning', 'AI']
      }
    },
    {
      title: 'Database Indexing Strategies',
      content: 'Database indexing is crucial for query performance...',
      contentType: 'text',
      metadata: {
        category: 'Database',
        difficulty: 'intermediate',
        tags: ['database', 'indexing', 'performance']
      }
    }
  ];
  
  const result = await apiCall('POST', '/ingest/bulk', { documents });
  console.log('Bulk ingest results:', result);
  return result;
}

// Example 10: File Upload (if you have a test file)
async function uploadTestFile() {
  console.log('\n=== File Upload ===');
  
  try {
    // Create a test file
    const fs = require('fs');
    const path = require('path');
    const testFilePath = path.join(__dirname, 'test-document.txt');
    
    const testContent = `This is a test document for the PG Vector Search API.
    
It contains information about vector databases, semantic search, and machine learning.
The document will be processed and stored in the database with vector embeddings.
This allows for semantic similarity search using the pgvector extension.`;
    
    fs.writeFileSync(testFilePath, testContent);
    
    const result = await uploadFile('/ingest/file', testFilePath, {
      title: 'Test Document',
      metadata: JSON.stringify({
        category: 'Test',
        source: 'API Example'
      })
    });
    
    console.log('File upload result:', result);
    
    // Clean up test file
    fs.unlinkSync(testFilePath);
    
    return result;
  } catch (error) {
    console.log('File upload example skipped (no test file available)');
    return null;
  }
}

// Example 11: Get Search Statistics
async function getSearchStats() {
  console.log('\n=== Search Statistics ===');
  
  const stats = await apiCall('GET', '/search/stats');
  console.log('Search statistics:', stats);
  return stats;
}

// Main execution function
async function runExamples() {
  try {
    console.log('🚀 PG Vector Search API Examples');
    console.log('================================');
    
    // Check if API is running
    try {
      await apiCall('GET', '/health');
      console.log('✅ API is running');
    } catch (error) {
      console.error('❌ API is not running. Please start the server first.');
      return;
    }
    
    // Run examples
    const documentId = await createDocument();
    await semanticSearch();
    await fullTextSearch();
    await hybridSearch();
    await metadataSearch();
    await getSimilarDocuments(documentId);
    await getAllDocuments();
    await updateDocument(documentId);
    await bulkIngest();
    await uploadTestFile();
    await getSearchStats();
    
    console.log('\n✅ All examples completed successfully!');
    
  } catch (error) {
    console.error('❌ Error running examples:', error);
  }
}

// Run examples if this file is executed directly
if (require.main === module) {
  runExamples();
}

module.exports = {
  createDocument,
  semanticSearch,
  fullTextSearch,
  hybridSearch,
  metadataSearch,
  getSimilarDocuments,
  getAllDocuments,
  updateDocument,
  bulkIngest,
  uploadTestFile,
  getSearchStats
};
