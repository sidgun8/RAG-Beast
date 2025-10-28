# PG Vector Search API - Python Implementation

A comprehensive data ingestion pipeline and semantic search system using PostgreSQL with pgvector extension, built with Python and FastAPI, featuring EmbeddingGemma support.

## Features

- **Semantic Search**: Vector similarity search using embeddings
- **Full-Text Search**: PostgreSQL full-text search capabilities
- **Hybrid Search**: Combines semantic and text search for optimal results
- **Data Ingestion**: Support for multiple file formats (TXT, PDF, DOCX, CSV, JSON)
- **REST API**: Complete RESTful API for all operations (FastAPI-powered)
- **Metadata Search**: Search documents by metadata fields
- **Similar Documents**: Find documents similar to a given document
- **EmbeddingGemma Integration**: Google's state-of-the-art embedding model

## Prerequisites

- Python 3.8 or higher
- PostgreSQL (v12 or higher)
- pgvector extension installed
- Optional: OpenAI API key or Hugging Face API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pg-vector-search
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   # Database Configuration
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=vector_search
   DB_USER=postgres
   DB_PASSWORD=your_password

   # Server Configuration
   PORT=3000
   HOST=0.0.0.0
   NODE_ENV=development

   # Embedding Configuration
   EMBEDDING_MODEL=embeddinggemma
   EMBEDDING_DIMENSIONS=768
   EMBEDDING_DEVICE=cpu
   USE_LOCAL_EMBEDDINGS=true

   # API Keys (Optional)
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

5. **Set up the database**
   ```bash
   python scripts/setup_database.py
   ```

6. **Test the system**
   ```bash
   python scripts/test_functionality.py
   ```

7. **Ingest sample data (optional)**
   ```bash
   python scripts/ingest_data.py
   ```

8. **Start the server**
   ```bash
   python app.py
   # or
   python scripts/start_server.py
   ```

## API Endpoints

### Search Endpoints

#### Semantic Search
```http
POST /api/search/semantic
Content-Type: application/json

{
  "query": "machine learning algorithms",
  "limit": 10,
  "similarityThreshold": 0.7
}
```

#### Full-Text Search
```http
POST /api/search/text
Content-Type: application/json

{
  "query": "database optimization",
  "limit": 10
}
```

#### Hybrid Search
```http
POST /api/search/hybrid
Content-Type: application/json

{
  "query": "vector similarity search",
  "limit": 10,
  "semanticWeight": 0.7,
  "textWeight": 0.3
}
```

#### Metadata Search
```http
POST /api/search/metadata
Content-Type: application/json

{
  "metadata": {
    "category": "Database",
    "difficulty": "advanced"
  },
  "limit": 10
}
```

#### Similar Documents
```http
GET /api/search/similar/{documentId}?limit=5
```

#### Search Statistics
```http
GET /api/search/stats
```

### Document Management Endpoints

#### Get All Documents
```http
GET /api/documents?limit=50&offset=0
```

#### Get Specific Document
```http
GET /api/documents/{id}
```

#### Create Document
```http
POST /api/documents
Content-Type: application/json

{
  "title": "Document Title",
  "content": "Document content...",
  "contentType": "text",
  "metadata": {
    "category": "AI/ML",
    "tags": ["machine learning", "AI"]
  }
}
```

#### Update Document
```http
PUT /api/documents/{id}
Content-Type: application/json

{
  "title": "Updated Title",
  "content": "Updated content...",
  "metadata": {
    "category": "Updated Category"
  }
}
```

#### Delete Document
```http
DELETE /api/documents/{id}
```

### Data Ingestion Endpoints

#### Upload Single File
```http
POST /api/ingest/file
Content-Type: multipart/form-data

file: [file]
title: "Document Title"
metadata: {"category": "AI/ML"}
```

#### Upload Multiple Files
```http
POST /api/ingest/files
Content-Type: multipart/form-data

files: [file1, file2, ...]
```

#### Ingest Text Content
```http
POST /api/ingest/text
Content-Type: application/json

{
  "title": "Document Title",
  "content": "Document content...",
  "contentType": "text",
  "metadata": {
    "category": "AI/ML"
  }
}
```

#### Bulk Ingest
```http
POST /api/ingest/bulk
Content-Type: application/json

{
  "documents": [
    {
      "title": "Document 1",
      "content": "Content 1...",
      "metadata": {"category": "AI/ML"}
    },
    {
      "title": "Document 2", 
      "content": "Content 2...",
      "metadata": {"category": "Database"}
    }
  ]
}
```

## Supported File Types

- **Text files** (.txt)
- **PDF documents** (.pdf)
- **Word documents** (.docx)
- **CSV files** (.csv)
- **JSON files** (.json)

## Database Schema

The system uses a single `documents` table with the following structure:

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  content_type VARCHAR(50),
  file_path TEXT,
  metadata JSONB,
  embedding VECTOR(768),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## EmbeddingGemma Integration

This project uses [EmbeddingGemma](https://huggingface.co/blog/embeddinggemma), Google's new state-of-the-art embedding model, providing several advantages:

- **Completely Free**: No API costs, runs locally
- **High Quality**: State-of-the-art performance on MTEB benchmarks
- **Multilingual**: Supports 100+ languages out of the box
- **Efficient**: Only 308M parameters, under 200MB when quantized
- **Open Source**: Based on Google's open research
- **Fast**: Optimized for production workloads

### EmbeddingGemma Features

- **308M Parameters**: Lightweight yet powerful
- **768 Dimensions**: Optimal for semantic search
- **2K Context Window**: Handles long documents
- **Matryoshka Learning**: Can truncate to 512, 256, or 128 dimensions
- **Local Inference**: No API calls required

### Configuration

EmbeddingGemma works out of the box with no API keys required:

```env
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_DIMENSIONS=768
EMBEDDING_DEVICE=cpu  # or cuda for GPU acceleration
USE_LOCAL_EMBEDDINGS=true
```

Optional: Add Hugging Face API key for faster inference:
```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

## FastAPI Benefits

This API is built with FastAPI, providing several advantages:

- **High Performance**: FastAPI is one of the fastest Python web frameworks
- **Automatic Documentation**: Built-in OpenAPI/Swagger documentation
- **Type Safety**: Automatic type checking and validation
- **Async Support**: Native async/await support
- **Built-in Validation**: Pydantic-based request/response validation
- **Security**: Built-in security features and CORS support
- **File Uploads**: Efficient multipart handling

## Development

### Project Structure

```
src/
├── services/
│   ├── embedding_service.py    # Embedding generation
│   ├── text_processor.py       # Text processing utilities
│   ├── document_service.py     # Document CRUD operations
│   ├── database_service.py     # Database operations
│   └── file_processor.py       # File processing utilities
├── routes/
│   ├── search.py              # Search endpoints
│   ├── documents.py           # Document management endpoints
│   └── ingest.py              # Data ingestion endpoints
└── main.py                    # Main application

scripts/
├── setup_database.py          # Database setup
├── ingest_data.py             # Sample data ingestion
├── test_functionality.py      # Test script
└── start_server.py            # Server startup
```

### Adding New File Types

To support additional file types, extend the `FileProcessor` class in `src/services/file_processor.py`.

### Custom Embedding Models

To use different embedding models, modify the `EmbeddingService` class in `src/services/embedding_service.py`.

## Scripts

- `python scripts/setup_database.py` - Set up database schema
- `python scripts/ingest_data.py` - Ingest sample data
- `python scripts/test_functionality.py` - Test all functionality
- `python scripts/start_server.py` - Start the server
- `python app.py` - Main application entry point

## Troubleshooting

### Common Issues

1. **pgvector extension not found**
   - Install pgvector: `CREATE EXTENSION vector;`
   - Ensure PostgreSQL version is 12+

2. **Embedding model errors**
   - Check if EmbeddingGemma model is available
   - Verify API key configuration if using external services
   - Test with a simple query first

3. **Performance issues**
   - Check database indexes
   - Monitor query execution plans
   - Consider adjusting similarity thresholds

### Health Check

```http
GET /health
```

Returns server status and version information.

## Migration from Node.js

This Python implementation maintains API compatibility with the original Node.js version while providing:

- Better performance with FastAPI
- Native async/await support
- Improved type safety
- Enhanced EmbeddingGemma integration
- Better error handling and logging

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/api`
- Open an issue on GitHub
