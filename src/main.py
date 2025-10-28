import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from .routes import search, documents, ingest
from .services.database_service import DatabaseService
from .services.document_service import DocumentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting PG Vector Search API...")
    
    # Initialize services
    db_service = DatabaseService()
    await db_service.initialize()
    
    # Set up database if needed
    try:
        await db_service.setup_database()
        logger.info("Database setup completed")
    except Exception as e:
        logger.warning(f"Database setup warning: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PG Vector Search API...")
    await db_service.close()


# Create FastAPI app
app = FastAPI(
    title="PG Vector Search API",
    description="Data ingestion pipeline and semantic search using pgvector",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
uploads_dir = "src/uploads"
os.makedirs(uploads_dir, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Include routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",  # This would be datetime.now().isoformat()
        "version": "1.0.0"
    }


@app.get("/api")
async def api_documentation():
    """API documentation endpoint"""
    return {
        "name": "PG Vector Search API",
        "version": "1.0.0",
        "description": "Data ingestion pipeline and semantic search using pgvector",
        "endpoints": {
            "search": {
                "POST /api/search/semantic": "Semantic search using vector similarity",
                "POST /api/search/text": "Full-text search using PostgreSQL FTS",
                "POST /api/search/hybrid": "Hybrid search with RRF (Reciprocal Rank Fusion) combining semantic and text search",
                "POST /api/search/rag": "RAG search - Retrieves relevant chunks and generates answer using AWS Bedrock LLM",
                "POST /api/search/metadata": "Search by document metadata",
                "GET /api/search/similar/{documentId}": "Find similar documents",
                "GET /api/search/stats": "Get search statistics"
            },
            "documents": {
                "GET /api/documents": "Get all documents",
                "GET /api/documents/{id}": "Get specific document",
                "POST /api/documents": "Create new document",
                "PUT /api/documents/{id}": "Update document",
                "DELETE /api/documents/{id}": "Delete document"
            },
            "ingest": {
                "POST /api/ingest/file": "Upload and process single file",
                "POST /api/ingest/files": "Upload and process multiple files",
                "POST /api/ingest/text": "Ingest text content directly",
                "POST /api/ingest/bulk": "Bulk ingest from JSON array"
            }
        }
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "availableEndpoints": "/api"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        reload=os.getenv("NODE_ENV") == "development"
    )
