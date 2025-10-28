from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ..services.document_service import DocumentService
from ..services.llm_service import get_llm_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize document service
document_service = DocumentService()


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")


class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic search")
    text_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for text search")
    rrf_k: int = Field(60, ge=1, le=1000, description="RRF parameter k")


class MetadataSearchRequest(BaseModel):
    metadata: Dict[str, Any] = Field(..., description="Metadata query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")


class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve for context")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold for chunks")
    chunk_type: str = Field("semantic", description="Type of search: 'semantic', 'text', or 'hybrid'")
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for semantic search (hybrid only)")
    text_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for text search (hybrid only)")
    rrf_k: int = Field(60, ge=1, le=1000, description="RRF parameter k for hybrid search")
    model_id: Optional[str] = Field(None, description="AWS Bedrock model ID (defaults to configured model)")
    max_tokens: int = Field(1000, ge=100, le=4000, description="Maximum tokens in LLM response")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for LLM")


@router.post("/semantic")
async def semantic_search(request: SemanticSearchRequest):
    """Semantic search using vector similarity"""
    try:
        results = await document_service.search_documents(
            query=request.query,
            search_type='semantic',
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "semantic",
            "algorithm": "Vector Similarity Search",
            "threshold": request.similarity_threshold
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during semantic search")


@router.post("/text")
async def full_text_search(request: TextSearchRequest):
    """Full-text search using PostgreSQL FTS"""
    try:
        results = await document_service.search_documents(
            query=request.query,
            search_type='text',
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "full-text",
            "algorithm": "PostgreSQL Full-Text Search"
        }
        
    except Exception as e:
        logger.error(f"Full-text search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during full-text search")


@router.post("/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """Hybrid search with RRF combining semantic and text search"""
    try:
        if request.semantic_weight + request.text_weight != 1:
            raise HTTPException(
                status_code=400,
                detail="semantic_weight and text_weight must sum to 1"
            )
        
        if request.rrf_k < 1 or request.rrf_k > 1000:
            raise HTTPException(
                status_code=400,
                detail="rrf_k must be between 1 and 1000"
            )
        
        results = await document_service.hybrid_search(
            query=request.query,
            limit=request.limit,
            semantic_weight=request.semantic_weight,
            text_weight=request.text_weight,
            rrf_k=request.rrf_k
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "hybrid-rrf",
            "algorithm": "Reciprocal Rank Fusion (RRF)",
            "parameters": {
                "weights": {
                    "semantic": request.semantic_weight,
                    "text": request.text_weight
                },
                "rrf_k": request.rrf_k,
                "description": f"RRF combines rankings using formula: 1/(k + rank) where k={request.rrf_k}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search with RRF error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during hybrid search with RRF")


@router.post("/metadata")
async def metadata_search(request: MetadataSearchRequest):
    """Search by document metadata"""
    try:
        results = await document_service.search_documents(
            query="",  # Not used for metadata search
            search_type='metadata',
            limit=request.limit,
            metadata=request.metadata
        )
        
        return {
            "metadata": request.metadata,
            "results": results,
            "total": len(results),
            "search_type": "metadata",
            "algorithm": "PostgreSQL JSON Query"
        }
        
    except Exception as e:
        logger.error(f"Metadata search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during metadata search")


@router.get("/similar/{document_id}")
async def get_similar_documents(
    document_id: str,
    limit: int = Query(5, ge=1, le=50, description="Maximum number of similar documents")
):
    """Find documents similar to a given document"""
    try:
        results = await document_service.get_similar_documents(document_id, limit)
        
        return {
            "document_id": document_id,
            "results": results,
            "total": len(results),
            "search_type": "similar"
        }
        
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Similar documents error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error finding similar documents")


@router.post("/rag")
async def rag_search(request: RAGSearchRequest):
    """
    RAG (Retrieval-Augmented Generation) search that retrieves relevant chunks 
    and generates an answer using AWS Bedrock LLM
    """
    try:
        # Step 1: Retrieve relevant chunks using selected search type
        logger.info(f"RAG Search: Retrieving chunks using {request.chunk_type} search for query: {request.query}")
        
        # Validate chunk_type
        if request.chunk_type not in ['semantic', 'text', 'hybrid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunk_type: {request.chunk_type}. Must be 'semantic', 'text', or 'hybrid'"
            )
        
        # For hybrid search, validate weights if provided
        if request.chunk_type == 'hybrid':
            if request.semantic_weight is not None and request.text_weight is not None:
                if abs((request.semantic_weight + request.text_weight) - 1.0) > 0.01:
                    raise HTTPException(
                        status_code=400,
                        detail="semantic_weight and text_weight must sum to 1 for hybrid search"
                    )
            elif request.semantic_weight is not None or request.text_weight is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Both semantic_weight and text_weight must be provided together for hybrid search"
                )
        
        # Retrieve chunks based on chunk_type
        if request.chunk_type == 'semantic':
            chunks = await document_service.search_documents(
                query=request.query,
                search_type='semantic',
                limit=request.limit,
                similarity_threshold=request.similarity_threshold
            )
        elif request.chunk_type == 'text':
            chunks = await document_service.search_documents(
                query=request.query,
                search_type='text',
                limit=request.limit
            )
        else:  # hybrid
            semantic_weight = request.semantic_weight or 0.7
            text_weight = request.text_weight or 0.3
            chunks = await document_service.hybrid_search(
                query=request.query,
                limit=request.limit,
                semantic_weight=semantic_weight,
                text_weight=text_weight,
                rrf_k=request.rrf_k
            )
        
        if not chunks:
            return {
                "query": request.query,
                "answer": "I couldn't find any relevant information to answer your question.",
                "chunks_used": [],
                "total_chunks": 0,
                "search_type": "rag"
            }
        
        # Step 2: Generate answer using LLM with retrieved chunks
        logger.info(f"RAG Search: Generating answer using {len(chunks)} chunks")
        llm_service = get_llm_service()
        
        answer, model_used = llm_service.generate_answer(
            query=request.query,
            context_chunks=chunks,
            model_id=request.model_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "query": request.query,
            "answer": answer,
            "chunks_used": [
                {
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "content_preview": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
                    "similarity_score": chunk.get("similarity_score")
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks),
            "search_type": "rag",
            "chunk_type": request.chunk_type,
            "model_used": model_used,
            "parameters": {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "similarity_threshold": request.similarity_threshold,
                "chunk_type": request.chunk_type
            }
        }
        
    except ValueError as e:
        logger.error(f"RAG search validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        logger.error(f"RAG search permission error: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG search: {str(e)}")


@router.get("/stats")
async def get_search_stats():
    """Get search statistics"""
    try:
        stats = await document_service.get_search_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Search stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting search statistics")
