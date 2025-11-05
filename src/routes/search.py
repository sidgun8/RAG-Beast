from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json

from ..services.document_service import DocumentService
from ..services.llm_service import get_llm_service
from ..services.tenant_config_service import get_tenant_config_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
document_service = DocumentService()
tenant_config_service = get_tenant_config_service()


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
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter to narrow search results")
    # LLM parameters
    model_id: Optional[str] = Field(None, description="AWS Bedrock model ID (defaults to configured model)")
    max_tokens: int = Field(1000, ge=100, le=4000, description="Maximum tokens in LLM response")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for LLM")


@router.post("/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    tenant_id: str = Query("default", description="Tenant identifier")
):
    """Semantic search using vector similarity"""
    try:
        results = await document_service.search_documents(
            query=request.query,
            search_type='semantic',
            limit=request.limit,
            tenant_id=tenant_id,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "semantic",
            "algorithm": "Vector Similarity Search",
            "threshold": request.similarity_threshold,
            "tenant_id": tenant_id
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during semantic search")


@router.post("/text")
async def full_text_search(
    request: TextSearchRequest,
    tenant_id: str = Query("default", description="Tenant identifier")
):
    """Full-text search using PostgreSQL FTS"""
    try:
        results = await document_service.search_documents(
            query=request.query,
            search_type='text',
            limit=request.limit,
            tenant_id=tenant_id
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "full-text",
            "algorithm": "PostgreSQL Full-Text Search",
            "tenant_id": tenant_id
        }
        
    except Exception as e:
        logger.error(f"Full-text search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during full-text search")


@router.post("/hybrid")
async def hybrid_search(
    request: HybridSearchRequest,
    tenant_id: str = Query("default", description="Tenant identifier")
):
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
            rrf_k=request.rrf_k,
            tenant_id=tenant_id
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "search_type": "hybrid-rrf",
            "algorithm": "Reciprocal Rank Fusion (RRF)",
            "tenant_id": tenant_id,
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
async def metadata_search(
    request: MetadataSearchRequest,
    tenant_id: str = Query("default", description="Tenant identifier")
):
    """Search by document metadata"""
    try:
        results = await document_service.search_documents(
            query="",  # Not used for metadata search
            search_type='metadata',
            limit=request.limit,
            tenant_id=tenant_id,
            metadata=request.metadata
        )
        
        return {
            "metadata": request.metadata,
            "results": results,
            "total": len(results),
            "search_type": "metadata",
            "algorithm": "PostgreSQL JSON Query",
            "tenant_id": tenant_id
        }
        
    except Exception as e:
        logger.error(f"Metadata search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during metadata search")


@router.get("/similar/{document_id}")
async def get_similar_documents(
    document_id: str,
    limit: int = Query(5, ge=1, le=50, description="Maximum number of similar documents"),
    tenant_id: str = Query("default", description="Tenant identifier")
):
    """Find documents similar to a given document"""
    try:
        results = await document_service.get_similar_documents(document_id, limit, tenant_id)
        
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


def _filter_by_metadata(chunks: List[Dict[str, Any]], metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter chunks by metadata constraints.
    Supports nested metadata matching using PostgreSQL @> containment operator logic.
    """
    if not metadata_filter:
        return chunks
    
    filtered_chunks = []
    for chunk in chunks:
        chunk_metadata = chunk.get('metadata', {})
        
        # Handle case where metadata is a JSON string
        if isinstance(chunk_metadata, str):
            try:
                chunk_metadata = json.loads(chunk_metadata)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not chunk_metadata:
            continue
        
        # Check if all filter keys match
        matches = True
        for key, value in metadata_filter.items():
            if key not in chunk_metadata or chunk_metadata[key] != value:
                matches = False
                break
        
        if matches:
            filtered_chunks.append(chunk)
    
    return filtered_chunks


@router.post("/rag")
async def rag_search(
    request: RAGSearchRequest,
    tenant_id: str = Query("default", description="Tenant identifier")
):
    """
    RAG (Retrieval-Augmented Generation) search that retrieves relevant chunks 
    and generates an answer using AWS Bedrock LLM
    
    Supports:
    - Multi-tenant isolation and configuration
    - Optional metadata filtering
    - Multiple search types (semantic, text, hybrid)
    - Configurable LLM parameters
    """
    try:
        # Load tenant configuration
        tenant_config = tenant_config_service.get_tenant_config(tenant_id)
        
        # Apply request overrides or use tenant defaults
        effective_search_type = request.chunk_type or tenant_config.search_type
        
        # Log configuration being used
        logger.info(f"RAG Search for tenant '{tenant_id}': {effective_search_type} search for query: {request.query}")
        
        # Validate chunk_type
        if effective_search_type not in ['semantic', 'text', 'hybrid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunk_type: {effective_search_type}. Must be 'semantic', 'text', or 'hybrid'"
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
        
        # Retrieve chunks based on chunk_type (get more if filtering by metadata)
        retrieval_limit = request.limit * 3 if request.metadata_filter else request.limit
        
        if effective_search_type == 'semantic':
            chunks = await document_service.search_documents(
                query=request.query,
                search_type='semantic',
                limit=retrieval_limit,
                tenant_id=tenant_id,
                similarity_threshold=request.similarity_threshold
            )
        elif effective_search_type == 'text':
            chunks = await document_service.search_documents(
                query=request.query,
                search_type='text',
                limit=retrieval_limit,
                tenant_id=tenant_id
            )
        else:  # hybrid
            semantic_weight = request.semantic_weight or tenant_config.semantic_weight
            text_weight = request.text_weight or tenant_config.text_weight
            chunks = await document_service.hybrid_search(
                query=request.query,
                limit=retrieval_limit,
                semantic_weight=semantic_weight,
                text_weight=text_weight,
                rrf_k=request.rrf_k,
                tenant_id=tenant_id
            )
        
        # Step 1.5: Apply metadata filtering if provided
        if request.metadata_filter:
            logger.info(f"RAG Search: Applying metadata filter: {request.metadata_filter}")
            chunks = _filter_by_metadata(chunks, request.metadata_filter)
            # Trim to requested limit after filtering
            chunks = chunks[:request.limit]
        
        if not chunks:
            return {
                "query": request.query,
                "answer": "I couldn't find any relevant information to answer your question.",
                "chunks_used": [],
                "total_chunks": 0,
                "search_type": "rag",
                "tenant_id": tenant_id,
                "metadata_filter": request.metadata_filter
            }
        
        # Step 2: Generate answer using LLM with retrieved chunks
        logger.info(f"RAG Search: Generating answer using {len(chunks)} chunks")
        llm_service = get_llm_service()
        
        # Get LLM config from tenant (with request overrides)
        llm_config = tenant_config_service.get_llm_config(tenant_id)
        effective_model_id = request.model_id or llm_config['llm_model_id']
        effective_max_tokens = request.max_tokens if request.max_tokens != 1000 else llm_config['llm_max_tokens']
        effective_temperature = request.temperature if request.temperature != 0.7 else llm_config['llm_temperature']
        
        logger.info(f"Using LLM: {effective_model_id} (max_tokens={effective_max_tokens}, temp={effective_temperature})")
        
        answer, model_used = llm_service.generate_answer(
            query=request.query,
            context_chunks=chunks,
            model_id=effective_model_id,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature
        )
        
        # Build response
        response_data = {
            "query": request.query,
            "answer": answer,
            "chunks_used": [
                {
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "content_preview": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
                    "similarity_score": chunk.get("similarity_score"),
                    "metadata": json.loads(chunk.get("metadata", "{}")) if isinstance(chunk.get("metadata"), str) else chunk.get("metadata", {})
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks),
            "search_type": "rag",
            "chunk_type": effective_search_type,
            "tenant_id": tenant_id,
            "model_used": model_used,
            "tenant_config": {
                "embedding_model": tenant_config.embedding_model,
                "embedding_dimensions": tenant_config.embedding_dimensions,
                "search_type": effective_search_type,
                "llm_model_id": llm_config['llm_model_id']
            },
            "llm_config": {
                "model_id": effective_model_id,
                "max_tokens": effective_max_tokens,
                "temperature": effective_temperature
            },
            "parameters": {
                "similarity_threshold": request.similarity_threshold,
                "chunk_type": effective_search_type
            }
        }
        
        # Add metadata filter info if used
        if request.metadata_filter:
            response_data["metadata_filter"] = request.metadata_filter
            response_data["parameters"]["metadata_filter"] = request.metadata_filter
        
        return response_data
        
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
