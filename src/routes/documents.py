from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ..services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize document service
document_service = DocumentService()


class DocumentCreateRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    content_type: str = Field("text", description="Content type")
    file_path: Optional[str] = Field(None, description="File path if from file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


@router.get("/")
async def get_all_documents(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents"),
    offset: int = Query(0, ge=0, description="Number of documents to skip")
):
    """Get all documents with pagination"""
    try:
        documents = await document_service.get_all_documents(limit, offset)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Get documents error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting documents")


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID"""
    try:
        document = await document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting document")


@router.post("/")
async def create_document(request: DocumentCreateRequest):
    """Create a new document"""
    try:
        result = await document_service.add_document(
            title=request.title,
            content=request.content,
            content_type=request.content_type,
            file_path=request.file_path,
            metadata=request.metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Create document error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error creating document")


@router.put("/{document_id}")
async def update_document(document_id: str, request: DocumentUpdateRequest):
    """Update a document"""
    try:
        # Remove None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        result = await document_service.update_document(document_id, updates)
        
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update document error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error updating document")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = await document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully", "id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error deleting document")
