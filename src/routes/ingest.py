import os
import tempfile
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ..services.document_service import DocumentService
from ..services.file_processor import FileProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
document_service = DocumentService()
file_processor = FileProcessor()


class TextIngestRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    content_type: str = Field("text", description="Content type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class BulkIngestRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to ingest")


@router.post("/file")
async def upload_single_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}")
):
    """Upload and process a single file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Check file type
        if not file_processor.is_supported(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Parse metadata
        try:
            import json
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text content
            extracted_text = await file_processor.extract_text_from_file(temp_file_path)
            
            # Process and store document
            result = await document_service.add_document(
                title=title or file.filename,
                content=extracted_text,
                content_type=file_processor.get_file_type(file.filename)[1:],  # Remove the dot
                file_path=None,
                metadata={
                    **metadata_dict,
                    "original_name": file.filename,
                    "file_size": len(content)
                }
            )
            
            return {
                "message": "File processed and stored successfully",
                "document": result
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing file")


@router.post("/files")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        results = []
        errors = []
        
        for file in files:
            try:
                if not file.filename:
                    errors.append({
                        "filename": file.filename or "unknown",
                        "error": "No filename provided"
                    })
                    continue
                
                if not file_processor.is_supported(file.filename):
                    errors.append({
                        "filename": file.filename,
                        "error": "Unsupported file type"
                    })
                    continue
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Extract text content
                    extracted_text = await file_processor.extract_text_from_file(temp_file_path)
                    
                    # Process and store document
                    result = await document_service.add_document(
                        title=file.filename,
                        content=extracted_text,
                        content_type=file_processor.get_file_type(file.filename)[1:],  # Remove the dot
                        file_path=None,
                        metadata={
                            "original_name": file.filename,
                            "file_size": len(content)
                        }
                    )
                    
                    results.append(result)
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up file: {e}")
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "message": f"Processed {len(results)} files successfully",
            "results": results,
            "errors": errors if errors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiple file upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing files")


@router.post("/text")
async def ingest_text_content(request: TextIngestRequest):
    """Ingest text content directly"""
    try:
        result = await document_service.add_document(
            title=request.title,
            content=request.content,
            content_type=request.content_type,
            file_path=None,
            metadata=request.metadata
        )
        
        return {
            "message": "Text content processed and stored successfully",
            "document": result
        }
        
    except Exception as e:
        logger.error(f"Text ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Error processing text content")


@router.post("/bulk")
async def bulk_ingest(request: BulkIngestRequest):
    """Bulk ingest from JSON array"""
    try:
        results = []
        errors = []
        
        for i, doc_data in enumerate(request.documents):
            try:
                # Validate required fields
                if "title" not in doc_data or "content" not in doc_data:
                    errors.append({
                        "index": i,
                        "error": "Missing required fields: title and content"
                    })
                    continue
                
                result = await document_service.add_document(
                    title=doc_data["title"],
                    content=doc_data["content"],
                    content_type=doc_data.get("content_type", "text"),
                    file_path=None,
                    metadata=doc_data.get("metadata", {})
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                errors.append({
                    "index": i,
                    "error": str(e)
                })
        
        return {
            "message": f"Processed {len(results)} documents successfully",
            "results": results,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Bulk ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Error processing bulk documents")
