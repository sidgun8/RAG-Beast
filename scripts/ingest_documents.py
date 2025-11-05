#!/usr/bin/env python3
"""
General-purpose document ingestion script with tenant support
Ingest text documents, PDFs, or other files into the vector database

Usage:
    python scripts/ingest_documents.py --tenant tenant_a --file document.txt
    python scripts/ingest_documents.py --tenant default --directory ./documents/
    python scripts/ingest_documents.py --help
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.document_service import DocumentService
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def ingest_file(doc_service: DocumentService, file_path: str, tenant_id: str, 
                     content_type: str = 'text', metadata: Optional[Dict] = None,
                     chunking_strategy: Optional[str] = None,
                     chunk_size: Optional[int] = None,
                     chunk_overlap: Optional[int] = None):
    """
    Ingest a single file into the database
    
    Args:
        doc_service: Document service instance
        file_path: Path to the file to ingest
        tenant_id: Tenant identifier
        content_type: Type of content (text, pdf, json, etc.)
        metadata: Optional metadata dictionary
        chunking_strategy: Optional chunking strategy override
        chunk_size: Optional chunk size override
        chunk_overlap: Optional chunk overlap override
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from filename
        title = Path(file_path).stem
        
        # Add file metadata
        file_metadata = {
            'filename': Path(file_path).name,
            'file_type': Path(file_path).suffix,
            'file_size': os.path.getsize(file_path),
            **(metadata or {})
        }
        
        # Ingest document with chunking options
        result = await doc_service.add_document(
            title=title,
            content=content,
            content_type=content_type,
            file_path=str(file_path),
            metadata=file_metadata,
            tenant_id=tenant_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunking_info = f" [{result.get('chunking_strategy', 'default')}, {result.get('chunks', 0)} chunks]"
        logger.info(f"✓ Ingested: {title} (ID: {result['id']}) → tenant '{tenant_id}'{chunking_info}")
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to ingest {file_path}: {e}")
        return None


async def ingest_directory(doc_service: DocumentService, directory: str, tenant_id: str,
                          content_type: str = 'text', recursive: bool = False,
                          chunking_strategy: Optional[str] = None,
                          chunk_size: Optional[int] = None,
                          chunk_overlap: Optional[int] = None):
    """
    Ingest all files from a directory
    
    Args:
        doc_service: Document service instance
        directory: Path to directory
        tenant_id: Tenant identifier
        content_type: Type of content
        recursive: Whether to search recursively
        chunking_strategy: Optional chunking strategy override
        chunk_size: Optional chunk size override
        chunk_overlap: Optional chunk overlap override
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return
    
    # Get all text files
    if recursive:
        files = list(dir_path.rglob("*.txt")) + list(dir_path.rglob("*.md"))
    else:
        files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))
    
    if not files:
        logger.warning(f"No files found in {directory}")
        return
    
    logger.info(f"Found {len(files)} files to ingest")
    
    success_count = 0
    error_count = 0
    
    for file_path in files:
        result = await ingest_file(
            doc_service, 
            str(file_path), 
            tenant_id, 
            content_type,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if result:
            success_count += 1
        else:
            error_count += 1
    
    logger.info(f"\nIngestion complete for tenant '{tenant_id}'!")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {error_count}")


async def ingest_from_text(doc_service: DocumentService, title: str, content: str, 
                          tenant_id: str, metadata: Optional[Dict] = None):
    """
    Ingest content directly from text
    
    Args:
        doc_service: Document service instance
        title: Document title
        content: Document content
        tenant_id: Tenant identifier
        metadata: Optional metadata
    """
    try:
        result = await doc_service.add_document(
            title=title,
            content=content,
            content_type='text',
            metadata=metadata or {},
            tenant_id=tenant_id
        )
        
        logger.info(f"✓ Ingested: {title} (ID: {result['id']}) → tenant '{tenant_id}'")
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to ingest '{title}': {e}")
        return None


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ingest documents into the vector database with tenant support'
    )
    
    parser.add_argument(
        '--tenant',
        type=str,
        default='default',
        help='Tenant ID (default: default)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path to a single file to ingest'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to a directory of files to ingest'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search directory'
    )
    
    parser.add_argument(
        '--content-type',
        type=str,
        default='text',
        help='Content type (default: text)'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        help='Title for direct text input (use with --content)'
    )
    
    parser.add_argument(
        '--content',
        type=str,
        help='Direct content to ingest (use with --title)'
    )
    
    parser.add_argument(
        '--chunking-strategy',
        type=str,
        choices=['semantic', 'sliding_window', 'recursive'],
        help='Override chunking strategy (semantic, sliding_window, recursive)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Override chunk size in characters'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        help='Override chunk overlap in characters'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.file, args.directory, args.content]):
        parser.error("Must specify --file, --directory, or --content")
    
    if args.content and not args.title:
        parser.error("--content requires --title")
    
    try:
        # Initialize services
        logger.info("Initializing document service...")
        doc_service = DocumentService()
        await doc_service.initialize()
        
        logger.info(f"Target tenant: '{args.tenant}'")
        
        # Display chunking configuration if overrides provided
        if args.chunking_strategy or args.chunk_size or args.chunk_overlap:
            logger.info("Chunking overrides:")
            if args.chunking_strategy:
                logger.info(f"  Strategy: {args.chunking_strategy}")
            if args.chunk_size:
                logger.info(f"  Size: {args.chunk_size}")
            if args.chunk_overlap:
                logger.info(f"  Overlap: {args.chunk_overlap}")
        
        # Ingest based on input type
        if args.file:
            logger.info(f"Ingesting file: {args.file}")
            await ingest_file(
                doc_service, 
                args.file, 
                args.tenant, 
                args.content_type,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            
        elif args.directory:
            logger.info(f"Ingesting directory: {args.directory}")
            await ingest_directory(
                doc_service, 
                args.directory, 
                args.tenant,
                args.content_type,
                args.recursive,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            
        elif args.content:
            logger.info(f"Ingesting direct content: {args.title}")
            await ingest_from_text(doc_service, args.title, args.content, args.tenant)
        
        logger.info("\n✅ Ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            await doc_service.close()
        except:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

