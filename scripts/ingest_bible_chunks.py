#!/usr/bin/env python3
"""
Bulk ingestion script for Bible chunks
Reads JSON files from bible_chunks/ folder and ingests them into the database
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.document_service import DocumentService
from services.embedding_service import get_embedding_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def load_bible_chunks(bible_chunks_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON files from bible_chunks directory"""
    chunks = []
    bible_dir = Path(bible_chunks_dir)
    
    if not bible_dir.exists():
        logger.error(f"Directory {bible_chunks_dir} does not exist")
        return chunks
    
    # Get all JSON files recursively
    json_files = sorted(bible_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {bible_chunks_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract relevant information
                chunk = {
                    'id': data.get('id', ''),
                    'reference': data.get('reference', ''),
                    'bookId': data.get('bookId', ''),
                    'number': data.get('number', ''),
                    'content': data.get('content', ''),
                    'verseCount': data.get('verseCount', 0),
                    'file_path': str(json_file)
                }
                
                if chunk['content'].strip():  # Only add non-empty chunks
                    chunks.append(chunk)
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {json_file}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(chunks)} Bible chunks")
    return chunks


async def ingest_bible_chunks(chunks: List[Dict[str, Any]], doc_service: DocumentService, tenant_id: str = 'default'):
    """
    Ingest Bible chunks into the database for a specific tenant
    
    Args:
        chunks: List of Bible chunk dictionaries
        doc_service: Document service instance
        tenant_id: Tenant identifier (default: 'default')
    """
    total = len(chunks)
    success_count = 0
    error_count = 0
    
    logger.info(f"Starting ingestion of {total} Bible chunks for tenant '{tenant_id}'...")
    
    for idx, chunk in enumerate(chunks, 1):
        try:
            # Create title from reference
            title = f"{chunk['reference']}"
            
            # Create metadata
            metadata = {
                'bibleId': chunk.get('id', ''),
                'bookId': chunk['bookId'],
                'chapterNumber': chunk['number'],
                'verseCount': chunk.get('verseCount', 0),
                'file_path': chunk['file_path'],
                'source': 'Bible KJV',
                'reference': chunk['reference']
            }
            
            # Ingest the document with tenant_id
            result = await doc_service.add_document(
                title=title,
                content=chunk['content'],
                content_type='bible_chunk',
                file_path=chunk['file_path'],
                metadata=metadata,
                tenant_id=tenant_id  # Assign to specific tenant
            )
            
            success_count += 1
            
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total} chunks ingested ({success_count} successful, {error_count} errors)")
        
        except Exception as e:
            error_count += 1
            logger.error(f"Failed to ingest chunk {idx}: {e}")
            continue
    
    logger.info(f"\nIngestion complete for tenant '{tenant_id}'!")
    logger.info(f"Total chunks: {total}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")


async def main():
    """Main entry point"""
    try:
        # Initialize services
        logger.info("Initializing services...")
        doc_service = DocumentService()
        await doc_service.initialize()
        
        # Get Bible chunks directory
        bible_chunks_dir = os.path.join(os.path.dirname(__file__), '..', 'bible_chunks')
        
        if not os.path.exists(bible_chunks_dir):
            logger.error(f"Bible chunks directory not found: {bible_chunks_dir}")
            logger.info("Please ensure bible_chunks folder is in the project root")
            return
        
        # Load all Bible chunks
        chunks = await load_bible_chunks(bible_chunks_dir)
        
        if not chunks:
            logger.warning("No Bible chunks found to ingest")
            return
        
        # Ask for tenant_id
        print("\n" + "=" * 60)
        print("Tenant Selection")
        print("=" * 60)
        print("Enter the tenant ID for this ingestion.")
        print("Available tenants: default, tenant_a, tenant_b, tenant_c")
        print("Or create a new tenant by entering a new ID.")
        tenant_id = input("\nTenant ID (press Enter for 'default'): ").strip()
        
        # Default to 'default' if empty
        if not tenant_id:
            tenant_id = 'default'
        
        # Validate tenant_id format
        if not tenant_id.replace('_', '').replace('-', '').isalnum():
            logger.error("Invalid tenant ID. Use only alphanumeric characters, hyphens, and underscores.")
            return
        
        # Ask for confirmation
        logger.info(f"\nReady to ingest {len(chunks)} Bible chunks into tenant '{tenant_id}'")
        response = input("Do you want to proceed? (y/n): ")
        
        if response.lower() != 'y':
            logger.info("Ingestion cancelled")
            return
        
        # Ingest chunks with tenant_id
        await ingest_bible_chunks(chunks, doc_service, tenant_id)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        # Cleanup
        try:
            await doc_service.close()
        except:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

