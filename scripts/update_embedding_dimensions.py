#!/usr/bin/env python3
"""
Update database to use 384 dimensions for embeddings
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.database_service import DatabaseService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def update_embedding_dimensions():
    """Update database to use 384 dimensions"""
    db_service = DatabaseService()
    
    try:
        await db_service.initialize()
        
        logger.info("Updating database to use 384 dimensions...")
        
        async with db_service._pool.acquire() as conn:
            # Drop the existing table
            await conn.execute('DROP TABLE IF EXISTS documents CASCADE;')
            logger.info('Dropped existing documents table')
            
            # Create new table with 384 dimensions
            await conn.execute('''
                CREATE TABLE documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type VARCHAR(50),
                    file_path TEXT,
                    metadata JSONB,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            logger.info('Created documents table with 384D embeddings')
            
            # Recreate indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            ''')
            logger.info('Created vector similarity index')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS documents_content_fts_idx 
                ON documents USING gin(to_tsvector('english', title || ' ' || content));
            ''')
            logger.info('Created full-text search index')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS documents_metadata_idx 
                ON documents USING gin(metadata);
            ''')
            logger.info('Created metadata index')
            
            # Recreate triggers
            await conn.execute('''
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            ''')
            
            await conn.execute('''
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            ''')
            logger.info('Created triggers')
        
        logger.info("✅ Database updated to use 384 dimensions!")
        
    except Exception as e:
        logger.error(f"❌ Error updating database: {e}")
        sys.exit(1)
    finally:
        await db_service.close()


if __name__ == "__main__":
    asyncio.run(update_embedding_dimensions())
