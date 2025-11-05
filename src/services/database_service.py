import os
import asyncio
import asyncpg
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for PostgreSQL with pgvector support"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.database = os.getenv('DB_NAME', 'vector_search')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD')
        self.embedding_dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))
        
        self._pool = None
        self._initialized = True
    
    async def initialize(self):
        """Initialize database connection pool"""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=1,
                    max_size=10
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
    
    async def setup_database(self):
        """Set up database schema and indexes"""
        try:
            async with self._pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                logger.info('✓ pgvector extension enabled')
                
                # Create documents table
                await conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_type VARCHAR(50),
                        file_path TEXT,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dimensions}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')
                logger.info(f'✓ documents table created with {self.embedding_dimensions}D embeddings')
                
                # Create search indexes
                await self._create_indexes(conn)
                
                # Create triggers
                await self._create_triggers(conn)
                
                logger.info('Database setup completed successfully!')
                
        except Exception as e:
            logger.error(f'Error setting up database: {e}')
            raise
    
    async def _create_indexes(self, conn):
        """Create database indexes"""
        # Vector similarity index
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        ''')
        logger.info('✓ vector similarity index created')
        
        # Full-text search index
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_content_fts_idx 
            ON documents USING gin(to_tsvector('english', title || ' ' || content));
        ''')
        logger.info('✓ full-text search index created')
        
        # Metadata index
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_metadata_idx 
            ON documents USING gin(metadata);
        ''')
        logger.info('✓ metadata index created')
    
    async def _create_triggers(self, conn):
        """Create database triggers"""
        # Create function to update updated_at timestamp
        await conn.execute('''
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        ''')
        
        # Create trigger for updated_at
        await conn.execute('''
            DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
            CREATE TRIGGER update_documents_updated_at
                BEFORE UPDATE ON documents
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        ''')
        logger.info('✓ triggers created')
    
    async def add_document(self, title: str, content: str, content_type: str = 'text', 
                          file_path: Optional[str] = None, metadata: Optional[Dict] = None,
                          embedding: Optional[List[float]] = None, tenant_id: str = 'default') -> Dict[str, Any]:
        """Add a new document to the database"""
        if metadata is None:
            metadata = {}
        
        document_id = str(uuid.uuid4())
        
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Insert main document
                    query = '''
                        INSERT INTO documents (id, title, content, content_type, file_path, metadata, embedding, tenant_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id, created_at
                    '''
                    
                    embedding_str = f"[{','.join(map(str, embedding))}]" if embedding else None
                    
                    result = await conn.fetchrow(query,
                        document_id,
                        title,
                        content,
                        content_type,
                        file_path,
                        json.dumps(metadata),
                        embedding_str,
                        tenant_id
                    )
                    
                    return {
                        'id': result['id'],
                        'title': title,
                        'created_at': result['created_at']
                    }
                    
                except Exception as e:
                    logger.error(f'Error adding document: {e}')
                    raise
    
    async def get_document(self, document_id: str, tenant_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        async with self._pool.acquire() as conn:
            query = 'SELECT * FROM documents WHERE id = $1 AND tenant_id = $2'
            result = await conn.fetchrow(query, document_id, tenant_id)
            
            if result:
                return dict(result)
            return None
    
    async def get_all_documents(self, limit: int = 50, offset: int = 0, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Get all documents with pagination"""
        async with self._pool.acquire() as conn:
            query = '''
                SELECT id, title, content_type, file_path, metadata, created_at, updated_at, tenant_id
                FROM documents 
                WHERE content_type != 'chunk' AND tenant_id = $1
                ORDER BY created_at DESC 
                LIMIT $2 OFFSET $3
            '''
            results = await conn.fetch(query, tenant_id, limit, offset)
            return [dict(row) for row in results]
    
    async def update_document(self, document_id: str, updates: Dict[str, Any], tenant_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Update a document"""
        allowed_fields = ['title', 'content', 'metadata']
        update_fields = []
        values = []
        param_count = 1
        
        for field, value in updates.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = ${param_count}")
                if field == 'metadata':
                    values.append(json.dumps(value))
                else:
                    values.append(value)
                param_count += 1
        
        if not update_fields:
            raise ValueError('No valid fields to update')
        
        values.append(document_id)
        values.append(tenant_id)
        query = f'''
            UPDATE documents 
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ${param_count} AND tenant_id = ${param_count + 1}
            RETURNING *
        '''
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(query, *values)
            return dict(result) if result else None
    
    async def delete_document(self, document_id: str, tenant_id: str = 'default') -> bool:
        """Delete a document and its chunks"""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Delete chunks first
                    await conn.execute(
                        "DELETE FROM documents WHERE metadata->>'parentId' = $1 AND tenant_id = $2", 
                        document_id, tenant_id
                    )
                    
                    # Delete main document
                    result = await conn.execute(
                        "DELETE FROM documents WHERE id = $1 AND tenant_id = $2", 
                        document_id, tenant_id
                    )
                    
                    return result == "DELETE 1"
                    
                except Exception as e:
                    logger.error(f'Error deleting document: {e}')
                    raise
    
    async def semantic_search(self, query_embedding: List[float], limit: int = 10, 
                            similarity_threshold: float = 0.7, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        async with self._pool.acquire() as conn:
            query = '''
                SELECT 
                    id,
                    title,
                    content,
                    content_type,
                    file_path,
                    metadata,
                    created_at,
                    tenant_id,
                    1 - (embedding <=> $1) as similarity_score
                FROM documents 
                WHERE 1 - (embedding <=> $1) > $2 AND tenant_id = $4
                ORDER BY embedding <=> $1
                LIMIT $3
            '''
            
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            results = await conn.fetch(query, embedding_str, similarity_threshold, limit, tenant_id)
            
            return [dict(row) for row in results]
    
    async def full_text_search(self, query: str, limit: int = 10, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Perform full-text search"""
        async with self._pool.acquire() as conn:
            query_sql = '''
                SELECT 
                    id,
                    title,
                    content,
                    content_type,
                    file_path,
                    metadata,
                    created_at,
                    tenant_id,
                    ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', $1)) as rank
                FROM documents 
                WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', $1)
                  AND tenant_id = $3
                ORDER BY rank DESC
                LIMIT $2
            '''
            
            results = await conn.fetch(query_sql, query, limit, tenant_id)
            return [dict(row) for row in results]
    
    async def search_by_metadata(self, metadata_query: Dict[str, Any], limit: int = 10, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Search documents by metadata"""
        async with self._pool.acquire() as conn:
            query = '''
                SELECT 
                    id,
                    title,
                    content,
                    content_type,
                    file_path,
                    metadata,
                    created_at,
                    tenant_id
                FROM documents 
                WHERE metadata @> $1 AND tenant_id = $3
                ORDER BY created_at DESC
                LIMIT $2
            '''
            
            results = await conn.fetch(query, json.dumps(metadata_query), limit, tenant_id)
            return [dict(row) for row in results]
    
    async def get_similar_documents(self, document_id: str, limit: int = 5, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        async with self._pool.acquire() as conn:
            # First get the document's embedding
            doc_query = 'SELECT embedding, tenant_id FROM documents WHERE id = $1 AND tenant_id = $2'
            doc_result = await conn.fetchrow(doc_query, document_id, tenant_id)
            
            if not doc_result:
                raise ValueError('Document not found')
            
            embedding = doc_result['embedding']
            
            # Find similar documents (within same tenant)
            similar_query = '''
                SELECT 
                    id,
                    title,
                    content,
                    content_type,
                    file_path,
                    metadata,
                    created_at,
                    tenant_id,
                    1 - (embedding <=> $1) as similarity_score
                FROM documents 
                WHERE id != $2 AND embedding IS NOT NULL AND tenant_id = $4
                ORDER BY embedding <=> $1
                LIMIT $3
            '''
            
            results = await conn.fetch(similar_query, embedding, document_id, limit, tenant_id)
            return [dict(row) for row in results]
    
    async def get_search_stats(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get search statistics for a tenant"""
        async with self._pool.acquire() as conn:
            stats_query = '''
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(CASE WHEN content_type = 'chunk' THEN 1 END) as total_chunks,
                    COUNT(CASE WHEN content_type != 'chunk' THEN 1 END) as main_documents,
                    AVG(LENGTH(content)) as avg_content_length,
                    MAX(created_at) as latest_document,
                    tenant_id
                FROM documents
                WHERE tenant_id = $1
                GROUP BY tenant_id
            '''
            
            result = await conn.fetchrow(stats_query, tenant_id)
            return dict(result) if result else {
                'total_documents': 0,
                'total_chunks': 0,
                'main_documents': 0,
                'avg_content_length': 0,
                'latest_document': None,
                'tenant_id': tenant_id
            }
