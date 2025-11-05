import asyncio
from typing import List, Dict, Any, Optional
from .database_service import DatabaseService
from .embedding_service import get_embedding_service
from .text_processor import TextProcessor
from .chunking_service import ChunkingService
from .tenant_config_service import get_tenant_config_service
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """Document service for managing documents and their embeddings"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.db_service = DatabaseService()
        self.embedding_service = get_embedding_service()
        self.text_processor = TextProcessor()
        self.chunking_service = ChunkingService(self.embedding_service)
        self.tenant_config_service = get_tenant_config_service()
        self._initialized = True
    
    async def initialize(self):
        """Initialize the document service"""
        await self.db_service.initialize()
    
    async def close(self):
        """Close the document service"""
        await self.db_service.close()
    
    async def add_document(self, title: str, content: str, content_type: str = 'text', 
                          file_path: Optional[str] = None, metadata: Optional[Dict] = None, 
                          tenant_id: str = 'default',
                          # Chunking override parameters
                          chunking_strategy: Optional[str] = None,
                          chunk_size: Optional[int] = None,
                          chunk_overlap: Optional[int] = None,
                          delimiters: Optional[List[str]] = None,
                          semantic_similarity_threshold: Optional[float] = None,
                          recursive_separators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a new document with embedding and tenant-aware chunking
        
        Args:
            title: Document title
            content: Document content
            content_type: Type of content
            file_path: Optional file path
            metadata: Optional metadata
            tenant_id: Tenant identifier
            chunking_strategy: Override chunking strategy (semantic, sliding_window, recursive)
            chunk_size: Override chunk size
            chunk_overlap: Override chunk overlap
            delimiters: Override delimiters for sliding_window
            semantic_similarity_threshold: Override threshold for semantic chunking
            recursive_separators: Override separators for recursive chunking
        """
        if metadata is None:
            metadata = {}
        
        try:
            # Get tenant's chunking configuration
            tenant_config = self.tenant_config_service.get_tenant_config(tenant_id)
            chunking_config = self.tenant_config_service.get_chunking_config(tenant_id)
            
            # Apply overrides or use tenant defaults
            effective_strategy = chunking_strategy or chunking_config['chunking_strategy']
            effective_chunk_size = chunk_size or chunking_config['chunk_size']
            effective_chunk_overlap = chunk_overlap or chunking_config['chunk_overlap']
            effective_delimiters = delimiters or chunking_config['delimiters']
            effective_sem_threshold = semantic_similarity_threshold or chunking_config['semantic_similarity_threshold']
            effective_rec_separators = recursive_separators or chunking_config['recursive_separators']
            
            logger.info(f"Adding document '{title}' for tenant '{tenant_id}' with {effective_strategy} chunking")
            
            # Clean the text
            cleaned_text = self.text_processor.clean_text(content)
            
            # Get summary and metadata
            summary = self.text_processor.extract_summary(cleaned_text)
            key_phrases = self.text_processor.extract_key_phrases(cleaned_text)
            entities = self.text_processor.extract_entities(cleaned_text)
            
            # Chunk the document using tenant's strategy
            chunks = await self.chunking_service.chunk_text(
                text=cleaned_text,
                strategy=effective_strategy,
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
                delimiters=effective_delimiters,
                semantic_similarity_threshold=effective_sem_threshold,
                recursive_separators=effective_rec_separators,
                tenant_id=tenant_id
            )
            
            # Get embedding for the main content (tenant-aware)
            if hasattr(self.embedding_service, 'get_embedding_for_tenant'):
                embedding = await self.embedding_service.get_embedding_for_tenant(cleaned_text, tenant_id)
            else:
                embedding = await self.embedding_service.get_embedding(cleaned_text)
            
            # Add document to database
            result = await self.db_service.add_document(
                title=title,
                content=cleaned_text,
                content_type=content_type,
                file_path=file_path,
                metadata={
                    **metadata,
                    'summary': summary,
                    'key_phrases': key_phrases[:5],  # Top 5 key phrases
                    'chunking_strategy': effective_strategy,
                    'chunk_count': len(chunks)
                },
                embedding=embedding,
                tenant_id=tenant_id
            )
            
            # Process chunks separately
            if len(chunks) > 1:
                await self._process_chunks(result['id'], chunks, tenant_id, effective_strategy)
            
            return {
                'id': result['id'],
                'title': title,
                'summary': summary,
                'key_phrases': key_phrases,
                'entities': entities,
                'chunks': len(chunks),
                'chunking_strategy': effective_strategy,
                'created_at': result['created_at']
            }
            
        except Exception as e:
            logger.error(f'Error adding document: {e}')
            raise
    
    async def _process_chunks(self, parent_id: str, chunks: List[Dict[str, Any]], 
                             tenant_id: str = 'default', strategy: str = 'sliding_window'):
        """Process document chunks with embeddings"""
        for index, chunk in enumerate(chunks):
            try:
                if hasattr(self.embedding_service, 'get_embedding_for_tenant'):
                    chunk_embedding = await self.embedding_service.get_embedding_for_tenant(chunk['text'], tenant_id)
                else:
                    chunk_embedding = await self.embedding_service.get_embedding(chunk['text'])
                
                chunk_metadata = {
                    'parentId': str(parent_id),
                    'chunkIndex': index,
                    'start': chunk.get('start', 0),
                    'end': chunk.get('end', 0),
                    'chunking_strategy': strategy
                }
                
                # Add strategy-specific metadata
                if 'metadata' in chunk:
                    chunk_metadata.update(chunk['metadata'])
                
                await self.db_service.add_document(
                    title=f"Chunk {index + 1} of {len(chunks)}",
                    content=chunk['text'],
                    content_type='chunk',
                    file_path=None,
                    metadata=chunk_metadata,
                    embedding=chunk_embedding,
                    tenant_id=tenant_id
                )
                
            except Exception as e:
                logger.error(f'Error processing chunk {index}: {e}')
                continue
    
    async def get_document(self, document_id: str, tenant_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return await self.db_service.get_document(document_id, tenant_id)
    
    async def get_all_documents(self, limit: int = 50, offset: int = 0, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Get all documents with pagination"""
        return await self.db_service.get_all_documents(limit, offset, tenant_id)
    
    async def update_document(self, document_id: str, updates: Dict[str, Any], tenant_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Update a document"""
        # If content is being updated, regenerate the embedding
        if 'content' in updates:
            try:
                if hasattr(self.embedding_service, 'get_embedding_for_tenant'):
                    embedding = await self.embedding_service.get_embedding_for_tenant(updates['content'], tenant_id)
                else:
                    embedding = await self.embedding_service.get_embedding(updates['content'])
                # Add embedding to updates
                updates['embedding'] = embedding
            except Exception as e:
                logger.error(f'Error generating embedding for updated content: {e}')
                # Continue without embedding update
        
        return await self.db_service.update_document(document_id, updates, tenant_id)
    
    async def delete_document(self, document_id: str, tenant_id: str = 'default') -> bool:
        """Delete a document and its chunks"""
        return await self.db_service.delete_document(document_id, tenant_id)
    
    async def search_documents(self, query: str, search_type: str = 'semantic', 
                             limit: int = 10, tenant_id: str = 'default', **kwargs) -> List[Dict[str, Any]]:
        """Search documents using various methods"""
        if search_type == 'semantic':
            return await self._semantic_search(query, limit, tenant_id, **kwargs)
        elif search_type == 'text':
            return await self._full_text_search(query, limit, tenant_id)
        elif search_type == 'metadata':
            return await self._metadata_search(kwargs.get('metadata', {}), limit, tenant_id)
        else:
            raise ValueError(f'Unknown search type: {search_type}')
    
    async def _semantic_search(self, query: str, limit: int, tenant_id: str = 'default', 
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        try:
            # Get embedding for the query (tenant-aware)
            if hasattr(self.embedding_service, 'get_embedding_for_tenant'):
                query_embedding = await self.embedding_service.get_embedding_for_tenant(query, tenant_id)
            else:
                query_embedding = await self.embedding_service.get_embedding(query)
            
            # Perform vector similarity search
            results = await self.db_service.semantic_search(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                tenant_id=tenant_id
            )
            
            return results
            
        except Exception as e:
            logger.error(f'Error in semantic search: {e}')
            raise
    
    async def _full_text_search(self, query: str, limit: int, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Perform full-text search"""
        try:
            results = await self.db_service.full_text_search(query, limit, tenant_id)
            return results
            
        except Exception as e:
            logger.error(f'Error in full-text search: {e}')
            raise
    
    async def _metadata_search(self, metadata_query: Dict[str, Any], limit: int, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Search by metadata"""
        try:
            results = await self.db_service.search_by_metadata(metadata_query, limit, tenant_id)
            return results
            
        except Exception as e:
            logger.error(f'Error in metadata search: {e}')
            raise
    
    async def get_similar_documents(self, document_id: str, limit: int = 5, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            results = await self.db_service.get_similar_documents(document_id, limit, tenant_id)
            return results
            
        except Exception as e:
            logger.error(f'Error finding similar documents: {e}')
            raise
    
    async def get_search_stats(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get search statistics"""
        try:
            stats = await self.db_service.get_search_stats(tenant_id)
            return stats
            
        except Exception as e:
            logger.error(f'Error getting search stats: {e}')
            raise
    
    async def hybrid_search(self, query: str, limit: int = 10, semantic_weight: float = 0.7, 
                          text_weight: float = 0.3, rrf_k: int = 60, tenant_id: str = 'default') -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and text search"""
        try:
            # Get both semantic and full-text results
            semantic_results = await self._semantic_search(query, limit * 3, tenant_id, 0.3)  # Lower threshold for more results
            text_results = await self._full_text_search(query, limit * 3, tenant_id)
            
            # Create a map to combine results with RRF
            result_map = {}
            
            # Add semantic results with RRF scoring
            for index, result in enumerate(semantic_results):
                rrf_score = 1 / (rrf_k + index + 1)  # RRF formula: 1/(k + rank)
                result_map[result['id']] = {
                    **result,
                    'semantic_score': result.get('similarity_score', 0),
                    'semantic_rank': index + 1,
                    'text_score': 0,
                    'text_rank': 0,
                    'rrf_score': rrf_score * semantic_weight,
                    'combined_score': result.get('similarity_score', 0) * semantic_weight
                }
            
            # Add or update with text results using RRF
            for index, result in enumerate(text_results):
                rrf_score = 1 / (rrf_k + index + 1)  # RRF formula: 1/(k + rank)
                doc_id = result['id']
                
                if doc_id in result_map:
                    existing = result_map[doc_id]
                    existing['text_score'] = result.get('rank', 0)
                    existing['text_rank'] = index + 1
                    existing['rrf_score'] += rrf_score * text_weight
                    # Update combined score with RRF
                    existing['combined_score'] = existing['rrf_score'] + (existing['semantic_score'] * semantic_weight * 0.3)
                else:
                    result_map[doc_id] = {
                        **result,
                        'semantic_score': 0,
                        'semantic_rank': 0,
                        'text_score': result.get('rank', 0),
                        'text_rank': index + 1,
                        'rrf_score': rrf_score * text_weight,
                        'combined_score': rrf_score * text_weight
                    }
            
            # Sort by RRF score first, then by combined score
            combined_results = sorted(
                result_map.values(),
                key=lambda x: (-x['rrf_score'], -x['combined_score'])
            )[:limit]
            
            # Add RRF metadata to results
            for i, result in enumerate(combined_results):
                result['rrf_rank'] = i + 1
                result['search_metadata'] = {
                    'semantic_rank': result['semantic_rank'],
                    'text_rank': result['text_rank'],
                    'rrf_score': result['rrf_score'],
                    'rrf_k': rrf_k,
                    'weights': {
                        'semantic': semantic_weight,
                        'text': text_weight
                    }
                }
            
            return combined_results
            
        except Exception as e:
            logger.error(f'Error in hybrid search with RRF: {e}')
            raise
