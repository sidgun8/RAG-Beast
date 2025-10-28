import asyncio
from typing import List, Dict, Any, Optional
from .database_service import DatabaseService
from .embedding_service import get_embedding_service
from .text_processor import TextProcessor
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
        self._initialized = True
    
    async def initialize(self):
        """Initialize the document service"""
        await self.db_service.initialize()
    
    async def close(self):
        """Close the document service"""
        await self.db_service.close()
    
    async def add_document(self, title: str, content: str, content_type: str = 'text', 
                          file_path: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add a new document with embedding"""
        if metadata is None:
            metadata = {}
        
        try:
            # Process the document
            processed_doc = self.text_processor.process_document(content, {
                **metadata,
                'content_type': content_type,
                'file_path': file_path
            })
            
            # Get embedding for the main content
            embedding = await self.embedding_service.get_embedding(processed_doc['cleaned_text'])
            
            # Add document to database
            result = await self.db_service.add_document(
                title=title,
                content=processed_doc['cleaned_text'],
                content_type=content_type,
                file_path=file_path,
                metadata=processed_doc['metadata'],
                embedding=embedding
            )
            
            # If document has chunks, process them separately
            if len(processed_doc['chunks']) > 1:
                await self._process_chunks(result['id'], processed_doc['chunks'])
            
            return {
                'id': result['id'],
                'title': title,
                'summary': processed_doc['summary'],
                'key_phrases': processed_doc['key_phrases'],
                'entities': processed_doc['entities'],
                'chunks': len(processed_doc['chunks']),
                'created_at': result['created_at']
            }
            
        except Exception as e:
            logger.error(f'Error adding document: {e}')
            raise
    
    async def _process_chunks(self, parent_id: str, chunks: List[Dict[str, Any]]):
        """Process document chunks with embeddings"""
        for index, chunk in enumerate(chunks):
            try:
                chunk_embedding = await self.embedding_service.get_embedding(chunk['text'])
                
                await self.db_service.add_document(
                    title=f"Chunk {index + 1} of {len(chunks)}",
                    content=chunk['text'],
                    content_type='chunk',
                    file_path=None,
                    metadata={
                        'parentId': str(parent_id),
                        'chunkIndex': index,
                        'start': chunk['start'],
                        'end': chunk['end']
                    },
                    embedding=chunk_embedding
                )
                
            except Exception as e:
                logger.error(f'Error processing chunk {index}: {e}')
                continue
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return await self.db_service.get_document(document_id)
    
    async def get_all_documents(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all documents with pagination"""
        return await self.db_service.get_all_documents(limit, offset)
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a document"""
        # If content is being updated, regenerate the embedding
        if 'content' in updates:
            try:
                embedding = await self.embedding_service.get_embedding(updates['content'])
                # Add embedding to updates
                updates['embedding'] = embedding
            except Exception as e:
                logger.error(f'Error generating embedding for updated content: {e}')
                # Continue without embedding update
        
        return await self.db_service.update_document(document_id, updates)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        return await self.db_service.delete_document(document_id)
    
    async def search_documents(self, query: str, search_type: str = 'semantic', 
                             limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search documents using various methods"""
        if search_type == 'semantic':
            return await self._semantic_search(query, limit, **kwargs)
        elif search_type == 'text':
            return await self._full_text_search(query, limit)
        elif search_type == 'metadata':
            return await self._metadata_search(kwargs.get('metadata', {}), limit)
        else:
            raise ValueError(f'Unknown search type: {search_type}')
    
    async def _semantic_search(self, query: str, limit: int, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        try:
            # Get embedding for the query
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Perform vector similarity search
            results = await self.db_service.semantic_search(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f'Error in semantic search: {e}')
            raise
    
    async def _full_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform full-text search"""
        try:
            results = await self.db_service.full_text_search(query, limit)
            return results
            
        except Exception as e:
            logger.error(f'Error in full-text search: {e}')
            raise
    
    async def _metadata_search(self, metadata_query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search by metadata"""
        try:
            results = await self.db_service.search_by_metadata(metadata_query, limit)
            return results
            
        except Exception as e:
            logger.error(f'Error in metadata search: {e}')
            raise
    
    async def get_similar_documents(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            results = await self.db_service.get_similar_documents(document_id, limit)
            return results
            
        except Exception as e:
            logger.error(f'Error finding similar documents: {e}')
            raise
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            stats = await self.db_service.get_search_stats()
            return stats
            
        except Exception as e:
            logger.error(f'Error getting search stats: {e}')
            raise
    
    async def hybrid_search(self, query: str, limit: int = 10, semantic_weight: float = 0.7, 
                          text_weight: float = 0.3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and text search"""
        try:
            # Get both semantic and full-text results
            semantic_results = await self._semantic_search(query, limit * 3, 0.3)  # Lower threshold for more results
            text_results = await self._full_text_search(query, limit * 3)
            
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
