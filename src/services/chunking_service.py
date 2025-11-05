"""
Chunking Service - Implements multiple text chunking strategies

Strategies:
1. Semantic Chunking: Groups semantically similar sentences using embeddings
2. Sliding Window: Enhanced sliding window with custom delimiters and overlap
3. Recursive Chunking: LangChain-style recursive splitting by hierarchical separators
"""
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking text using various strategies"""
    
    def __init__(self, embedding_service=None):
        """
        Initialize chunking service
        
        Args:
            embedding_service: Optional embedding service for semantic chunking
        """
        self.embedding_service = embedding_service
    
    async def chunk_text(
        self,
        text: str,
        strategy: str = "sliding_window",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        delimiters: Optional[List[str]] = None,
        semantic_similarity_threshold: float = 0.7,
        recursive_separators: Optional[List[str]] = None,
        tenant_id: str = 'default'
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using specified strategy
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy (semantic, sliding_window, recursive)
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks (for sliding_window)
            delimiters: Custom delimiters for sliding_window
            semantic_similarity_threshold: Threshold for semantic chunking
            recursive_separators: Separators for recursive chunking
            
        Returns:
            List of chunk dictionaries with 'text', 'start', 'end', 'metadata'
        """
        if not text or not text.strip():
            return []
        
        if strategy == "semantic":
            return await self._semantic_chunking(
                text,
                chunk_size,
                semantic_similarity_threshold,
                tenant_id
            )
        elif strategy == "sliding_window":
            return self._sliding_window_chunking(
                text, 
                chunk_size, 
                chunk_overlap, 
                delimiters or ["\n\n", "\n", ".", "?", "!"]
            )
        elif strategy == "recursive":
            return self._recursive_chunking(
                text, 
                chunk_size,
                recursive_separators or ["\n\n", "\n", " ", ""]
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    async def _semantic_chunking(
        self, 
        text: str, 
        target_chunk_size: int,
        similarity_threshold: float,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking using embeddings to group similar sentences
        
        Algorithm:
        1. Split text into sentences
        2. Generate embeddings for each sentence
        3. Group consecutive sentences with similarity > threshold
        4. Combine groups into chunks, respecting target size
        """
        if not self.embedding_service:
            logger.warning("Semantic chunking requires embedding service, falling back to sliding_window")
            return self._sliding_window_chunking(text, target_chunk_size, 100, ["\n\n", "\n", "."])
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'metadata': {'strategy': 'semantic', 'sentence_count': len(sentences)}
            }]
        
        try:
            # Generate embeddings for each sentence
            logger.info(f"Generating embeddings for {len(sentences)} sentences")
            embeddings = []
            for sentence in sentences:
                if sentence.strip():
                    # Use tenant-aware embeddings if available
                    if hasattr(self.embedding_service, 'get_embedding_for_tenant'):
                        emb = await self.embedding_service.get_embedding_for_tenant(sentence, tenant_id)
                    else:
                        emb = await self.embedding_service.get_embedding(sentence)
                    embeddings.append(emb)
                else:
                    embeddings.append(None)
            
            # Group sentences by semantic similarity
            chunks = []
            current_chunk_sentences = [sentences[0]]
            current_chunk_start = 0
            
            for i in range(1, len(sentences)):
                if embeddings[i] is None or embeddings[i-1] is None:
                    # Skip empty sentences
                    current_chunk_sentences.append(sentences[i])
                    continue
                
                # Calculate similarity between consecutive sentences
                sim = cosine_similarity(
                    [embeddings[i-1]], 
                    [embeddings[i]]
                )[0][0]
                
                # Calculate current chunk size
                current_chunk_text = ' '.join(current_chunk_sentences)
                
                # Decide whether to continue current chunk or start new one
                if sim >= similarity_threshold and len(current_chunk_text) < target_chunk_size * 1.5:
                    # Continue current chunk
                    current_chunk_sentences.append(sentences[i])
                else:
                    # Finalize current chunk and start new one
                    chunk_text = ' '.join(current_chunk_sentences).strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'start': current_chunk_start,
                            'end': current_chunk_start + len(chunk_text),
                            'metadata': {
                                'strategy': 'semantic',
                                'sentence_count': len(current_chunk_sentences),
                                'avg_similarity': sim
                            }
                        })
                    current_chunk_sentences = [sentences[i]]
                    current_chunk_start = current_chunk_start + len(chunk_text) + 1
            
            # Add final chunk
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start': current_chunk_start,
                        'end': current_chunk_start + len(chunk_text),
                        'metadata': {
                            'strategy': 'semantic',
                            'sentence_count': len(current_chunk_sentences)
                        }
                    })
            
            logger.info(f"Semantic chunking created {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}, falling back to sliding_window")
            return self._sliding_window_chunking(text, target_chunk_size, 100, ["\n\n", "\n", "."])
    
    def _sliding_window_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        delimiters: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Sliding window chunking with custom delimiters
        
        Algorithm:
        1. Try to split at delimiters in priority order
        2. Maintain overlap between consecutive chunks
        3. Respect chunk size while preferring natural boundaries
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to find a good split point using delimiters
            if end < text_length:
                split_point = self._find_best_split_point(
                    text, 
                    start, 
                    end, 
                    delimiters,
                    min_ratio=0.5  # At least 50% of chunk_size
                )
                if split_point > start:
                    end = split_point
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'metadata': {
                        'strategy': 'sliding_window',
                        'overlap': overlap if start > 0 else 0
                    }
                })
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= text_length or (end >= text_length and start < text_length):
                break
        
        logger.info(f"Sliding window chunking created {len(chunks)} chunks")
        return chunks
    
    def _recursive_chunking(
        self,
        text: str,
        chunk_size: int,
        separators: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Recursive chunking using hierarchical separators
        
        Algorithm:
        1. Try to split by first separator
        2. If pieces are too large, recursively split with next separator
        3. Continue until all chunks are within size limit
        """
        chunks = self._recursive_split(text, chunk_size, separators, 0)
        
        # Add metadata and position information
        position = 0
        result = []
        for chunk in chunks:
            chunk_text = chunk.strip()
            if chunk_text:
                result.append({
                    'text': chunk_text,
                    'start': position,
                    'end': position + len(chunk_text),
                    'metadata': {
                        'strategy': 'recursive'
                    }
                })
                position += len(chunk_text) + 1
        
        logger.info(f"Recursive chunking created {len(result)} chunks")
        return result
    
    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        separators: List[str],
        depth: int
    ) -> List[str]:
        """
        Recursively split text by separators
        """
        # Base case: no more separators or text is small enough
        if depth >= len(separators) or len(text) <= chunk_size:
            return [text] if text else []
        
        separator = separators[depth]
        
        # Split by current separator
        if separator:
            pieces = text.split(separator)
        else:
            # Empty separator means split by character
            pieces = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        result = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                # Piece is small enough
                if piece:
                    result.append(piece)
            else:
                # Piece is too large, recurse with next separator
                sub_chunks = self._recursive_split(
                    piece,
                    chunk_size,
                    separators,
                    depth + 1
                )
                result.extend(sub_chunks)
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting using regex
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_split_point(
        self,
        text: str,
        start: int,
        end: int,
        delimiters: List[str],
        min_ratio: float = 0.5
    ) -> int:
        """
        Find the best split point within range using delimiters
        
        Args:
            text: Full text
            start: Start position
            end: End position
            delimiters: List of delimiters in priority order
            min_ratio: Minimum ratio of chunk size to accept split
            
        Returns:
            Best split position, or end if no good split found
        """
        min_pos = start + int((end - start) * min_ratio)
        
        # Try each delimiter in priority order
        for delimiter in delimiters:
            # Search backwards from end for delimiter
            chunk = text[start:end]
            last_pos = chunk.rfind(delimiter)
            
            if last_pos != -1:
                actual_pos = start + last_pos + len(delimiter)
                # Only accept if it's past the minimum position
                if actual_pos >= min_pos:
                    return actual_pos
        
        # No good split point found
        return end

