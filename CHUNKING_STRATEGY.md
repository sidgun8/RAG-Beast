# Chunking Strategy Documentation

## Overview

This document describes the chunking strategy used in the PG Vector Search system for processing documents into manageable chunks for vector embedding and retrieval.

## Chunking Implementation

The chunking strategy is implemented in `src/services/text_processor.py` in the `chunk_text()` method.

### Location
- **File**: `src/services/text_processor.py`
- **Method**: `chunk_text(text, max_chunk_size, overlap)`
- **Lines**: 170-195

## Strategy Type: **Sliding Window with Sentence Boundary Detection**

### Key Characteristics

1. **Fixed Size Chunks with Overlap**
   - Default chunk size: **1000 characters**
   - Default overlap: **100 characters**
   - Overlap prevents context loss at chunk boundaries

2. **Sentence-Aware Splitting**
   - Attempts to break chunks at sentence boundaries (`.`, `?`, `!`)
   - Searches backwards from the target chunk size to find sentence end
   - Only breaks at sentence boundary if it's within the second half of the target size (50% rule)

3. **Sliding Window Approach**
   - Each next chunk starts from `end - overlap` of the previous chunk
   - This creates overlap between consecutive chunks

### Algorithm Flow

```
1. Start from position 0
2. Set end = start + max_chunk_size (1000 chars)
3. If end < text length:
   - Search backwards for sentence endings (., ?, !)
   - Find the last sentence end within the chunk range
   - If sentence end found and it's beyond 50% of chunk size:
     - Set end = sentence_end + 1
4. Extract chunk from start to end
5. Set next start = end - overlap
6. Repeat until entire text is processed
```

### Visual Example

```
Original Text (10000 characters):
[================================================================================]

After chunking (1000 char chunks, 100 char overlap):

Chunk 1 (chars 0-1000):
[================================================]
           Overlap
[================]
           |
Chunk 2 (chars 900-1900):
[================][==================================]
           Overlap
                  |
Chunk 3 (chars 1800-2800):
                           [===============]
```

## Configuration Parameters

### Default Values
```python
max_chunk_size: int = 1000  # Maximum characters per chunk
overlap: int = 100          # Overlap between chunks
```

### Why These Values?

1. **1000 characters**: 
   - Balances context preservation with embedding quality
   - Typical embedding models work well with 100-500 tokens (~400-2000 characters)
   - Large enough to maintain context, small enough for precise retrieval

2. **100 characters overlap**:
   - ~10% overlap ensures important context spans chunk boundaries
   - Prevents critical information from being split between chunks
   - Common industry practice (5-20% overlap)

## Sentence Boundary Detection

The chunker uses **Smart Boundary Detection**:

```python
# Find all sentence endings in the chunk range
last_sentence_end = text.rfind('.', start, end)
last_question_end = text.rfind('?', start, end)
last_exclamation_end = text.rfind('!', start, end)

# Use the latest sentence end
last_end = max(last_sentence_end, last_question_end, last_exclamation_end)

# Only use sentence boundary if it's in the second half of chunk
if last_end > start + max_chunk_size * 0.5:
    end = last_end + 1
```

### Benefits:
- **Preserves sentence integrity** - No mid-sentence breaks
- **Maintains readability** - Chunks are more natural
- **Better embeddings** - Complete sentences embed better than fragments
- **50% rule** - Ensures chunks aren't too small

### Fallback Behavior:
- If no sentence boundary is found in the second half, uses exact 1000 chars
- If text is shorter than chunk size, creates single chunk

## Integration with Vector Search

### Chunk Processing Flow

```
1. Document Ingested
   ↓
2. TextProcessor.chunk_text() called
   ↓
3. Chunks created with metadata:
   - text: The chunk content
   - start: Character position in original
   - end: Character position in original
   ↓
4. Each chunk gets its own embedding
   ↓
5. Chunks stored as separate documents with:
   - parentId: Reference to original document
   - chunkIndex: Sequential index
   - start/end: Position metadata
   ↓
6. Chunks indexed in vector database
   ↓
7. Semantic search retrieves relevant chunks
```

### Code Location (document_service.py)

```python
async def _process_chunks(self, parent_id: str, chunks: List[Dict[str, Any]]):
    """Process document chunks with embeddings"""
    for index, chunk in enumerate(chunks):
        chunk_embedding = await self.embedding_service.get_embedding(chunk['text'])
        
        await self.db_service.add_document(
            title=f"Chunk {index + 1} of {len(chunks)}",
            content=chunk['text'],
            content_type='chunk',
            metadata={
                'parentId': str(parent_id),
                'chunkIndex': index,
                'start': chunk['start'],
                'end': chunk['end']
            },
            embedding=chunk_embedding
        )
```

## Advantages of This Strategy

### 1. **Context Preservation**
- Overlap ensures no information is lost at boundaries
- Sentence-aware splitting maintains semantic coherence

### 2. **Retrieval Quality**
- Smaller, focused chunks improve retrieval precision
- Multiple retrieval opportunities for similar content (due to overlap)

### 3. **Flexibility**
- Configurable chunk size and overlap
- Can be adjusted based on document types

### 4. **Traceability**
- Start/end positions allow reconstruction of original document
- Parent-child relationship maintained

## Limitations

### 1. **Fixed Character Count**
- Doesn't consider semantic boundaries (paragraphs, sections)
- May split related concepts

### 2. **Overlap Redundancy**
- Slightly increases storage and indexing overhead
- Similar content retrieved multiple times

### 3. **Simple Sentence Detection**
- Only looks for sentence punctuation
- Doesn't handle abbreviations or lists well

## Alternative Strategies (Not Currently Used)

### 1. **Semantic Chunking**
```python
# Would use NLP to identify semantic boundaries
# More complex, better for structured documents
```

### 2. **Token-Based Chunking**
```python
# Count tokens instead of characters
# Better aligned with LLM tokenizers
```

### 3. **Recursive Character Splitter**
```python
# Tries multiple separators hierarchically
# Paragraphs → Sentences → Words
```

### 4. **Fixed Token Chunks**
```python
# Uses actual tokenizer to count tokens
# More accurate for embedding models
```

## Recommendations for Improvement

### Short Term
1. **Add paragraph-aware chunking**:
   - Detect paragraphs using double newlines
   - Prefer paragraph boundaries over sentence boundaries

2. **Token-based sizing**:
   - Use actual tokenizer to ensure chunk size limits
   - More accurate than character counting

3. **Configurable chunking strategy**:
   - Allow different strategies per document type
   - User-configurable chunk sizes

### Medium Term
1. **Hybrid semantic/character chunking**
   - Use NLP to identify semantic boundaries
   - Combine with character limits

2. **Adaptive overlap**
   - Increase overlap for technical content
   - Reduce for narrative content

3. **Chunk quality scoring**
   - Rate chunk coherence and completeness
   - Prioritize high-quality chunks

## Usage Example

```python
from src.services.text_processor import TextProcessor

processor = TextProcessor()

text = "Your long document text here..."

# Chunk with defaults (1000 chars, 100 overlap)
chunks = processor.chunk_text(text)

# Custom chunk size and overlap
chunks = processor.chunk_text(text, max_chunk_size=500, overlap=50)

for chunk in chunks:
    print(f"Chunk: {chunk['text'][:50]}...")
    print(f"Position: {chunk['start']} to {chunk['end']}")
```

## Summary

The current chunking strategy uses a **sliding window approach with sentence boundary detection**. It's simple, effective, and well-suited for general-purpose document processing. The strategy prioritizes:

✅ **Simplicity** - Easy to understand and maintain  
✅ **Reliability** - Predictable behavior  
✅ **Context preservation** - Overlap prevents information loss  
✅ **Natural boundaries** - Sentence-aware splitting  

For production use, consider enhancements like paragraph awareness, token-based sizing, or semantic boundary detection based on your specific requirements.

