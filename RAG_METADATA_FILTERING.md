# RAG Metadata Filtering Guide

## Overview

The RAG (Retrieval-Augmented Generation) endpoint now supports **metadata filtering**, allowing you to narrow search results to specific document types, categories, or other metadata attributes before generating answers.

This is particularly useful when you have a large collection of documents and want to:
- Search only within specific categories (e.g., specific Bible books)
- Filter by document source or version
- Limit results to specific chapters, sections, or time periods
- Combine semantic/text search with structured metadata constraints

## How It Works

### Search Flow with Metadata Filtering

```
User Query → Vector/Text Search → Metadata Filtering → LLM Generation → Answer
```

1. **Initial Retrieval**: The system performs semantic, text, or hybrid search to find relevant chunks
2. **Metadata Filtering**: Results are filtered to match your metadata criteria
3. **Answer Generation**: The LLM generates an answer using only the filtered chunks

### Why Post-Filter Instead of Pre-Filter?

The system retrieves more chunks initially (3x the requested limit when using metadata filters) to ensure you still get enough results after filtering. This approach:
- Maintains search quality by finding the most relevant content first
- Ensures sufficient context for answer generation
- Handles sparse metadata matches gracefully

## API Reference

### Endpoint

```
POST /api/search/rag
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The question to answer |
| `limit` | integer | No | 5 | Number of chunks to retrieve (1-20) |
| `similarity_threshold` | float | No | 0.5 | Minimum similarity score (0.0-1.0) |
| `chunk_type` | string | No | "semantic" | Search type: "semantic", "text", or "hybrid" |
| `metadata_filter` | object | No | null | **NEW**: Metadata attributes to filter by |
| `semantic_weight` | float | No | 0.7 | Weight for semantic search (hybrid only) |
| `text_weight` | float | No | 0.3 | Weight for text search (hybrid only) |
| `rrf_k` | integer | No | 60 | RRF parameter for hybrid search |
| `model_id` | string | No | null | AWS Bedrock model ID |
| `max_tokens` | integer | No | 1000 | Maximum tokens in response (100-4000) |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-1.0) |

### Metadata Filter Format

The `metadata_filter` parameter is a JSON object where keys are metadata field names and values are the required values:

```json
{
  "metadata_filter": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

**Matching Logic**: All specified fields must match exactly (AND logic). The filter uses PostgreSQL's JSON containment operator semantics.

## Bible Chunks Metadata

For Bible chunks ingested using the `ingest_bible_chunks.py` script, the following metadata is available:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `bookId` | string | Book abbreviation | "1CO", "GEN", "PSA" |
| `chapterNumber` | string | Chapter number | "1", "13", "119" |
| `verseCount` | integer | Number of verses | 13 |
| `source` | string | Bible version | "Bible KJV" |
| `reference` | string | Human-readable reference | "1 Corinthians 13" |
| `bibleId` | string | Unique chunk identifier | "1CO.13" |

## Examples

### Example 1: Search in a Specific Book

Search for content about love only in 1 Corinthians:

```bash
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does it teach about love?",
    "limit": 3,
    "metadata_filter": {
      "bookId": "1CO"
    }
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it teach about love?",
        "limit": 3,
        "metadata_filter": {"bookId": "1CO"}
    }
)

result = response.json()
print(result['answer'])
```

### Example 2: Search in a Specific Chapter

Find content about the Spirit in Romans Chapter 8:

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it say about the Spirit?",
        "limit": 5,
        "metadata_filter": {
            "bookId": "ROM",
            "chapterNumber": "8"
        }
    }
)
```

### Example 3: Filter by Source

Search only in KJV Bible:

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "teachings about wisdom",
        "limit": 5,
        "metadata_filter": {"source": "Bible KJV"}
    }
)
```

### Example 4: Hybrid Search with Metadata Filter

Combine hybrid search with metadata filtering:

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "prayers about protection",
        "limit": 5,
        "chunk_type": "hybrid",
        "semantic_weight": 0.6,
        "text_weight": 0.4,
        "metadata_filter": {"bookId": "PSA"}
    }
)
```

### Example 5: No Metadata Filter

Search across all documents (default behavior):

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does the Bible say about faith?",
        "limit": 5
        # No metadata_filter - searches everything
    }
)
```

## Response Format

### With Metadata Filter

```json
{
  "query": "What does it teach about love?",
  "answer": "According to 1 Corinthians 13, love is patient...",
  "chunks_used": [
    {
      "id": "uuid-123",
      "title": "1 Corinthians 13",
      "content_preview": "Though I speak with the tongues...",
      "similarity_score": 0.89,
      "metadata": {
        "bookId": "1CO",
        "chapterNumber": "13",
        "verseCount": 13,
        "source": "Bible KJV",
        "reference": "1 Corinthians 13"
      }
    }
  ],
  "total_chunks": 1,
  "search_type": "rag",
  "chunk_type": "semantic",
  "model_used": "anthropic.claude-3-haiku-20240307-v1:0",
  "metadata_filter": {
    "bookId": "1CO"
  },
  "parameters": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "similarity_threshold": 0.5,
    "chunk_type": "semantic",
    "metadata_filter": {
      "bookId": "1CO"
    }
  }
}
```

### Key Response Fields

- `metadata_filter`: The filter that was applied (null if no filter)
- `chunks_used[].metadata`: Full metadata for each chunk used
- `parameters.metadata_filter`: Filter included in parameters for reference

## Use Cases

### Bible Study Application

```python
# Search for parables only in the Gospels
def search_gospels(query):
    gospels = ["MAT", "MAR", "LUK", "JOH"]
    
    # Search each gospel and combine results
    all_results = []
    for gospel in gospels:
        response = requests.post(
            "http://localhost:8000/api/search/rag",
            json={
                "query": query,
                "limit": 2,
                "metadata_filter": {"bookId": gospel}
            }
        )
        all_results.append(response.json())
    
    return all_results

# Use it
results = search_gospels("Tell me about the parables")
```

### Topical Study

```python
# Study a topic across Old Testament books
def study_topic_in_ot(topic, books):
    for book in books:
        response = requests.post(
            "http://localhost:8000/api/search/rag",
            json={
                "query": topic,
                "limit": 3,
                "metadata_filter": {"bookId": book}
            }
        )
        
        result = response.json()
        print(f"\n=== {book} ===")
        print(result['answer'])

# Study prophecy in major prophets
study_topic_in_ot(
    "prophecies about the Messiah",
    ["ISA", "JER", "EZE", "DAN"]
)
```

### Chapter-by-Chapter Analysis

```python
# Analyze each chapter of a book
def analyze_book(book_id, query):
    for chapter in range(1, 30):  # Adjust range as needed
        response = requests.post(
            "http://localhost:8000/api/search/rag",
            json={
                "query": query,
                "limit": 1,
                "metadata_filter": {
                    "bookId": book_id,
                    "chapterNumber": str(chapter)
                }
            }
        )
        
        result = response.json()
        if result['total_chunks'] > 0:
            print(f"Chapter {chapter}: {result['answer'][:100]}...")

analyze_book("ROM", "main themes")
```

## Performance Considerations

### Retrieval Strategy

When using metadata filters, the system:
1. Retrieves 3x the requested limit initially
2. Applies metadata filtering
3. Returns up to the requested limit after filtering

This ensures good results even with restrictive filters.

### Optimization Tips

1. **Be Specific**: Use precise metadata values
   - Good: `{"bookId": "1CO"}`
   - Less Good: Searching without filters when you know the book

2. **Combine Search Types**: Use hybrid search with metadata filters for best results
   ```python
   {
     "chunk_type": "hybrid",
     "metadata_filter": {"bookId": "PSA"}
   }
   ```

3. **Adjust Thresholds**: Lower similarity thresholds when using metadata filters
   ```python
   {
     "similarity_threshold": 0.3,  # Lower when metadata filters are strict
     "metadata_filter": {"bookId": "1CO", "chapterNumber": "13"}
   }
   ```

## Custom Metadata

When ingesting your own documents, you can define custom metadata:

```python
await doc_service.add_document(
    title="My Document",
    content="Document content...",
    metadata={
        "category": "research",
        "topic": "AI",
        "year": "2024",
        "author": "John Doe"
    }
)
```

Then filter by your custom metadata:

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "AI research findings",
        "metadata_filter": {
            "category": "research",
            "year": "2024"
        }
    }
)
```

## Running Examples

Try the included example scripts:

### Bible-Specific Examples
```bash
python examples/bible_rag_metadata.py
```

This runs 7 comprehensive examples showing:
- Search in specific books
- Chapter-level filtering
- Source filtering
- Hybrid search with metadata
- Advanced parameter combinations

### General RAG Examples
```bash
python examples/rag_usage.py
```

Updated to include metadata filtering examples (Examples 5-6).

## Troubleshooting

### No Results with Metadata Filter

**Problem**: Getting zero results when you expect some.

**Solutions**:
1. Check metadata field names match exactly
2. Verify metadata values are correct (case-sensitive)
3. Lower the `similarity_threshold`
4. Increase the `limit`
5. Check that documents with that metadata exist

```python
# Debug: Check what metadata exists
response = requests.post(
    "http://localhost:8000/api/search/metadata",
    json={
        "metadata": {"bookId": "1CO"},
        "limit": 10
    }
)
print(response.json())
```

### Too Few Results

**Problem**: Getting fewer results than expected.

**Solution**: The system retrieves 3x the limit when filtering, but if this still isn't enough:

```python
# Increase the limit
{
  "query": "...",
  "limit": 10,  # Will retrieve up to 30, then filter
  "metadata_filter": {"bookId": "PSA"}
}
```

### Metadata Not in Response

**Problem**: Metadata not showing in `chunks_used`.

**Solution**: Updated! Metadata is now included in each chunk:

```python
for chunk in result['chunks_used']:
    print(chunk['metadata'])  # This now works!
```

## Summary

Metadata filtering in RAG:
- ✅ Narrows search to specific document subsets
- ✅ Maintains search quality through over-retrieval
- ✅ Works with semantic, text, and hybrid search
- ✅ Supports multiple metadata attributes (AND logic)
- ✅ Fully backward compatible (optional parameter)
- ✅ Includes metadata in response for transparency

For more examples and code, see:
- `examples/bible_rag_metadata.py` - Bible-specific examples
- `examples/rag_usage.py` - General RAG examples with metadata
- `scripts/ingest_bible_chunks.py` - How metadata is set during ingestion

