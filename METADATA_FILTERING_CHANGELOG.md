# Metadata Filtering for RAG - Changelog

## Summary

Enhanced the RAG (Retrieval-Augmented Generation) pipeline to support **metadata filtering**, allowing users to narrow search results to specific document types, categories, or other metadata attributes before generating answers.

## Changes Made

### 1. Backend Changes

#### `src/routes/search.py`

**Modified `RAGSearchRequest` class:**
- Added `metadata_filter` parameter (Optional[Dict[str, Any]])
- Allows users to specify metadata constraints for filtering results

**Added `_filter_by_metadata()` helper function:**
- Filters chunks based on metadata constraints
- Uses exact matching (AND logic) for all specified fields
- Handles missing metadata gracefully

**Updated `rag_search()` endpoint:**
- Implements over-retrieval strategy (3x limit) when metadata filtering is used
- Applies metadata filtering after initial search
- Includes metadata in response chunks
- Returns metadata_filter in response for transparency

**Key improvements:**
```python
# Before: No metadata filtering
chunks = await document_service.search_documents(
    query=request.query,
    search_type='semantic',
    limit=request.limit
)

# After: With metadata filtering support
retrieval_limit = request.limit * 3 if request.metadata_filter else request.limit
chunks = await document_service.search_documents(
    query=request.query,
    search_type='semantic',
    limit=retrieval_limit
)

if request.metadata_filter:
    chunks = _filter_by_metadata(chunks, request.metadata_filter)
    chunks = chunks[:request.limit]
```

### 2. Example Code

#### `examples/rag_usage.py`

**Added two new examples:**
- **Example 5**: RAG with single metadata filter (search in specific book)
- **Example 6**: RAG with multiple metadata filters (combined filters)

Shows metadata filtering with:
- Semantic search
- Hybrid search
- Multiple filter attributes

#### `examples/bible_rag_metadata.py` (NEW)

**Created comprehensive Bible-specific examples:**
- 7 different usage patterns
- Demonstrates filtering by book, chapter, source
- Shows hybrid search integration
- Includes reusable helper functions
- Covers advanced parameter combinations

### 3. Documentation

#### `RAG_METADATA_FILTERING.md` (NEW)

**Complete guide covering:**
- Overview and how it works
- API reference with all parameters
- Bible chunks metadata reference
- 5+ detailed examples (cURL, Python, JavaScript)
- Use cases and practical applications
- Performance considerations and optimization tips
- Troubleshooting guide
- Custom metadata usage

#### `METADATA_FILTER_CHEATSHEET.md` (NEW)

**Quick reference guide with:**
- Basic syntax examples
- Complete Bible book ID reference (66 books)
- Common filter patterns
- Code snippets in Python, cURL, JavaScript
- Best practices and troubleshooting quick fixes
- Testing commands

#### `RAG_README.md` (UPDATED)

**Added metadata filtering section:**
- Overview in main workflow
- Link to comprehensive documentation
- Quick example snippet

### 4. Response Format Changes

**Enhanced chunk information:**
```json
{
  "chunks_used": [
    {
      "id": "...",
      "title": "...",
      "content_preview": "...",
      "similarity_score": 0.89,
      "metadata": {           // NEW: Now included
        "bookId": "1CO",
        "chapterNumber": "13",
        "source": "Bible KJV"
      }
    }
  ],
  "metadata_filter": {       // NEW: Shows applied filter
    "bookId": "1CO"
  },
  "parameters": {
    "metadata_filter": {...} // NEW: Included in parameters
  }
}
```

## Features

### ✅ What Works

1. **Single Field Filtering**
   ```python
   {"metadata_filter": {"bookId": "1CO"}}
   ```

2. **Multiple Field Filtering (AND logic)**
   ```python
   {"metadata_filter": {"bookId": "ROM", "chapterNumber": "8"}}
   ```

3. **Works with All Search Types**
   - Semantic search
   - Text search
   - Hybrid search

4. **Over-Retrieval Strategy**
   - Retrieves 3x the limit when filtering
   - Ensures sufficient results after filtering

5. **Metadata in Response**
   - Shows what filter was applied
   - Includes metadata for each chunk
   - Full transparency

6. **Backward Compatible**
   - Optional parameter
   - No breaking changes
   - Existing code continues to work

## Usage Examples

### Basic Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it teach about love?",
        "limit": 5,
        "metadata_filter": {"bookId": "1CO"}
    }
)

result = response.json()
print(result['answer'])
```

### Advanced Usage

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "teachings about the Spirit",
        "limit": 5,
        "chunk_type": "hybrid",
        "semantic_weight": 0.7,
        "text_weight": 0.3,
        "similarity_threshold": 0.4,
        "metadata_filter": {
            "bookId": "ROM",
            "chapterNumber": "8"
        }
    }
)
```

### No Filter (Default)

```python
# Still works - searches all documents
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does the Bible say about faith?",
        "limit": 5
    }
)
```

## Testing

### Run Examples

```bash
# Bible-specific examples
python examples/bible_rag_metadata.py

# General RAG examples (includes metadata filtering)
python examples/rag_usage.py
```

### Manual Testing

```bash
# Test service is running
curl http://localhost:8000/health

# Test basic RAG (no filter)
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "faith", "limit": 3}'

# Test with metadata filter
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "love",
    "metadata_filter": {"bookId": "1CO"},
    "limit": 5
  }'
```

## Bible Metadata Available

For documents ingested via `scripts/ingest_bible_chunks.py`:

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `bookId` | string | `"1CO"` | Book abbreviation |
| `chapterNumber` | string | `"13"` | Chapter number |
| `verseCount` | integer | `13` | Number of verses |
| `source` | string | `"Bible KJV"` | Bible version |
| `reference` | string | `"1 Corinthians 13"` | Human-readable reference |
| `bibleId` | string | `"1CO.13"` | Unique chunk ID |

## Performance Impact

### Retrieval Strategy

- **Without filter**: Retrieves exactly `limit` chunks
- **With filter**: Retrieves `limit * 3` chunks, then filters to `limit`

### Benefits

- ✅ Maintains search quality
- ✅ Handles sparse metadata matches
- ✅ No database schema changes needed
- ✅ Works with existing indexes

### Considerations

- Slightly more chunks retrieved initially when filtering
- Negligible performance impact for typical use cases
- Can adjust multiplier if needed (currently 3x)

## Future Enhancements (Optional)

Potential improvements for future versions:

1. **OR Logic Support**
   ```python
   # Search in multiple books
   {"metadata_filter": {"bookId": ["1CO", "ROM", "EPH"]}}
   ```

2. **Range Filters**
   ```python
   # Chapters 1-10
   {"metadata_filter": {"chapterNumber": {"gte": "1", "lte": "10"}}}
   ```

3. **Database-Level Filtering**
   - Add metadata filtering to SQL queries
   - Could improve performance for very restrictive filters
   - Requires database service modifications

4. **Partial Matching**
   ```python
   # Contains "Corinthians"
   {"metadata_filter": {"reference": {"contains": "Corinthians"}}}
   ```

## Migration Guide

### For Existing Users

**No changes required!** The feature is fully backward compatible.

All existing RAG requests will continue to work exactly as before:

```python
# This still works perfectly
requests.post(
    "http://localhost:8000/api/search/rag",
    json={"query": "faith", "limit": 5}
)
```

### To Start Using Metadata Filtering

Simply add the `metadata_filter` parameter:

```python
# Add this parameter to your existing requests
requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "faith",
        "limit": 5,
        "metadata_filter": {"bookId": "ROM"}  # NEW
    }
)
```

## Files Changed/Added

### Modified Files
- `src/routes/search.py` - Added metadata filtering logic
- `examples/rag_usage.py` - Added 2 new examples
- `RAG_README.md` - Added metadata filtering overview

### New Files
- `examples/bible_rag_metadata.py` - Comprehensive Bible examples
- `RAG_METADATA_FILTERING.md` - Complete documentation
- `METADATA_FILTER_CHEATSHEET.md` - Quick reference
- `METADATA_FILTERING_CHANGELOG.md` - This file

## Links

- **Full Documentation**: [RAG_METADATA_FILTERING.md](RAG_METADATA_FILTERING.md)
- **Quick Reference**: [METADATA_FILTER_CHEATSHEET.md](METADATA_FILTER_CHEATSHEET.md)
- **Python Examples**: [examples/bible_rag_metadata.py](examples/bible_rag_metadata.py)
- **General RAG Guide**: [RAG_README.md](RAG_README.md)
- **Bible Ingestion**: [BIBLE_INGESTION_GUIDE.md](BIBLE_INGESTION_GUIDE.md)

## Questions?

Run the examples to see it in action:
```bash
python examples/bible_rag_metadata.py
```

Check the documentation for detailed explanations:
```bash
cat RAG_METADATA_FILTERING.md
```

Quick reference for common patterns:
```bash
cat METADATA_FILTER_CHEATSHEET.md
```

