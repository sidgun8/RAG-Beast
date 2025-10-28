# Bible Chunks Ingestion Guide

## Overview

This guide explains how to ingest your Bible chunks (both `bible_chunks/` and `verse_chunks/` folders) into the vector search system so you can answer questions about them using the RAG endpoint.

## Data Structure

You have two folders with JSON files:

### `bible_chunks/` folder
- Contains chapter-level chunks
- Structure: `{bookId}/{bookId}.{chapter}.json`
- Example: `1JN/1JN.1.json`
- Each file contains a chapter with multiple verses

**Example JSON Structure:**
```json
{
  "id": "1JN.1",
  "bookId": "1JN",
  "reference": "1 John 1",
  "number": "1",
  "content": "[1] That which was from the beginning...",
  "verseCount": 10
}
```

### `verse_chunks/` folder
- Contains verse-level chunks
- Similar structure but with individual verses

## Ingestion Options

### Option 1: Ingest All Bible Chunks (Recommended)

Run the ingestion script:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the Bible ingestion script
python scripts/ingest_bible_chunks.py
```

**What it does:**
1. Scans `bible_chunks/` folder for all JSON files
2. Loads each chunk with metadata
3. Creates embeddings for each chunk
4. Stores in PostgreSQL with vector index
5. Shows progress (successful/errors)

### Option 2: Ingest via API

You can also ingest chunks individually via the API:

```bash
curl -X POST "http://localhost:8000/api/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "1 John 1",
    "content": "Chapter content here...",
    "content_type": "bible_chunk",
    "metadata": {
      "bookId": "1JN",
      "chapterNumber": "1",
      "source": "Bible KJV"
    }
  }'
```

## After Ingestion

Once ingested, you can query the Bible using the RAG endpoint:

### Example: Asking Questions

```bash
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does John say about Jesus in chapter 1?",
    "limit": 5,
    "chunk_type": "semantic"
  }'
```

### Example: Finding Similar Passages

```bash
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "love and fellowship",
    "limit": 10,
    "similarity_threshold": 0.6
  }'
```

## Usage Examples

### 1. Semantic Search

Find passages by meaning:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/semantic",
    json={
        "query": "What does the Bible say about forgiveness?",
        "limit": 5,
        "similarity_threshold": 0.7
    }
)

results = response.json()
for result in results['results']:
    print(f"{result['title']}: {result['content'][:100]}...")
    print(f"Similarity: {result['similarity_score']}")
```

### 2. RAG Question Answering

Ask natural language questions:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "Explain the concept of fellowship in the Bible",
        "limit": 5,
        "chunk_type": "hybrid"
    }
)

result = response.json()
print(f"Question: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Used {result['total_chunks']} Bible passages")
```

### 3. Search Specific Books

Use metadata search to find passages from specific books:

```python
response = requests.post(
    "http://localhost:8000/api/search/metadata",
    json={
        "metadata": {
            "bookId": "1JN"
        },
        "limit": 10
    }
)

results = response.json()
for result in results['results']:
    print(result['reference'])
```

## Checking Your Data

### Get Statistics

```bash
curl http://localhost:8000/api/search/stats
```

Returns:
- Total documents
- Total chunks
- Documents by type
- Average chunk count

### List All Documents

```bash
curl http://localhost:8000/api/documents
```

### Search Documents

```bash
curl "http://localhost:8000/api/documents?limit=10"
```

## Query Types

### 1. Semantic Search (Meaning-based)

Best for: Finding conceptually similar passages

```json
{
  "query": "What does the Bible teach about love?",
  "limit": 10,
  "similarity_threshold": 0.7
}
```

### 2. Text Search (Keyword-based)

Best for: Finding exact words or phrases

```json
{
  "query": "Jesus Christ",
  "limit": 10
}
```

### 3. Hybrid Search (Combined)

Best for: Comprehensive results

```json
{
  "query": "What is the Gospel message?",
  "limit": 10,
  "semantic_weight": 0.7,
  "text_weight": 0.3
}
```

### 4. RAG (Question Answering)

Best for: Getting comprehensive answers

```json
{
  "query": "What is the main message of 1 John?",
  "limit": 5,
  "chunk_type": "semantic"
}
```

## Tips for Best Results

### 1. Query Formulation

**Good queries:**
- "What does John teach about love?"
- "Explain the concept of forgiveness"
- "What is the Gospel according to John?"

**Avoid:**
- "love" (too broad)
- "What chapter is..." (use search instead)
- Single words without context

### 2. Similarity Threshold

- **0.8-0.9**: Very strict, fewer but highly relevant results
- **0.6-0.7**: Balanced (default)
- **0.4-0.5**: Lenient, more results but less precise

### 3. Chunk Retrieval

- **semantic**: Best for conceptual questions
- **text**: Best for specific word/phrase searches
- **hybrid**: Best for general questions

### 4. Result Limits

Start with `limit: 5` for RAG questions, increase if needed for broader search.

## Troubleshooting

### No Results Found

If you get no results:

1. **Check if data is ingested:**
   ```bash
   curl http://localhost:8000/api/search/stats
   ```

2. **Lower the similarity threshold:**
   ```json
   {
     "similarity_threshold": 0.5
   }
   ```

3. **Try text search instead:**
   ```json
   {
     "chunk_type": "text"
   }
   ```

### Poor Quality Results

1. **Increase the limit** to get more context:
   ```json
   {
     "limit": 10
   }
   ```

2. **Use hybrid search** for better coverage:
   ```json
   {
     "chunk_type": "hybrid",
     "semantic_weight": 0.7,
     "text_weight": 0.3
   }
   ```

3. **Refine your query** to be more specific

### Error: "AWS credentials not found"

Make sure your `.env` file has:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BEDROCK_MODEL_ID=us.meta.llama4-scout-17b-instruct-v1:0
```

## Example Queries

### Find Specific Topics

```bash
# Questions about forgiveness
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the Bible say about forgiveness?",
    "limit": 5
  }'

# Questions about love
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should Christians show love to others?",
    "limit": 5
  }'
```

### Find Similar Passages

```bash
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Walking in the light and truth",
    "limit": 10,
    "similarity_threshold": 0.6
  }'
```

### Search by Reference

```bash
# Find specific book/chapter
curl -X POST "http://localhost:8000/api/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "1 John chapter 1",
    "limit": 10
  }'
```

## Next Steps

1. **Ingest your data**: Run `python scripts/ingest_bible_chunks.py`
2. **Test search**: Try the example queries above
3. **Ask questions**: Use the RAG endpoint for Q&A
4. **Refine**: Adjust parameters based on your results

## Full Example

Complete workflow:

```bash
# 1. Start the server
python app.py

# 2. Ingest Bible chunks
python scripts/ingest_bible_chunks.py

# 3. Ask a question
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of 1 John?",
    "limit": 5,
    "chunk_type": "semantic"
  }'
```

## Support

For issues:
1. Check server logs for errors
2. Verify database connection
3. Ensure AWS credentials are set
4. Check that embeddings are generated correctly

