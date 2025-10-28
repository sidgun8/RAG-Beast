# Semantic Search Documentation

## Overview

This document explains how **Semantic Search** works in the PG Vector Search system, including embeddings, vector similarity, and the underlying mathematics.

## What is Semantic Search?

Semantic search understands the **meaning** and **context** of your query, not just keyword matches. It finds conceptually similar content even when exact words don't match.

**Example:**
- **Query**: "How do neural networks learn?"
- **Text Search finds**: Documents with exact words "neural", "networks", "learn"
- **Semantic Search finds**: Documents about AI training, machine learning, deep learning, model optimization, etc.

## Implementation Location

- **Embedding Service**: `src/services/embedding_service.py` - Generates embeddings
- **Database Service**: `src/services/database_service.py` - Vector similarity search
- **Document Service**: `src/services/document_service.py` - Orchestrates search
- **API Endpoint**: `src/routes/search.py` - Exposes semantic search endpoint

## How Semantic Search Works

### High-Level Flow

```
User Query: "How does machine learning work?"
    ↓
1. Convert query to embedding vector
    ↓
2. Calculate similarity with all document embeddings
    ↓
3. Return most similar documents
```

## Step-by-Step Process

### Step 1: Text to Embedding

When you send a query, the system converts it into a numerical vector (embedding):

```python
# User Query
query = "How does machine learning work?"

# Convert to embedding
embedding = [0.123, -0.456, 0.789, ..., 0.234]  # 768 dimensions
```

**How it works:**
1. Tokenization: Split text into tokens
2. Model processing: Pass through embedding model
3. Vector extraction: Get final embedding vector
4. Normalization: Scale to unit length

### Step 2: Vector Similarity Search

Once we have the query embedding, we search for similar document embeddings:

```sql
SELECT 
    id,
    title,
    content,
    ...,
    1 - (embedding <=> $1) as similarity_score
FROM documents 
WHERE 1 - (embedding <=> $1) > $2
ORDER BY embedding <=> $1
LIMIT $3
```

**Key Components:**

#### A. Cosine Distance (`<=>` operator)

The `<=>` operator calculates **cosine distance** between vectors:

```
cosine_distance = 1 - cosine_similarity
```

**Cosine Similarity Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- `A` = Query embedding vector
- `B` = Document embedding vector
- `·` = Dot product
- `||A||` = Vector magnitude

#### B. Distance to Similarity Conversion

```sql
1 - (embedding <=> $1) as similarity_score
```

This converts distance to similarity:
- **Distance 0** → Similarity 1 (identical)
- **Distance 1** → Similarity 0 (opposite)

### Step 3: Thresholding

Only documents above the threshold are returned:

```sql
WHERE 1 - (embedding <=> $1) > $2
```

**Default threshold**: 0.7 (70% similarity)

### Step 4: Ranking and Limiting

Results are sorted by similarity and limited:

```sql
ORDER BY embedding <=> $1  -- Order by distance (ascending)
LIMIT $3                    -- Top N results
```

## Embedding Models

### Current Model: EmbeddingGemma

**Location**: `models/embeddinggemma/`

**Properties:**
- **Dimensions**: 768 (configurable to 256, 512, 768)
- **Model**: Google's EmbeddingGemma
- **Purpose**: General-purpose semantic embeddings
- **Language**: English

**How embeddings capture meaning:**

```
Query: "Apple fruit"
Embedding: [0.12, -0.34, 0.56, ...]

Document: "Red delicious apples"
Embedding: [0.13, -0.33, 0.57, ...]

Document: "Car manufacturing"
Embedding: [-0.45, 0.23, -0.12, ...]

→ Apple fruit is closer to Red delicious apples than Car manufacturing
```

### Embedding Generation Process

```python
# In embedding_service.py
async def get_embedding(self, text: str) -> List[float]:
    """Generate embedding for text"""
    # 1. Tokenize
    inputs = self._tokenizer(text, max_length=512, ...)
    
    # 2. Pass through model
    outputs = self._model(**inputs)
    
    # 3. Extract embedding (mean pooling)
    embedding = outputs.last_hidden_state.mean(dim=1)
    
    # 4. Reduce dimensions if needed
    if self.embedding_dimensions < 768:
        embedding = embedding[:self.embedding_dimensions]
    
    return embedding
```

## Vector Similarity Mathematics

### Cosine Similarity

Measures the angle between two vectors in high-dimensional space:

```
Cosine Similarity = (A · B) / (||A|| × ||B||)

where:
- A and B are embedding vectors
- · denotes dot product
- ||A|| is the magnitude (length) of vector A
```

**Visual representation:**
```
Vector A: [0.5, 0.5]
Vector B: [0.8, 0.2]

Angle between A and B determines similarity
- Small angle → High similarity
- Large angle → Low similarity
```

**Result range**: -1 to 1
- **1**: Vectors point in same direction (very similar)
- **0**: Vectors are perpendicular (unrelated)
- **-1**: Vectors point in opposite directions (opposite meaning)

### Why Cosine Similarity?

1. **Direction over magnitude**: Focuses on semantic similarity, not document length
2. **Normalized**: Results are in fixed range (-1 to 1)
3. **Efficient**: Fast computation with matrix operations
4. **Interpretable**: Easy to understand as "angle between meanings"

### Distance Metrics Used

```sql
-- Cosine distance (what we use)
embedding <=> $1  -- Returns 0 (identical) to 1 (opposite)

-- Can also use:
-- L2 distance
embedding <-> $1  -- Euclidean distance

-- Inner product
embedding <#> $1  -- Negative dot product
```

## Indexing for Performance

### IVFFlat Index

For fast vector search, an **IVFFlat index** is created:

```sql
CREATE INDEX documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

**What is IVFFlat?**
- **IVF**: Inverted File Index
- **Flat**: Compares against all vectors in a cluster
- **Speed**: Approximate nearest neighbor search

**How it works:**
1. **Clustering**: Divides vectors into 100 clusters
2. **Training**: Learns cluster centroids
3. **Search**: Finds closest clusters first
4. **Comparison**: Compares against vectors in those clusters only

**Benefits:**
- **10-100x faster** than linear search
- **Trade-off**: Approximate results (99% accuracy typical)

## Complete Semantic Search Flow

```
1. User Query: "machine learning algorithms"
   ↓
2. EmbeddingService.get_embedding()
   - Tokenize query
   - Pass through EmbeddingGemma model
   - Extract 768-dimensional vector
   ↓
3. Vector: [0.123, -0.456, ..., 0.789]
   ↓
4. DatabaseService.semantic_search()
   - Calculate cosine distance to all documents
   - Filter by similarity threshold (>0.7)
   - Order by distance (most similar first)
   - Limit to top N results
   ↓
5. Results with similarity scores
   ↓
6. API returns formatted response
```

## Code Implementation

### API Endpoint

```python
@router.post("/semantic")
async def semantic_search(request: SemanticSearchRequest):
    results = await document_service.search_documents(
        query=request.query,
        search_type='semantic',
        limit=request.limit,
        similarity_threshold=request.similarity_threshold
    )
    return results
```

### Document Service

```python
async def _semantic_search(self, query: str, limit: int, 
                           similarity_threshold: float = 0.7):
    # Get embedding for query
    query_embedding = await self.embedding_service.get_embedding(query)
    
    # Perform vector similarity search
    results = await self.db_service.semantic_search(
        query_embedding=query_embedding,
        limit=limit,
        similarity_threshold=similarity_threshold
    )
    return results
```

### Database Service

```python
async def semantic_search(self, query_embedding: List[float], 
                         limit: int, similarity_threshold: float):
    query = '''
        SELECT 
            ...,
            1 - (embedding <=> $1) as similarity_score
        FROM documents 
        WHERE 1 - (embedding <=> $1) > $2
        ORDER BY embedding <=> $1
        LIMIT $3
    '''
    results = await conn.fetch(query, embedding, threshold, limit)
    return results
```

## Similarity Scores Explained

### What Does a Score Mean?

**Score 0.95** (95% similar):
- Very similar meaning
- Example: "machine learning" vs "ML algorithms"

**Score 0.80** (80% similar):
- Related concepts
- Example: "neural networks" vs "deep learning"

**Score 0.60** (60% similar):
- Somewhat related
- Example: "AI" vs "machine learning"

**Score 0.40** (40% similar):
- Weak relation
- Example: "data science" vs "statistics"

**Score 0.20** (20% similar):
- Barely related
- Example: "database" vs "machine learning"

### Default Threshold

**Threshold 0.7 (70%)**:
- Only returns documents with ≥70% similarity
- Balances precision and recall
- Can be adjusted per query

## Advantages of Semantic Search

### 1. **Conceptual Understanding**
- Finds related concepts, not just keywords
- Handles synonyms and paraphrasing

### 2. **Natural Language Queries**
- No need to think of exact keywords
- Query in your own words

### 3. **Cross-Language Similarity**
- Similar concepts in different languages
- (Works best with multilingual models)

### 4. **Context-Aware**
- Understands word meaning in context
- "Apple" (fruit) vs "Apple" (company)

### 5. **Fuzzy Matching**
- Finds documents even with different wording
- Typo-resistant to some degree

## Limitations

### 1. **Computational Cost**
- Requires embedding generation
- Vector similarity calculations
- Slower than keyword search

### 2. **Storage Overhead**
- Each document needs 768-dim vector
- Increased database size

### 3. **Model-Dependent Quality**
- Quality depends on embedding model
- May miss domain-specific nuances

### 4. **Interpretability**
- Hard to explain why documents match
- Black box decision making

### 5. **Language Support**
- Currently optimized for English
- Other languages need appropriate models

## Configuration

### Embedding Dimensions

```env
EMBEDDING_DIMENSIONS=768  # Full dimensions
# Or reduced:
EMBEDDING_DIMENSIONS=512  # Matryoshka reduction
EMBEDDING_DIMENSIONS=256  # Further reduction
```

**Trade-offs:**
- **Higher dimensions**: More accurate, slower
- **Lower dimensions**: Faster, less accurate

### Similarity Threshold

```python
{
  "query": "your query",
  "similarity_threshold": 0.7  # 0.0 to 1.0
}
```

**Guidelines:**
- **0.8-0.9**: Very strict (few results)
- **0.6-0.7**: Balanced (default)
- **0.4-0.5**: Lenient (many results)

## API Usage

### Endpoint

```
POST /api/search/semantic
```

### Request

```json
{
  "query": "How do neural networks learn?",
  "limit": 10,
  "similarity_threshold": 0.7
}
```

### Response

```json
{
  "query": "How do neural networks learn?",
  "results": [
    {
      "id": "uuid-here",
      "title": "Introduction to Neural Networks",
      "content": "Neural networks learn through...",
      "similarity_score": 0.923
    }
  ],
  "total": 1,
  "search_type": "semantic",
  "algorithm": "Vector Similarity Search",
  "threshold": 0.7
}
```

## Comparison: Semantic vs Text Search

| Aspect | Semantic Search | Text Search |
|--------|----------------|-------------|
| **Method** | Vector similarity | Keyword matching |
| **Understanding** | Conceptual | Lexical |
| **Query** | "How does AI learn?" | "AI learning" |
| **Finds** | Related concepts | Exact terms |
| **Speed** | Moderate | Very fast |
| **Synonyms** | Yes | No |
| **Context** | Yes | Limited |
| **Use Case** | Natural queries | Known terms |

## Best Practices

### 1. **Combine with Text Search**
```python
# Use hybrid search for best results
chunk_type="hybrid"
```

### 2. **Adjust Threshold Dynamically**
```python
# Lower threshold if few results
if len(results) < 5:
    threshold = 0.5
```

### 3. **Chunk Documents Appropriately**
```python
# Smaller chunks → More precise retrieval
max_chunk_size=500
```

### 4. **Use Appropriate Models**
```python
# Domain-specific models for better results
EMBEDDING_MODEL=your_custom_model
```

## Advanced Topics

### 1. Dense vs Sparse Embeddings

**Dense Embeddings** (what we use):
- Every dimension has a value
- Compact (768 values)
- Good for semantic similarity

**Sparse Embeddings**:
- Most values are 0
- Large vectors
- Good for exact term matching

### 2. Multi-Embedding Search

Combine multiple embeddings:
```python
# Search across different models
results = []
for model in ['query_model', 'context_model']:
    results.extend(semantic_search(query, model))
```

### 3. Hybrid Retrieval

Combine multiple signals:
```python
# Score = semantic_score + keyword_score + recency_score
final_score = (
    0.6 * semantic_similarity +
    0.2 * keyword_matches +
    0.2 * recency_normalized
)
```

## Summary

Semantic search in this system:

✅ **Uses EmbeddingGemma** - Google's state-of-the-art model  
✅ **768-Dimensional Vectors** - Captures rich semantic meaning  
✅ **Cosine Similarity** - Measures conceptual similarity  
✅ **IVFFlat Indexing** - Fast approximate nearest neighbor search  
✅ **Configurable Threshold** - Balance precision and recall  

**Key Advantages:**
- Understands meaning, not just keywords
- Finds related concepts automatically
- Handles synonyms and paraphrasing
- Natural language queries

**Use when:**
- Asking conceptual questions
- Need to find similar ideas
- Don't know exact keywords
- Want intelligent document discovery

**Combine with:**
- Text search for exact terms
- Hybrid search for best results
- Metadata filters for domain-specific search
