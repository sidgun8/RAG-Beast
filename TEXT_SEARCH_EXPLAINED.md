# Full-Text Search Documentation

## Overview

This document explains how **Full-Text Search (FTS)** works in the PG Vector Search system, including the underlying PostgreSQL FTS mechanisms, query processing, and ranking algorithms.

## What is Full-Text Search?

Full-text search enables users to search for **exact word matches** and **phrase matches** in document content, as opposed to semantic search which finds conceptually similar content.

**Semantic Search**: "Find documents about artificial intelligence"  
**Text Search**: "Find documents containing the exact words 'artificial' and 'intelligence'"

## Implementation Location

- **Database Service**: `src/services/database_service.py` → `full_text_search()` method
- **API Endpoint**: `src/routes/search.py` → `full_text_search()` route
- **SQL Query**: Lines 285-297 in database_service.py

## How It Works

### Step 1: Query Processing

When you send a search query like `"machine learning algorithms"`:

```python
# User Query
query = "machine learning algorithms"
```

### Step 2: PostgreSQL Full-Text Search

The system uses **PostgreSQL's built-in Full-Text Search (FTS)** capabilities:

#### A. Text Vectorization (`to_tsvector`)

PostgreSQL converts the document text into a **tsvector** (text search vector):

```sql
to_tsvector('english', title || ' ' || content)
```

**What happens:**
1. **Tokenization**: Splits text into individual words
2. **Normalization**: Converts to lowercase
3. **Stemming**: Reduces words to root forms (e.g., "learning" → "learn")
4. **Stop word removal**: Removes common words (the, a, an, etc.)
5. **Position tracking**: Records positions of each word

**Example:**
```
Input: "Machine learning is fun. Learning algorithms are complex."

to_tsvector output:
'algorithm':4 'complex':6 'fun':3 'learn':1,2 'machin':1
```

#### B. Query Parsing (`plainto_tsquery`)

The user's query is converted into a **tsquery** (text search query):

```sql
plainto_tsquery('english', $1)
```

**What happens:**
1. **Tokenization**: Splits query into words
2. **Stemming**: Reduces to root forms
3. **Phrase handling**: Treats as "AND" of terms
4. **Operators**: Converts to boolean query

**Example:**
```
Input: "machine learning algorithms"

plainto_tsquery output:
'machin' & 'learn' & 'algorithm'
```

### Step 3: Matching (`@@` operator)

Documents are matched using the `@@` operator:

```sql
WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', $1)
```

**How it works:**
- Compares the tsvector (document) with tsquery (search terms)
- Returns documents where **ALL** query terms match
- Position-aware matching

### Step 4: Ranking (`ts_rank`)

Results are ranked by relevance:

```sql
ts_rank(
    to_tsvector('english', title || ' ' || content), 
    plainto_tsquery('english', $1)
) as rank
```

**Ranking factors:**
1. **Term frequency**: How often query terms appear
2. **Term rarity**: Rare words get higher scores
3. **Document length**: Shorter documents ranked higher
4. **Term proximity**: Closer terms get higher scores

## Complete SQL Query

Here's the full text search query used in the system:

```sql
SELECT 
    id,
    title,
    content,
    content_type,
    file_path,
    metadata,
    created_at,
    ts_rank(
        to_tsvector('english', title || ' ' || content), 
        plainto_tsquery('english', $1)
    ) as rank
FROM documents 
WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', $1)
ORDER BY rank DESC
LIMIT $2
```

## Key Components Explained

### 1. `to_tsvector()`

**Purpose**: Convert text to a searchable vector

**Parameters:**
- First: Language (`'english'`)
- Second: Text to convert (title + content)

**Returns**: TSVector with normalized tokens and positions

**Example:**
```sql
SELECT to_tsvector('english', 'Machine learning is fascinating');

Result: 'fascin':3 'learn':1 'machin':1
```

### 2. `plainto_tsquery()`

**Purpose**: Convert plain text query to search query

**Parameters:**
- First: Language
- Second: User query

**Returns**: TSQuery object with boolean operators

**Example:**
```sql
SELECT plainto_tsquery('english', 'machine learning');

Result: 'machin' & 'learn'
```

### 3. `ts_rank()`

**Purpose**: Calculate relevance score

**Returns**: Float score (0.0 to 1.0+)

**Scoring factors:**
- Term frequency (tf)
- Inverse document frequency (idf)
- Document length normalization

### 4. `@@` Operator

**Purpose**: Match operator for text search

**Usage**: `tsvector @@ tsquery`

**Returns**: Boolean (true if matches)

## Indexing for Performance

### GIN Index Creation

For fast text search, a **GIN index** (Generalized Inverted Index) is created:

```sql
CREATE INDEX IF NOT EXISTS documents_content_fts_idx 
ON documents USING gin(
    to_tsvector('english', title || ' ' || content)
);
```

**What is a GIN index?**
- **Inverted index**: Maps words to document locations
- **Fast lookups**: O(log n) search time
- **Space efficient**: Only stores unique terms

**Index structure (simplified):**
```
'machine'    -> [doc1, doc3, doc7]
'learning'   -> [doc1, doc2, doc5]
'algorithm'  -> [doc1, doc4, doc6]
```

## Language Support

### English (`'english'`)

**Features:**
- English stemmer (Porter's algorithm)
- English stop words removal
- Case-insensitive matching

**Available languages in PostgreSQL:**
- `simple` - No stemming, basic matching
- `english` - English language
- `spanish` - Spanish
- `french` - French
- And 20+ more languages

## Query Types

### 1. Phrase Search (AND)

```python
query = "machine learning"
# Finds documents with BOTH "machine" AND "learning"
```

### 2. OR Search

```sql
-- You can modify to support OR:
SELECT to_tsquery('english', 'machine | learning | algorithms');
-- Matches ANY of the terms
```

### 3. NOT Search

```sql
SELECT to_tsquery('english', 'machine & !learning');
-- Matches "machine" but NOT "learning"
```

### 4. Phrase Proximity

```sql
-- Use phraseto_tsquery for exact phrase matching
phraseto_tsquery('english', 'machine learning');
```

## Full Text Search vs Semantic Search

| Feature | Full-Text Search | Semantic Search |
|---------|-----------------|-----------------|
| **Method** | Keyword matching | Vector similarity |
| **Query Type** | Exact words | Natural language |
| **Speed** | Very fast (indexed) | Moderate (vector search) |
| **Accuracy** | Exact matches only | Conceptual similarity |
| **Use Case** | Known terms | Unknown concepts |
| **Index Type** | GIN (text) | IVFFlat (vectors) |

### Example Comparison

**Query**: "AI and ML techniques"

**Full-Text Search:**
- Searches for: "AI", "ML", "techniques"
- Finds documents with exact terms
- Misses: "artificial intelligence", "machine learning"

**Semantic Search:**
- Searches for: AI concepts, machine learning concepts
- Finds documents even without exact terms
- Understands synonyms and related concepts

## Hybrid Search

The system combines both approaches for optimal results:

```python
# In hybrid search:
results = await document_service.hybrid_search(
    query=query,
    limit=10,
    semantic_weight=0.7,  # 70% semantic
    text_weight=0.3       # 30% text
)
```

**Benefits:**
- Semantic for conceptual understanding
- Text for exact term matching
- Best of both worlds

## Performance Optimization

### 1. Index Usage

```sql
-- Creates GIN index for fast lookups
CREATE INDEX documents_content_fts_idx 
ON documents USING gin(to_tsvector('english', title || ' ' || content));
```

### 2. Query Optimization

```sql
-- Only computes tsvector once per document
ts_rank(to_tsvector(...), plainto_tsquery(...))
```

### 3. Limit Results

```sql
LIMIT 10  -- Only return top 10 results
```

## API Usage

### Endpoint

```
POST /api/search/text
```

### Request Body

```json
{
  "query": "machine learning algorithms",
  "limit": 10
}
```

### Response

```json
{
  "query": "machine learning algorithms",
  "results": [
    {
      "id": "uuid-here",
      "title": "ML Document",
      "content": "Machine learning algorithms are...",
      "content_type": "document",
      "file_path": null,
      "metadata": {},
      "created_at": "2024-01-01T00:00:00",
      "rank": 0.856
    }
  ],
  "total": 1,
  "search_type": "full-text",
  "algorithm": "PostgreSQL Full-Text Search"
}
```

## Advanced Features

### 1. Custom Ranking

```sql
-- Weight title more heavily than content
ts_rank_cd(
    setweight(to_tsvector('english', title), 'A') ||
    setweight(to_tsvector('english', content), 'B'),
    plainto_tsquery('english', $1)
) as rank
```

### 2. Highlighting Results

```sql
SELECT 
    ts_headline('english', content, plainto_tsquery('english', $1))
FROM documents
```

### 3. Multiple Languages

```sql
-- Search in multiple languages
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
   OR to_tsvector('spanish', content) @@ plainto_tsquery('spanish', $1)
```

## Code Flow

```
1. User sends query → "machine learning"
   ↓
2. API endpoint receives request
   ↓
3. DatabaseService.full_text_search() called
   ↓
4. PostgreSQL executes FTS query:
   - Creates tsvector from document content
   - Creates tsquery from user query
   - Matches documents using @@ operator
   - Ranks results using ts_rank()
   ↓
5. Returns top N results ordered by rank
   ↓
6. API returns formatted response
```

## Limitations

### 1. Exact Term Matching
- Only finds exact word matches (after stemming)
- Doesn't understand synonyms or context

### 2. Language Dependent
- Configured for English language
- Other languages need different configuration

### 3. No Typo Tolerance
- "machne" won't find "machine"
- Consider using trigram similarity for typo handling

### 4. Limited Context
- Doesn't understand document structure
- No awareness of titles vs body text prioritization

## Best Practices

### 1. Use Hybrid Search
```python
# Combine semantic + text for best results
chunk_type="hybrid"
```

### 2. Optimize Indexes
```sql
-- Ensure GIN index exists
CREATE INDEX documents_content_fts_idx 
ON documents USING gin(to_tsvector('english', content));
```

### 3. Set Appropriate Limits
```python
limit=10  # Don't fetch too many results
```

### 4. Combine with Semantic Search
```python
# Use text search for known terms
# Use semantic search for concepts
```

## Summary

Full-text search in this system uses:

✅ **PostgreSQL FTS** - Industry-standard text search  
✅ **GIN Indexes** - Fast lookups with inverted indexes  
✅ **Stemming & Normalization** - Handles word variations  
✅ **Ranking Algorithm** - Relevance-based scoring  
✅ **Language Support** - Configurable for different languages  

**Key Advantages:**
- Fast exact term matching
- Handles word variations (learning, learns, learned)
- Indexed for performance
- Integrated with PostgreSQL

**Use when:**
- Searching for specific terms
- Need exact keyword matching
- Have fast response time requirements
- Working with known vocabulary
