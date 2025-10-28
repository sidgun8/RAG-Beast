# Metadata Filtering Quick Start Guide

## 🚀 Get Started in 30 Seconds

### Before (No Filtering)
```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it say about love?",
        "limit": 5
    }
)
```
**Result**: Searches across ALL documents

### After (With Filtering)
```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it say about love?",
        "limit": 5,
        "metadata_filter": {"bookId": "1CO"}  # ← ADD THIS
    }
)
```
**Result**: Searches ONLY in 1 Corinthians ✨

## 📊 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG with Metadata Filtering                  │
└─────────────────────────────────────────────────────────────────┘

    User Query: "What does it say about love?"
    Filter: {"bookId": "1CO"}
         │
         ▼
    ┌────────────────────┐
    │  1. Search Phase   │
    │  Retrieve 15 chunks│  ← Gets 3x more to compensate for filtering
    │  (semantic/hybrid) │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ 2. Filter Phase    │
    │ Keep only chunks   │  ← Filters by metadata
    │ where bookId="1CO" │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ 3. Limit Phase     │
    │ Take top 5 chunks  │  ← Returns requested limit
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ 4. Generate Answer │
    │ LLM uses filtered  │  ← AWS Bedrock generates answer
    │ chunks as context  │
    └─────────┬──────────┘
              │
              ▼
         Answer + Sources
```

## 🎯 Common Use Cases

### 1. Search in One Book
```python
{"metadata_filter": {"bookId": "1CO"}}
```
✅ Find content about love in 1 Corinthians

### 2. Search in Specific Chapter
```python
{"metadata_filter": {"bookId": "ROM", "chapterNumber": "8"}}
```
✅ Find content about the Spirit in Romans 8

### 3. Search by Source
```python
{"metadata_filter": {"source": "Bible KJV"}}
```
✅ Only search KJV Bible documents

### 4. No Filter (Search Everything)
```python
{}  # or omit metadata_filter entirely
```
✅ Search across all ingested documents

## 📚 Available Metadata (Bible Chunks)

| What You Want | Filter | Example |
|---------------|--------|---------|
| Specific book | `{"bookId": "XXX"}` | `{"bookId": "1CO"}` |
| Specific chapter | `{"bookId": "XXX", "chapterNumber": "Y"}` | `{"bookId": "ROM", "chapterNumber": "8"}` |
| Bible version | `{"source": "Bible KJV"}` | `{"source": "Bible KJV"}` |

### Popular Book IDs

| Book | ID | Book | ID |
|------|-----|------|-----|
| Genesis | `GEN` | Matthew | `MAT` |
| Psalms | `PSA` | John | `JOH` |
| Isaiah | `ISA` | Romans | `ROM` |
| 1 Corinthians | `1CO` | Ephesians | `EPH` |
| Revelation | `REV` | Hebrews | `HEB` |

**Full list**: See [METADATA_FILTER_CHEATSHEET.md](METADATA_FILTER_CHEATSHEET.md)

## 💻 Code Examples

### Python Helper Function
```python
import requests

def ask_bible(question, book=None, chapter=None):
    """Quick Bible question with optional filtering"""
    data = {"query": question, "limit": 5}
    
    if book:
        data["metadata_filter"] = {"bookId": book}
        if chapter:
            data["metadata_filter"]["chapterNumber"] = str(chapter)
    
    response = requests.post(
        "http://localhost:8000/api/search/rag",
        json=data
    )
    return response.json()['answer']

# Usage examples
print(ask_bible("What is love?", book="1CO"))
print(ask_bible("The fruit of the Spirit", book="GAL", chapter="5"))
print(ask_bible("Creation account", book="GEN", chapter="1"))
```

### cURL One-Liner
```bash
# Search for "grace" in Ephesians
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "grace", "metadata_filter": {"bookId": "EPH"}}'
```

### JavaScript/TypeScript
```javascript
async function searchBible(question, book = null) {
  const body = { query: question, limit: 5 };
  
  if (book) {
    body.metadata_filter = { bookId: book };
  }
  
  const res = await fetch('http://localhost:8000/api/search/rag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  
  const data = await res.json();
  return data.answer;
}

// Usage
const answer = await searchBible("What is grace?", "EPH");
```

## 🧪 Test It Out

### 1. Start the Server
```bash
python app.py
```

### 2. Run Example Script
```bash
python examples/bible_rag_metadata.py
```

### 3. Try Your Own Query
```bash
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "YOUR QUESTION HERE",
    "metadata_filter": {"bookId": "1CO"}
  }'
```

## 📖 Learn More

| Document | Purpose |
|----------|---------|
| [RAG_METADATA_FILTERING.md](RAG_METADATA_FILTERING.md) | Complete documentation |
| [METADATA_FILTER_CHEATSHEET.md](METADATA_FILTER_CHEATSHEET.md) | Quick reference |
| [examples/bible_rag_metadata.py](examples/bible_rag_metadata.py) | Code examples |
| [METADATA_FILTERING_CHANGELOG.md](METADATA_FILTERING_CHANGELOG.md) | What changed |

## 🔧 Troubleshooting

### No results?
- ✅ Check book ID is correct (e.g., `"1CO"` not `"1 COR"`)
- ✅ Lower similarity threshold: `"similarity_threshold": 0.3`
- ✅ Increase limit: `"limit": 10`

### Wrong results?
- ✅ Verify metadata field names are exact
- ✅ Remember: filters use AND logic (all must match)
- ✅ Check that documents with that metadata exist

### Chapter number not working?
- ✅ Use string, not number: `"chapterNumber": "8"` ✅
- ✅ Not: `"chapterNumber": 8` ❌

## ✨ Pro Tips

1. **Lower threshold with strict filters**
   ```python
   {
     "similarity_threshold": 0.3,  # Lower for better recall
     "metadata_filter": {"bookId": "1CO", "chapterNumber": "13"}
   }
   ```

2. **Use hybrid search for better results**
   ```python
   {
     "chunk_type": "hybrid",
     "semantic_weight": 0.7,
     "text_weight": 0.3,
     "metadata_filter": {"bookId": "PSA"}
   }
   ```

3. **Compare across books**
   ```python
   books = ["MAT", "MAR", "LUK", "JOH"]
   for book in books:
       result = ask_bible("Tell me about miracles", book=book)
       print(f"{book}: {result[:100]}...")
   ```

## 🎉 That's It!

You now have metadata filtering in your RAG pipeline!

**Key Takeaway**: Just add `"metadata_filter": {"field": "value"}` to any RAG request.

---

**Need help?** Check the full docs or run the examples!

