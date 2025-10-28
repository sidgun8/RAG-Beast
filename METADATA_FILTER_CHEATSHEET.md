# RAG Metadata Filter Cheat Sheet

Quick reference for using metadata filters in RAG searches.

## Basic Syntax

```python
{
  "query": "your question here",
  "metadata_filter": {
    "field1": "value1",
    "field2": "value2"
  }
}
```

## Bible Metadata Fields

| Field | Type | Example Values |
|-------|------|----------------|
| `bookId` | string | `"1CO"`, `"GEN"`, `"PSA"`, `"MAT"` |
| `chapterNumber` | string | `"1"`, `"13"`, `"119"` |
| `source` | string | `"Bible KJV"` |
| `reference` | string | `"1 Corinthians 13"`, `"Genesis 1"` |

## Common Filter Patterns

### 1. Search in One Book

```json
{
  "query": "What does it say about love?",
  "metadata_filter": {"bookId": "1CO"}
}
```

### 2. Search in Specific Chapter

```json
{
  "query": "What does it teach?",
  "metadata_filter": {
    "bookId": "ROM",
    "chapterNumber": "8"
  }
}
```

### 3. Search in Specific Bible Version

```json
{
  "query": "wisdom",
  "metadata_filter": {"source": "Bible KJV"}
}
```

### 4. No Filter (Search All)

```json
{
  "query": "faith",
  "limit": 5
}
```

## Bible Book IDs Reference

### Old Testament

#### Pentateuch
- `GEN` - Genesis
- `EXO` - Exodus
- `LEV` - Leviticus
- `NUM` - Numbers
- `DEU` - Deuteronomy

#### Historical Books
- `JOS` - Joshua
- `JDG` - Judges
- `RUT` - Ruth
- `1SA` - 1 Samuel
- `2SA` - 2 Samuel
- `1KI` - 1 Kings
- `2KI` - 2 Kings
- `1CH` - 1 Chronicles
- `2CH` - 2 Chronicles
- `EZR` - Ezra
- `NEH` - Nehemiah
- `EST` - Esther

#### Wisdom Literature
- `JOB` - Job
- `PSA` - Psalms
- `PRO` - Proverbs
- `ECC` - Ecclesiastes
- `SNG` - Song of Solomon

#### Major Prophets
- `ISA` - Isaiah
- `JER` - Jeremiah
- `LAM` - Lamentations
- `EZE` - Ezekiel
- `DAN` - Daniel

#### Minor Prophets
- `HOS` - Hosea
- `JOE` - Joel
- `AMO` - Amos
- `OBA` - Obadiah
- `JON` - Jonah
- `MIC` - Micah
- `NAH` - Nahum
- `HAB` - Habakkuk
- `ZEP` - Zephaniah
- `HAG` - Haggai
- `ZEC` - Zechariah
- `MAL` - Malachi

### New Testament

#### Gospels
- `MAT` - Matthew
- `MAR` - Mark
- `LUK` - Luke
- `JOH` - John

#### History
- `ACT` - Acts

#### Pauline Epistles
- `ROM` - Romans
- `1CO` - 1 Corinthians
- `2CO` - 2 Corinthians
- `GAL` - Galatians
- `EPH` - Ephesians
- `PHP` - Philippians
- `COL` - Colossians
- `1TH` - 1 Thessalonians
- `2TH` - 2 Thessalonians
- `1TI` - 1 Timothy
- `2TI` - 2 Timothy
- `TIT` - Titus
- `PHM` - Philemon

#### General Epistles
- `HEB` - Hebrews
- `JAS` - James
- `1PE` - 1 Peter
- `2PE` - 2 Peter
- `1JN` - 1 John
- `2JN` - 2 John
- `3JN` - 3 John
- `JUD` - Jude

#### Prophecy
- `REV` - Revelation

## Python Examples

### Quick Function

```python
import requests

def ask_bible(question, book=None, chapter=None):
    """Quick helper function for Bible questions"""
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

# Usage
print(ask_bible("What is love?", book="1CO"))
print(ask_bible("The beatitudes", book="MAT", chapter=5))
print(ask_bible("creation story", book="GEN", chapter=1))
```

### Compare Across Books

```python
def compare_books(question, books):
    """Compare a topic across multiple books"""
    for book in books:
        result = ask_bible(question, book=book)
        print(f"\n=== {book} ===")
        print(result[:200] + "...")

# Usage: Compare how different books discuss faith
compare_books("faith", ["ROM", "HEB", "JAS"])
```

## cURL Examples

### Simple Filter

```bash
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "love",
    "metadata_filter": {"bookId": "1CO"}
  }'
```

### Chapter-Specific

```bash
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Spirit",
    "metadata_filter": {
      "bookId": "ROM",
      "chapterNumber": "8"
    }
  }'
```

### Hybrid Search with Filter

```bash
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "salvation",
    "chunk_type": "hybrid",
    "semantic_weight": 0.7,
    "text_weight": 0.3,
    "metadata_filter": {"bookId": "EPH"}
  }'
```

## JavaScript/TypeScript

```javascript
async function searchBible(question, bookId = null, chapterNum = null) {
  const body = {
    query: question,
    limit: 5
  };
  
  if (bookId) {
    body.metadata_filter = { bookId };
    if (chapterNum) {
      body.metadata_filter.chapterNumber = String(chapterNum);
    }
  }
  
  const response = await fetch('http://localhost:8000/api/search/rag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  
  const data = await response.json();
  return data.answer;
}

// Usage
const answer = await searchBible("What is grace?", "EPH");
console.log(answer);
```

## Best Practices

### ✅ DO

- Use specific book IDs: `{"bookId": "1CO"}`
- Combine with appropriate chunk types: `"chunk_type": "hybrid"`
- Lower similarity threshold when filtering: `"similarity_threshold": 0.3`
- Check available metadata first if unsure

### ❌ DON'T

- Use invalid book IDs: `{"bookId": "1 COR"}` ❌ (use `"1CO"` ✅)
- Mix string/number types: `{"chapterNumber": 8}` ❌ (use `"8"` ✅)
- Set threshold too high with strict filters: May get no results
- Forget to increase limit for very specific filters

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| No results | Lower `similarity_threshold` to 0.3 or 0.4 |
| Too few results | Increase `limit` (e.g., to 10) |
| Wrong book | Check book ID against reference table |
| Wrong chapter | Ensure chapter number is a string: `"13"` not `13` |

## Testing Commands

```bash
# Test the service is running
curl http://localhost:8000/health

# Test basic RAG (no filter)
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "faith", "limit": 3}'

# Test with metadata filter
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "love", "metadata_filter": {"bookId": "1CO"}}'

# Run Python examples
python examples/bible_rag_metadata.py
```

## Links

- Full Documentation: [RAG_METADATA_FILTERING.md](RAG_METADATA_FILTERING.md)
- Python Examples: [examples/bible_rag_metadata.py](examples/bible_rag_metadata.py)
- RAG Guide: [RAG_README.md](RAG_README.md)
- Bible Ingestion: [BIBLE_INGESTION_GUIDE.md](BIBLE_INGESTION_GUIDE.md)

