# Bug Fix: Metadata JSON String Handling

## Issue

**Error**: `"string indices must be integers, not 'str'"`

**Symptom**: RAG metadata filtering was failing with an internal server error when trying to filter by metadata.

**Root Cause**: PostgreSQL returns JSONB/JSON columns as JSON strings in some cases, but the code was trying to access them as dictionaries directly without parsing.

## The Problem

In `src/routes/search.py`, the `_filter_by_metadata()` function was doing:

```python
chunk_metadata = chunk.get('metadata', {})
# ...
if key not in chunk_metadata or chunk_metadata[key] != value:  # ❌ ERROR HERE
```

When `chunk_metadata` was a JSON string like `'{"bookId": "1CO"}'`, trying to access `chunk_metadata[key]` would fail because you can't use string keys on a string—only on a dictionary.

## The Fix

### 1. Added JSON Import

```python
import json
```

### 2. Updated `_filter_by_metadata()` Function

Added JSON string parsing before accessing metadata:

```python
def _filter_by_metadata(chunks: List[Dict[str, Any]], metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not metadata_filter:
        return chunks
    
    filtered_chunks = []
    for chunk in chunks:
        chunk_metadata = chunk.get('metadata', {})
        
        # ✅ NEW: Handle case where metadata is a JSON string
        if isinstance(chunk_metadata, str):
            try:
                chunk_metadata = json.loads(chunk_metadata)
            except (json.JSONDecodeError, TypeError):
                continue  # Skip chunks with invalid JSON
        
        if not chunk_metadata:
            continue
        
        # Now chunk_metadata is guaranteed to be a dict
        matches = True
        for key, value in metadata_filter.items():
            if key not in chunk_metadata or chunk_metadata[key] != value:
                matches = False
                break
        
        if matches:
            filtered_chunks.append(chunk)
    
    return filtered_chunks
```

### 3. Updated Response Building

Also fixed the response building to handle JSON string metadata:

```python
"metadata": json.loads(chunk.get("metadata", "{}")) if isinstance(chunk.get("metadata"), str) else chunk.get("metadata", {})
```

## Testing

Created comprehensive tests including:

1. ✅ Metadata as dictionary (original behavior)
2. ✅ Metadata as JSON string (PostgreSQL behavior)
3. ✅ Invalid JSON strings (graceful handling)
4. ✅ Multiple field filtering with JSON strings

Run tests with:
```bash
python3 verify_metadata_fix.py
```

All tests pass! ✅

## Files Changed

1. `src/routes/search.py` - Added JSON parsing logic
2. `tests/test_metadata_filter.py` - Added tests for JSON string cases
3. `verify_metadata_fix.py` - Standalone verification script

## Impact

- ✅ **Backward Compatible**: Dictionary metadata still works
- ✅ **Handles PostgreSQL**: JSON string metadata now parsed correctly
- ✅ **Error Handling**: Invalid JSON gracefully skipped
- ✅ **No Breaking Changes**: Existing functionality preserved

## Before vs After

### Before (Broken)

```python
# PostgreSQL returns: metadata = '{"bookId": "1CO"}'
chunk_metadata = '{"bookId": "1CO"}'
chunk_metadata["bookId"]  # ❌ TypeError: string indices must be integers
```

### After (Fixed)

```python
# PostgreSQL returns: metadata = '{"bookId": "1CO"}'
chunk_metadata = '{"bookId": "1CO"}'
if isinstance(chunk_metadata, str):
    chunk_metadata = json.loads(chunk_metadata)  # Now it's a dict!
chunk_metadata["bookId"]  # ✅ Returns "1CO"
```

## Usage

Now RAG metadata filtering works correctly:

```python
response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "What does it say about love?",
        "metadata_filter": {"bookId": "1CO"}  # ✅ Works now!
    }
)
```

## Verification

Run the verification script to confirm the fix:

```bash
python3 verify_metadata_fix.py
```

Expected output:
```
============================================================
Metadata Filtering Fix Verification
============================================================
Testing JSON string metadata parsing...
✅ JSON string metadata test passed!
Testing dict metadata...
✅ Dict metadata test passed!
Testing invalid JSON handling...
✅ Invalid JSON handling test passed!
Testing multiple field filtering...
✅ Multiple field filtering test passed!

============================================================
✅ ALL TESTS PASSED!
============================================================
```

## Status

**FIXED** ✅ - The error is now resolved and metadata filtering works with both dictionary and JSON string metadata.

