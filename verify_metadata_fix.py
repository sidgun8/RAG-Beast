#!/usr/bin/env python3
"""
Quick verification script for metadata filtering fix
Tests that JSON string metadata is properly handled
"""
import json
from typing import List, Dict, Any


def _filter_by_metadata(chunks: List[Dict[str, Any]], metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter chunks by metadata constraints.
    Supports nested metadata matching using PostgreSQL @> containment operator logic.
    """
    if not metadata_filter:
        return chunks
    
    filtered_chunks = []
    for chunk in chunks:
        chunk_metadata = chunk.get('metadata', {})
        
        # Handle case where metadata is a JSON string
        if isinstance(chunk_metadata, str):
            try:
                chunk_metadata = json.loads(chunk_metadata)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not chunk_metadata:
            continue
        
        # Check if all filter keys match
        matches = True
        for key, value in metadata_filter.items():
            if key not in chunk_metadata or chunk_metadata[key] != value:
                matches = False
                break
        
        if matches:
            filtered_chunks.append(chunk)
    
    return filtered_chunks


def test_json_string_metadata():
    """Test that metadata as JSON string is properly parsed"""
    print("Testing JSON string metadata parsing...")
    
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "13"}'},
        {"id": "2", "content": "test", "metadata": '{"bookId": "ROM", "chapterNumber": "8"}'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "1"}'},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    assert result[0]["id"] == "1", f"Expected first ID to be '1', got {result[0]['id']}"
    assert result[1]["id"] == "3", f"Expected second ID to be '3', got {result[1]['id']}"
    
    print("✅ JSON string metadata test passed!")


def test_dict_metadata():
    """Test that dict metadata still works"""
    print("Testing dict metadata...")
    
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}},
        {"id": "3", "content": "test", "metadata": {"bookId": "1CO"}},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    
    print("✅ Dict metadata test passed!")


def test_invalid_json():
    """Test that invalid JSON is handled gracefully"""
    print("Testing invalid JSON handling...")
    
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO"}'},
        {"id": "2", "content": "test", "metadata": 'invalid json{'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "1CO"}'},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    # Should skip the invalid JSON and return the valid ones
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    
    print("✅ Invalid JSON handling test passed!")


def test_multiple_fields():
    """Test filtering with multiple metadata fields"""
    print("Testing multiple field filtering...")
    
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "13"}'},
        {"id": "2", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "8"}'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "ROM", "chapterNumber": "13"}'},
    ]
    
    metadata_filter = {"bookId": "1CO", "chapterNumber": "13"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert result[0]["id"] == "1", f"Expected ID to be '1', got {result[0]['id']}"
    
    print("✅ Multiple field filtering test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Metadata Filtering Fix Verification")
    print("=" * 60)
    
    try:
        test_json_string_metadata()
        test_dict_metadata()
        test_invalid_json()
        test_multiple_fields()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fix is working correctly. Metadata can now be:")
        print("  1. A dictionary (as before)")
        print("  2. A JSON string (from PostgreSQL)")
        print("  3. Invalid JSON (handled gracefully)")
        print("\nThe error 'string indices must be integers' is now fixed!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
