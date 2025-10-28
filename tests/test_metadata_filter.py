"""
Unit tests for metadata filtering in RAG search
"""
import pytest
import json
from src.routes.search import _filter_by_metadata


def test_filter_by_metadata_single_field():
    """Test filtering by a single metadata field"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}},
        {"id": "3", "content": "test", "metadata": {"bookId": "1CO"}},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 2
    assert all(chunk["metadata"]["bookId"] == "1CO" for chunk in result)


def test_filter_by_metadata_multiple_fields():
    """Test filtering by multiple metadata fields"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "ROM", "chapterNumber": "8"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM", "chapterNumber": "9"}},
        {"id": "3", "content": "test", "metadata": {"bookId": "1CO", "chapterNumber": "8"}},
    ]
    
    metadata_filter = {"bookId": "ROM", "chapterNumber": "8"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_by_metadata_no_filter():
    """Test that no filter returns all chunks"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}},
    ]
    
    result = _filter_by_metadata(chunks, None)
    
    assert len(result) == 2


def test_filter_by_metadata_empty_filter():
    """Test that empty filter returns all chunks"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}},
    ]
    
    result = _filter_by_metadata(chunks, {})
    
    assert len(result) == 2


def test_filter_by_metadata_no_matches():
    """Test that no matches returns empty list"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}},
    ]
    
    metadata_filter = {"bookId": "GEN"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 0


def test_filter_by_metadata_missing_metadata():
    """Test that chunks without metadata are filtered out"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}},
        {"id": "2", "content": "test"},  # No metadata
        {"id": "3", "content": "test", "metadata": {}},  # Empty metadata
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_by_metadata_partial_match():
    """Test that partial matches are not included (AND logic)"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "ROM", "chapterNumber": "8", "source": "KJV"}},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM", "chapterNumber": "8"}},
        {"id": "3", "content": "test", "metadata": {"bookId": "ROM"}},
    ]
    
    metadata_filter = {"bookId": "ROM", "chapterNumber": "8", "source": "KJV"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    # Only chunk 1 has all three fields
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_by_metadata_preserves_order():
    """Test that filtering preserves the original order"""
    chunks = [
        {"id": "1", "content": "test", "metadata": {"bookId": "1CO"}, "score": 0.9},
        {"id": "2", "content": "test", "metadata": {"bookId": "ROM"}, "score": 0.8},
        {"id": "3", "content": "test", "metadata": {"bookId": "1CO"}, "score": 0.7},
        {"id": "4", "content": "test", "metadata": {"bookId": "1CO"}, "score": 0.6},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 3
    assert [chunk["id"] for chunk in result] == ["1", "3", "4"]
    # Scores should be in descending order (original order preserved)
    assert result[0]["score"] == 0.9
    assert result[1]["score"] == 0.7
    assert result[2]["score"] == 0.6


def test_filter_by_metadata_json_string():
    """Test that metadata as JSON string is properly parsed"""
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "13"}'},
        {"id": "2", "content": "test", "metadata": '{"bookId": "ROM", "chapterNumber": "8"}'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "1"}'},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "3"


def test_filter_by_metadata_json_string_multiple_fields():
    """Test filtering with JSON string metadata and multiple fields"""
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "13"}'},
        {"id": "2", "content": "test", "metadata": '{"bookId": "1CO", "chapterNumber": "8"}'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "ROM", "chapterNumber": "13"}'},
    ]
    
    metadata_filter = {"bookId": "1CO", "chapterNumber": "13"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_by_metadata_invalid_json_string():
    """Test that invalid JSON strings are handled gracefully"""
    chunks = [
        {"id": "1", "content": "test", "metadata": '{"bookId": "1CO"}'},
        {"id": "2", "content": "test", "metadata": 'invalid json{'},
        {"id": "3", "content": "test", "metadata": '{"bookId": "1CO"}'},
    ]
    
    metadata_filter = {"bookId": "1CO"}
    result = _filter_by_metadata(chunks, metadata_filter)
    
    # Should skip the invalid JSON and return the valid ones
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "3"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

