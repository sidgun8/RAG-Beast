#!/usr/bin/env python3
"""
Example usage of RAG search with metadata filtering for Bible chunks

This demonstrates how to use metadata filters to narrow RAG searches to:
- Specific Bible books
- Specific chapters
- Specific sources (e.g., KJV, NIV)
- Combinations of metadata attributes

The metadata stored for each Bible chunk includes:
- bibleId: Unique identifier for the chunk
- bookId: Book abbreviation (e.g., "1CO", "GEN", "MAT")
- chapterNumber: Chapter number as string
- verseCount: Number of verses in the chunk
- source: Bible version (e.g., "Bible KJV")
- reference: Human-readable reference (e.g., "1 Corinthians 13")
"""
import requests
import json
from typing import Dict, Any, Optional


# Configuration
BASE_URL = "http://localhost:8000"
RAG_ENDPOINT = f"{BASE_URL}/api/search/rag"


def rag_search_with_metadata(
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    chunk_type: str = "semantic",
    similarity_threshold: float = 0.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform RAG search with optional metadata filtering
    
    Args:
        query: The question to ask
        metadata_filter: Dictionary of metadata attributes to filter by
        limit: Number of chunks to retrieve
        chunk_type: Type of search ('semantic', 'text', or 'hybrid')
        similarity_threshold: Minimum similarity score
        **kwargs: Additional RAG parameters (max_tokens, temperature, etc.)
    
    Returns:
        Dictionary containing the answer and metadata
    """
    request_data = {
        "query": query,
        "limit": limit,
        "chunk_type": chunk_type,
        "similarity_threshold": similarity_threshold,
        **kwargs
    }
    
    if metadata_filter:
        request_data["metadata_filter"] = metadata_filter
    
    response = requests.post(RAG_ENDPOINT, json=request_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def example_1_search_specific_book():
    """Example 1: Search only in 1 Corinthians"""
    print("=" * 80)
    print("Example 1: Search in specific book (1 Corinthians)")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="What does it teach about love?",
        metadata_filter={"bookId": "1CO"},
        limit=3,
        similarity_threshold=0.5
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\n{'Sources Used':=^80}")
    for i, chunk in enumerate(result['chunks_used'], 1):
        metadata = chunk.get('metadata', {})
        print(f"\n{i}. {metadata.get('reference', 'N/A')} (Score: {chunk.get('similarity_score', 0):.3f})")
        print(f"   Preview: {chunk['content_preview'][:100]}...")
    print()


def example_2_search_multiple_books():
    """Example 2: Compare content across multiple books"""
    print("\n" + "=" * 80)
    print("Example 2: Search in Genesis")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="What happened in the beginning?",
        metadata_filter={"bookId": "GEN"},
        limit=5,
        similarity_threshold=0.4
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\nChunks found: {result['total_chunks']}")
    print()


def example_3_hybrid_search_with_metadata():
    """Example 3: Use hybrid search with metadata filtering"""
    print("\n" + "=" * 80)
    print("Example 3: Hybrid search in Psalms")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="prayers about protection and deliverance",
        metadata_filter={"bookId": "PSA"},
        limit=5,
        chunk_type="hybrid",
        semantic_weight=0.6,
        text_weight=0.4,
        similarity_threshold=0.3
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Search Type: {result['chunk_type']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\n{'Sources':=^80}")
    for chunk in result['chunks_used']:
        metadata = chunk.get('metadata', {})
        print(f"- {metadata.get('reference', 'N/A')}")
    print()


def example_4_specific_chapter():
    """Example 4: Search in a specific chapter"""
    print("\n" + "=" * 80)
    print("Example 4: Search in Romans Chapter 8")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="What does it say about the Spirit?",
        metadata_filter={
            "bookId": "ROM",
            "chapterNumber": "8"
        },
        limit=3,
        similarity_threshold=0.4
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print()


def example_5_by_source():
    """Example 5: Filter by Bible version/source"""
    print("\n" + "=" * 80)
    print("Example 5: Search in KJV Bible")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="teachings about wisdom",
        metadata_filter={"source": "Bible KJV"},
        limit=5,
        similarity_threshold=0.5
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\n{'References from':=^80}")
    for chunk in result['chunks_used']:
        metadata = chunk.get('metadata', {})
        print(f"- {metadata.get('reference', 'N/A')}")
    print()


def example_6_no_metadata_filter():
    """Example 6: Search without metadata filter (search all documents)"""
    print("\n" + "=" * 80)
    print("Example 6: Search across all Bible books (no filter)")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="What does the Bible say about faith?",
        metadata_filter=None,  # No filter - search everything
        limit=5,
        similarity_threshold=0.5
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: None (searching all documents)")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\n{'References from different books':=^80}")
    for chunk in result['chunks_used']:
        metadata = chunk.get('metadata', {})
        print(f"- {metadata.get('reference', 'N/A')} (Book: {metadata.get('bookId', 'N/A')})")
    print()


def example_7_advanced_parameters():
    """Example 7: Advanced RAG with metadata filtering"""
    print("\n" + "=" * 80)
    print("Example 7: Advanced RAG with custom model and parameters")
    print("=" * 80)
    
    result = rag_search_with_metadata(
        query="Explain the concept of grace and mercy",
        metadata_filter={"bookId": "EPH"},  # Ephesians
        limit=3,
        chunk_type="semantic",
        similarity_threshold=0.4,
        max_tokens=1500,
        temperature=0.5,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Filter: {result.get('metadata_filter', {})}")
    print(f"Model: {result['model_used']}")
    print(f"\n{'Answer':=^80}")
    print(result['answer'])
    print(f"\n{'Parameters':=^80}")
    print(json.dumps(result['parameters'], indent=2))
    print()


def check_service_status():
    """Check if the service is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Service not available: {e}")
        return False


def main():
    """Run all examples"""
    print("Bible RAG Search with Metadata Filtering Examples")
    print("=" * 80)
    
    # Check service status
    if not check_service_status():
        print("\n❌ Service is not running!")
        print(f"Please start the server first: python app.py")
        return
    
    print("✅ Service is running\n")
    
    # Run examples
    try:
        example_1_search_specific_book()
        example_2_search_multiple_books()
        example_3_hybrid_search_with_metadata()
        example_4_specific_chapter()
        example_5_by_source()
        example_6_no_metadata_filter()
        example_7_advanced_parameters()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

