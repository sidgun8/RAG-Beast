#!/usr/bin/env python3
"""
Example usage of the RAG search endpoint
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
RAG_ENDPOINT = f"{BASE_URL}/api/search/rag"


def test_rag_search():
    """Test the RAG search endpoint"""
    
    # Example 1: Simple RAG search
    print("=" * 80)
    print("Example 1: Simple RAG Search")
    print("=" * 80)
    
    query = "What are the main topics discussed in the documents?"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 3,
            "similarity_threshold": 0.5
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nChunks Used: {result['total_chunks']}")
        print(f"Model: {result['model_used']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Example 2: RAG with text search (instead of semantic)
    print("\n" + "=" * 80)
    print("Example 2: RAG with Text Search")
    print("=" * 80)
    
    query = "Explain the concept in detail with examples"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 5,
            "similarity_threshold": 0.7,
            "chunk_type": "text",  # Using text search instead of semantic
            "max_tokens": 1500,
            "temperature": 0.5
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\nChunks Used ({result['total_chunks']}):")
        for i, chunk in enumerate(result['chunks_used'], 1):
            print(f"\n{i}. {chunk['title']}")
            print(f"   Score: {chunk['similarity_score']:.3f}")
            print(f"   Preview: {chunk['content_preview'][:100]}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Example 3: RAG with hybrid search
    print("\n" + "=" * 80)
    print("Example 3: RAG with Hybrid Search")
    print("=" * 80)
    
    query = "What are the key findings and recommendations?"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 3,
            "similarity_threshold": 0.6,
            "chunk_type": "hybrid",
            "semantic_weight": 0.7,
            "text_weight": 0.3,
            "rrf_k": 60,
            "max_tokens": 1000
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nChunk Type: {result['chunk_type']}")
        print(f"Model Used: {result['model_used']}")
        print(f"\nParameters Used:")
        print(json.dumps(result['parameters'], indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Example 4: RAG with specific model
    print("\n" + "=" * 80)
    print("Example 4: RAG with Specific Model")
    print("=" * 80)
    
    query = "Summarize the main points"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 3,
            "similarity_threshold": 0.6,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 1000
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nParameters Used:")
        print(json.dumps(result['parameters'], indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Example 5: RAG with Metadata Filter (Bible-specific)
    print("\n" + "=" * 80)
    print("Example 5: RAG with Metadata Filter - Search only in specific book")
    print("=" * 80)
    
    query = "What does it say about love?"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 5,
            "similarity_threshold": 0.5,
            "metadata_filter": {
                "bookId": "1CO"  # Search only in 1 Corinthians
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nMetadata Filter: {result.get('metadata_filter', {})}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nChunks Used ({result['total_chunks']}):")
        for i, chunk in enumerate(result['chunks_used'], 1):
            metadata = chunk.get('metadata', {})
            print(f"\n{i}. {chunk['title']}")
            print(f"   Book: {metadata.get('bookId', 'N/A')}")
            print(f"   Chapter: {metadata.get('chapterNumber', 'N/A')}")
            print(f"   Reference: {metadata.get('reference', 'N/A')}")
            print(f"   Score: {chunk.get('similarity_score', 0):.3f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Example 6: RAG with Multiple Metadata Filters
    print("\n" + "=" * 80)
    print("Example 6: RAG with Multiple Metadata Filters")
    print("=" * 80)
    
    query = "Tell me about faith and hope"
    
    response = requests.post(
        RAG_ENDPOINT,
        json={
            "query": query,
            "limit": 5,
            "similarity_threshold": 0.4,
            "chunk_type": "hybrid",
            "semantic_weight": 0.7,
            "text_weight": 0.3,
            "metadata_filter": {
                "source": "Bible KJV",
                "bookId": "1CO"
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nMetadata Filter: {json.dumps(result.get('metadata_filter', {}), indent=2)}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nTotal Chunks: {result['total_chunks']}")
        print(f"Search Type: {result['chunk_type']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def check_service_status():
    """Check if the service is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Service not available: {e}")
        return False


if __name__ == "__main__":
    print("RAG Search Example")
    print("=" * 80)
    
    # Check service status
    if not check_service_status():
        print("\n❌ Service is not running!")
        print(f"Please start the server first: python app.py")
        exit(1)
    
    print("✅ Service is running\n")
    
    # Run examples
    try:
        test_rag_search()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
