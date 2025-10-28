#!/usr/bin/env python3
"""
Test script for PG Vector Search API functionality
"""

import asyncio
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.document_service import DocumentService
from services.embedding_service import get_embedding_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embedding_service():
    """Test embedding service functionality"""
    logger.info("Testing embedding service...")
    
    try:
        embedding_service = get_embedding_service()
        
        # Test single embedding
        test_text = "This is a test document for semantic search."
        embedding = await embedding_service.get_embedding(test_text)
        
        logger.info(f"✓ Single embedding generated: {len(embedding)} dimensions")
        
        # Test batch embeddings
        test_texts = [
            "Machine learning algorithms",
            "Database optimization techniques",
            "Python web development"
        ]
        batch_embeddings = await embedding_service.get_batch_embeddings(test_texts)
        
        logger.info(f"✓ Batch embeddings generated: {len(batch_embeddings)} embeddings")
        
        # Get model info
        model_info = embedding_service.get_model_info()
        logger.info(f"✓ Model info: {model_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding service test failed: {e}")
        return False


async def test_document_service():
    """Test document service functionality"""
    logger.info("Testing document service...")
    
    try:
        document_service = DocumentService()
        await document_service.initialize()
        
        # Test adding a document
        test_doc = {
            "title": "Test Document",
            "content": "This is a test document for the vector search system. It contains information about machine learning and artificial intelligence.",
            "content_type": "text",
            "metadata": {
                "category": "Test",
                "tags": ["test", "machine learning", "AI"]
            }
        }
        
        result = await document_service.add_document(
            title=test_doc["title"],
            content=test_doc["content"],
            content_type=test_doc["content_type"],
            metadata=test_doc["metadata"]
        )
        
        logger.info(f"✓ Document added: {result['id']}")
        
        # Test retrieving the document
        retrieved_doc = await document_service.get_document(result["id"])
        if retrieved_doc:
            logger.info("✓ Document retrieved successfully")
        else:
            logger.error("❌ Failed to retrieve document")
            return False
        
        # Test semantic search
        search_results = await document_service.search_documents(
            query="machine learning algorithms",
            search_type="semantic",
            limit=5
        )
        
        logger.info(f"✓ Semantic search returned {len(search_results)} results")
        
        # Test full-text search
        text_results = await document_service.search_documents(
            query="artificial intelligence",
            search_type="text",
            limit=5
        )
        
        logger.info(f"✓ Full-text search returned {len(text_results)} results")
        
        # Test hybrid search
        hybrid_results = await document_service.hybrid_search(
            query="AI and machine learning",
            limit=5
        )
        
        logger.info(f"✓ Hybrid search returned {len(hybrid_results)} results")
        
        # Test similar documents
        similar_results = await document_service.get_similar_documents(result["id"], limit=3)
        logger.info(f"✓ Similar documents search returned {len(similar_results)} results")
        
        # Test search stats
        stats = await document_service.get_search_stats()
        logger.info(f"✓ Search stats: {stats}")
        
        # Clean up test document
        await document_service.delete_document(str(result["id"]))
        logger.info("✓ Test document cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document service test failed: {e}")
        return False
    finally:
        await document_service.close()


async def test_database_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    
    try:
        from services.database_service import DatabaseService
        
        db_service = DatabaseService()
        await db_service.initialize()
        
        # Test basic query
        stats = await db_service.get_search_stats()
        logger.info(f"✓ Database connection successful: {stats}")
        
        await db_service.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting PG Vector Search API tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Embedding Service", test_embedding_service),
        ("Document Service", test_document_service),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if await test_func():
                logger.info(f"PASSED {test_name} test")
                passed += 1
            else:
                logger.error(f"FAILED {test_name} test")
        except Exception as e:
            logger.error(f"FAILED {test_name} test with exception: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("SUCCESS: All tests passed!")
        return True
    else:
        logger.error("FAILED: Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
