#!/usr/bin/env python3
"""
Sample data ingestion script for PG Vector Search API
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

import nltk
nltk.download('punkt_tab')


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.document_service import DocumentService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_sample_data():
    """Ingest sample documents for testing"""
    document_service = DocumentService()
    
    try:
        await document_service.initialize()
        
        logger.info("Starting data ingestion...")
        
        # Sample documents for testing
        sample_documents = [
            {
                "title": "Introduction to Machine Learning",
                "content": """Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. It involves the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Unsupervised learning finds hidden patterns in data without labeled examples. Reinforcement learning learns through interaction with an environment using rewards and penalties.""",
                "content_type": "text",
                "metadata": {
                    "category": "AI/ML",
                    "difficulty": "beginner",
                    "tags": ["machine learning", "AI", "algorithms", "data science"]
                }
            },
            {
                "title": "PostgreSQL Vector Search with pgvector",
                "content": """PostgreSQL with the pgvector extension provides powerful vector similarity search capabilities. pgvector is an open-source vector similarity search for PostgreSQL that supports exact and approximate nearest neighbor search.

Vector embeddings are numerical representations of text, images, or other data that capture semantic meaning. They enable similarity search by converting high-dimensional data into dense vectors that can be compared using distance metrics like cosine similarity, L2 distance, or inner product.

The pgvector extension adds a vector data type to PostgreSQL and provides operators for vector similarity search. You can create indexes using different algorithms like IVFFlat and HNSW for efficient approximate nearest neighbor search on large datasets.

Common use cases include semantic search, recommendation systems, image similarity, and any application requiring similarity search on high-dimensional data. The integration with PostgreSQL means you can combine vector search with traditional SQL queries, full-text search, and other database features.""",
                "content_type": "text",
                "metadata": {
                    "category": "Database",
                    "difficulty": "intermediate",
                    "tags": ["postgresql", "vector search", "embeddings", "similarity search"]
                }
            },
            {
                "title": "Python REST API Development",
                "content": """Python is a versatile programming language that's particularly well-suited for building REST APIs due to its simplicity, extensive library ecosystem, and strong community support. FastAPI and Flask are popular frameworks for creating REST APIs in Python.

A REST API (Representational State Transfer) is an architectural style for designing networked applications. REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs. They are stateless, cacheable, and have a uniform interface.

When building REST APIs with Python, you typically use frameworks like FastAPI or Flask to handle routing, middleware, and HTTP request/response handling. FastAPI provides automatic API documentation, type validation, and high performance, while Flask offers simplicity and flexibility.

Key principles of good REST API design include using proper HTTP status codes, implementing consistent error handling, versioning your API, implementing authentication and authorization, and following RESTful naming conventions for endpoints.""",
                "content_type": "text",
                "metadata": {
                    "category": "Web Development",
                    "difficulty": "intermediate",
                    "tags": ["python", "rest api", "fastapi", "flask", "backend"]
                }
            },
            {
                "title": "Semantic Search Implementation",
                "content": """Semantic search is a search technique that understands the intent and contextual meaning of search queries, rather than just matching keywords. It uses natural language processing and machine learning to provide more relevant search results.

Traditional keyword-based search relies on exact matches between query terms and indexed content. Semantic search, on the other hand, understands synonyms, context, and intent. For example, a search for "automobile" would also return results about "cars" and "vehicles".

Implementation typically involves converting text into vector embeddings using pre-trained models like Word2Vec, GloVe, or transformer-based models like BERT. These embeddings capture semantic relationships between words and phrases.

The search process involves: 1) Converting the search query into a vector embedding, 2) Comparing this embedding against stored document embeddings using similarity metrics, 3) Ranking results by similarity score, and 4) Returning the most relevant documents.

Semantic search is particularly powerful when combined with traditional full-text search in a hybrid approach, providing both keyword matching and semantic understanding for comprehensive search results.""",
                "content_type": "text",
                "metadata": {
                    "category": "Search",
                    "difficulty": "advanced",
                    "tags": ["semantic search", "NLP", "embeddings", "vector search", "machine learning"]
                }
            },
            {
                "title": "Database Performance Optimization",
                "content": """Database performance optimization is crucial for maintaining fast and efficient applications. There are several strategies and techniques to improve database performance, from query optimization to hardware considerations.

Indexing is one of the most important performance optimization techniques. Proper indexing can dramatically improve query performance by reducing the number of rows that need to be examined. Common index types include B-tree indexes for equality and range queries, hash indexes for equality lookups, and specialized indexes like GIN and GiST for complex data types.

Query optimization involves analyzing and improving SQL queries to reduce execution time and resource usage. This includes using appropriate JOIN strategies, avoiding SELECT *, using LIMIT clauses, and understanding query execution plans.

Connection pooling helps manage database connections efficiently, reducing the overhead of establishing and closing connections. Caching frequently accessed data in memory can significantly improve response times.

For PostgreSQL specifically, configuration tuning of parameters like shared_buffers, effective_cache_size, and work_mem can have substantial performance impacts. Regular maintenance tasks like VACUUM and ANALYZE are essential for optimal performance.""",
                "content_type": "text",
                "metadata": {
                    "category": "Database",
                    "difficulty": "advanced",
                    "tags": ["database optimization", "performance", "indexing", "postgresql", "query optimization"]
                }
            }
        ]
        
        logger.info(f"Ingesting {len(sample_documents)} sample documents...")
        
        for i, doc in enumerate(sample_documents):
            logger.info(f"Processing document {i + 1}/{len(sample_documents)}: {doc['title']}")
            
            result = await document_service.add_document(
                title=doc["title"],
                content=doc["content"],
                content_type=doc["content_type"],
                file_path=None,
                metadata=doc["metadata"]
            )
            
            logger.info(f"✓ Document created with ID: {result['id']}")
        
        logger.info("✅ Data ingestion completed successfully!")
        logger.info("You can now test the search functionality using the API endpoints.")
        
    except Exception as e:
        logger.error(f"❌ Error during data ingestion: {e}")
        sys.exit(1)
    finally:
        await document_service.close()


if __name__ == "__main__":
    asyncio.run(ingest_sample_data())
