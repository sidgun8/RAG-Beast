import os
import sys
import json
import pytest
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def mock_document_service(monkeypatch):
    class DummyService:
        async def search_documents(self, query: str, search_type: str = 'semantic', limit: int = 10, **kwargs):
            if search_type == 'semantic':
                return [
                    {"id": "1", "title": "Doc A", "content": "About love and light", "similarity_score": 0.91},
                    {"id": "2", "title": "Doc B", "content": "About truth", "similarity_score": 0.82}
                ][:limit]
            if search_type == 'text':
                return [
                    {"id": "3", "title": "Keyword Match", "content": "Jesus Christ" , "rank": 0.9}
                ][:limit]
            return []
        async def hybrid_search(self, **kwargs):
            return [
                {"id": "1", "title": "Hybrid A", "content": "blended", "rrf_score": 0.5}
            ]
    from src.routes import search as search_router
    monkeypatch.setattr(search_router, 'document_service', DummyService())


@pytest.fixture
def mock_llm_service(monkeypatch):
    class DummyLLM:
        def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], **kwargs):
            return (f"Answer to: {query}. Using {len(context_chunks)} chunks.", kwargs.get('model_id') or 'dummy-model')
    from src.routes import search as search_router
    monkeypatch.setattr(search_router, 'get_llm_service', lambda: DummyLLM())


def test_semantic_search(mock_document_service):
    resp = client.post('/api/search/semantic', json={
        "query": "love and light",
        "limit": 2,
        "similarity_threshold": 0.7
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['search_type'] == 'semantic'
    assert data['total'] == 2
    assert data['results'][0]['similarity_score'] >= 0.8


def test_text_search(mock_document_service):
    resp = client.post('/api/search/text', json={
        "query": "Jesus",
        "limit": 1
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['search_type'] == 'full-text'
    assert data['total'] == 1


def test_rag_search_semantic(mock_document_service, mock_llm_service):
    resp = client.post('/api/search/rag', json={
        "query": "What is fellowship?",
        "limit": 2,
        "similarity_threshold": 0.5,
        "chunk_type": "semantic",
        "model_id": "anthropic.claude-v2:1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['search_type'] == 'rag'
    assert data['chunk_type'] == 'semantic'
    assert 'answer' in data
    assert data['model_used'] == 'anthropic.claude-v2:1'


def test_rag_search_hybrid(mock_document_service, mock_llm_service):
    resp = client.post('/api/search/rag', json={
        "query": "Explain Gospel",
        "limit": 1,
        "chunk_type": "hybrid",
        "semantic_weight": 0.7,
        "text_weight": 0.3
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['chunk_type'] == 'hybrid'
    assert data['total_chunks'] == 1
