import os
import sys
import asyncio
import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable (so `import src.*` works)
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src.main import app


@pytest.fixture(scope='session')
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope='session')
def test_client():
    return TestClient(app)
