import os
import sys
import io
import json
import pytest
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.services.file_processor import FileProcessor


@pytest.mark.asyncio
async def test_extract_pdf_text_mocks_pages():
    fp = FileProcessor()
    dummy_texts = ["Page 1 text", "Page 2 text"]

    class DummyPage:
        def __init__(self, t):
            self._t = t
        def extract_text(self):
            return self._t

    class DummyReader:
        def __init__(self, pages):
            self.pages = [DummyPage(t) for t in pages]

    with patch('src.services.file_processor.PyPDF2.PdfReader', return_value=DummyReader(dummy_texts)):
        # Need a real file path for open(); use temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            result = await fp._extract_pdf(tmp.name)
            assert "Page 1 text" in result
            assert "Page 2 text" in result


@pytest.mark.asyncio
async def test_extract_json_returns_pretty_string(tmp_path):
    fp = FileProcessor()
    p = tmp_path / 'sample.json'
    data = {"a": 1, "b": {"c": 2}}
    p.write_text(json.dumps(data))
    out = await fp._extract_json(str(p))
    assert "\n" in out
    assert '"a": 1' in out


@pytest.mark.asyncio
async def test_extract_csv_to_text(tmp_path):
    fp = FileProcessor()
    p = tmp_path / 'sample.csv'
    p.write_text("col1,col2\nval1,val2\n")
    out = await fp._extract_csv(str(p))
    assert 'col1: val1' in out
    assert 'col2: val2' in out
