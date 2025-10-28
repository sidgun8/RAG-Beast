import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.services.text_processor import TextProcessor


def test_clean_text_basic():
    tp = TextProcessor()
    raw = "Hello,  world!\n\nThis\tis   a   test."
    cleaned = tp.clean_text(raw)
    assert "  " not in cleaned
    assert "\n" not in cleaned
    assert cleaned.endswith("test.")


def test_chunk_text_overlap_and_bounds():
    tp = TextProcessor()
    text = ("Sentence one. " * 50) + ("Sentence two! " * 50) + ("Question three? " * 50)
    chunks = tp.chunk_text(text, max_chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # Ensure overlap applied
    for i in range(1, len(chunks)):
        assert chunks[i]['start'] <= chunks[i-1]['end']
        assert chunks[i]['start'] >= chunks[i-1]['end'] - 60  # allow some tolerance
    # Ensure reasonable end bounds
    for ch in chunks:
        assert 0 <= ch['start'] < ch['end'] <= len(text)
        assert len(ch['text']) <= 350
