# Testing Guide

## Prerequisites

- Python 3.12 (matches project venv)
- Virtual environment activated
- Dependencies installed:
  ```bash
  pip install -r requirements.txt
  pip install pytest pytest-asyncio httpx
  ```

## Test Commands

- Run all tests:
  ```bash
  pytest
  ```

- Run unit tests only:
  ```bash
  pytest tests/unit -q
  ```

- Run integration tests only:
  ```bash
  pytest tests/integration -q
  ```

## Structure

- `tests/unit/`
  - `test_text_processor.py`: chunking and cleaning
  - `test_file_processor.py`: PDF/CSV/JSON extraction (PDF mocked)
- `tests/integration/`
  - `test_search_endpoints.py`: `/api/search/*` endpoints with mocked services

## Notes

- Integration tests use `fastapi.testclient` and monkeypatch the service layer to avoid real DB and Bedrock calls.
- PDF extraction is tested via mocks; real OCR is not included.
- Ensure `.env` is present for app import; sensitive values are not used during tests.

## Troubleshooting

- If import errors occur, verify `src` is in `sys.path` (handled in conftest.py).
- If tests hang, ensure no external network calls are happening; mocks should prevent this.
- For Mac MPS issues with torch during imports, set:
  ```bash
  export USE_LOCAL_EMBEDDINGS=true
  export EMBEDDING_DEVICE=cpu
  ```
