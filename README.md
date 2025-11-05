# Multi-Tenant RAG System with Configurable Chunking

A production-ready PostgreSQL-based RAG (Retrieval-Augmented Generation) system with multi-tenant support, intelligent chunking strategies, and flexible embedding options.

## Features

- **Multi-Tenant Isolation** - Complete data and configuration isolation per tenant
- **3 Chunking Strategies** - Semantic, Sliding Window, and Recursive chunking
- **Flexible Embeddings** - Support for local (EmbeddingGemma) and API-based models (OpenAI, HuggingFace)
- **Variable Dimensions** - Configure embedding dimensions per tenant (256-1536)
- **Hybrid Search** - Combine semantic + full-text search with RRF
- **Metadata Filtering** - Filter results by structured metadata
- **AWS Bedrock LLM** - Generate answers using Claude/other models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Database

```bash
# Run migration to add tenant_id support
python scripts/add_tenant_id_migration.py
```

### 3. Configure Tenants

Edit `config/tenants.yaml`:

```yaml
tenants:
  my_tenant:
    # Embeddings (for ingestion)
    embedding_model: "embeddinggemma"  # or "openai", "huggingface"
    embedding_dimensions: 768
    
    # Search (for RAG)
    search_type: "semantic"  # or "text", "hybrid"
    
    # Chunking (for ingestion)
    chunking_strategy: "semantic"  # or "sliding_window", "recursive"
    chunk_size: 1000
    chunk_overlap: 100
```

### 4. Ingest Documents

```bash
# Ingest with tenant's default chunking
python scripts/ingest_documents.py --tenant my_tenant --file document.txt

# Override chunking strategy
python scripts/ingest_documents.py \
  --tenant my_tenant \
  --file document.txt \
  --chunking-strategy semantic \
  --chunk-size 800
```

### 5. Query with RAG

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag?tenant_id=my_tenant",
    json={
        "query": "What are the main points?",
        "limit": 5,
        "chunk_type": "semantic",  # or use tenant's default
        "metadata_filter": {"category": "research"}  # optional
    }
)

print(response.json()['answer'])
```

## Configuration

### Tenant Configuration (`config/tenants.yaml`)

Each tenant can configure:

**Embedding** (used during ingestion):
- `embedding_model`: "embeddinggemma" | "openai" | "huggingface"
- `embedding_dimensions`: 256-1536 (depends on model)

**Search** (used during RAG queries):
- `search_type`: "semantic" | "text" | "hybrid"
- `semantic_weight`: 0.0-1.0 (for hybrid)
- `text_weight`: 0.0-1.0 (for hybrid)

**LLM** (used for RAG answer generation):
- `llm_model_id`: AWS Bedrock model ID
  - "us.anthropic.claude-opus-4-1-20250805-v1:0" (most capable)
  - "us.meta.llama4-scout-17b-instruct-v1:0" (fast, cost-effective)
  - "anthropic.claude-3-haiku-20240307-v1:0" (fast, affordable)
- `llm_max_tokens`: 100-4000 (default: 1000)
- `llm_temperature`: 0.0-1.0 (default: 0.7)

**Chunking** (used during ingestion):
- `chunking_strategy`: "semantic" | "sliding_window" | "recursive"
- `chunk_size`: Target chunk size in characters
- `chunk_overlap`: Overlap between chunks (sliding_window only)
- `semantic_similarity_threshold`: 0.0-1.0 (semantic only)

**Layered Embeddings (optional)**:
- Define an `embedding_pipeline` to combine multiple embedders and optionally project dimensions.
- Example:
```yaml
embedding_pipeline:
  combine: concat          # or weighted_sum
  normalize: true          # L2 normalize each component before combine
  components:
    - type: sentence_transformer
      name: "odunola/sentence-transformers-bible-reference-final"
      weight: 1.0
    - type: embeddinggemma
      name: "google/embeddinggemma-300m"
      weight: 0.5
  projector:
    type: matryoshka
    target_dims: 768
```

### Environment Variables (`.env`)

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vector_search
DB_USER=postgres
DB_PASSWORD=your_password

# AWS Bedrock (for RAG)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# Optional: API Keys for cloud embeddings
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_hf_key
```

## Chunking Strategies

### Semantic Chunking
- Groups semantically similar sentences using embeddings
- **Best for**: Articles, essays, narratives
- **Pros**: Preserves complete thoughts
- **Cons**: Slower (requires sentence embeddings)

### Sliding Window
- Fixed-size chunks with configurable overlap
- **Best for**: General documents, fast bulk import
- **Pros**: Fast, predictable sizes
- **Cons**: May split related ideas

### Recursive Chunking
- Hierarchical splitting (paragraphs → sentences → words)
- **Best for**: Code, technical docs, structured content
- **Pros**: Respects document structure
- **Cons**: Less suitable for unstructured text

## API Endpoints

### Ingestion

**Ingest Text**
```bash
POST /api/ingest/text
{
  "title": "Document Title",
  "content": "Document content...",
  "tenant_id": "my_tenant",
  "chunking_strategy": "semantic",  # optional override
  "chunk_size": 800  # optional override
}
```

**Upload File**
```bash
POST /api/ingest/file
FormData:
  - file: document.pdf
  - tenant_id: my_tenant
  - chunking_strategy: semantic  # optional
```

### Search

**RAG Search** (Retrieval + Generation)
```bash
POST /api/search/rag?tenant_id=my_tenant
{
  "query": "Your question",
  "limit": 5,
  "chunk_type": "semantic",  # or use tenant default
  "metadata_filter": {"category": "research"},  # optional
  "model_id": "anthropic.claude-3-haiku-20240307-v1:0"  # optional LLM override
}
```

Note: Both ingestion and query embeddings use the tenant's configured embedder (or layered pipeline) consistently.

**Semantic Search**
```bash
POST /api/search/semantic?tenant_id=my_tenant
{
  "query": "search query",
  "limit": 10
}
```

**Hybrid Search**
```bash
POST /api/search/hybrid?tenant_id=my_tenant
{
  "query": "search query",
  "limit": 10,
  "semantic_weight": 0.7,
  "text_weight": 0.3
}
```

## Multi-Tenancy

### Data Isolation

Documents are isolated per tenant:

```python
# Ingest for tenant_a
await doc_service.add_document(
    title="Doc A",
    content="Content...",
    tenant_id="tenant_a"
)

# Query tenant_a - only sees their data
results = await doc_service.search_documents(
    query="search",
    tenant_id="tenant_a"
)
# ✓ Returns only tenant_a documents
# ✗ Cannot see tenant_b documents
```

### Configuration Per Tenant

Different tenants can have different settings:

```yaml
tenants:
  research_team:
    embedding_model: "embeddinggemma"
    chunking_strategy: "semantic"
    search_type: "hybrid"
  
  legal_dept:
    embedding_model: "openai"
    chunking_strategy: "sliding_window"
    search_type: "text"
  
  code_docs:
    embedding_model: "huggingface"
    chunking_strategy: "recursive"
    search_type: "semantic"

  BibleIQ:
    embedding_pipeline:
      combine: concat
      normalize: true
      components:
        - type: sentence_transformer
          name: "odunola/sentence-transformers-bible-reference-final"
          weight: 1.0
        - type: embeddinggemma
          name: "google/embeddinggemma-300m"
          weight: 0.5
      projector:
        type: matryoshka
        target_dims: 768
    search_type: "semantic"
```

## Scripts

### Validate Configuration
```bash
python scripts/validate_config.py
```

### Ingest Documents
```bash
# Single file
python scripts/ingest_documents.py --tenant my_tenant --file doc.txt

# Directory (recursive)
python scripts/ingest_documents.py --tenant my_tenant --directory ./docs/ --recursive

# With chunking override
python scripts/ingest_documents.py \
  --tenant my_tenant \
  --file doc.txt \
  --chunking-strategy semantic \
  --chunk-size 800
```

### Ingest Bible
```bash
python scripts/ingest_bible_chunks.py
# → Prompts for tenant selection
# → Uses tenant's chunking configuration
```

### Clean Database
```bash
python scripts/clean_database.py
```

## Examples

### Basic Ingestion
```python
from services.document_service import DocumentService

doc_service = DocumentService()
await doc_service.initialize()

result = await doc_service.add_document(
    title="My Document",
    content="Document content here...",
    tenant_id="my_tenant",
    chunking_strategy="semantic",  # optional override
    chunk_size=800
)

print(f"Created {result['chunks']} chunks using {result['chunking_strategy']}")
```

### RAG Query
```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag?tenant_id=my_tenant",
    json={
        "query": "What are the key findings?",
        "limit": 5,
        "metadata_filter": {"year": "2024"}
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Based on {len(result['chunks_used'])} chunks")
```


## Best Practices

### Chunking
1. **Default**: Use `sliding_window` for general documents
2. **Articles**: Use `semantic` for better context preservation
3. **Code**: Use `recursive` to respect structure
4. **Chunk Size**: Keep between 500-1500 characters for best results

### Multi-Tenancy
1. **Always specify tenant_id** during ingestion and search
2. **Configure defaults** in `config/tenants.yaml`
3. **Override when needed** via API or CLI flags
4. **Test isolation** by querying different tenants

### Performance
1. **Semantic chunking** is slower (requires embeddings per sentence)
2. **Sliding window** is fastest for bulk imports
3. **Local embeddings** (EmbeddingGemma) avoid API costs
4. **Hybrid search** provides best quality but is slower than single-method search

## Troubleshooting

### Configuration Not Found
If tenant config isn't found, the system uses the `default` tenant.

**Solution**: Ensure `config/tenants.yaml` exists and contains your tenant.

### No Results from Search
**Check**:
1. Did you ingest data for this tenant?
2. Is the tenant_id correct in your query?
3. Run: `curl "http://localhost:8000/api/search/stats?tenant_id=my_tenant"`

### Chunks Too Large/Small
**Solution**: Adjust `chunk_size` in tenant config or pass `--chunk-size` flag during ingestion.

### Semantic Chunking Slow
**Solution**: Use `sliding_window` for bulk ingestion, `semantic` only for high-value documents.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Documents  │────▶│  ChunkingService │────▶│  PostgreSQL  │
│             │     │  (3 strategies)  │     │  + pgvector  │
└─────────────┘     └──────────────────┘     └──────────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │ EmbeddingService│    │  Vector Search  │
                    │  (multi-model)  │    │   + Full-Text   │
                    └─────────────────┘    └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  AWS Bedrock    │
                                            │  (LLM Response) │
                                            └─────────────────┘
```

## Project Structure

```
.
├── config/
│   └── tenants.yaml          # Multi-tenant configuration
├── src/
│   ├── services/
│   │   ├── chunking_service.py      # 3 chunking strategies
│   │   ├── document_service.py      # Document CRUD + tenant isolation
│   │   ├── embedding_service.py     # Multi-model embeddings
│   │   ├── database_service.py      # PostgreSQL + pgvector
│   │   ├── tenant_config_service.py # Config management
│   │   └── llm_service.py          # AWS Bedrock integration
│   └── routes/
│       ├── search.py         # Search + RAG endpoints
│       └── ingest.py         # Ingestion endpoints
├── scripts/
│   ├── ingest_documents.py   # General document ingestion
│   ├── ingest_bible_chunks.py# Bible-specific ingestion
│   ├── validate_config.py    # Config validation
│   └── clean_database.py     # Database cleanup
└── examples/
    └── tenant_ingestion_example.py  # Usage examples
```

## License

MIT

## Documentation

- **README.md** - This comprehensive guide
- **LLM_MODELS.md** - Per-tenant LLM model configuration guide
- **config/tenants.yaml** - Tenant configuration file
- **API Docs**: `http://localhost:8000/docs` (when server running)

## Support

- **Configuration**: See `config/tenants.yaml`
- **LLM Models**: See `LLM_MODELS.md`
- **Scripts**: See `scripts/` directory for utilities

---

**Ready to use!** Configure your tenants, ingest your documents, and start querying with RAG.
