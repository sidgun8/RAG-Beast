# Per-Tenant LLM Model Configuration

## Overview

Each tenant can now configure their own LLM model for RAG answer generation. The system supports AWS Bedrock models with automatic fallbacks and per-request overrides.

## Configuration

### In `config/tenants.yaml`

```yaml
tenants:
  my_tenant:
    # ... other settings ...
    
    # LLM Configuration
    llm_model_id: "us.anthropic.claude-opus-4-1-20250805-v1:0"
    llm_max_tokens: 1500
    llm_temperature: 0.7
```

## Available Models

### AWS Bedrock Models

**Claude Models** (Recommended):
- `us.anthropic.claude-opus-4-1-20250805-v1:0` - Most capable, highest cost
- `anthropic.claude-3-sonnet-20240229-v1:0` - Balanced performance and cost
- `anthropic.claude-3-haiku-20240307-v1:0` - Fastest, most affordable
- `anthropic.claude-v2:1` - Legacy Claude

**Llama Models**:
- `us.meta.llama4-scout-17b-instruct-v1:0` - Fast, cost-effective

**Other Models**:
- `ai21.j2-ultra-v1` - AI21 Jurassic
- `amazon.titan-text-express-v1` - Amazon Titan

### Local Open-Source Models (Coming Soon)

For Apple Silicon (MPS) users, these models can run locally:
- **Llama 3.2** (1B, 3B) - Very fast, runs on laptop
- **Phi-3** (3.8B) - Microsoft's efficient model
- **Mistral 7B** - Powerful open model
- **Gemma 2** (2B, 7B) - Google's open model

*Note: Local model support requires additional implementation*

## Tenant Examples

### Example 1: Research Team (Claude Opus)
```yaml
tenant_a:
  embedding_model: "embeddinggemma"
  search_type: "hybrid"
  
  # Premium LLM for high-quality answers
  llm_model_id: "us.anthropic.claude-opus-4-1-20250805-v1:0"
  llm_max_tokens: 1500
  llm_temperature: 0.7
```

### Example 2: General Use (Llama Scout)
```yaml
tenant_b:
  embedding_model: "openai"
  search_type: "semantic"
  
  # Fast, cost-effective LLM
  llm_model_id: "us.meta.llama4-scout-17b-instruct-v1:0"
  llm_max_tokens: 1000
  llm_temperature: 0.7
```

### Example 3: Quick Answers (Claude Haiku)
```yaml
tenant_c:
  embedding_model: "huggingface"
  search_type: "text"
  
  # Fastest, most affordable Claude
  llm_model_id: "anthropic.claude-3-haiku-20240307-v1:0"
  llm_max_tokens: 800
  llm_temperature: 0.5
```

## Layered Embeddings (Advanced)

Tenants can optionally define an `embedding_pipeline` to combine multiple embedders and refine dimensions:

```yaml
BibleIQ:
  embedding_pipeline:
    combine: concat  # or weighted_sum
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
```

Notes:
- `combine` controls how vectors are merged (concatenate or weighted sum)
- `normalize` applies L2-normalization per component before combining
- `projector` can truncate to a target dimension (matryoshka) or use PCA if available

## Usage

### Default Behavior (Uses Tenant Config)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag?tenant_id=tenant_a",
    json={
        "query": "What are the main findings?"
    }
)

# Uses tenant_a's configured LLM (Claude Opus)
result = response.json()
print(result['answer'])
print(f"Model used: {result['model_used']}")
print(f"LLM config: {result['llm_config']}")
```

### Override Per Request

```python
response = requests.post(
    "http://localhost:8000/api/search/rag?tenant_id=tenant_a",
    json={
        "query": "What are the main findings?",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",  # Override
        "max_tokens": 500,  # Override
        "temperature": 0.3  # Override
    }
)
```

## Parameters

### `llm_model_id`
- **Type**: string
- **Default**: `us.meta.llama4-scout-17b-instruct-v1:0`
- **Description**: AWS Bedrock model identifier

### `llm_max_tokens`
- **Type**: integer
- **Range**: 100-4000
- **Default**: 1000
- **Description**: Maximum response length

### `llm_temperature`
- **Type**: float
- **Range**: 0.0-1.0
- **Default**: 0.7
- **Description**: 
  - `0.0` = Deterministic, focused answers
  - `0.7` = Balanced creativity
  - `1.0` = Maximum creativity/randomness

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Research/Analysis** | Claude Opus | Most capable, best understanding |
| **General Q&A** | Llama Scout | Fast, good quality, cost-effective |
| **Quick Lookups** | Claude Haiku | Fastest, affordable, good for simple queries |
| **Creative Writing** | Claude Opus (temp=0.9) | Best creative capabilities |
| **Factual Queries** | Any model (temp=0.3) | Lower temperature for accuracy |
| **Long Documents** | Claude Sonnet | Balanced quality and speed |

## Cost Optimization

**For budget-conscious deployments:**
1. Use **Llama Scout** as default (lowest cost)
2. Reserve **Claude Opus** for complex queries only
3. Use **Claude Haiku** for simple lookups
4. Reduce `max_tokens` to minimum needed

**Example cost-optimized config:**
```yaml
default:
  llm_model_id: "us.meta.llama4-scout-17b-instruct-v1:0"
  llm_max_tokens: 800
  llm_temperature: 0.7
```

## Fallback Behavior

If the primary model fails, the system automatically tries:
1. Your configured model
2. `us.meta.llama4-scout-17b-instruct-v1:0`
3. `anthropic.claude-v2:1`
4. `anthropic.claude-3-haiku-20240307-v1:0`

This ensures high availability even if a model is unavailable.

## Response Fields

When querying with RAG, you get:

```json
{
  "answer": "The generated answer...",
  "model_used": "us.anthropic.claude-opus-4-1-20250805-v1:0",
  "llm_config": {
    "model_id": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "max_tokens": 1500,
    "temperature": 0.7
  },
  "tenant_config": {
    "llm_model_id": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    ...
  }
}
```

## Best Practices

1. **Match model to task complexity**
   - Simple queries → Haiku
   - General use → Llama Scout
   - Complex analysis → Opus

2. **Tune temperature appropriately**
   - Factual: 0.0-0.3
   - Balanced: 0.5-0.7
   - Creative: 0.8-1.0

3. **Set appropriate max_tokens**
   - Short answers: 500-800
   - Normal: 1000-1500
   - Detailed: 2000-4000

4. **Use per-request overrides sparingly**
   - Configure defaults in tenant config
   - Override only when needed

5. **Monitor costs**
   - Start with cheaper models
   - Upgrade selectively based on quality needs

## Troubleshooting

### Model Not Available
If you see errors about model availability:
1. Check AWS Bedrock console for model access
2. Verify region supports the model
3. System will automatically fall back to available models

### Unexpected Costs
- Reduce `max_tokens`
- Switch to more economical model (Llama Scout or Haiku)
- Lower `temperature` (fewer tokens needed for focused answers)

### Poor Quality Answers
- Upgrade to better model (Sonnet or Opus)
- Increase `max_tokens` for more detailed answers
- Adjust `temperature` (lower for factual, higher for creative)

---

**Need help?** Check `README.md` for full system documentation.

