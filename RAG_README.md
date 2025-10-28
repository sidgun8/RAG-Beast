# RAG (Retrieval-Augmented Generation) with AWS Bedrock

This document explains how to use the RAG search functionality that combines semantic search with AWS Bedrock LLM for generating intelligent answers.

## Overview

The RAG endpoint (`POST /api/search/rag`) performs the following steps:
1. **Retrieve**: Use semantic/vector search to find the most relevant document chunks
2. **Augment**: Pass these chunks as context to an AWS Bedrock LLM
3. **Generate**: The LLM generates a comprehensive answer based on the retrieved context

## Setup

### 1. AWS Credentials

Add the following environment variables to your `.env` file:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
```

### 2. Install Dependencies

```bash
pip install boto3 botocore
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 3. AWS Bedrock Access

Make sure:
- Your AWS account has access to Bedrock
- The models you want to use are enabled in your AWS Bedrock console
- Your IAM user/role has the `bedrock:InvokeModel` permission

## Available Models

The service supports multiple AWS Bedrock models:

### Claude (Anthropic)
- `anthropic.claude-3-haiku-20240307-v1:0` (default - fast and cost-effective)
- `anthropic.claude-3-sonnet-20240229-v1:0` (balanced)
- `anthropic.claude-3-opus-20240229-v1:0` (most capable)
- `anthropic.claude-v2:1`

### Llama 2 (Meta)
- `meta.llama2-13b-chat-v1`
- `meta.llama2-70b-chat-v1`

### Jurassic-2 (AI21)
- `ai21.j2-ultra-v1`
- `ai21.j2-mid-v1`

### Titan (Amazon)
- `amazon.titan-text-express-v1`

**Note**: Model availability varies by AWS region.

## API Usage

### Endpoint: `POST /api/search/rag`

#### Request Body

```json
{
  "query": "What are the key findings from the research?",
  "limit": 5,
  "similarity_threshold": 0.5,
  "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
  "max_tokens": 1000,
  "temperature": 0.7
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The question to answer |
| `limit` | int | 5 | Number of chunks to retrieve (1-20) |
| `similarity_threshold` | float | 0.5 | Minimum similarity score (0.0-1.0) |
| `model_id` | string | null | AWS Bedrock model ID (uses default if null) |
| `max_tokens` | int | 1000 | Maximum tokens in response (100-4000) |
| `temperature` | float | 0.7 | Sampling temperature for creativity (0.0-1.0) |

#### Response

```json
{
  "query": "What are the key findings from the research?",
  "answer": "Based on the retrieved documents...",
  "chunks_used": [
    {
      "id": "doc-123",
      "title": "Research Document 1",
      "content_preview": "The research findings indicate...",
      "similarity_score": 0.92
    }
  ],
  "total_chunks": 1,
  "search_type": "rag",
  "model_used": "anthropic.claude-3-haiku-20240307-v1:0",
  "parameters": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "similarity_threshold": 0.5
  }
}
```

## Example Usage

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/search/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of using vector search?",
    "limit": 3,
    "similarity_threshold": 0.7
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search/rag",
    json={
        "query": "Explain the concept of semantic search",
        "limit": 5,
        "similarity_threshold": 0.6,
        "max_tokens": 800,
        "temperature": 0.5
    }
)

result = response.json()
print(f"Question: {result['query']}")
print(f"\nAnswer: {result['answer']}")
print(f"\nUsing {result['total_chunks']} document chunks")
```

### Using JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/search/rag', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What is the main topic discussed?',
    limit: 5,
    similarity_threshold: 0.5,
    max_tokens: 1000
  })
});

const result = await response.json();
console.log('Answer:', result.answer);
```

## How It Works

### Step 1: Retrieval
- Query is converted to an embedding using your configured embedding model
- Vector similarity search finds the most relevant document chunks
- Results are filtered by `similarity_threshold` and limited by `limit`

### Step 2: Context Preparation
- Retrieved chunks are formatted with metadata (title, score, content)
- A prompt is constructed that includes:
  - Instructions for the LLM
  - The retrieved context documents
  - The user's question

### Step 3: Generation
- The prompt is sent to AWS Bedrock
- The LLM generates an answer based on the provided context
- The response is returned to the client

## Best Practices

### 1. Chunk Retrieval
- Use `limit` to control how many chunks to include (typically 3-10)
- Set `similarity_threshold` appropriately (0.5-0.8 for most cases)
- Too many chunks can overwhelm the LLM and increase costs

### 2. LLM Parameters
- **Max Tokens**: Start with 1000, increase if answers are truncated
- **Temperature**: 
  - 0.0-0.3 for factual, deterministic answers
  - 0.5-0.7 for balanced responses (default)
  - 0.8-1.0 for creative, varied responses

### 3. Cost Optimization
- Claude Haiku is the most cost-effective for most use cases
- Use fewer chunks (`limit`) when possible
- Reduce `max_tokens` if responses are consistently shorter

### 4. Quality Tips
- Ensure your documents are well-structured and relevant
- Use appropriate `similarity_threshold` to filter noise
- Test different models to find the best fit for your use case

## Troubleshooting

### Error: "AWS credentials not found"
- Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env`
- Verify the credentials are correct

### Error: "Access denied to AWS Bedrock"
- Check IAM permissions - you need `bedrock:InvokeModel`
- Ensure Bedrock is enabled in your AWS region
- Verify the model is available in your region

### Error: "Model may not be available in region"
- Some models aren't available in all AWS regions
- Use a different model or change your AWS region
- Check AWS Bedrock console for available models

### Poor Answer Quality
- Try increasing `similarity_threshold` to get higher-quality chunks
- Increase `limit` to provide more context
- Experiment with different models (try Claude Sonnet or Opus)
- Adjust `temperature` for different response styles

## Cost Estimates

Approximate costs (as of 2024):

- **Claude Haiku**: ~$0.25 per 1M input tokens, $1.25 per 1M output tokens
- **Claude Sonnet**: ~$3 per 1M input tokens, $15 per 1M output tokens
- **Llama 2 70B**: ~$0.35 per 1M input tokens, $0.35 per 1M output tokens

A typical RAG query with 5 chunks (5000 tokens) and a 500-token answer using Claude Haiku costs approximately **$0.002** per request.

## Advanced Usage

### Custom Prompts

You can modify the prompt template in `src/services/llm_service.py` in the `_create_prompt` method to customize how the LLM responds.

### Multi-Turn Conversations

For conversation history, you can extend the service to:
1. Store conversation context
2. Include previous messages in the prompt
3. Reference past chunks for consistency

### Hybrid Retrieval

Combine the RAG endpoint with other search methods:
1. Use semantic search for initial retrieval
2. Use metadata filters for additional filtering
3. Send the best results to the LLM

## Support

For issues or questions:
1. Check AWS Bedrock documentation: https://docs.aws.amazon.com/bedrock/
2. Review AWS Bedrock model pricing: https://aws.amazon.com/bedrock/pricing/
3. Check IAM permissions and Bedrock access policies
