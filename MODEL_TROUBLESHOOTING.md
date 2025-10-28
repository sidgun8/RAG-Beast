# AWS Bedrock Model Troubleshooting

## Problem: "Model may not be available in region"

This error occurs when the specified AWS Bedrock model is not available in your configured AWS region.

## Solutions

### 1. Enable Models in AWS Bedrock Console

1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to "Model access" in the left sidebar
3. Click "Request model access" for the models you want to use
4. Wait for approval (usually instant for most models)

### 2. Check Model Availability by Region

Model availability varies by region. Common models and their typical availability:

#### Claude Models (Anthropic)
- `anthropic.claude-v2:1` - Available in: us-east-1, us-west-2, eu-west-1, ap-northeast-1
- `anthropic.claude-3-haiku-20240307-v1:0` - Available in: us-east-1, us-west-2
- `anthropic.claude-3-sonnet-20240229-v1:0` - Available in: us-east-1, us-west-2
- `anthropic.claude-3-opus-20240229-v1:0` - Available in: us-east-1

#### Llama Models (Meta)
- `meta.llama2-13b-chat-v1` - Available in: us-east-1, us-west-2, eu-west-1
- `meta.llama2-70b-chat-v1` - Available in: us-east-1

#### Jurassic Models (AI21)
- `ai21.j2-ultra-v1` - Available in: us-east-1, us-west-2
- `ai21.j2-mid-v1` - Available in: us-east-1, us-west-2

#### Titan Models (Amazon)
- `amazon.titan-text-express-v1` - Available in: us-east-1, us-west-2, ap-southeast-1, eu-central-1

### 3. Change AWS Region

Update your `.env` file to use a region where your desired models are available:

```env
AWS_REGION=us-east-1  # Most models available here
```

Or for other regions:

```env
AWS_REGION=us-west-2  # Good alternative
AWS_REGION=eu-west-1  # European region
```

### 4. Update Model Configuration

The system now automatically tries fallback models if the primary model fails. You can still specify which model to use:

```env
# In .env file
BEDROCK_MODEL_ID=anthropic.claude-v2:1  # Most widely available
```

### 5. Test Available Models

Run this Python script to test which models work in your region:

```python
import boto3
import os
from botocore.exceptions import ClientError

region = os.getenv('AWS_REGION', 'us-east-1')
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

bedrock = boto3.client(
    'bedrock-runtime',
    region_name=region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

models_to_test = [
    'anthropic.claude-v2:1',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'meta.llama2-13b-chat-v1',
    'amazon.titan-text-express-v1'
]

print(f"Testing models in region: {region}\n")

for model in models_to_test:
    try:
        # Try to invoke the model with a minimal request
        if 'claude' in model:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}]
            }
        else:
            continue  # Skip non-Claude for this simple test
            
        response = bedrock.invoke_model(
            modelId=model,
            body=body
        )
        print(f"✅ {model} - AVAILABLE")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ {model} - {error_code}")
    except Exception as e:
        print(f"❌ {model} - {str(e)}")
```

## Automatic Fallback

The RAG service now automatically tries multiple models if the primary model fails:

1. First tries the requested model (or default)
2. Falls back to Claude 2.1 (most widely available)
3. Falls back to Claude 3 Haiku
4. Falls back to Claude 3 Sonnet
5. Returns error only if all fail

## Recommendations

1. **For maximum compatibility**: Use `anthropic.claude-v2:1` as it's available in most regions
2. **For best performance in US**: Use Claude 3 models in `us-east-1`
3. **For European users**: Use Claude 2.1 in `eu-west-1`

## Checking Model Availability via API

You can also check model availability programmatically:

```python
import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1')
response = bedrock.list_foundation_models()

for model in response['modelSummaries']:
    print(f"{model['modelId']} - {model['providerName']}")
```

## Still Having Issues?

1. Verify your AWS credentials are correct
2. Check your IAM user/role has `bedrock:InvokeModel` permission
3. Ensure Bedrock is enabled in your AWS account
4. Contact AWS support if models show as available but you still get errors

