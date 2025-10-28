import os
import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with AWS Bedrock LLM models"""
    
    def __init__(self):
        """Initialize the LLM service with AWS credentials from environment"""
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        # Default model - check for BEDROCK_MODEL_ID or MODEL_ID
        model_id = os.getenv('BEDROCK_MODEL_ID') or os.getenv('MODEL_ID')
        
        # Fallback models that are widely available
        default_models = [
            'us.meta.llama4-scout-17b-instruct-v1:0',  # Your specified model
            'anthropic.claude-v2:1',  # Claude 2.1 (older but more widely available)
            'anthropic.claude-3-haiku-20240307-v1:0',  # Claude 3 Haiku
        ]
        
        self.model_id = model_id or default_models[0]
        self.fallback_models = [model_id] + [m for m in default_models if m != model_id] if model_id else default_models
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], 
                       model_id: Optional[str] = None, 
                       max_tokens: int = 1000,
                       temperature: float = 0.7) -> tuple[str, str]:
        """
        Generate an answer using AWS Bedrock based on query and context chunks
        
        Args:
            query: The user's question
            context_chunks: List of document chunks retrieved from vector search
            model_id: Optional model ID (defaults to configured model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            Generated answer as a string
        """
        # Prepare context from chunks
        context_text = self._format_context(context_chunks)
        
        # Create the prompt
        prompt = self._create_prompt(query, context_text)
        
        # Use specified model or default
        primary_model = model_id or self.model_id
        
        # Try primary model, then fallback models
        models_to_try = [primary_model] + [m for m in self.fallback_models if m != primary_model]
        
        last_error = None
        
        for model in models_to_try:
            try:
                logger.info(f"Trying model: {model}")
                
                # Check which model to use based on model_id
                model_lower = model.lower()
                if 'claude' in model_lower:
                    answer = self._invoke_claude(prompt, model, max_tokens, temperature)
                elif 'llama' in model_lower or 'scout' in model_lower:
                    # Meta Llama models
                    answer = self._invoke_llama(prompt, model, max_tokens, temperature)
                elif 'jurassic' in model_lower:
                    answer = self._invoke_jurassic(prompt, model, max_tokens, temperature)
                elif 'titan' in model_lower:
                    answer = self._invoke_titan(prompt, model, max_tokens, temperature)
                else:
                    # Default to Claude
                    logger.warning(f"Unknown model type, trying Claude invocation for {model}")
                    answer = self._invoke_claude(prompt, model, max_tokens, temperature)
                
                # Return the answer and the model that was used
                return answer, model
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.warning(f"AWS Bedrock error with model {model}: {error_code}")
                last_error = e
                
                # If this is the last model, raise the error
                if model == models_to_try[-1]:
                    if error_code == 'ValidationException':
                        raise ValueError(
                            f"No available models in region {self.region}. "
                            f"Tried: {models_to_try}. Please check AWS Bedrock console to enable models."
                        )
                    elif error_code == 'AccessDeniedException':
                        raise PermissionError("Access denied to AWS Bedrock. Please check your IAM permissions.")
                    else:
                        raise Exception(f"AWS Bedrock error: {e}")
                # Otherwise, continue to next model
                continue
                
        # If we get here, all models failed
        raise Exception(f"Failed to invoke any model. Last error: {last_error}")
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks into a readable string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            title = chunk.get('title', 'Unknown')
            similarity = chunk.get('similarity_score', 0)
            
            context_parts.append(
                f"[Document {i}] {title}\n"
                f"Relevance Score: {similarity:.3f}\n"
                f"{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM"""
        return f"""You are a helpful AI assistant. Use the following retrieved documents to answer the user's question.

<documents>
{context}
</documents>

<question>
{query}
</question>

Instructions:
1. Answer the question based ONLY on the information provided in the documents above.
2. If the documents don't contain enough information to answer the question, say so clearly.
3. Cite which document(s) you used for your answer by referencing the document number.
4. Be concise and accurate.
5. If the question cannot be answered with the given context, politely explain what information is missing.

Answer:"""
    
    def _invoke_claude(self, prompt: str, model_id: str, max_tokens: int, temperature: float) -> str:
        """Invoke Claude model via Bedrock"""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    def _invoke_llama(self, prompt: str, model_id: str, max_tokens: int, temperature: float) -> str:
        """Invoke Llama model via Bedrock"""
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['generation']
    
    def _invoke_jurassic(self, prompt: str, model_id: str, max_tokens: int, temperature: float) -> str:
        """Invoke Jurassic model via Bedrock"""
        body = {
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['completions'][0]['data']['text']
    
    def _invoke_titan(self, prompt: str, model_id: str, max_tokens: int, temperature: float) -> str:
        """Invoke Titan model via Bedrock"""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['results'][0]['outputText']


# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """Get singleton instance of LLM service"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
