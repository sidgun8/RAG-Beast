import os
import asyncio
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
from functools import lru_cache
from math import sqrt
from .tenant_config_service import get_tenant_config_service


class EmbeddingService:
    """Embedding service with support for multiple models including EmbeddingGemma"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'embeddinggemma')
        self.embedding_dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))
        self.device = os.getenv('EMBEDDING_DEVICE', 'cpu')
        self.use_local = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true'
        
        # Initialize models lazily
        self._model = None
        self._tokenizer = None
        self._sentence_transformer = None
        # Cache for arbitrary SentenceTransformer models loaded by name
        self._custom_models = {}
        self._model_initialized = False
        self._initialized = True

    # ---------- Utility helpers for layered embeddings ----------
    def _l2_normalize(self, vec: Union[List[float], np.ndarray]) -> np.ndarray:
        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr
        return arr / norm

    def _combine_vectors(self, vectors: List[Union[List[float], np.ndarray]],
                         weights: Optional[List[float]] = None,
                         method: str = 'concat') -> np.ndarray:
        if not vectors:
            return np.array([], dtype=float)
        if method == 'concat':
            return np.concatenate([np.array(v, dtype=float) for v in vectors], axis=-1)
        if method == 'weighted_sum':
            arrs = [np.array(v, dtype=float) for v in vectors]
            if weights is None:
                weights = [1.0] * len(arrs)
            w = np.array(weights, dtype=float)
            if len(w) != len(arrs):
                raise ValueError('weights length must equal number of vectors for weighted_sum')
            # Ensure all vectors same length
            dim = len(arrs[0])
            if any(len(a) != dim for a in arrs):
                raise ValueError('All vectors must have same dimension for weighted_sum')
            stacked = np.vstack(arrs)
            return (w[:, None] * stacked).sum(axis=0)
        raise ValueError(f"Unknown combine method: {method}")

    def _project_vector(self, vec: Union[List[float], np.ndarray], projector_cfg: Optional[dict]) -> np.ndarray:
        """Apply optional projection/refinement to a vector.
        Supports:
          - type: 'matryoshka' → truncate to target_dims
          - type: 'pca' → requires sklearn; fit is ephemeral per process (not persisted)
          - type: 'none' or missing → no-op
        """
        if not projector_cfg or not isinstance(projector_cfg, dict):
            return np.array(vec, dtype=float)
        ptype = projector_cfg.get('type', 'none')
        target_dims = projector_cfg.get('target_dims')
        arr = np.array(vec, dtype=float)
        if ptype == 'none' or target_dims is None:
            return arr
        if ptype == 'matryoshka':
            return arr[: int(target_dims)]
        if ptype == 'pca':
            try:
                from sklearn.decomposition import PCA  # type: ignore
                # Fit PCA on the single vector is degenerate; instead, no-op if insufficient dims
                if arr.ndim == 1:
                    # Without a dataset, approximate by truncation if needed
                    return arr[: int(target_dims)]
                pca = PCA(n_components=int(target_dims))
                reduced = pca.fit_transform(arr)
                return reduced
            except Exception:
                # Fallback to truncation if sklearn not available or errors
                return arr[: int(target_dims)]
        return arr
    
    def _ensure_model_initialized(self):
        """Ensure the model is initialized (lazy loading)"""
        if self._model_initialized:
            return
            
        print('Initializing embedding model...')
        self._initialize_model()
        self._model_initialized = True
    
    def _initialize_model(self):
        """Initialize the embedding model based on configuration"""
        if self.use_local:
            print('Using local EmbeddingGemma (offline mode)')
            # Check if MPS is available and requested
            if self.device == 'mps' and torch.backends.mps.is_available():
                print('Running on MPS (Apple Silicon GPU)')
                self.device = torch.device('mps')
            else:
                print('Running on CPU')
                self.device = torch.device('cpu')
            # Try to load from local path first, then from Hugging Face
            self._load_embeddinggemma_local()
        elif self.huggingface_api_key:
            print('Using Hugging Face API for EmbeddingGemma')
            print('GPU acceleration available via Hugging Face API')
        else:
            print('WARNING: No embedding service configured - using local mode')
            self.use_local = True
            # Try to load EmbeddingGemma from Hugging Face, fallback to sentence transformer
            self._load_embeddinggemma_from_huggingface()
    
    def _load_embeddinggemma_local(self):
        """Load EmbeddingGemma model locally"""
        try:
            model_path = "./models/embeddinggemma"
            if os.path.exists(model_path):
                print(f"Loading EmbeddingGemma from {model_path}...")
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModel.from_pretrained(model_path).to(self.device)
                print(f"EmbeddingGemma loaded successfully on {self.device}")
            else:
                print("WARNING: Local EmbeddingGemma model not found, trying Hugging Face...")
                self._load_embeddinggemma_from_huggingface()
        except Exception as e:
            print(f"ERROR: Error loading EmbeddingGemma: {e}")
            self._load_fallback_model()
    
    def _load_embeddinggemma_from_huggingface(self):
        """Load EmbeddingGemma model from Hugging Face"""
        try:
            print("Loading EmbeddingGemma from Hugging Face...")
            model_name = "google/embeddinggemma-300m"
            # Add memory optimization settings
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                low_cpu_mem_usage=True
            )
            self._model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                low_cpu_mem_usage=True
            )
            print("EmbeddingGemma loaded from Hugging Face successfully")
        except Exception as e:
            print(f"ERROR: Error loading EmbeddingGemma from Hugging Face: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback embedding model"""
        try:
            print("Loading fallback embedding model...")
            # Use the lightest available model
            self._sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("Fallback model loaded successfully")
        except Exception as e:
            print(f"ERROR: Error loading fallback model: {e}")
            print("Trying to create a simple embedding service...")
            try:
                # Create a simple embedding service that doesn't require heavy models
                self._create_simple_embedding_service()
                print("Simple embedding service created successfully")
            except Exception as e2:
                print(f"ERROR: Error creating simple embedding service: {e2}")
                raise
    
    def _create_simple_embedding_service(self):
        """Create a simple embedding service using basic text processing"""
        print("Creating simple embedding service using TF-IDF...")
        # This will be a simple hash-based embedding for testing
        self._simple_embedding = True
        print("Simple embedding service ready (using hash-based embeddings)")
    
    async def _get_simple_embedding(self, text: str) -> List[float]:
        """Get a simple hash-based embedding"""
        import hashlib
        # Create a simple hash-based embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to float list (384 dimensions)
        embedding = []
        for i in range(0, len(hash_bytes), 2):
            if i + 1 < len(hash_bytes):
                val = int.from_bytes(hash_bytes[i:i+2], 'big') / 65535.0
                embedding.append(val)
        
        # Pad or truncate to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        embedding = embedding[:384]
        
        return embedding
    
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Ensure model is initialized before use
            self._ensure_model_initialized()
            
            if self.use_local:
                return await self._get_local_embedding(text, model)
            else:
                return await self._get_api_embedding(text, model)
        except Exception as e:
            print(f'Error getting embedding: {e}')
            raise
    
    async def _get_local_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding using local model"""
        if hasattr(self, '_simple_embedding') and self._simple_embedding:
            return await self._get_simple_embedding(text)
        elif self._model and self._tokenizer:
            return await self._get_embeddinggemma_embedding(text)
        elif self._sentence_transformer:
            return await self._get_sentence_transformer_embedding(text)
        else:
            raise Exception("No local embedding model available")
    
    async def _get_embeddinggemma_embedding(self, text: str) -> List[float]:
        """Get embedding using EmbeddingGemma model"""
        try:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Mean pooling and move back to CPU for conversion to list
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
            
            # Apply Matryoshka dimension reduction if needed
            if self.embedding_dimensions < 768:
                embedding = embedding[:self.embedding_dimensions]
            
            return embedding
        except Exception as e:
            print(f"Error getting EmbeddingGemma embedding: {e}")
            raise
    
    async def _get_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Get embedding using sentence transformer"""
        try:
            embedding = self._sentence_transformer.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting sentence transformer embedding: {e}")
            raise

    async def _get_custom_sentence_transformer_embedding(self, text: str, model_name: str) -> List[float]:
        """Get embedding using a custom SentenceTransformer model by name."""
        try:
            if model_name not in self._custom_models:
                print(f"Loading custom SentenceTransformer model: {model_name}...")
                self._custom_models[model_name] = SentenceTransformer(model_name)
                print(f"Custom model '{model_name}' loaded")
            embedding = self._custom_models[model_name].encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting custom sentence transformer embedding ({model_name}): {e}")
            raise
    
    async def _get_api_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding using API services"""
        embedding_model = model or self.embedding_model
        
        if embedding_model == 'embeddinggemma':
            return await self._get_embeddinggemma_api_embedding(text)
        elif embedding_model == 'openai':
            return await self._get_openai_embedding(text)
        elif embedding_model == 'huggingface':
            return await self._get_huggingface_embedding(text)
        else:
            # Auto-detect
            if self.huggingface_api_key:
                return await self._get_embeddinggemma_api_embedding(text)
            elif self.openai_api_key:
                return await self._get_openai_embedding(text)
            else:
                raise Exception('No embedding service available. Please provide API keys.')
    
    async def _get_embeddinggemma_api_embedding(self, text: str) -> List[float]:
        """Get embedding using EmbeddingGemma via Hugging Face API"""
        try:
            if not self.huggingface_api_key:
                raise Exception('Hugging Face API key not provided for EmbeddingGemma API')
            
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/embeddinggemma-300m",
                headers=headers,
                json={"inputs": text}
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            
            return response.json()
        except Exception as e:
            print(f"EmbeddingGemma API embedding error: {e}")
            raise
    
    async def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API"""
        try:
            if not self.openai_api_key:
                raise Exception('OpenAI API key not provided')
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={
                    "input": text,
                    "model": "text-embedding-ada-002"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API request failed: {response.status_code}")
            
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            raise
    
    async def _get_huggingface_embedding(self, text: str) -> List[float]:
        """Get embedding using Hugging Face API"""
        try:
            if not self.huggingface_api_key:
                raise Exception('Hugging Face API key not provided')
            
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
            response = requests.post(
                "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                headers=headers,
                json={"inputs": text}
            )
            
            if response.status_code != 200:
                raise Exception(f"Hugging Face API request failed: {response.status_code}")
            
            return response.json()
        except Exception as e:
            print(f"Hugging Face embedding error: {e}")
            raise
    
    async def get_batch_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Ensure model is initialized before use
            self._ensure_model_initialized()
            
            if self.use_local:
                return await self._get_local_batch_embeddings(texts, model)
            else:
                return await self._get_api_batch_embeddings(texts, model)
        except Exception as e:
            print(f'Error getting batch embeddings: {e}')
            raise
    
    async def _get_local_batch_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Get batch embeddings using local model"""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text, model)
            embeddings.append(embedding)
        return embeddings
    
    async def _get_api_batch_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Get batch embeddings using API"""
        embedding_model = model or self.embedding_model
        
        if embedding_model == 'openai' and self.openai_api_key:
            return await self._get_openai_batch_embeddings(texts)
        else:
            # Process one by one for other services
            embeddings = []
            for text in texts:
                embedding = await self.get_embedding(text, embedding_model)
                embeddings.append(embedding)
            return embeddings

    async def _get_custom_batch_embeddings(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Get batch embeddings using a custom SentenceTransformer model by name."""
        try:
            if model_name not in self._custom_models:
                print(f"Loading custom SentenceTransformer model: {model_name}...")
                self._custom_models[model_name] = SentenceTransformer(model_name)
                print(f"Custom model '{model_name}' loaded")
            embeddings = self._custom_models[model_name].encode(texts, convert_to_tensor=False)
            # sentence-transformers may return numpy array
            return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        except Exception as e:
            print(f"Error getting custom batch embeddings ({model_name}): {e}")
            raise
    
    async def _get_openai_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings using OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={
                    "input": texts,
                    "model": "text-embedding-ada-002"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API request failed: {response.status_code}")
            
            return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            print(f"OpenAI batch embedding error: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        # Ensure model is initialized to get accurate info
        if not self._model_initialized:
            return {
                "name": "Not Initialized",
                "dimensions": self.embedding_dimensions,
                "device": self.device,
                "offline": self.use_local,
                "note": "Model will be initialized on first use"
            }
            
        if self.use_local:
            if hasattr(self, '_simple_embedding') and self._simple_embedding:
                return {
                    "name": "Simple Hash Embedding (Local)",
                    "dimensions": 384,
                    "device": self.device,
                    "offline": True,
                    "note": "Simple hash-based embedding for testing"
                }
            elif self._model and self._tokenizer:
                return {
                    "name": "EmbeddingGemma (Local)",
                    "dimensions": self.embedding_dimensions,
                    "device": self.device,
                    "offline": True,
                    "note": "Local EmbeddingGemma model"
                }
            elif self._sentence_transformer:
                return {
                    "name": "paraphrase-MiniLM-L6-v2 (Local)",
                    "dimensions": 384,
                    "device": self.device,
                    "offline": True,
                    "note": "Fallback sentence transformer model"
                }
        else:
            return {
                "name": f"EmbeddingGemma (API)",
                "dimensions": self.embedding_dimensions,
                "device": "cloud",
                "offline": False,
                "note": "Hugging Face API"
            }

    # ---------- Tenant-aware embedding APIs ----------

    async def get_embedding_for_tenant(self, text: str, tenant_id: str) -> List[float]:
        """Compute embedding using tenant's configured pipeline or simple model."""
        tenant_cfg_service = get_tenant_config_service()
        tenant_cfg = tenant_cfg_service.get_tenant_config(tenant_id)

        # If pipeline defined, compute layered embedding
        if tenant_cfg.has_pipeline():
            pipeline = tenant_cfg.get_embedding_pipeline()
            components = pipeline.get('components', [])
            combine = pipeline.get('combine', 'concat')
            normalize_all = pipeline.get('normalize', True)
            projector = pipeline.get('projector')

            vectors: List[np.ndarray] = []
            weights: List[float] = []
            for comp in components:
                ctype = comp.get('type', 'sentence_transformer')
                name = comp.get('name')
                weight = float(comp.get('weight', 1.0))
                comp_norm = comp.get('normalize')
                do_norm = normalize_all if comp_norm is None else bool(comp_norm)

                # Produce component embedding
                if ctype == 'sentence_transformer':
                    if not name:
                        raise ValueError('sentence_transformer component requires a name')
                    vec = await self._get_custom_sentence_transformer_embedding(text, name)
                elif ctype == 'embeddinggemma':
                    vec = await self.get_embedding(text, model='embeddinggemma')
                elif ctype == 'openai':
                    vec = await self.get_embedding(text, model='openai')
                elif ctype == 'huggingface':
                    vec = await self.get_embedding(text, model='huggingface')
                else:
                    raise ValueError(f"Unknown component type: {ctype}")

                arr = np.array(vec, dtype=float)
                if do_norm:
                    arr = self._l2_normalize(arr)
                vectors.append(arr)
                weights.append(weight)

            combined = self._combine_vectors(vectors, weights if combine == 'weighted_sum' else None, combine)
            projected = self._project_vector(combined, projector)
            return projected.tolist()

        # Fallback: simple model per existing behavior
        vec = await self.get_embedding(text, tenant_cfg.embedding_model)
        return vec

    async def get_batch_embeddings_for_tenant(self, texts: List[str], tenant_id: str) -> List[List[float]]:
        """Batch variant of tenant-aware embeddings."""
        tenant_cfg_service = get_tenant_config_service()
        tenant_cfg = tenant_cfg_service.get_tenant_config(tenant_id)

        if tenant_cfg.has_pipeline():
            # For simplicity and consistency, compute per text (component-level batch could be optimized later)
            results: List[List[float]] = []
            for t in texts:
                vec = await self.get_embedding_for_tenant(t, tenant_id)
                results.append(vec)
            return results

        # Simple fallback
        embs = await self.get_batch_embeddings(texts, tenant_cfg.embedding_model)
        return embs


# Singleton instance
@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get singleton instance of EmbeddingService"""
    return EmbeddingService()
