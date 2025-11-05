"""
Tenant Configuration Service
Manages multi-tenant configurations for RAG system
"""
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


class TenantConfig:
    """Configuration for a single tenant"""
    
    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        self.tenant_id = tenant_id
        self._config = config
        
        # Embedding configuration
        self.embedding_model = config.get('embedding_model', 'embeddinggemma')
        self.embedding_dimensions = config.get('embedding_dimensions', 768)
        self.use_local = config.get('use_local', True)
        self.device = config.get('device', 'cpu')
        # Optional layered embedding pipeline (overrides simple embedding_model when present)
        self.embedding_pipeline = config.get('embedding_pipeline') or None
        
        # API Keys (with environment variable interpolation)
        self.openai_api_key = self._resolve_env_var(config.get('openai_api_key', ''))
        self.huggingface_api_key = self._resolve_env_var(config.get('huggingface_api_key', ''))
        
        # Search configuration
        self.search_type = config.get('search_type', 'semantic')
        self.default_similarity_threshold = config.get('default_similarity_threshold', 0.5)
        self.default_limit = config.get('default_limit', 5)
        
        # Hybrid search configuration
        self.semantic_weight = config.get('semantic_weight', 0.7)
        self.text_weight = config.get('text_weight', 0.3)
        self.rrf_k = config.get('rrf_k', 60)
        
        # Chunking configuration
        self.chunking_strategy = config.get('chunking_strategy', 'sliding_window')
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 100)
        self.delimiters = config.get('delimiters', ["\n\n", "\n", ".", "?", "!"])
        self.semantic_similarity_threshold = config.get('semantic_similarity_threshold', 0.7)
        self.recursive_separators = config.get('recursive_separators', ["\n\n", "\n", " ", ""])
        
        # LLM configuration for RAG
        self.llm_model_id = config.get('llm_model_id', 'us.meta.llama4-scout-17b-instruct-v1:0')
        self.llm_max_tokens = config.get('llm_max_tokens', 1000)
        self.llm_temperature = config.get('llm_temperature', 0.7)
    
    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable references like ${VAR_NAME}"""
        if not value:
            return ''
        
        # Pattern to match ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        return re.sub(pattern, replacer, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'tenant_id': self.tenant_id,
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions,
            'use_local': self.use_local,
            'device': self.device,
            'embedding_pipeline': self.embedding_pipeline,
            'search_type': self.search_type,
            'default_similarity_threshold': self.default_similarity_threshold,
            'default_limit': self.default_limit,
            'semantic_weight': self.semantic_weight,
            'text_weight': self.text_weight,
            'rrf_k': self.rrf_k,
            'chunking_strategy': self.chunking_strategy,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'llm_model_id': self.llm_model_id,
            'llm_max_tokens': self.llm_max_tokens,
            'llm_temperature': self.llm_temperature,
            'has_openai_key': bool(self.openai_api_key),
            'has_huggingface_key': bool(self.huggingface_api_key)
        }

    # ---- Embedding pipeline helpers ----
    def has_pipeline(self) -> bool:
        """Whether this tenant defines an embedding pipeline"""
        return bool(self.embedding_pipeline and isinstance(self.embedding_pipeline, dict))

    def get_embedding_pipeline(self) -> Dict[str, Any]:
        """Return a normalized embedding pipeline configuration with sensible defaults.

        Structure:
        {
          'components': [
            { 'type': 'sentence_transformer'|'embeddinggemma'|'openai'|'huggingface',
              'name': str|None,
              'weight': float|None,
              'normalize': bool|None
            }, ...
          ],
          'combine': 'concat'|'weighted_sum',
          'normalize': bool,  # normalize each component before combine
          'projector': { 'type': 'matryoshka'|'pca'|'none', 'target_dims': int|None }
        }
        """
        if not self.has_pipeline():
            return {}

        pipeline = dict(self.embedding_pipeline or {})

        # Defaults
        combine = pipeline.get('combine', 'concat')
        normalize = pipeline.get('normalize', True)
        components = pipeline.get('components', []) or []
        projector = pipeline.get('projector', {}) or {}

        # Normalize components
        norm_components: List[Dict[str, Any]] = []
        for comp in components:
            c = dict(comp or {})
            c.setdefault('type', 'sentence_transformer')
            c.setdefault('name', None)
            c.setdefault('weight', 1.0)
            c.setdefault('normalize', None)  # fallback to pipeline-level normalize if None
            norm_components.append(c)

        # Normalize projector
        proj = {
            'type': projector.get('type', 'none'),
            'target_dims': projector.get('target_dims')
        }

        return {
            'components': norm_components,
            'combine': combine,
            'normalize': normalize,
            'projector': proj
        }


class TenantConfigService:
    """Service to manage tenant configurations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TenantConfigService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._configs: Dict[str, TenantConfig] = {}
        self._config_path = None
        self._initialized = True
        
        # Try to load config
        self._load_config()
    
    def _load_config(self):
        """Load tenant configurations from YAML file"""
        # Try different possible locations
        possible_paths = [
            Path('config/tenants.yaml'),
            Path('config/tenants.yml'),
            Path('../config/tenants.yaml'),
            Path(__file__).parent.parent.parent / 'config' / 'tenants.yaml',
        ]
        
        for path in possible_paths:
            if path.exists():
                self._config_path = path
                break
        
        if not self._config_path:
            logger.warning("No tenant configuration file found. Using default configuration.")
            self._create_default_config()
            return
        
        try:
            with open(self._config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data or 'tenants' not in data:
                logger.warning("Invalid tenant configuration format. Using default.")
                self._create_default_config()
                return
            
            # Load all tenant configs
            for tenant_id, config in data['tenants'].items():
                self._configs[tenant_id] = TenantConfig(tenant_id, config)
            
            logger.info(f"Loaded {len(self._configs)} tenant configurations from {self._config_path}")
            
            # Ensure default tenant exists
            if 'default' not in self._configs:
                logger.warning("No 'default' tenant found. Creating one.")
                self._create_default_config()
        
        except Exception as e:
            logger.error(f"Error loading tenant configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a default tenant configuration"""
        default_config = {
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'embeddinggemma'),
            'embedding_dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '768')),
            'use_local': os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true',
            'device': os.getenv('EMBEDDING_DEVICE', 'cpu'),
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY', ''),
            'search_type': 'semantic',
            'default_similarity_threshold': 0.5,
            'default_limit': 5,
            'semantic_weight': 0.7,
            'text_weight': 0.3,
            'rrf_k': 60,
            'chunking_strategy': 'sliding_window',
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'delimiters': ["\n\n", "\n", ".", "?", "!"],
            'semantic_similarity_threshold': 0.7,
            'recursive_separators': ["\n\n", "\n", " ", ""],
            'llm_model_id': 'us.meta.llama4-scout-17b-instruct-v1:0',
            'llm_max_tokens': 1000,
            'llm_temperature': 0.7
        }
        
        self._configs['default'] = TenantConfig('default', default_config)
        logger.info("Created default tenant configuration from environment variables")
    
    def get_tenant_config(self, tenant_id: str = 'default') -> TenantConfig:
        """Get configuration for a specific tenant"""
        if tenant_id not in self._configs:
            logger.warning(f"Tenant '{tenant_id}' not found. Using default configuration.")
            tenant_id = 'default'
        
        return self._configs[tenant_id]
    
    def get_embedding_config(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get embedding configuration for a tenant"""
        config = self.get_tenant_config(tenant_id)
        return {
            'embedding_model': config.embedding_model,
            'embedding_dimensions': config.embedding_dimensions,
            'use_local': config.use_local,
            'device': config.device,
            'openai_api_key': config.openai_api_key,
            'huggingface_api_key': config.huggingface_api_key
        }
    
    def get_search_config(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get search configuration for a tenant"""
        config = self.get_tenant_config(tenant_id)
        return {
            'search_type': config.search_type,
            'default_similarity_threshold': config.default_similarity_threshold,
            'default_limit': config.default_limit,
            'semantic_weight': config.semantic_weight,
            'text_weight': config.text_weight,
            'rrf_k': config.rrf_k
        }
    
    def get_chunking_config(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get chunking configuration for a tenant"""
        config = self.get_tenant_config(tenant_id)
        return {
            'chunking_strategy': config.chunking_strategy,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'delimiters': config.delimiters,
            'semantic_similarity_threshold': config.semantic_similarity_threshold,
            'recursive_separators': config.recursive_separators
        }
    
    def get_llm_config(self, tenant_id: str = 'default') -> Dict[str, Any]:
        """Get LLM configuration for a tenant"""
        config = self.get_tenant_config(tenant_id)
        return {
            'llm_model_id': config.llm_model_id,
            'llm_max_tokens': config.llm_max_tokens,
            'llm_temperature': config.llm_temperature
        }
    
    def list_tenants(self) -> List[str]:
        """List all configured tenant IDs"""
        return list(self._configs.keys())
    
    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant configuration exists"""
        return tenant_id in self._configs
    
    def reload_config(self):
        """Reload configuration from file"""
        self._configs.clear()
        self._load_config()
        logger.info("Tenant configuration reloaded")


# Singleton instance
_tenant_config_service = None

def get_tenant_config_service() -> TenantConfigService:
    """Get singleton instance of TenantConfigService"""
    global _tenant_config_service
    if _tenant_config_service is None:
        _tenant_config_service = TenantConfigService()
    return _tenant_config_service

