#!/usr/bin/env python3
"""
Configuration validation script for tenant configurations
Validates tenants.yaml syntax and settings
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.tenant_config_service import get_tenant_config_service


def validate_config():
    """Validate tenant configuration"""
    print("=" * 60)
    print("Tenant Configuration Validator")
    print("=" * 60)
    
    try:
        # Load config service
        print("\n1. Loading configuration...")
        config_service = get_tenant_config_service()
        
        # Check if config file exists
        if not config_service._config_path:
            print("⚠️  No configuration file found")
            print("   Using default configuration from environment variables")
            return
        
        print(f"✓ Configuration loaded from: {config_service._config_path}")
        
        # List all tenants
        print("\n2. Validating tenants...")
        tenants = config_service.list_tenants()
        
        if not tenants:
            print("❌ No tenants found in configuration!")
            return False
        
        print(f"✓ Found {len(tenants)} tenant(s): {', '.join(tenants)}")
        
        # Validate each tenant
        print("\n3. Checking tenant configurations...")
        all_valid = True
        
        for tenant_id in tenants:
            print(f"\n   Tenant: {tenant_id}")
            tenant_config = config_service.get_tenant_config(tenant_id)
            
            # Validate embedding configuration
            valid_models = ['embeddinggemma', 'openai', 'huggingface']
            if getattr(tenant_config, 'embedding_pipeline', None):
                print("   ✓ Using embedding_pipeline (overrides embedding_model)")
                # Validate pipeline schema
                try:
                    pipeline = tenant_config.get_embedding_pipeline()
                    components = pipeline.get('components', [])
                    combine = pipeline.get('combine', 'concat')
                    projector = pipeline.get('projector', {})
                    normalize = pipeline.get('normalize', True)

                    if combine not in ['concat', 'weighted_sum']:
                        raise ValueError(f"combine must be 'concat' or 'weighted_sum', got {combine}")
                    if not isinstance(normalize, bool):
                        raise ValueError("normalize must be boolean")
                    if not components:
                        raise ValueError("components list must not be empty")

                    allowed_types = {'sentence_transformer', 'embeddinggemma', 'openai', 'huggingface'}
                    for idx, comp in enumerate(components):
                        ctype = comp.get('type')
                        if ctype not in allowed_types:
                            raise ValueError(f"component[{idx}].type invalid: {ctype}")
                        if ctype == 'sentence_transformer' and not comp.get('name'):
                            raise ValueError(f"component[{idx}].name required for sentence_transformer")
                        w = float(comp.get('weight', 1.0))
                        if w <= 0:
                            raise ValueError(f"component[{idx}].weight must be > 0")
                        if comp.get('normalize') not in [None, True, False]:
                            raise ValueError(f"component[{idx}].normalize must be boolean or omitted")

                    ptype = projector.get('type', 'none')
                    if ptype not in ['none', 'matryoshka', 'pca']:
                        raise ValueError(f"projector.type invalid: {ptype}")
                    if ptype != 'none':
                        td = projector.get('target_dims')
                        if not isinstance(td, int) or td <= 0:
                            raise ValueError("projector.target_dims must be positive integer when projector is used")

                    print("   ✓ embedding_pipeline schema valid")
                except Exception as e:
                    print(f"   ❌ embedding_pipeline invalid: {e}")
                    all_valid = False
            else:
                if tenant_config.embedding_model not in valid_models:
                    print(f"   ❌ Invalid embedding_model: {tenant_config.embedding_model}")
                    print(f"      Must be one of: {', '.join(valid_models)}")
                    all_valid = False
                else:
                    print(f"   ✓ Embedding model: {tenant_config.embedding_model}")
            
            # Validate dimensions
            if not (128 <= tenant_config.embedding_dimensions <= 1536):
                print(f"   ❌ Invalid embedding_dimensions: {tenant_config.embedding_dimensions}")
                print(f"      Must be between 128 and 1536")
                all_valid = False
            else:
                print(f"   ✓ Embedding dimensions: {tenant_config.embedding_dimensions}")
            
            # Validate search type
            valid_search_types = ['semantic', 'text', 'hybrid']
            if tenant_config.search_type not in valid_search_types:
                print(f"   ❌ Invalid search_type: {tenant_config.search_type}")
                print(f"      Must be one of: {', '.join(valid_search_types)}")
                all_valid = False
            else:
                print(f"   ✓ Search type: {tenant_config.search_type}")
            
            # Check API keys if using cloud models
            if not tenant_config.use_local:
                if tenant_config.embedding_model == 'openai' and not tenant_config.openai_api_key:
                    print(f"   ⚠️  WARNING: OpenAI model selected but OPENAI_API_KEY not set")
                elif tenant_config.embedding_model == 'huggingface' and not tenant_config.huggingface_api_key:
                    print(f"   ⚠️  WARNING: HuggingFace model selected but HUGGINGFACE_API_KEY not set")
            
            # Validate hybrid search params
            if tenant_config.search_type == 'hybrid':
                total_weight = tenant_config.semantic_weight + tenant_config.text_weight
                if abs(total_weight - 1.0) > 0.01:
                    print(f"   ❌ Hybrid search weights don't sum to 1.0: {total_weight}")
                    all_valid = False
                else:
                    print(f"   ✓ Hybrid weights: semantic={tenant_config.semantic_weight}, text={tenant_config.text_weight}")
        
        # Summary
        print("\n" + "=" * 60)
        if all_valid:
            print("✅ All tenant configurations are valid!")
            print("=" * 60)
            return True
        else:
            print("❌ Some tenant configurations have errors")
            print("=" * 60)
            return False
    
    except Exception as e:
        print(f"\n❌ Error validating configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_tenant_summary():
    """Print a summary of all tenant configurations"""
    try:
        config_service = get_tenant_config_service()
        tenants = config_service.list_tenants()
        
        print("\n" + "=" * 60)
        print("Tenant Configuration Summary")
        print("=" * 60)
        
        for tenant_id in tenants:
            config = config_service.get_tenant_config(tenant_id)
            print(f"\n{tenant_id}:")
            print(f"  Embedding: {config.embedding_model} ({config.embedding_dimensions}D)")
            print(f"  Search: {config.search_type}")
            print(f"  Mode: {'Local' if config.use_local else 'API'}")
            if config.search_type == 'hybrid':
                print(f"  Weights: {config.semantic_weight}/{config.text_weight}")
    
    except Exception as e:
        print(f"Error printing summary: {e}")


if __name__ == "__main__":
    try:
        # Validate configuration
        is_valid = validate_config()
        
        # Print summary if valid
        if is_valid:
            print_tenant_summary()
        
        # Exit with appropriate code
        sys.exit(0 if is_valid else 1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

