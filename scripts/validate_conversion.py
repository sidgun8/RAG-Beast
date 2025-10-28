#!/usr/bin/env python3
"""
Validation script to test the Python conversion
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    modules_to_test = [
        'src.services.embedding_service',
        'src.services.document_service', 
        'src.services.database_service',
        'src.services.text_processor',
        'src.services.file_processor',
        'src.routes.search',
        'src.routes.documents',
        'src.routes.ingest',
        'src.main'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                failed_imports.append(module_name)
                print(f"X {module_name} - Module not found")
            else:
                print(f"OK {module_name} - OK")
        except Exception as e:
            failed_imports.append(module_name)
            print(f"❌ {module_name} - Error: {e}")
    
    return len(failed_imports) == 0


def test_file_structure():
    """Test that all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'src/__init__.py',
        'src/main.py',
        'src/services/__init__.py',
        'src/services/embedding_service.py',
        'src/services/document_service.py',
        'src/services/database_service.py',
        'src/services/text_processor.py',
        'src/services/file_processor.py',
        'src/routes/__init__.py',
        'src/routes/search.py',
        'src/routes/documents.py',
        'src/routes/ingest.py',
        'scripts/setup_database.py',
        'scripts/ingest_data.py',
        'scripts/test_functionality.py',
        'scripts/start_server.py',
        'scripts/validate_conversion.py',
        'app.py',
        'requirements.txt',
        'env.example',
        'README_PYTHON.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - OK")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - Missing")
    
    return len(missing_files) == 0


def test_requirements():
    """Test that requirements.txt has necessary packages"""
    print("\n🔍 Testing requirements.txt...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt - Missing")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'psycopg2-binary',
        'sqlalchemy',
        'torch',
        'transformers',
        'sentence-transformers',
        'numpy',
        'requests',
        'python-dotenv',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package in content:
            print(f"✅ {package} - Found")
        else:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    return len(missing_packages) == 0


def test_environment_config():
    """Test environment configuration"""
    print("\n🔍 Testing environment configuration...")
    
    if not os.path.exists('env.example'):
        print("❌ env.example - Missing")
        return False
    
    with open('env.example', 'r') as f:
        content = f.read()
    
    required_vars = [
        'DB_HOST',
        'DB_PORT', 
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD',
        'PORT',
        'EMBEDDING_MODEL',
        'EMBEDDING_DIMENSIONS'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if var in content:
            print(f"✅ {var} - Found")
        else:
            missing_vars.append(var)
            print(f"❌ {var} - Missing")
    
    return len(missing_vars) == 0


def main():
    """Run all validation tests"""
    print("Starting Python conversion validation...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Environment Config", test_environment_config),
        ("Module Imports", test_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print(f"\n--- Validation Results ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up database: python scripts/setup_database.py")
        print("3. Test functionality: python scripts/test_functionality.py")
        print("4. Start server: python app.py")
        return True
    else:
        print("❌ Some validation tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
