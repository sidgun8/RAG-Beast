#!/usr/bin/env python3
"""
Main application entry point for PG Vector Search API
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app

if __name__ == "__main__":
    reload = os.getenv("NODE_ENV") == "development"
    
    print(f"Starting PG Vector Search API on ")
    print(f"API Documentation:")
    print(f"Health Check: ")
    
    uvicorn.run(
        "src.main:app",
        reload=reload,
        log_level="info"
    )
