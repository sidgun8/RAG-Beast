#!/usr/bin/env python3
"""
Start the PG Vector Search API server
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
load_dotenv()


def start_server():
    """Start the FastAPI server"""
    reload = os.getenv("NODE_ENV") == "development"
    
    print(f"🚀 Starting PG Vector Search API on")
    print(f"📚 API Documentation")
    print(f"🏥 Health Check: ")
    
    uvicorn.run(
        "src.main:app",
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
