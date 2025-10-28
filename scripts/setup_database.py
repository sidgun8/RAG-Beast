#!/usr/bin/env python3
"""
Database setup script for PG Vector Search API
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.database_service import DatabaseService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_database():
    """Set up database schema and indexes"""
    db_service = DatabaseService()
    
    try:
        logger.info("Setting up database...")
        
        # Initialize database connection
        await db_service.initialize()
        
        # Set up database schema
        await db_service.setup_database()
        
        logger.info("✅ Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error setting up database: {e}")
        sys.exit(1)
    finally:
        await db_service.close()


if __name__ == "__main__":
    asyncio.run(setup_database())
