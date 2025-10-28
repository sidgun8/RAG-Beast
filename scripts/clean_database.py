#!/usr/bin/env python3
"""
Database cleaning script - Delete all data from PostgreSQL database
"""
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.database_service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def clean_database():
    """Delete all data from documents table"""
    db_service = DatabaseService()
    
    try:
        logger.info("🧹 Starting database cleanup...")
        
        # Initialize database connection
        await db_service.initialize()
        
        async with db_service._pool.acquire() as conn:
            # Get count before deletion
            count = await conn.fetchval('SELECT COUNT(*) FROM documents;')
            logger.info(f"Found {count} documents in database")
            
            if count > 0:
                # Delete all documents
                await conn.execute('DELETE FROM documents;')
                logger.info(f"✓ Deleted all {count} documents from database")
            else:
                logger.info("Database is already empty")
            
            logger.info("✅ Database cleaned successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    finally:
        await db_service.close()


if __name__ == "__main__":
    print("⚠️  This will DELETE ALL DATA from your database!")
    response = input("Are you sure? Type 'DELETE ALL' to confirm: ")
    
    if response == "DELETE ALL":
        asyncio.run(clean_database())
    else:
        logger.info("Cancelled")

