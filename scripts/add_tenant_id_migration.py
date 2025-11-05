#!/usr/bin/env python3
"""
Database migration to add tenant_id support
Adds tenant_id column to documents table and creates index
"""
import asyncio
import asyncpg
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_migration():
    """Run the migration to add tenant_id column"""
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'vector_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    print("Connecting to database...")
    conn = await asyncpg.connect(**db_config)
    
    try:
        # Check if column already exists
        check_column = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' 
            AND column_name = 'tenant_id'
        """
        
        result = await conn.fetchval(check_column)
        
        if result:
            print("✓ tenant_id column already exists. Skipping...")
        else:
            print("Adding tenant_id column to documents table...")
            
            # Add tenant_id column with default value
            await conn.execute("""
                ALTER TABLE documents 
                ADD COLUMN tenant_id VARCHAR(255) DEFAULT 'default' NOT NULL
            """)
            
            print("✓ Added tenant_id column")
        
        # Check if index exists
        check_index = """
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'documents' 
            AND indexname = 'idx_documents_tenant_id'
        """
        
        index_result = await conn.fetchval(check_index)
        
        if index_result:
            print("✓ Index on tenant_id already exists. Skipping...")
        else:
            print("Creating index on tenant_id...")
            
            # Create index for performance
            await conn.execute("""
                CREATE INDEX idx_documents_tenant_id 
                ON documents(tenant_id)
            """)
            
            print("✓ Created index idx_documents_tenant_id")
        
        # Verify migration
        print("\nVerifying migration...")
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM documents WHERE tenant_id = 'default'
        """)
        
        print(f"✓ Found {count} documents with tenant_id='default'")
        
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add tenant_id Support")
    print("=" * 60)
    print()
    
    try:
        asyncio.run(run_migration())
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

