"""Database migration utilities for PostgreSQL."""

import asyncio
import logging
from typing import Dict, List
from sqlalchemy import text

from .client import postgres_client
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database schema migrations for PostgreSQL."""
    
    def __init__(self, client=None):
        self.client = client or postgres_client
    
    async def check_connection(self) -> bool:
        """Test database connection."""
        return await self.client.check_connection()
    
    async def create_database_if_not_exists(self, database_name: str = "mess_ai") -> bool:
        """Create database if it doesn't exist (requires superuser privileges)."""
        try:
            async with self.client.get_session() as session:
                # Check if database exists
                result = await session.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                    {"dbname": database_name}
                )
                
                if result.scalar():
                    logger.info(f"Database {database_name} already exists")
                    return True
                
                # Create database (this might require different connection)
                await session.execute(text(f"CREATE DATABASE {database_name}"))
                await session.commit()
                logger.info(f"Created database {database_name}")
                return True
                
        except Exception as e:
            logger.warning(f"Could not create database {database_name}: {e}")
            logger.info("Database might already exist or require different privileges")
            return True  # Assume it exists
    
    async def create_extensions(self) -> bool:
        """Create required PostgreSQL extensions."""
        extensions = ["uuid-ossp"]
        
        try:
            async with self.client.get_session() as session:
                for ext in extensions:
                    await session.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                
                await session.commit()
                logger.info("PostgreSQL extensions created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create extensions: {e}")
            return False
    
    async def create_tables(self) -> bool:
        """Create all database tables."""
        try:
            # Create extensions first
            await self.create_extensions()
            
            # Create all tables
            success = await self.client.create_tables()
            
            if success:
                logger.info("Database tables created successfully")
            else:
                logger.error("Failed to create database tables")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    async def drop_tables(self) -> bool:
        """Drop all database tables (for development/testing)."""
        try:
            success = await self.client.drop_tables()
            
            if success:
                logger.info("Database tables dropped successfully")
            else:
                logger.error("Failed to drop database tables")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            return False
    
    async def check_tables_exist(self) -> Dict[str, bool]:
        """Check which tables exist in the database."""
        try:
            table_info = await self.client.get_table_info()
            existing_tables = {table['name'] for table in table_info}
            
            required_tables = [
                'recordings', 'embeddings', 'embedding_chunks', 'similarity_scores',
                'composers', 'works', 'tags', 'recording_tags'
            ]
            
            return {table: table in existing_tables for table in required_tables}
            
        except Exception as e:
            logger.error(f"Failed to check tables: {e}")
            return {}
    
    async def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        return await self.client.get_table_counts()
    
    async def create_indexes(self) -> bool:
        """Create additional performance indexes."""
        indexes = [
            # Recording indexes
            "CREATE INDEX IF NOT EXISTS idx_recordings_dataset ON recordings(dataset_source)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_metadata ON recordings USING GIN(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_title ON recordings(title)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC)",
            
            # Embedding indexes  
            "CREATE INDEX IF NOT EXISTS idx_embeddings_recording ON embeddings(recording_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name, model_version)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_feature_type ON embeddings(feature_type)",
            
            # Embedding chunk indexes
            "CREATE INDEX IF NOT EXISTS idx_embedding_chunks_embedding ON embedding_chunks(embedding_id, chunk_index)",
            
            # Similarity score indexes
            "CREATE INDEX IF NOT EXISTS idx_similarity_scores_a ON similarity_scores(recording_id_a)",
            "CREATE INDEX IF NOT EXISTS idx_similarity_scores_b ON similarity_scores(recording_id_b)", 
            "CREATE INDEX IF NOT EXISTS idx_similarity_scores_model ON similarity_scores(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_similarity_scores_score ON similarity_scores(similarity_score DESC)",
            
            # Future indexes for extended models
            "CREATE INDEX IF NOT EXISTS idx_works_composer ON works(composer_id)",
            "CREATE INDEX IF NOT EXISTS idx_recording_tags_recording ON recording_tags(recording_id)",
            "CREATE INDEX IF NOT EXISTS idx_recording_tags_tag ON recording_tags(tag_id)",
        ]
        
        try:
            async with self.client.get_session() as session:
                for index_sql in indexes:
                    await session.execute(text(index_sql))
                
                await session.commit()
                logger.info("Database indexes created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    async def create_triggers(self) -> bool:
        """Create database triggers for automatic timestamp updates."""
        trigger_sql = """
        -- Function to update updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        -- Trigger for recordings table
        DROP TRIGGER IF EXISTS update_recordings_updated_at ON recordings;
        CREATE TRIGGER update_recordings_updated_at 
            BEFORE UPDATE ON recordings
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
        -- Trigger for composers table
        DROP TRIGGER IF EXISTS update_composers_updated_at ON composers;
        CREATE TRIGGER update_composers_updated_at 
            BEFORE UPDATE ON composers  
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
        -- Trigger for works table
        DROP TRIGGER IF EXISTS update_works_updated_at ON works;
        CREATE TRIGGER update_works_updated_at 
            BEFORE UPDATE ON works
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        
        try:
            async with self.client.get_session() as session:
                for statement in trigger_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        await session.execute(text(statement))
                
                await session.commit()
                logger.info("Database triggers created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create triggers: {e}")
            return False
    
    async def migrate(self, force_recreate: bool = False) -> bool:
        """Run the full migration process."""
        logger.info("Starting PostgreSQL database migration...")
        
        # Check connection
        if not await self.check_connection():
            logger.error("Cannot connect to database. Check your configuration.")
            return False
        
        # Drop tables if force recreate
        if force_recreate:
            logger.info("Force recreating tables...")
            await self.drop_tables()
        
        # Check existing tables
        table_status = await self.check_tables_exist()
        logger.info(f"Table status: {table_status}")
        
        # Create tables if needed
        core_tables = ['recordings', 'embeddings', 'embedding_chunks', 'similarity_scores']
        core_missing = not all(table_status.get(table, False) for table in core_tables)
        
        if core_missing or force_recreate:
            logger.info("Creating database schema...")
            if not await self.create_tables():
                return False
            
            logger.info("Creating indexes...")
            if not await self.create_indexes():
                logger.warning("Failed to create indexes, but continuing...")
            
            logger.info("Creating triggers...")
            if not await self.create_triggers():
                logger.warning("Failed to create triggers, but continuing...")
        else:
            logger.info("All core tables already exist")
        
        # Show final status
        counts = await self.get_table_counts()
        logger.info(f"Migration complete. Table counts: {counts}")
        
        return True
    
    async def reset_database(self) -> bool:
        """Reset the entire database (drop and recreate everything)."""
        logger.warning("Resetting entire database - ALL DATA WILL BE LOST!")
        
        success = await self.migrate(force_recreate=True)
        
        if success:
            logger.info("Database reset completed successfully")
        else:
            logger.error("Database reset failed")
        
        return success


async def run_migration(force_recreate: bool = False):
    """Standalone migration runner."""
    logging.basicConfig(level=logging.INFO)
    
    migrator = DatabaseMigrator()
    success = await migrator.migrate(force_recreate=force_recreate)
    
    if success:
        print("✅ Database migration completed successfully")
    else:
        print("❌ Database migration failed")
        exit(1)


async def reset_database():
    """Standalone database reset runner."""
    logging.basicConfig(level=logging.INFO)
    
    migrator = DatabaseMigrator()
    success = await migrator.reset_database()
    
    if success:
        print("✅ Database reset completed successfully")
    else:
        print("❌ Database reset failed")
        exit(1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        asyncio.run(reset_database())
    elif len(sys.argv) > 1 and sys.argv[1] == "force":
        asyncio.run(run_migration(force_recreate=True))
    else:
        asyncio.run(run_migration())