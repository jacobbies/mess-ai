"""PostgreSQL database client using SQLAlchemy."""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from contextlib import asynccontextmanager

from .config import AsyncSessionLocal, engine
from .models import Base

logger = logging.getLogger(__name__)


class PostgreSQLClient:
    """Client for PostgreSQL database operations."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = AsyncSessionLocal
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session context manager."""
        async with self.SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def check_connection(self) -> bool:
        """Test database connection."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def create_tables(self) -> bool:
        """Create all tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    async def drop_tables(self) -> bool:
        """Drop all tables (for testing/development)."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            return False
    
    async def get_table_info(self) -> List[Dict[str, Any]]:
        """Get information about existing tables."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("""
                    SELECT 
                        table_name,
                        table_schema
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """))
                
                return [{'name': row.table_name, 'schema': row.table_schema} for row in result]
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return []
    
    async def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        tables = ['recordings', 'embeddings', 'embedding_chunks', 'similarity_scores']
        counts = {}
        
        async with self.get_session() as session:
            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    counts[table] = result.scalar() or 0
                except Exception as e:
                    logger.debug(f"Table {table} might not exist: {e}")
                    counts[table] = 0
        
        return counts
    
    async def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL and return results."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params or {})
                
                # Handle different result types
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    # For INSERT/UPDATE/DELETE operations
                    await session.commit()
                    return [{'affected_rows': result.rowcount}]
                    
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            raise


# Global client instance
postgres_client = PostgreSQLClient()


# Base repository class for PostgreSQL
class BaseRepository:
    """Base repository class using SQLAlchemy."""
    
    def __init__(self, client: PostgreSQLClient = None):
        self.client = client or postgres_client
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.client.get_session() as session:
            yield session