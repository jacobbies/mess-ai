"""PostgreSQL database configuration and connection management."""

import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """PostgreSQL database configuration class."""
    
    def __init__(self):
        # PostgreSQL connection parameters
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', '5432'))
        self.database = os.getenv('POSTGRES_DATABASE', 'mess_ai')
        self.username = os.getenv('POSTGRES_USERNAME', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD')
        
        # Connection pool settings
        self.pool_size = int(os.getenv('POSTGRES_POOL_SIZE', '20'))
        self.max_overflow = int(os.getenv('POSTGRES_MAX_OVERFLOW', '0'))
        self.pool_timeout = int(os.getenv('POSTGRES_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('POSTGRES_POOL_RECYCLE', '3600'))
        self.echo = os.getenv('POSTGRES_ECHO', 'false').lower() == 'true'
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration is present."""
        if not self.password:
            raise ValueError(
                "Missing required environment variable: POSTGRES_PASSWORD\n"
                "Please set this in your .env file or environment."
            )
    
    @property
    def database_url(self) -> str:
        """Get the database URL for SQLAlchemy."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_database_url(self) -> str:
        """Get the async database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        return bool(self.password and self.host and self.database)
    
    def get_connection_info(self) -> dict:
        """Get connection information for logging/debugging."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'username': self.username,
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
        }


# Global configuration instance
config = DatabaseConfig()

# Create async engine with optimized settings for RDS
engine = create_async_engine(
    config.async_database_url,
    pool_pre_ping=True,  # Important for AWS RDS
    pool_size=config.pool_size,
    max_overflow=config.max_overflow,
    pool_timeout=config.pool_timeout,
    pool_recycle=config.pool_recycle,  # Recycle connections before timeout
    echo=config.echo,
    future=True
)

# Session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db_session() -> AsyncSession:
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_database_config() -> DatabaseConfig:
    """Get the global database configuration."""
    return config


def check_database_setup() -> tuple[bool, str]:
    """
    Check if the database is properly set up.
    
    Returns:
        tuple: (is_ready, message)
    """
    try:
        if not config.is_configured:
            return False, (
                "Database not configured. Please set environment variables:\n"
                "- POSTGRES_HOST\n"
                "- POSTGRES_PASSWORD (required)\n"
                "- POSTGRES_DATABASE (optional, defaults to mess_ai)\n"
                "- POSTGRES_USERNAME (optional, defaults to postgres)\n"
                "- POSTGRES_PORT (optional, defaults to 5432)"
            )
        
        return True, "Database configuration is valid"
        
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error checking database setup: {e}"


def create_env_template():
    """Create a template .env file for development."""
    template = """# PostgreSQL Configuration
# Copy this to .env and fill in your actual values

# Required: Database connection details
POSTGRES_HOST=your-rds-endpoint.region.rds.amazonaws.com
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DATABASE=mess_ai
POSTGRES_USERNAME=postgres
POSTGRES_PORT=5432

# Optional: Connection pool settings
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=0
POSTGRES_POOL_TIMEOUT=30
POSTGRES_POOL_RECYCLE=3600
POSTGRES_ECHO=false

# For local development, you can use:
# POSTGRES_HOST=localhost
# POSTGRES_PASSWORD=your-local-password
"""
    
    env_file = ".env.template"
    with open(env_file, 'w') as f:
        f.write(template)
    
    return env_file


if __name__ == "__main__":
    # CLI for checking database setup
    is_ready, message = check_database_setup()
    
    print("Database Configuration Check")
    print("=" * 40)
    print(f"Status: {'✅ Ready' if is_ready else '❌ Not Ready'}")
    print(f"Message: {message}")
    
    if is_ready:
        print("\nConnection Info:")
        info = config.get_connection_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nTo get started, create a .env file with your PostgreSQL configuration.")
        template_file = create_env_template()
        print(f"Template created: {template_file}")