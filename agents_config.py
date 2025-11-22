"""Configuration for ADK agents and services."""

import os
from services import PostgresMemoryService

# Database configuration
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/agent_state")
ASYNC_DB_URL = os.getenv("ASYNC_DATABASE_URL", "postgresql+asyncpg://postgres@localhost:5432/agent_state")

# Memory service factory
def create_memory_service():
    """Create and return a PostgresMemoryService instance."""
    return PostgresMemoryService(db_url=ASYNC_DB_URL)
