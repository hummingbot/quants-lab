"""
Simple database manager for QuantsLab tasks.
Reads database configuration from environment variables and provides shared database client instances.
"""
import os
from typing import Optional
import logging

from core.services.mongodb_client import MongoClient
# TimescaleClient removed - using parquet files for time-series data

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages shared database connections for tasks."""
    
    def __init__(self):
        self._mongodb_client: Optional[MongoClient] = None

    async def get_mongodb_client(self) -> Optional[MongoClient]:
        """Get MongoDB client instance."""
        if self._mongodb_client is None:
            # Build MongoDB URI from environment variables
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = os.getenv('MONGO_PORT', '27017')
            mongo_user = os.getenv('MONGO_USER', 'admin')
            mongo_password = os.getenv('MONGO_PASSWORD', 'admin')
            mongo_database = os.getenv('MONGO_DATABASE', 'quants_lab')
            
            # Build MongoDB URI with authentication
            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_database}?authSource=admin&retryWrites=true&w=majority"
            
            try:
                self._mongodb_client = MongoClient(
                    uri=mongo_uri,
                    database=mongo_database
                )
                await self._mongodb_client.connect()
                logger.info("MongoDB client initialized from environment variables")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB client: {e}")
                return None
                
        return self._mongodb_client

    async def cleanup(self):
        """Cleanup database connections."""
        if self._mongodb_client:
            await self._mongodb_client.disconnect()
            self._mongodb_client = None


# Global database manager instance
db_manager = DatabaseManager()