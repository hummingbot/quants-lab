"""
Simple database manager for QuantsLab tasks.
Reads config/database.yml and provides shared database client instances.
"""
import yaml
from pathlib import Path
from typing import Optional
import logging

from core.services.mongodb_client import MongoClient
from core.services.timescale_client import TimescaleClient

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages shared database connections for tasks."""
    
    def __init__(self):
        self._config = None
        self._mongodb_client: Optional[MongoClient] = None
        self._timescale_client: Optional[TimescaleClient] = None
        
    def _load_config(self) -> dict:
        """Load database configuration from config/database.yml"""
        if self._config is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "database.yml"
            
            try:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                    logger.info(f"Loaded database config from {config_path}")
            except FileNotFoundError:
                logger.warning(f"Database config not found: {config_path}")
                self._config = {}
            except Exception as e:
                logger.error(f"Error loading database config: {e}")
                self._config = {}
                
        return self._config
    
    async def get_mongodb_client(self) -> Optional[MongoClient]:
        """Get MongoDB client instance."""
        if self._mongodb_client is None:
            config = self._load_config()
            mongo_config = config.get('mongodb', {})
            
            if mongo_config:
                try:
                    self._mongodb_client = MongoClient(
                        uri=mongo_config.get('uri'),
                        database=mongo_config.get('database', 'quants_lab')
                    )
                    await self._mongodb_client.connect()
                    logger.info("MongoDB client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize MongoDB client: {e}")
                    return None
                    
        return self._mongodb_client
    
    async def get_timescale_client(self) -> Optional[TimescaleClient]:
        """Get TimescaleDB client instance.""" 
        if self._timescale_client is None:
            config = self._load_config()
            ts_config = config.get('timescaledb', {})
            
            if ts_config:
                try:
                    self._timescale_client = TimescaleClient(
                        host=ts_config.get('host', 'localhost'),
                        port=ts_config.get('port', 5432),
                        user=ts_config.get('user', 'admin'),
                        password=ts_config.get('password', 'admin'),
                        database=ts_config.get('database', 'timescaledb')
                    )
                    await self._timescale_client.connect()
                    logger.info("TimescaleDB client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize TimescaleDB client: {e}")
                    return None
                    
        return self._timescale_client
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self._mongodb_client:
            await self._mongodb_client.disconnect()
            self._mongodb_client = None
            
        if self._timescale_client:
            await self._timescale_client.close()
            self._timescale_client = None


# Global database manager instance
db_manager = DatabaseManager()