import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

from core.services.mongodb_client import MongoClient
from geckoterminal_py import GeckoTerminalAsyncClient
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
load_dotenv()


class PoolsScreenerTask(BaseTask):
    """Pool screening task using v2.0 BaseTask interface."""
    
    def __init__(self, config):
        super().__init__(config)
        self.gt = None
        self.mongo_client = None
        
        # Configuration with defaults
        self.network = self.config.config.get("network", "solana")
        self.quote_asset = self.config.config.get("quote_asset", "SOL")
        self.min_pool_age_days = self.config.config.get("min_pool_age_days", 2)
        self.min_fdv = self.config.config.get("min_fdv", 70_000)
        self.max_fdv = self.config.config.get("max_fdv", 5_000_000)
        self.min_volume_24h = self.config.config.get("min_volume_24h", 150_000)
        self.min_liquidity = self.config.config.get("min_liquidity", 50_000)
        self.min_transactions_24h = self.config.config.get("min_transactions_24h", 300)

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            # Check required environment variables/config
            mongo_config = self.config.config.get("mongo_config", {})
            if not mongo_config.get("uri"):
                logging.error("MongoDB URI not configured")
                return False
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            # Initialize GeckoTerminal client
            self.gt = GeckoTerminalAsyncClient()
            
            # Initialize MongoDB client with configuration
            mongo_config = self.config.config.get("mongo_config", {})
            self.mongo_client = MongoClient(
                uri=mongo_config.get("uri", ""),
                database=mongo_config.get("db", "strategies"),
            )
            
            # Connect to MongoDB
            await self.mongo_client.connect()
            
            logging.info(f"Setup completed for {context.task_name}")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            if self.mongo_client:
                await self.mongo_client.disconnect()
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    def clean_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich pools dataframe with calculated metrics"""
        try:
            pools["fdv_usd"] = pd.to_numeric(pools["fdv_usd"])
            pools["volume_usd_h24"] = pd.to_numeric(pools["volume_usd_h24"])
            pools["reserve_in_usd"] = pd.to_numeric(pools["reserve_in_usd"])
            pools["pool_created_at"] = pd.to_datetime(pools["pool_created_at"]).dt.tz_localize(None)
            pools["base"] = pools["name"].apply(lambda x: x.split("/")[0].strip())
            pools["quote"] = pools["name"].apply(lambda x: x.split("/")[1].strip())
            
            # Calculate ratios with safe division
            pools["volume_liquidity_ratio"] = pools.apply(
                lambda x: x["volume_usd_h24"] / x["reserve_in_usd"] if x["reserve_in_usd"] != 0 else 0, 
                axis=1
            )
            pools["fdv_liquidity_ratio"] = pools.apply(
                lambda x: x["fdv_usd"] / x["reserve_in_usd"] if x["reserve_in_usd"] != 0 else 0, 
                axis=1
            )
            pools["fdv_volume_ratio"] = pools.apply(
                lambda x: x["fdv_usd"] / x["volume_usd_h24"] if x["volume_usd_h24"] != 0 else 0, 
                axis=1
            )
            
            pools["transactions_h24_buys"] = pd.to_numeric(pools["transactions_h24_buys"])
            pools["transactions_h24_sells"] = pd.to_numeric(pools["transactions_h24_sells"])
            pools["price_change_percentage_h1"] = pd.to_numeric(pools["price_change_percentage_h1"])
            pools["price_change_percentage_h24"] = pd.to_numeric(pools["price_change_percentage_h24"])
            
            # Filter by quote asset
            pools = pools[pools['quote'] == self.quote_asset]
                
            return pools
        except Exception as e:
            logging.error(f"Error cleaning pools data: {str(e)}")
            return pd.DataFrame()

    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        """Filter pools based on configured criteria"""
        try:
            min_date = datetime.now() - pd.Timedelta(days=self.min_pool_age_days)
            
            filtered_pools = pools[
                (pools["pool_created_at"] > min_date) &
                (pools["fdv_usd"] >= self.min_fdv) & 
                (pools["fdv_usd"] <= self.max_fdv) &
                (pools["volume_usd_h24"] >= self.min_volume_24h) &
                (pools["reserve_in_usd"] >= self.min_liquidity) &
                (pools["transactions_h24_buys"] >= self.min_transactions_24h) & 
                (pools["transactions_h24_sells"] >= self.min_transactions_24h)
            ]
            
            return filtered_pools
        except Exception as e:
            logging.error(f"Error filtering pools: {str(e)}")
            return pd.DataFrame()

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        try:
            # Fetch data
            top_pools = await self.gt.get_top_pools_by_network(self.network)
            new_pools = await self.gt.get_new_pools_by_network(self.network)
            
            # Clean and filter data
            cleaned_top = self.clean_pools(top_pools.copy())
            cleaned_new = self.clean_pools(new_pools.copy())
            
            filtered_top = self.filter_pools(cleaned_top.copy())
            filtered_new = self.filter_pools(cleaned_new.copy())
            
            # Create document to store
            document = {
                'timestamp': datetime.now(timezone.utc),
                'execution_id': context.execution_id,
                'trending_pools': cleaned_top.to_dict('records') if not cleaned_top.empty else [],
                'filtered_trending_pools': filtered_top.to_dict('records') if not filtered_top.empty else [],
                'new_pools': cleaned_new.to_dict('records') if not cleaned_new.empty else [],
                'filtered_new_pools': filtered_new.to_dict('records') if not filtered_new.empty else []
            }
            
            # Store data using MongoDBClient
            await self.mongo_client.insert_documents(collection_name="pools", documents=[document])
            
            # Prepare result
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "network": self.network,
                "quote_asset": self.quote_asset,
                "stats": {
                    "top_pools_total": len(cleaned_top),
                    "top_pools_filtered": len(filtered_top),
                    "new_pools_total": len(cleaned_new),
                    "new_pools_filtered": len(filtered_new),
                }
            }
            
            logging.info(f"Screening completed for execution {context.execution_id}")
            logging.info(f"Top pools: {len(cleaned_top)} (filtered: {len(filtered_top)})")
            logging.info(f"New pools: {len(cleaned_new)} (filtered: {len(filtered_new)})")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing market screener task: {str(e)}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ PoolsScreenerTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Top pools: {stats.get('top_pools_filtered', 0)}/{stats.get('top_pools_total', 0)}")
        logging.info(f"  - New pools: {stats.get('new_pools_filtered', 0)}/{stats.get('new_pools_total', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ PoolsScreenerTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 PoolsScreenerTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Build MongoDB URI from environment
    mongo_uri = (
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
        f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'admin')}@"
        f"{os.getenv('MONGO_HOST', 'localhost')}:"
        f"{os.getenv('MONGO_PORT', '27017')}/"
    )
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="pools_screener_test",
        enabled=True,
        task_class="tasks.data_collection.pools_screener.PoolsScreenerTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "mongo_config": {
                "uri": mongo_uri,
                "db": "strategies"
            },
            "network": "solana",
            "quote_asset": "SOL",
            "min_pool_age_days": 2,
            "min_fdv": 70_000,
            "max_fdv": 5_000_000,
            "min_volume_24h": 150_000,
            "min_liquidity": 50_000,
            "min_transactions_24h": 300
        }
    )
    
    # Create and run task
    task = PoolsScreenerTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        print(f"Result: {result.result_data}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())