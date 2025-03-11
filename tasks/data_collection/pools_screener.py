import asyncio
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

from core.services.mongodb_client import MongoClient
from geckoterminal_py import GeckoTerminalAsyncClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
load_dotenv()


class PoolsScreenerTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.name = "market_screener"
        self.gt = GeckoTerminalAsyncClient()
        
        # Initialize MongoDB client with Docker configuration
        self.mongo_client = MongoClient(
            uri=self.config.get("mongo_uri", ""),
            database=self.config.get("database", "strategies"),
        )
        # Configuration
        self.network = self.config.get("network", "solana")
        self.quote_asset = self.config.get("quote_asset", "SOL")
        self.min_pool_age_days = self.config.get("min_pool_age_days", 2)
        self.min_fdv = self.config.get("min_fdv", 70_000)
        self.max_fdv = self.config.get("max_fdv", 5_000_000)
        self.min_volume_24h = self.config.get("min_volume_24h", 150_000)
        self.min_liquidity = self.config.get("min_liquidity", 50_000)
        self.min_transactions_24h = self.config.get("min_transactions_24h", 300)

    async def pre_execute(self) -> None:
        """Pre-execution setup"""
        await self.mongo_client.connect()

    async def post_execute(self) -> None:
        """Post-execution cleanup"""
        await self.mongo_client.disconnect()

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

    async def execute(self) -> None:
        """Main execution logic"""
        try:
            await self.pre_execute()

            # Fetch data
            top_pools = await self.gt.get_top_pools_by_network(self.network)
            new_pools = await self.gt.get_new_pools_by_network(self.network)
            
            # Clean and filter data
            cleaned_top = self.clean_pools(top_pools.copy())
            cleaned_new = self.clean_pools(new_pools.copy())
            
            filtered_top = self.filter_pools(cleaned_top.copy())
            filtered_new = self.filter_pools(cleaned_new.copy())
            document = {
                'timestamp': datetime.utcnow(),
                'trending_pools': cleaned_top.to_dict('records') if not cleaned_top.empty else [],
                'filtered_trending_pools': filtered_top.to_dict(
                    'records') if not filtered_top.empty else [],
                'new_pools': cleaned_new.to_dict('records') if not cleaned_new.empty else [],
                'filtered_new_pools': filtered_new.to_dict(
                    'records') if not filtered_new.empty else []
            }
            # Store data using MongoDBClient
            await self.mongo_client.insert_documents(collection_name="pools",
                                                     documents=[document])
            logging.info(f"Screening completed at {datetime.now()}")
            logging.info(f"Top pools: {len(cleaned_top)} (filtered: {len(filtered_top)})")
            logging.info(f"New pools: {len(cleaned_new)} (filtered: {len(filtered_new)})")
            
        except Exception as e:
            logging.error(f"Error executing market screener task: {str(e)}")
        finally:
            await self.post_execute()


async def main():
    mongo_uri = (
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
        f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'admin')}@"
        f"{os.getenv('MONGO_HOST', 'localhost')}:"
        f"{os.getenv('MONGO_PORT', '27017')}/"
    )
    config = {
        "mongo_uri": mongo_uri,
        "network": "solana",
        "quote_asset": "SOL",
        "min_pool_age_days": 2,
        "min_fdv": 70_000,
        "max_fdv": 5_000_000,
        "min_volume_24h": 150_000,
        "min_liquidity": 50_000,
        "min_transactions_24h": 300
    }
    
    task = PoolsScreenerTask(
        name="Market Screener",
        frequency=timedelta(hours=1),
        config=config
    )
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())