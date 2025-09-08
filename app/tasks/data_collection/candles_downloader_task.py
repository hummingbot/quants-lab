import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

from core.data_sources import CLOBDataSource
from core.services.timescale_client import TimescaleClient
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class CandlesDownloaderTask(BaseTask):
    """Download OHLC candles data from exchanges and store in TimescaleDB."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.days_data_retention = task_config.get("days_data_retention", 7)
        self.intervals = task_config.get("intervals", ["1m"])
        self.quote_asset = task_config.get("quote_asset", "USDT")
        self.min_notional_size = Decimal(str(task_config.get("min_notional_size", 10.0)))
        
        # Initialize clients (will be connected in setup)
        self.clob = CLOBDataSource()
        self.timescale_client = None

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            # Check required configuration
            if not self.connector_name:
                logging.error("connector_name not configured")
                return False
                
            timescale_config = self.config.config.get("timescale_config", {})
            if not timescale_config:
                logging.error("timescale_config not provided")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            # Initialize TimescaleDB client
            timescale_config = self.config.config["timescale_config"]
            self.timescale_client = TimescaleClient(
                host=timescale_config.get("host", "localhost"),
                port=timescale_config.get("port", 5432),
                user=timescale_config.get("user", "admin"),
                password=timescale_config.get("password", "admin"),
                database=timescale_config.get("database", "timescaledb")
            )
            await self.timescale_client.connect()
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Quote asset: {self.quote_asset}")
            logging.info(f"Intervals: {self.intervals}")
            logging.info(f"Data retention: {self.days_data_retention} days")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            if self.timescale_client:
                await self.timescale_client.close()
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting candles downloader for {self.connector_name}")
        
        try:
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = pd.Timestamp(
                time.time() - self.days_data_retention * 24 * 60 * 60,
                unit="s"
            ).tz_localize(timezone.utc).timestamp()
            
            logging.info(f"Time range: {start_time} to {end_time}")
            
            # Get trading pairs
            trading_rules = await self.clob.get_trading_rules(self.connector_name)
            trading_pairs = trading_rules.get_all_trading_pairs()
            
            # Track statistics
            stats = {
                "pairs_processed": 0,
                "pairs_total": len(trading_pairs),
                "intervals_processed": 0,
                "candles_downloaded": 0,
                "errors": 0
            }
            
            # Process each trading pair
            for i, trading_pair in enumerate(trading_pairs):
                for interval in self.intervals:
                    try:
                        logging.info(f"Fetching candles for {trading_pair} [{i+1}/{len(trading_pairs)}] {interval}")
                        
                        # Setup table
                        table_name = self.timescale_client.get_ohlc_table_name(
                            self.connector_name, trading_pair, interval
                        )
                        await self.timescale_client.create_candles_table(table_name)
                        
                        # Get last timestamp to avoid duplicates
                        last_timestamp = await self.timescale_client.get_last_candle_timestamp(
                            connector_name=self.connector_name,
                            trading_pair=trading_pair,
                            interval=interval
                        )
                        fetch_start = last_timestamp if last_timestamp else start_time
                        
                        # Fetch candles
                        candles = await self.clob.get_candles(
                            self.connector_name,
                            trading_pair,
                            interval,
                            int(fetch_start),
                            int(end_time.timestamp())
                        )
                        
                        if candles.data.empty:
                            logging.info(f"No new candles for {trading_pair} {interval}")
                            continue
                        
                        # Store candles
                        await self.timescale_client.append_candles(
                            table_name=table_name,
                            candles=candles.data.values.tolist()
                        )
                        
                        # Clean up old data
                        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.days_data_retention)
                        await self.timescale_client.delete_candles(
                            connector_name=self.connector_name,
                            trading_pair=trading_pair,
                            interval=interval,
                            timestamp=cutoff_time.timestamp()
                        )
                        
                        stats["candles_downloaded"] += len(candles.data)
                        stats["intervals_processed"] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        stats["errors"] += 1
                        logging.exception(f"Error processing {trading_pair} {interval}: {e}")
                        continue
                
                stats["pairs_processed"] += 1
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connector": self.connector_name,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Candles download completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing candles downloader: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ CandlesDownloaderTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Intervals: {stats.get('intervals_processed', 0)}")
        logging.info(f"  - Candles: {stats.get('candles_downloaded', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ CandlesDownloaderTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 CandlesDownloaderTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Build TimescaleDB config from environment
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="candles_downloader_test",
        enabled=True,
        task_class="tasks.data_collection.candles_downloader_task.CandlesDownloaderTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "connector_name": "binance_perpetual",
            "quote_asset": "USDT",
            "intervals": ["15m", "1h"],
            "days_data_retention": 30,
            "min_notional_size": 10,
            "timescale_config": timescale_config
        }
    )
    
    # Create and run task
    task = CandlesDownloaderTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Downloaded {stats.get('candles_downloaded', 0)} candles")
        print(f"Processed {stats.get('pairs_processed', 0)} pairs")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())