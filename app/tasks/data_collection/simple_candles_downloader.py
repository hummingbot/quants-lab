import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd

from core.data_sources import CLOBDataSource
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


class SimpleCandlesDownloader(BaseTask):
    """Download OHLC candles for specific trading pairs and store as parquet files."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.days_data_retention = task_config.get("days_data_retention", 7)
        self.intervals = task_config.get("intervals", ["1m"])
        self.trading_pairs = task_config.get("trading_pairs", ["BTC-USDT"])
        
        # Initialize CLOB data source (handles parquet caching automatically)
        self.clob = CLOBDataSource()

    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution, including validation of prerequisites."""
        try:
            await super().setup(context)
            
            # Validate prerequisites
            if not self.connector_name:
                raise RuntimeError("connector_name not configured")
                
            if not self.trading_pairs:
                raise RuntimeError("trading_pairs not configured")
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Trading pairs: {self.trading_pairs}")
            logging.info(f"Intervals: {self.intervals}")
            logging.info(f"Data retention: {self.days_data_retention} days")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            await super().cleanup(context, result)
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting simple candles downloader for {self.connector_name}")
        
        try:
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = pd.Timestamp(
                time.time() - self.days_data_retention * 24 * 60 * 60,
                unit="s"
            ).tz_localize(timezone.utc).timestamp()
            
            logging.info(f"Time range: {start_time} to {end_time}")
            
            # Track statistics
            stats = {
                "pairs_processed": 0,
                "pairs_total": len(self.trading_pairs),
                "intervals_processed": 0,
                "candles_downloaded": 0,
                "errors": 0
            }
            
            # Process each trading pair and interval
            for i, trading_pair in enumerate(self.trading_pairs):
                for interval in self.intervals:
                    try:
                        logging.info(f"Fetching candles for {trading_pair} [{i+1}/{len(self.trading_pairs)}] {interval}")
                        
                        # Fetch candles using CLOB (handles caching automatically)
                        try:
                            candles = await self.clob.get_candles(
                                self.connector_name,
                                trading_pair,
                                interval,
                                int(start_time),
                                int(end_time.timestamp())
                            )
                        except KeyError as e:
                            logging.error(f"KeyError fetching candles for {trading_pair}: {e}")
                            stats["errors"] += 1
                            continue
                        
                        if candles.data.empty:
                            logging.info(f"No new candles for {trading_pair} {interval}")
                            continue
                        
                        stats["candles_downloaded"] += len(candles.data)
                        stats["intervals_processed"] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        stats["errors"] += 1
                        logging.exception(f"Error processing {trading_pair} {interval}: {e}")
                        continue
                
                stats["pairs_processed"] += 1
            
            # Save all cached data to parquet files
            logging.info("Saving candles cache to parquet files...")
            self.clob.dump_candles_cache()
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connector": self.connector_name,
                "trading_pairs": self.trading_pairs,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Simple candles download completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing simple candles downloader: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ SimpleCandlesDownloader succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Intervals: {stats.get('intervals_processed', 0)}")
        logging.info(f"  - Candles: {stats.get('candles_downloaded', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ SimpleCandlesDownloader failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 SimpleCandlesDownloader retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="simple_candles_downloader_test",
        enabled=True,
        task_class="tasks.data_collection.simple_candles_downloader.SimpleCandlesDownloader",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "connector_name": "binance_perpetual",
            "intervals": ["15m"],
            "days_data_retention": 7,
            "trading_pairs": ["BTC-USDT", "ETH-USDT"]
        }
    )
    
    # Create and run task
    task = SimpleCandlesDownloader(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Downloaded {stats.get('candles_downloaded', 0)} candles")
        print(f"Processed {stats.get('pairs_processed', 0)} pairs")
        print(f"Trading pairs: {result.result_data.get('trading_pairs', [])}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())