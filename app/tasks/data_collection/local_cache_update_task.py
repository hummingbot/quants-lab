import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_paths import data_paths
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)


class LocalCacheUpdateTask(BaseTask):
    """Update and refresh local Parquet cache files for candles data."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.selected_pairs = task_config.get("selected_pairs", [])
        self.intervals = task_config.get("intervals", ["1m", "5m", "15m", "1h"])
        self.days_to_fetch = task_config.get("days_to_fetch", 7)
        
        # Initialize CLOB data source
        self.clob = CLOBDataSource()
        
        # Use centralized data paths
        self.output_dir = data_paths.candles_dir
        

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            # Check required configuration
            if not self.connector_name:
                logging.error("connector_name not configured")
                return False
                
            if not self.selected_pairs:
                logging.error("selected_pairs not configured")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            await super().setup(context)
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Selected pairs: {len(self.selected_pairs)} pairs")
            logging.info(f"Output directory: {self.output_dir}")
            
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
        logging.info(f"Starting cache update for {self.connector_name}")
        
        try:
            # Track statistics
            stats = {
                "files_updated": 0,
                "total_candles_cached": 0,
                "pairs_processed": 0,
                "intervals_processed": 0,
                "errors": 0
            }

            # Calculate time range
            end_time = int(datetime.now(timezone.utc).timestamp())
            start_time = end_time - (self.days_to_fetch * 24 * 60 * 60)
            
            # Load existing cache
            logging.info("Loading existing cache...")
            self.clob.load_candles_cache()
            
            # Process each trading pair and interval
            for trading_pair in self.selected_pairs:
                stats["pairs_processed"] += 1
                
                for interval in self.intervals:
                    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
                    logging.info(f"{current_time} - Updating cache for {trading_pair} - {interval}")
                    
                    try:
                        # Fetch fresh candles (CLOB will merge with cache automatically)
                        candles = await self.clob.get_candles(
                            self.connector_name,
                            trading_pair,
                            interval,
                            start_time,
                            end_time
                        )
                        
                        if candles.data.empty:
                            logging.info(f"{current_time} - No data found for {trading_pair} - {interval}")
                            continue
                        
                        stats["intervals_processed"] += 1
                        stats["total_candles_cached"] += len(candles.data)
                        logging.info(f"{current_time} - Cached {len(candles.data)} candles")
                        
                    except Exception as e:
                        stats["errors"] += 1
                        logging.exception(f"{current_time} - Error updating cache for {trading_pair} - {interval}: {e}")
                        continue
            
            # Save all cached data to parquet files
            logging.info("Saving updated cache to parquet files...")
            self.clob.dump_candles_cache()
            stats["files_updated"] = len(self.clob._candles_cache)
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connector": self.connector_name,
                "output_directory": str(self.output_dir),
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Cache update completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing candles export: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ LocalCacheUpdateTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Files updated: {stats.get('files_updated', 0)}")
        logging.info(f"  - Candles cached: {stats.get('total_candles_cached', 0)}")
        logging.info(f"  - Pairs processed: {stats.get('pairs_processed', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ LocalCacheUpdateTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 LocalCacheUpdateTask retry attempt {attempt}: {error})


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="local_cache_update_test",
        enabled=True,
        task_class="tasks.data_collection.local_cache_update_task.LocalCacheUpdateTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "connector_name": "binance_perpetual",
            "intervals": ["1m", "5m", "15m", "1h"],
            "days_to_fetch": 7,
            "selected_pairs": [
                '1000SHIB-USDT', 'WLD-USDT', 'ACT-USDT', '1000BONK-USDT', 'DOGE-USDT', 'AGLD-USDT',
                'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
                'HMSTR-USDT', 'TROY-USDT', '1000PEPE-USDT', '1000X-USDT', 'PNUT-USDT', 'SOL-USDT',
                'XRP-USDT', 'SWELL-USDT'
            ]
        }
    )
    
    # Create and run task
    task = LocalCacheUpdateTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Updated {stats.get('files_updated', 0)} cache files")
        print(f"Total candles cached: {stats.get('total_candles_cached', 0)}")
        print(f"Pairs processed: {stats.get('pairs_processed', 0)}")
        print(f"Output directory: {result.result_data.get('output_directory', 'N/A')}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
