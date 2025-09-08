import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

from core.services.timescale_client import TimescaleClient
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
load_dotenv()


class LocalCacheExportTask(BaseTask):
    """Export candles data from TimescaleDB to local Parquet files."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.selected_pairs = task_config.get("selected_pairs", [])
        self.root_path = task_config.get("root_path", "")
        self.output_dir_name = task_config.get("output_dir", "data/candles")
        
        # Setup output directory
        self.output_dir = Path(os.path.join(self.root_path, self.output_dir_name))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client (will be connected in setup)
        self.timescale_client = None

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
            logging.info(f"Selected pairs: {len(self.selected_pairs)} pairs")
            logging.info(f"Output directory: {self.output_dir}")
            
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
        logging.info(f"Starting candles export for {self.connector_name}")
        
        try:
            # Track statistics
            stats = {
                "files_exported": 0,
                "total_candles_exported": 0,
                "pairs_processed": 0,
                "intervals_found": 0,
                "errors": 0
            }

            # Get available candles from database
            available_candles = await self.timescale_client.get_available_candles()
            
            for connector_name, trading_pair, interval in available_candles:
                # Filter by connector and selected pairs
                if connector_name != self.connector_name or trading_pair not in self.selected_pairs:
                    continue
                
                stats["intervals_found"] += 1
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
                logging.info(f"{current_time} - Exporting candles for {trading_pair} - {interval}")
                
                try:
                    # Get all candles from the database
                    candles = await self.timescale_client.get_all_candles(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        interval=interval
                    )
                    
                    candles_df: pd.DataFrame = candles.data
                    if candles_df.empty:
                        logging.info(f"{current_time} - No data found for {trading_pair} - {interval}")
                        continue
                    
                    # Process DataFrame
                    candles_df["timestamp"] = candles_df["timestamp"].apply(lambda x: x.timestamp())
                    candles_df.sort_values("timestamp", inplace=True)
                    
                    # Create filename and save
                    filename = f"{self.connector_name}|{trading_pair}|{interval}.parquet"
                    filepath = self.output_dir / filename
                    
                    # Save with compression
                    candles_df.to_parquet(
                        filepath,
                        engine='pyarrow',
                        compression='snappy',
                        index=True
                    )
                    
                    stats["files_exported"] += 1
                    stats["total_candles_exported"] += len(candles_df)
                    logging.info(f"{current_time} - Saved {len(candles_df)} candles to {filepath}")

                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"{current_time} - Error exporting {trading_pair} - {interval}: {e}")
                    continue
            
            # Count unique pairs processed
            processed_pairs = set()
            for connector_name, trading_pair, interval in available_candles:
                if connector_name == self.connector_name and trading_pair in self.selected_pairs:
                    processed_pairs.add(trading_pair)
            stats["pairs_processed"] = len(processed_pairs)
            
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
            
            logging.info(f"Candles export completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing candles export: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ LocalCacheExportTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Files exported: {stats.get('files_exported', 0)}")
        logging.info(f"  - Candles exported: {stats.get('total_candles_exported', 0)}")
        logging.info(f"  - Pairs processed: {stats.get('pairs_processed', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ LocalCacheExportTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 LocalCacheExportTask retry attempt {attempt}: {error}")


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
        name="local_cache_export_test",
        enabled=True,
        task_class="tasks.data_collection.local_cache_update_task.LocalCacheExportTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            "connector_name": "binance_perpetual",
            "output_dir": "data/candles",
            "timescale_config": timescale_config,
            "selected_pairs": [
                '1000SHIB-USDT', 'WLD-USDT', 'ACT-USDT', '1000BONK-USDT', 'DOGE-USDT', 'AGLD-USDT',
                'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
                'HMSTR-USDT', 'TROY-USDT', '1000PEPE-USDT', '1000X-USDT', 'PNUT-USDT', 'SOL-USDT',
                'XRP-USDT', 'SWELL-USDT'
            ]
        }
    )
    
    # Create and run task
    task = LocalCacheExportTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Exported {stats.get('files_exported', 0)} files")
        print(f"Total candles: {stats.get('total_candles_exported', 0)}")
        print(f"Pairs processed: {stats.get('pairs_processed', 0)}")
        print(f"Output directory: {result.result_data.get('output_directory', 'N/A')}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
