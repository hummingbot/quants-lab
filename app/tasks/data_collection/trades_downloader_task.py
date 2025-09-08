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
load_dotenv()


class TradesDownloaderTask(BaseTask):
    """Download trades data from exchanges and store in TimescaleDB with OHLC resampling."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.days_data_retention = task_config.get("days_data_retention", 7)
        self.quote_asset = task_config.get("quote_asset", "USDT")
        self.min_notional_size = Decimal(str(task_config.get("min_notional_size", 10.0)))
        self.resample_intervals = task_config.get("resample_intervals", ["1s"])
        
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
            logging.info(f"Min notional size: {self.min_notional_size}")
            logging.info(f"Data retention: {self.days_data_retention} days")
            logging.info(f"Resample intervals: {self.resample_intervals}")
            
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
        logging.info(f"Starting trades downloader for {self.connector_name}")
        
        try:
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = pd.Timestamp(
                time.time() - self.days_data_retention * 24 * 60 * 60,
                unit="s"
            ).tz_localize(timezone.utc)
            
            logging.info(f"Time range: {start_time} to {end_time}")
            
            # Get filtered trading pairs
            trading_rules = await self.clob.get_trading_rules(self.connector_name)
            trading_pairs = trading_rules.filter_by_quote_asset(self.quote_asset) \
                .filter_by_min_notional_size(self.min_notional_size) \
                .get_all_trading_pairs()
            
            # Track statistics
            stats = {
                "pairs_processed": 0,
                "pairs_total": len(trading_pairs),
                "trades_downloaded": 0,
                "ohlc_resampled": 0,
                "errors": 0
            }
            
            # Process each trading pair
            for i, trading_pair in enumerate(trading_pairs):
                try:
                    logging.info(f"Fetching trades for {trading_pair} [{i+1}/{len(trading_pairs)}]")
                    
                    # Setup table
                    table_name = self.timescale_client.get_trades_table_name(
                        self.connector_name, trading_pair
                    )
                    
                    # Get last trade ID to avoid duplicates
                    last_trade_id = await self.timescale_client.get_last_trade_id(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        table_name=table_name
                    )
                    
                    # Fetch trades
                    trades = await self.clob.get_trades(
                        self.connector_name,
                        trading_pair,
                        int(start_time.timestamp()),
                        int(end_time.timestamp()),
                        last_trade_id
                    )
                    
                    if trades.empty:
                        logging.info(f"No new trades for {trading_pair}")
                        continue
                    
                    # Prepare trades data
                    trades["connector_name"] = self.connector_name
                    trades["trading_pair"] = trading_pair
                    
                    trades_data = trades[
                        ["id", "connector_name", "trading_pair", "timestamp", "price", "volume", "sell_taker"]
                    ].values.tolist()
                    
                    # Store trades
                    await self.timescale_client.append_trades(
                        table_name=table_name,
                        trades=trades_data
                    )
                    
                    # Clean up old data
                    cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.days_data_retention)
                    await self.timescale_client.delete_trades(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        timestamp=cutoff_time.timestamp()
                    )
                    
                    # Resample to OHLC for configured intervals
                    for interval in self.resample_intervals:
                        try:
                            await self.timescale_client.compute_resampled_ohlc(
                                connector_name=self.connector_name,
                                trading_pair=trading_pair,
                                interval=interval
                            )
                            stats["ohlc_resampled"] += 1
                        except Exception as e:
                            logging.warning(f"OHLC resampling failed for {trading_pair} {interval}: {e}")
                    
                    stats["trades_downloaded"] += len(trades_data)
                    logging.info(f"Inserted {len(trades_data)} trades for {trading_pair}")
                    
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error processing {trading_pair}: {e}")
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
            
            logging.info(f"Trades download completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing trades downloader: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ TradesDownloaderTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Trades: {stats.get('trades_downloaded', 0)}")
        logging.info(f"  - OHLC resampled: {stats.get('ohlc_resampled', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ TradesDownloaderTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 TradesDownloaderTask retry attempt {attempt}: {error}")


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
        name="trades_downloader_test",
        enabled=True,
        task_class="tasks.data_collection.trades_downloader_task.TradesDownloaderTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "connector_name": "binance_perpetual",
            "quote_asset": "USDT",
            "min_notional_size": 10.0,
            "days_data_retention": 7,
            "resample_intervals": ["1s", "1m"],
            "timescale_config": timescale_config
        }
    )
    
    # Create and run task
    task = TradesDownloaderTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Downloaded {stats.get('trades_downloaded', 0)} trades")
        print(f"Processed {stats.get('pairs_processed', 0)} pairs")
        print(f"OHLC intervals resampled: {stats.get('ohlc_resampled', 0)}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())