import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pandas as pd

from core.data_sources import CLOBDataSource
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)


class TradesDownloaderTask(BaseTask):
    """Download trades data from exchanges and store as parquet files with OHLC resampling."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_name = task_config["connector_name"]
        self.days_data_retention = task_config.get("days_data_retention", 7)
        self.quote_asset = task_config.get("quote_asset", "USDT")
        self.min_notional_size = Decimal(str(task_config.get("min_notional_size", 10.0)))
        self.resample_intervals = task_config.get("resample_intervals", ["1m"])  # Default to 1m candles
        
        # Initialize CLOB data source (handles parquet caching automatically)
        self.clob = CLOBDataSource()

    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution, including validation of prerequisites."""
        try:
            await super().setup(context)
            
            # Validate prerequisites
            if not self.connector_name:
                raise RuntimeError("connector_name not configured")
            
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
            await super().cleanup(context, result)
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
                "candles_downloaded": 0,
                "ohlc_resampled": 0,
                "errors": 0
            }
            
            # Process each trading pair to generate OHLC candles from trades
            for i, trading_pair in enumerate(trading_pairs):
                try:
                    logging.info(f"Generating OHLC candles from trades for {trading_pair} [{i+1}/{len(trading_pairs)}]")
                    
                    # Generate OHLC candles from trades for each interval
                    for interval in self.resample_intervals:
                        try:
                            # Fetch candles (CLOB will use trades to generate OHLC if needed)
                            candles = await self.clob.get_candles(
                                self.connector_name,
                                trading_pair,
                                interval,
                                int(start_time.timestamp()),
                                int(end_time.timestamp()),
                                from_trades=True  # Generate from trades data
                            )
                            
                            if candles.data.empty:
                                logging.info(f"No candles generated from trades for {trading_pair} {interval}")
                                continue
                            
                            stats["candles_downloaded"] += len(candles.data)
                            stats["ohlc_resampled"] += 1
                            logging.info(f"Generated {len(candles.data)} candles from trades for {trading_pair} {interval}")
                            
                        except Exception as e:
                            logging.warning(f"OHLC generation failed for {trading_pair} {interval}: {e}")
                    
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error processing {trading_pair}: {e}")
                    continue
                
                stats["pairs_processed"] += 1
            
            # Save all cached candles data to parquet files
            logging.info("Saving candles cache to parquet files...")
            self.clob.dump_candles_cache()
            
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
        logging.info(f"  - Candles: {stats.get('candles_downloaded', 0)}")
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
            "resample_intervals": ["1m", "5m", "15m", "1h"]
        }
    )
    
    # Create and run task
    task = TradesDownloaderTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Generated {stats.get('candles_downloaded', 0)} candles from trades")
        print(f"Processed {stats.get('pairs_processed', 0)} pairs")
        print(f"OHLC intervals resampled: {stats.get('ohlc_resampled', 0)}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())