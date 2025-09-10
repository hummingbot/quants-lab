import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
# Removed dotenv - no longer needed

from core.data_sources import CLOBDataSource
from core.tasks import BaseTask, TaskContext

# No environment loading needed
logging.basicConfig(level=logging.INFO)


class ScreenerSikorTask(BaseTask):
    """Market screening task for volatility and volume imbalance analysis."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.host = task_config.get("host", "localhost")
        self.connector_name = task_config.get("connector_name", "binance_perpetual")
        self.interval = task_config.get("interval", "15m")
        self.volatility_window = task_config.get("volatility_window", 50)
        self.volume_window = task_config.get("volume_window", 50)
        
        # Initialize CLOB data source (handles parquet caching automatically)
        self.clob = CLOBDataSource()
        self.days_lookback = task_config.get("days_lookback", 7)

    @staticmethod
    def get_volatility(df, window):
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=window).std() * np.sqrt(window)
        return df['volatility'].iloc[-1]

    @staticmethod
    def get_volume_imbalance(df, window):
        # Calculate volume metrics
        df["volume_usd"] = df["volume"] * df["close"]
        df["buy_taker_volume_usd"] = df["taker_buy_base_volume"] * df["close"]
        df["sell_taker_volume_usd"] = df["volume_usd"] - df["buy_taker_volume_usd"]
        # Calculate buy/sell imbalance
        df["buy_sell_imbalance"] = df["buy_taker_volume_usd"] - df["sell_taker_volume_usd"]
        # Calculate rolling total volume
        rolling_total_volume_usd = df["volume_usd"].rolling(window=window, min_periods=1).sum()
        return rolling_total_volume_usd.iloc[-1]

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            if not self.host:
                logging.error("host not configured")
                return False
                
            if not self.connector_name:
                logging.error("connector_name not configured")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            self.ts_client = TimescaleClient(host=self.host)
            await self.ts_client.connect()
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Host: {self.host}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Interval: {self.interval}")
            logging.info(f"Windows: volatility={self.volatility_window}, volume={self.volume_window}")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            if self.ts_client:
                await self.ts_client.close()
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting market screening analysis")
        
        try:
            # Track statistics
            stats = {
                "candles_processed": 0,
                "candles_total": 0,
                "pairs_analyzed": 0,
                "errors": 0
            }

            # Get available candles from cache
            available_candles = set()
            for (connector, pair, interval) in self.clob.candles_cache.keys():
                if connector == self.connector_name and interval == self.interval:
                    available_candles.add((connector, pair, interval))
            
            filtered_candles = list(available_candles)
            stats["candles_total"] = len(filtered_candles)
            
            # Calculate time range
            end_time = int(datetime.now(timezone.utc).timestamp())
            start_time = end_time - (self.days_lookback * 24 * 60 * 60)
            
            # Process candles in batch
            candles_tasks = [
                self.clob.get_candles(candle[0], candle[1], candle[2], start_time, end_time)
                for candle in filtered_candles
            ]
            candles = await asyncio.gather(*candles_tasks)
            
            # Analyze each candle
            report = []
            for candle in candles:
                try:
                    trading_pair = candle.trading_pair
                    df = candle.data
                    
                    # Convert to appropriate types
                    df["close"] = df["close"].astype(float)
                    df["volume"] = df["volume"].astype(float)
                    df["taker_buy_base_volume"] = df["taker_buy_base_volume"].astype(float)
                    
                    # Calculate metrics
                    volatility = self.get_volatility(df, self.volatility_window)
                    volume_imbalance = self.get_volume_imbalance(df, self.volume_window)
                    
                    report.append({
                        "trading_pair": trading_pair,
                        "volatility": volatility,
                        "volume_imbalance": volume_imbalance
                    })
                    
                    stats["pairs_analyzed"] += 1
                    
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error analyzing {candle.trading_pair}: {e}")
                    continue
                
                stats["candles_processed"] += 1
            
            # Log the report
            report_df = pd.DataFrame(report)
            logging.info(f"Market screening results:\n{report_df}")
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connector": self.connector_name,
                "interval": self.interval,
                "report_data": report,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Market screening completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing market screening: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ ScreenerSikorTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Candles: {stats.get('candles_processed', 0)}/{stats.get('candles_total', 0)}")
        logging.info(f"  - Pairs analyzed: {stats.get('pairs_analyzed', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ ScreenerSikorTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 ScreenerSikorTask retry attempt {attempt}: {error}")

async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="screener_sikor_test",
        enabled=True,
        task_class="tasks.data_reporting.screener_task_sikor.ScreenerSikorTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=12.0
        ),
        config={
            "connector_name": "binance_perpetual",
            "interval": "15m",
            "days_lookback": 7,
            "volatility_window": 50,
            "volume_window": 50
        }
    )
    
    # Create and run task
    task = ScreenerSikorTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Processed {stats.get('candles_processed', 0)}/{stats.get('candles_total', 0)} candles")
        print(f"Pairs analyzed: {stats.get('pairs_analyzed', 0)}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        
        # Show sample report data
        report_data = result.result_data.get("report_data", [])
        if report_data:
            print(f"Sample results (first 3): {report_data[:3]}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
