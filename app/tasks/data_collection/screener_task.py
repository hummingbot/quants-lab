import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

from core.services.timescale_client import TimescaleClient
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
load_dotenv()


class MarketScreenerTask(BaseTask):
    """Calculate and store market screening metrics for available trading pairs."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.intervals = task_config.get("intervals", ["1m", "3m", "5m", "15m", "1h"])
        self.interval_mapping = {
            "1m": "one_min",
            "3m": "three_min",
            "5m": "five_min",
            "15m": "fifteen_min",
            "1h": "one_hour"
        }
        
        # Initialize client (will be connected in setup)
        self.ts_client = None

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            # Check required configuration
            timescale_config = self.config.config.get("timescale_config", {})
            if not timescale_config:
                logging.error("timescale_config not provided")
                return False
                
            if not self.intervals:
                logging.error("intervals not configured")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            # Initialize TimescaleDB client
            timescale_config = self.config.config.get("timescale_config", {})
            self.ts_client = TimescaleClient(
                host=timescale_config.get("host", "localhost"),
                port=timescale_config.get("port", 5432),
                user=timescale_config.get("user", "admin"),
                password=timescale_config.get("password", "admin"),
                database=timescale_config.get("database", "timescaledb")
            )
            await self.ts_client.connect()
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Intervals: {self.intervals}")
            
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
        logging.info("Starting market screening")
        
        try:
            # Track statistics
            stats = {
                "pairs_processed": 0,
                "pairs_total": 0,
                "metrics_calculated": 0,
                "errors": 0
            }

            # Get available pairs
            available_pairs = await self.ts_client.get_available_pairs()
            stats["pairs_total"] = len(available_pairs)

            for connector_name, trading_pair in available_pairs:
                try:
                    await self.process_pair(connector_name, trading_pair)
                    stats["pairs_processed"] += 1
                    stats["metrics_calculated"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error processing pair {trading_pair}: {e}")
                    continue
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Market screening completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing market screener: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ MarketScreenerTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Metrics calculated: {stats.get('metrics_calculated', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ MarketScreenerTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 MarketScreenerTask retry attempt {attempt}: {error}")

    async def process_pair(self, connector_name, trading_pair):
        """Process metrics for a single trading pair."""
        try:
            candles = await self.ts_client.get_candles(connector_name, trading_pair, interval="1h")
            screener_metrics = self.calculate_global_screener_metrics(
                candles_df=candles.data,
                connector_name=connector_name,
                trading_pair=trading_pair
            )

            interval_screener_metrics = await self.calculate_interval_metrics(connector_name, trading_pair)
            screener_metrics.update(interval_screener_metrics)
            screener_metrics = {key: json.dumps(value) if isinstance(value, dict) else value for key, value in screener_metrics.items()}

            await self.ts_client.append_screener_metrics(screener_metrics)

        except (ValueError, TypeError) as e:
            logging.exception(f"{self.now()} - Error calculating metrics for {trading_pair}\n {e}")
        except Exception as e:
            logging.exception(f"{self.now()} - Unexpected error processing pair {trading_pair}\n {e}")

    async def calculate_interval_metrics(self, connector_name, trading_pair):
        """Calculate metrics for each selected interval."""
        interval_screener_metrics = {}
        for selected_interval in self.intervals:
            mapped_interval = self.interval_mapping[selected_interval]
            try:
                candles = await self.ts_client.get_candles(connector_name, trading_pair, interval=selected_interval)
                interval_screener_metrics[mapped_interval] = self.calculate_interval_screener_metrics(candles.data)
            except Exception as e:
                logging.exception(
                    f"{self.now()} - Error processing interval {selected_interval} for {trading_pair}\n {e}")
                interval_screener_metrics[mapped_interval] = {}
        return interval_screener_metrics

    def calculate_global_screener_metrics(self, candles_df: pd.DataFrame, connector_name: str, trading_pair: str):
        df = candles_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")

        # 1. Price Analysis
        # Describe price statistics
        global_metrics = {
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "start_time": df.index.min().tz_localize("UTC"),  # or your desired timezone
            "end_time": df.index.max().tz_localize("UTC"),
            "price": df["close"].describe().to_dict()}

        # Price Change Over Periods (% CBO 24h, 1w, 1m)
        price_resampled = df['close'].resample('1d').last()  # Daily resampling for consistent periods
        global_metrics['price_cbo'] = {
            '24h': self.percent_change(price_resampled, 1),
            '1w': self.percent_change(price_resampled, 7),
            '1m': self.percent_change(price_resampled, 28)
        }

        # 2. Volume Analysis
        # 24h Volume USD
        last_24h = df.loc[df.index >= (df.index[-1] - timedelta(hours=24)), 'volume'].sum()
        global_metrics['volume_24h'] = last_24h

        # Volume % CBO for Different Periods
        volume_resampled = df['volume'].resample('1d').sum()
        global_metrics['volume_cbo'] = {
            '24h': self.percent_change(volume_resampled, 1),
            '1w': self.percent_change(volume_resampled, 7),
            '1m': self.percent_change(volume_resampled, 30)
        }
        return global_metrics

    @staticmethod
    def calculate_interval_screener_metrics(candles_df: pd.DataFrame):
        interval_metrics = {}

        df = candles_df.copy()
        df['atr_24h'] = ta.atr(df['high'], df['low'], df['close'], length=24)
        df['atr_1w'] = ta.atr(df['high'], df['low'], df['close'], length=7 * 24)
        df['natr_24h'] = ta.natr(df['high'], df['low'], df['close'], length=24)
        df['natr_1w'] = ta.natr(df['high'], df['low'], df['close'], length=7 * 24)
        interval_metrics['natr'] = df[['natr_24h', 'natr_1w']].describe().to_dict()

        # Bollinger Bands Width (50, 100, 200 / 2.0)
        for window in [50, 100, 200]:
            bb = ta.bbands(df['close'], length=window, std=2)
            interval_metrics[f'bb_width_{window}'] = bb[f'BBB_{window}_2.0'].mean() / 200
        return interval_metrics

    @staticmethod
    def percent_change(series, period):
        shifted = series.shift(period)
        if shifted.iloc[-1] == 0 or pd.isna(shifted.iloc[-1]):
            return None
        return (series.iloc[-1] - shifted.iloc[-1]) / shifted.iloc[-1]

    @staticmethod
    def now():
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')


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
        name="market_screener_test",
        enabled=True,
        task_class="tasks.data_collection.screener_task.MarketScreenerTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "intervals": ["1m", "3m", "5m", "15m", "1h"],
            "timescale_config": timescale_config
        }
    )
    
    # Create and run task
    task = MarketScreenerTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Processed {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)} pairs")
        print(f"Metrics calculated: {stats.get('metrics_calculated', 0)}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
