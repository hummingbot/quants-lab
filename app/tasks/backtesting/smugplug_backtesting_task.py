import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting

from core.backtesting.optimizer import StrategyOptimizer
from core.services.timescale_client import TimescaleClient
from core.tasks import BaseTask, TaskContext
from research_notebooks.research_notebooks.smugplug.smugplug_config_gen_simple import SmugPlugConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class SmugPlugBacktestingTask(BaseTask):
    """Backtesting task for SmugPlug strategy optimization with top markets screening."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.resolution = task_config.get("resolution", "1m")
        self.root_path = task_config.get('root_path', "")
        self.max_top_markets = task_config.get("max_top_markets", 50)
        self.n_trials = task_config.get("n_trials", 50)
        
        # TimescaleDB configuration
        self.timescale_host = task_config.get("timescale_host", "localhost")
        self.timescale_port = task_config.get("timescale_port", 5432)
        
        # Optuna configuration
        self.optuna_host = task_config.get("optuna_host", "localhost") 
        self.optuna_port = task_config.get("optuna_port", 5433)
        self.optuna_user = task_config.get("optuna_user", "admin")
        self.optuna_password = task_config.get("optuna_password", "admin")
        
        # Initialize clients (will be connected in setup)
        self.ts_client = None
        self.optimizer = None

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            # Check required configuration
            if not self.root_path:
                logging.error("root_path not configured")
                return False
                
            if not self.timescale_host:
                logging.error("timescale_host not configured")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Prerequisites validation failed: {e}")
            return False
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution."""
        try:
            # Initialize TimescaleDB client
            self.ts_client = TimescaleClient(
                host=self.timescale_host,
                port=self.timescale_port
            )
            await self.ts_client.connect()
            
            # Initialize strategy optimizer
            self.optimizer = StrategyOptimizer(
                engine="postgres",
                root_path=self.root_path,
                resolution=self.resolution,
                db_client=self.ts_client,
                db_host=self.optuna_host,
                db_port=self.optuna_port,
                db_user=self.optuna_user,
                db_pass=self.optuna_password
            )
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Resolution: {self.resolution}")
            logging.info(f"Max top markets: {self.max_top_markets}")
            logging.info(f"N trials: {self.n_trials}")
            
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

    def generate_top_markets_report(self, status_db_df: pd.DataFrame):
        """Generate top markets report based on volume."""
        df = status_db_df.copy()
        df.sort_values("volume_usd", ascending=False, inplace=True)
        screener_top_markets = df.head(self.max_top_markets)
        return screener_top_markets[["connector_name", "trading_pair", "from_timestamp", "to_timestamp"]]

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info("Starting SmugPlug backtesting with top markets screening")
        
        try:
            # Track statistics
            stats = {
                "markets_processed": 0,
                "markets_total": 0,
                "optimizations_completed": 0,
                "total_trials": 0,
                "markets_with_no_data": 0,
                "errors": 0
            }

            # Generate top markets report
            logging.info("Generating top markets report")
            metrics_df = await self.ts_client.get_db_status_df()
            top_markets_df = self.generate_top_markets_report(metrics_df)
            stats["markets_total"] = len(top_markets_df)
            
            logging.info(f"Optimizing strategy for top {stats['markets_total']} markets")
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            for index, row in top_markets_df.iterrows():
                try:
                    connector_name = row["connector_name"]
                    trading_pair = row["trading_pair"]
                    start_date = pd.Timestamp(row["from_timestamp"].timestamp(), unit="s")
                    end_date = pd.Timestamp(row["to_timestamp"].timestamp(), unit="s")
                    
                    logging.info(f"Optimizing strategy for {connector_name} {trading_pair}")
                    logging.info(f"Time range: {start_date} to {end_date}")
                    
                    # Create config generator
                    config_generator = SmugPlugConfigGenerator(
                        start_date=start_date,
                        end_date=end_date,
                        backtester=DirectionalTradingBacktesting()
                    )
                    config_generator.trading_pair = trading_pair
                    
                    # Fetch candles
                    logging.info(f"Fetching candles for {connector_name} {trading_pair}")
                    candles = await self.optimizer._db_client.get_candles(
                        connector_name, trading_pair,
                        self.resolution, start_date.timestamp(), end_date.timestamp()
                    )
                    
                    if len(candles.data) == 0:
                        stats["markets_with_no_data"] += 1
                        logging.warning(f"No candles data found for {connector_name} {trading_pair}")
                        continue
                    
                    # Setup backtester data
                    start_time = candles.data["timestamp"].min()
                    end_time = candles.data["timestamp"].max()
                    config_generator.backtester.backtesting_data_provider.candles_feeds[
                        f"{connector_name}_{trading_pair}_{self.resolution}"
                    ] = candles.data
                    config_generator.start = int(start_time)
                    config_generator.end = int(end_time)
                    
                    # Run optimization
                    study_name = f"smugplug_test_task_{today_str}_{trading_pair.replace('-', '_')}"
                    await self.optimizer.optimize(
                        study_name=study_name,
                        config_generator=config_generator,
                        n_trials=self.n_trials
                    )
                    
                    stats["optimizations_completed"] += 1
                    stats["total_trials"] += self.n_trials
                    logging.info(f"Completed optimization for {trading_pair}")
                    
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error optimizing {connector_name} {trading_pair}: {e}")
                    continue
                
                stats["markets_processed"] += 1
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "strategy": "smugplug",
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"SmugPlug backtesting completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing SmugPlug backtesting: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ SmugPlugBacktestingTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Markets: {stats.get('markets_processed', 0)}/{stats.get('markets_total', 0)}")
        logging.info(f"  - Optimizations: {stats.get('optimizations_completed', 0)}")
        logging.info(f"  - Total trials: {stats.get('total_trials', 0)}")
        if stats.get('markets_with_no_data', 0) > 0:
            logging.info(f"  - Markets with no data: {stats.get('markets_with_no_data', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ SmugPlugBacktestingTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 SmugPlugBacktestingTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="smugplug_backtesting_test",
        enabled=True,
        task_class="tasks.backtesting.smugplug_backtesting_task.SmugPlugBacktestingTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=12.0
        ),
        config={
            "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            "total_amount": 500,
            "activation_bounds": 0.002,
            "max_executors_per_side": 1,
            "cooldown_time": 0,
            "leverage": 20,
            "time_limit": 86400,  # 60 * 60 * 24
            "bb_lengths": [50, 100, 200],
            "bb_stds": [1.0, 2.0, 3.0],
            "intervals": ["1m"],
            "volume_threshold": 0.5,
            "volatility_threshold": 0.5,
            "ts_delta_multiplier": 0.2,
            "max_top_markets": 10,  # Reduced for testing
            "max_dca_amount_ratio": 5,
            "backtesting_resolution": "1m",
            "min_distance_between_orders": 0.01,
            "max_ts_sl_ratio": 0.5,
            "lookback_days": 7,
            "resolution": "1m",
            "n_trials": 5,  # Reduced for testing
            "timescale_host": os.getenv("TIMESCALE_HOST", "localhost"),
            "timescale_port": int(os.getenv("TIMESCALE_PORT", "5432")),
            "optuna_host": os.getenv("OPTUNA_HOST", "localhost"),
            "optuna_port": int(os.getenv("OPTUNA_DOCKER_PORT", "5433")),
            "optuna_user": "admin",
            "optuna_password": "admin"
        }
    )
    
    # Create and run task
    task = SmugPlugBacktestingTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Markets: {stats.get('markets_processed', 0)}/{stats.get('markets_total', 0)}")
        print(f"Optimizations: {stats.get('optimizations_completed', 0)}")
        print(f"Total trials: {stats.get('total_trials', 0)}")
        print(f"Strategy: {result.result_data.get('strategy', 'N/A')}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
