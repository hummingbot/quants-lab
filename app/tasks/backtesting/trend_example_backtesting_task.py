import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

from dotenv import load_dotenv
from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop

from app.controllers.directional_trading.trend_example import TrendExampleControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator, StrategyOptimizer
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class TrendExampleConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for MACD and Bollinger Bands optimization.
    """

    async def generate_config(self, trial) -> BacktestingConfig:
        # Suggest hyperparameters using the trial object
        interval = trial.suggest_categorical("interval", ["1m"])
        ema_short = trial.suggest_int("ema_short", 9, 30)
        ema_medium = trial.suggest_int("ema_medium", ema_short, 70)
        ema_long = trial.suggest_int("slow_ma", ema_medium, 200)
        total_amount_quote = 1000
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)
        take_profit = trial.suggest_float("take_profit", 0.04, 0.05, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.01)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.3, step=0.1)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15

        # Create the strategy configuration
        config = TrendExampleControllerConfig(
            connector_name=self.config["connector_name"],
            trading_pair=self.config["trading_pair"],
            interval=interval,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            total_amount_quote=Decimal(total_amount_quote),
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price),
                trailing_delta=Decimal(trailing_stop_trailing_delta),
            ),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)

class TrendExampleBacktestingTask(BaseTask):
    """Backtesting task for Trend Example strategy optimization."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.resolution = task_config.get("resolution", "1m")
        self.connector_name = task_config.get("connector_name", "binance_perpetual")
        self.selected_pairs = task_config.get("selected_pairs", ["1000BONK-USDT"])
        self.study_name_base = task_config.get("study_name", "trend_example")
        self.n_trials = task_config.get("n_trials", 50)
        self.lookback_days = task_config.get("lookback_days", 30)
        self.end_time_buffer_hours = task_config.get("end_time_buffer_hours", 6)
        
        # Initialize optimizer (will be set up in setup)
        self.optimizer = None

    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution, including validation of prerequisites."""
        # Call parent setup to initialize database and notification services
        await super().setup(context)
        
        try:
            # Validate prerequisites
            if not self.connector_name:
                raise RuntimeError("connector_name not configured")
                
            if not self.selected_pairs:
                raise RuntimeError("selected_pairs not configured")
            
            # Initialize strategy optimizer (no root_path needed, uses local SQLite)
            self.optimizer = StrategyOptimizer(
                resolution=self.resolution,
                load_cached_data=True,
                custom_backtester=DirectionalTradingBacktesting()
            )
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Resolution: {self.resolution}")
            logging.info(f"Trading pairs: {len(self.selected_pairs)} pairs")
            logging.info(f"Lookback days: {self.lookback_days}")
            logging.info(f"N trials: {self.n_trials}")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            # No specific cleanup needed for optimizer
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting Trend Example backtesting for {len(self.selected_pairs)} pairs")
        
        try:
            # Track statistics
            stats = {
                "pairs_processed": 0,
                "pairs_total": len(self.selected_pairs),
                "optimizations_completed": 0,
                "total_trials": 0,
                "errors": 0
            }

            today_str = datetime.now().strftime("%Y-%m-%d")
            
            for trading_pair in self.selected_pairs:
                try:
                    # Calculate time range dynamically
                    import time
                    import pandas as pd
                    
                    end_date = time.time() - (self.end_time_buffer_hours * 3600)
                    start_date = end_date - (self.lookback_days * 24 * 3600)
                    
                    logging.info(f"Optimizing strategy for {self.connector_name} {trading_pair}")
                    logging.info(f"Time range: {pd.to_datetime(start_date, unit='s')} to {pd.to_datetime(end_date, unit='s')}")
                    
                    # Create config generator
                    config_generator = TrendExampleConfigGenerator(
                        start_date=pd.to_datetime(start_date, unit="s"),
                        end_date=pd.to_datetime(end_date, unit="s"),
                        config={
                            "connector_name": self.connector_name,
                            "trading_pair": trading_pair
                        }
                    )
                    
                    # Run optimization
                    study_name = f"{self.study_name_base}_{today_str}_{trading_pair.replace('-', '_')}"
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
                    logging.exception(f"Error optimizing {trading_pair}: {e}")
                    continue
                
                stats["pairs_processed"] += 1
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connector": self.connector_name,
                "strategy": "trend_example",
                "lookback_days": self.lookback_days,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Trend Example backtesting completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing Trend Example backtesting: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ TrendExampleBacktestingTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Optimizations: {stats.get('optimizations_completed', 0)}")
        logging.info(f"  - Total trials: {stats.get('total_trials', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ TrendExampleBacktestingTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 TrendExampleBacktestingTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="trend_example_backtesting_test",
        enabled=True,
        task_class="tasks.backtesting.trend_example_backtesting_task.TrendExampleBacktestingTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=12.0
        ),
        config={
            "resolution": "1m",
            "connector_name": "binance_perpetual",
            "selected_pairs": ["1000BONK-USDT"],
            "study_name": "trend_example_test",
            "n_trials": 5,  # Reduced for testing
            "lookback_days": 7,
            "end_time_buffer_hours": 6
        }
    )
    
    # Create and run task
    task = TrendExampleBacktestingTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Processed {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)} pairs")
        print(f"Optimizations: {stats.get('optimizations_completed', 0)}")
        print(f"Total trials: {stats.get('total_trials', 0)}")
        print(f"Strategy: {result.result_data.get('strategy', 'N/A')}")
        print(f"Time range: {result.result_data.get('time_range', 'N/A')}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
