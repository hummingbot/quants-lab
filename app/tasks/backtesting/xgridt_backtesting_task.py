import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from hummingbot.strategy_v2.utils.distributions import Distributions

from app.controllers.directional_trading.xgridt import XGridTControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator, StrategyOptimizer
from core.tasks import BaseTask, TaskContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XGridTConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for XGridT optimization.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config.get("connector_name", "binance_perpetual")
        trading_pair = self.config.get("trading_pair", "PNUT-USDT")
        interval = self.config.get("interval", "1m")
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("interval", interval)
        ema_short = trial.suggest_int("ema_short", 9, 59)
        ema_medium = trial.suggest_int("ema_medium", ema_short + 10, 150)
        ema_long = trial.suggest_int("ema_long", ema_medium + 10, 201)
        donchian_channel_length = trial.suggest_int("donchian_channel_length", 50, 200, step=50)
        natr_length = 100
        natr_multiplier = 2.0
        tp_default = trial.suggest_float("tp_default", 0.04, 0.05, step=0.01)
        # Suggest hyperparameters using the trial object
        total_amount_quote = 1000
        max_executors_per_side = 1
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15

        # Create the strategy configuration
        # Creating the instance of the configuration and the controller
        config = XGridTControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            total_amount_quote=Decimal(total_amount_quote),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            donchian_channel_length=donchian_channel_length,
            natr_length=natr_length,
            natr_multiplier=natr_multiplier,
            tp_default=tp_default
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class XGridTBacktestingTask(BaseTask):
    """Backtesting task for XGridT strategy optimization."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.root_path = task_config.get('root_path', "")
        self.connector_name = task_config.get("connector_name", "binance_perpetual")
        self.selected_pairs = task_config.get("selected_pairs", ["1000BONK-USDT"])
        self.lookback_days = task_config.get("lookback_days", 20)
        self.end_time_buffer_hours = task_config.get("end_time_buffer_hours", 6)
        self.resolution = task_config.get("resolution", "1m")
        self.n_trials = task_config.get("n_trials", 200)
        
        # Initialize optimizer (will be set up in setup)
        self.optimizer = None

    async def validate_prerequisites(self) -> bool:
        """Validate task prerequisites before execution."""
        try:
            if not self.root_path:
                logging.error("root_path not configured")
                return False
                
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
            self.optimizer = StrategyOptimizer(root_path=self.root_path)
            
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
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting XGridT backtesting for {len(self.selected_pairs)} pairs")
        
        try:
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
                    # Calculate time range
                    end_date = time.time() - (self.end_time_buffer_hours * 3600)
                    start_date = end_date - (self.lookback_days * 24 * 3600)
                    
                    logging.info(f"Optimizing strategy for {self.connector_name} {trading_pair}")
                    logging.info(f"Time range: {pd.to_datetime(start_date, unit='s')} to {pd.to_datetime(end_date, unit='s')}")
                    
                    # Create config generator
                    config_generator = XGridTConfigGenerator(
                        start_date=pd.to_datetime(start_date, unit="s"),
                        end_date=pd.to_datetime(end_date, unit="s"),
                        config={"connector_name": self.connector_name, "trading_pair": trading_pair}
                    )
                    
                    # Load candles cache
                    self.optimizer.load_candles_cache_by_connector_pair(
                        connector_name=self.connector_name, 
                        trading_pair=trading_pair
                    )
                    
                    # Run optimization
                    study_name = f"xgridt_{today_str}_{trading_pair.replace('-', '_')}"
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
                "strategy": "xgridt",
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"XGridT backtesting completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing XGridT backtesting: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ XGridTBacktestingTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Optimizations: {stats.get('optimizations_completed', 0)}")
        logging.info(f"  - Total trials: {stats.get('total_trials', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ XGridTBacktestingTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 XGridTBacktestingTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="xgridt_backtesting_test",
        enabled=True,
        task_class="tasks.backtesting.xgridt_backtesting_task.XGridTBacktestingTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=12.0
        ),
        config={
            "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            "connector_name": "binance_perpetual",
            "total_amount": 100,
            "lookback_days": 20,
            "end_time_buffer_hours": 6,
            "resolution": "1m",
            "n_trials": 10,  # Reduced for testing
            "selected_pairs": ['1000BONK-USDT']
        }
    )
    
    # Create and run task
    task = XGridTBacktestingTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Processed {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)} pairs")
        print(f"Optimizations: {stats.get('optimizations_completed', 0)}")
        print(f"Total trials: {stats.get('total_trials', 0)}")
        print(f"Strategy: {result.result_data.get('strategy', 'N/A')}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
