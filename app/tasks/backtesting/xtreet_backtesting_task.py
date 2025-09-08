import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

from core.backtesting.optimizer import StrategyOptimizer
from core.tasks import BaseTask, TaskContext
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting
from research_notebooks.xtreet_bb.xtreet_config_gen_simple import XtreetConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XtreetBacktestingTask(BaseTask):
    """Backtesting task for Xtreet strategy optimization."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.resolution = task_config.get("resolution", "1s")
        self.root_path = task_config.get('root_path', "")
        self.connector_name = task_config.get("connector_name", "binance_perpetual")
        self.selected_pairs = task_config.get("selected_pairs", ["1000BONK-USDT"])
        self.engine = task_config.get("engine", "sqlite")
        self.study_name = task_config.get("study_name", "xtreet_test")
        self.n_trials = task_config.get("n_trials", 50)
        
        # Optuna configuration
        self.optuna_config = task_config.get("optuna_config", {})
        
        # Initialize optimizer (will be set up in setup)
        self.storage_name = None

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
            # Prepare storage configuration
            kwargs = {
                "root_path": self.root_path,
                "db_host": self.optuna_config.get("host", "localhost"),
                "db_port": self.optuna_config.get("port", 5433),
                "db_user": self.optuna_config.get("user", "admin"),
                "db_pass": self.optuna_config.get("password", "admin"),
                "database_name": self.optuna_config.get("database", "optimization_database"),
            }
            
            self.storage_name = StrategyOptimizer.get_storage_name(
                engine=self.engine,
                **kwargs
            )
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connector: {self.connector_name}")
            logging.info(f"Resolution: {self.resolution}")
            logging.info(f"Trading pairs: {self.selected_pairs}")
            logging.info(f"Engine: {self.engine}")
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
        logging.info(f"Starting Xtreet backtesting for {len(self.selected_pairs)} pairs")
        
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
                    logging.info(f"Optimizing strategy for {self.connector_name} {trading_pair}")
                    
                    # Initialize custom backtester and optimizer for this pair
                    custom_backtester = XtreetBacktesting()
                    optimizer = StrategyOptimizer(
                        storage_name=self.storage_name,
                        resolution=self.resolution,
                        root_path=self.root_path,
                        custom_backtester=custom_backtester,
                    )
                    
                    # Load candles cache
                    optimizer.load_candles_cache_by_connector_pair(self.connector_name, trading_pair)
                    
                    # Get date range from loaded candles
                    candles_key = f"{self.connector_name}_{trading_pair}_{self.resolution}"
                    candles = optimizer._backtesting_engine._bt_engine.backtesting_data_provider.candles_feeds[candles_key]
                    start_date = candles.index.min()
                    end_date = candles.index.max()
                    
                    logging.info(f"Time range: {start_date} to {end_date}")
                    
                    # Create config generator
                    config_generator = XtreetConfigGenerator(start_date=start_date, end_date=end_date)
                    config_generator.trading_pair = trading_pair
                    
                    # Run optimization
                    study_name = f"{self.study_name}_{today_str}_{trading_pair.replace('-', '_')}"
                    await optimizer.optimize(
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
                "strategy": "xtreet",
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Xtreet backtesting completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing Xtreet backtesting: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ XtreetBacktestingTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Pairs: {stats.get('pairs_processed', 0)}/{stats.get('pairs_total', 0)}")
        logging.info(f"  - Optimizations: {stats.get('optimizations_completed', 0)}")
        logging.info(f"  - Total trials: {stats.get('total_trials', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ XtreetBacktestingTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 XtreetBacktestingTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Build optuna config from environment
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": int(os.getenv("OPTUNA_PORT", "5433")),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="xtreet_backtesting_test",
        enabled=True,
        task_class="tasks.backtesting.xtreet_backtesting_task.XtreetBacktestingTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=12.0
        ),
        config={
            "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            "resolution": "1s",
            "optuna_config": optuna_config,
            "connector_name": "binance_perpetual",
            "selected_pairs": ["1000BONK-USDT"],
            "engine": "sqlite",
            "study_name": "xtreet_test_4",
            "n_trials": 5  # Reduced for testing
        }
    )
    
    # Create and run task
    task = XtreetBacktestingTask(config)
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
