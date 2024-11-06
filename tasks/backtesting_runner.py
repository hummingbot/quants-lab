import asyncio
import logging
import os
import sys
from datetime import timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.backtesting.xtreet_backtesting_task import BacktestingTask
    orchestrator = TaskOrchestrator()

    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "total_amount": 500,
        "activation_bounds": 0.002,
        "max_executors_per_side": 1,
        "cooldown_time": 0,
        "leverage": 20,
        "time_limit": 86400,  # 60 * 60 * 24
        "bb_lengths": [50, 100, 200],
        "bb_stds": [1.0, 1.4, 1.8, 2.0, 3.0],
        "intervals": ["1m", "5m", "15m"],
        "volume_threshold": 0.5,
        "volatility_threshold": 0.5,
        "ts_delta_multiplier": 0.2,
        "max_top_markets": 20,
        "max_dca_amount_ratio": 5,
        "backtesting_resolution": "1s",
        "min_distance_between_orders": 0.01,
        "max_ts_sl_ratio": 0.5,
        "lookback_days": 7,
        "resolution": "1s",
        "timescale_config": timescale_config,
        "optuna_config": optuna_config
    }

    backtesting_task = BacktestingTask("Backtesting", timedelta(hours=12), config)
    orchestrator.add_task(backtesting_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
