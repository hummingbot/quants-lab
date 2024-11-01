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

    # from tasks.data_collection.trades_downloader_task import TradesDownloaderTask
    # from tasks.data_reporting.data_reporting_task import ReportGeneratorTask
    orchestrator = TaskOrchestrator()

    backtesting_config = {
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
        "TIMESCALE_HOST": "63.250.52.93",
        "TIMESCALE_PORT": 5432,
        "TIMESCALE_USER": "admin",
        "TIMESCALE_PASSWORD": "admin",
        "OPTUNA_HOST": "localhost",
        "OPTUNA_DOCKER_PORT": 5432,
        "OPTUNA_USER": "admin",
        "OPTUNA_PASSWORD": "admin",
    }

    backtesting_task = BacktestingTask("Backtesting", timedelta(hours=12), backtesting_config)

    # trades_downloader_config = {
    #     'connector_name': 'binance_perpetual',
    #     'quote_asset': 'USDT',
    #     'min_notional_size': 10.0,
    #     'days_data_retention': 10
    # }
    # trades_downloader_task = TradesDownloaderTask("Trades Downloader Binance", timedelta(hours=5),
    #                                               trades_downloader_config)
    #
    # report_task = ReportGeneratorTask(
    #     name="Report Generator",
    #     config={
    #         "host": "localhost",
    #         "port": 5432,
    #         "user": "admin",
    #         "password": "admin",
    #         "database": "timescaledb",
    #     },
    #     frequency=timedelta(hours=12))

    orchestrator.add_task(backtesting_task)
    # orchestrator.add_task(trades_downloader_task)
    # orchestrator.add_task(report_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
