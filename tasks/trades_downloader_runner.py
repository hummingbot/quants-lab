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
    from tasks.data_collection.trades_downloader_task import TradesDownloaderTask

    orchestrator = TaskOrchestrator()

    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }

    trades_downloader_task = TradesDownloaderTask(
        name="Trades Downloader Binance",
        config={
            "timescale_config": timescale_config,
            "connector_name": "binance_perpetual",
            "quote_asset": "USDT",
            "min_notional_size": 10.0,
            "days_data_retention": 10,
            "max_trades_per_call": 1_000_000,
            "selected_pairs": None
        },
        frequency=timedelta(hours=5))

    orchestrator.add_task(trades_downloader_task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
