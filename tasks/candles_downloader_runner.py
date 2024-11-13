import asyncio
import logging
import os
import sys
from datetime import timedelta

from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.data_collection.candles_downloader_task import CandlesDownloaderTask
    orchestrator = TaskOrchestrator()
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }

    config = {
        "connector_name": "binance_perpetual",
        "quote_asset": "USDT",
        "intervals": ["1m", "3m", "5m", "15m", "1h"],
        "days_data_retention": 30,
        "min_notional_size": 10,
        "timescale_config": timescale_config
    }

    candles_downloader_task = CandlesDownloaderTask("Metrics Report", timedelta(hours=4), config)
    orchestrator.add_task(candles_downloader_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
