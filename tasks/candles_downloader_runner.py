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
        "db_host": os.getenv("TIMESCALE_HOST", "localhost"),
        "db_port": os.getenv("TIMESCALE_PORT", 5432),
        "db_user": os.getenv("TIMESCALE_USER", "admin"),
        "db_password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "db_name": os.getenv("TIMESCALE_DB", "timescaledb")
    }

    config = {
        "connector_name": "binance_perpetual",
        "quote_asset": "USDT",
        "intervals": ["1m", "3m", "5m", "15m", "1h"],
        "days_data_retention": 120,
        "min_notional_size": 10,
        "selected_pairs": None,
        "timescale_config": timescale_config
    }

    candles_downloader_task = CandlesDownloaderTask("Candles Downloader", timedelta(hours=1), config)
    orchestrator.add_task(candles_downloader_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
