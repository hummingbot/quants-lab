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

    config = {
        'connector_name': 'binance_perpetual',
        'quote_asset': 'USDT',
        'intervals': ['1m', '3m', '5m', '15m', '1h'],
        'days_data_retention': 30,
        'min_notional_size': 10,
        'db_host': os.getenv("TIMESCALE_HOST", 'localhost'),
        'db_port': 5432,
    }

    candles_downloader_task = CandlesDownloaderTask("Metrics Report", timedelta(hours=4), config)
    orchestrator.add_task(candles_downloader_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
