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

    trades_downloader_task = TradesDownloaderTask(
        name="Trades Downloader Binance",
        frequency=timedelta(hours=5),
        config={
            'connector_name': 'binance_perpetual',
            'quote_asset': 'USDT',
            'min_notional_size': 10.0,
            'days_data_retention': 3
        })

    orchestrator.add_task(trades_downloader_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
