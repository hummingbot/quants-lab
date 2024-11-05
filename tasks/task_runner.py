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
    from tasks.data_reporting.data_reporting_task import ReportGeneratorTask

    orchestrator = TaskOrchestrator()

    trades_downloader_config = {
        'connector_name': 'binance_perpetual',
        'quote_asset': 'USDT',
        'min_notional_size': 10.0,
        'days_data_retention': 10
    }
    trades_downloader_task = TradesDownloaderTask(
        "Trades Downloader Binance",
        timedelta(hours=5),
        trades_downloader_config)

    report_task = ReportGeneratorTask(
        name="Report Generator",
        config={
            "host": "localhost",
            "port": 5432,
            "user": "admin",
            "password": "admin",
            "database": "timescaledb",
        },
        frequency=timedelta(hours=24))

    orchestrator.add_task(trades_downloader_task)
    orchestrator.add_task(report_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
