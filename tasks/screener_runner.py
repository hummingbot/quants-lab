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
    from tasks.data_collection.screener_task import MarketScreenerTask
    orchestrator = TaskOrchestrator()

    config = {
        'connector_name': 'binance_perpetual',
        'intervals': ['1m', '3m', '5m', '15m', '1h'],
        'db_host': os.getenv("TIMESCALE_HOST", 'localhost'),
    }

    screener_task = MarketScreenerTask("Screener Report", timedelta(hours=4), config)
    orchestrator.add_task(screener_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
