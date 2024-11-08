import asyncio
import logging
import time
from datetime import timedelta, datetime, timezone
from typing import Dict, Any
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)


class DbStatusTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)

    async def execute(self):
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')
        logging.info(f"{now} - Generating metrics report for TimescaleDB at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        timescale_client = TimescaleClient(
            host=self.config["db_host"],
            port=self.config["db_port"],
        )
        await timescale_client.connect()
        available_trading_pairs = await timescale_client.get_available_pairs()
        for i, market_info in enumerate(available_trading_pairs):
            connector_name, trading_pair = market_info
            logging.info(f"{now} - Fetching trades for {trading_pair} [{i} from {len(available_trading_pairs)}]")
            try:
                logging.info(f"{now} - Updated metrics for {trading_pair}")
                await timescale_client.append_metrics(connector_name=connector_name, trading_pair=trading_pair)
            except Exception as e:
                logging.exception(f"{now} - An error occurred during the data load for trading pair {trading_pair}:\n {e}")
                continue

        await timescale_client.close()


if __name__ == "__main__":
    config = {
        'db_host': 'localhost',
        'db_port': 5432,
    }

    task = DbStatusTask("Metrics Report", timedelta(hours=1), config)
    asyncio.run(task.execute())
