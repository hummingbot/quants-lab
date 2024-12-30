import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from dotenv import load_dotenv

from core.data_sources.external_data.coin_glass import CoinGlassDataFeed
from core.services.coin_glass_data_client import CoinGlassClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
load_dotenv()


class CoinGlassDataDownloaderTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.days_data_retention = config.get("days_data_retention", 7)
        self.start_time = time.time() - self.days_data_retention * 24 * 60 * 60
        self.trading_pairs = config.get("trading_pairs", ["BTC-USDT"])
        self.intervals = config.get("interval", ["1d"])
        self.data_feed = CoinGlassDataFeed(config["api_key"])
        self.end_point = config.get("end_point", "liquidation_aggregated_history")
        self.connector_name = config.get("connector_name", "bybit_perpetual")
        self.limit = config.get("limit", 1000)

    async def execute(self):
        logging.info(
            f"{self.now()} - Starting data downloader for {self.end_point} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        end_time = datetime.now(timezone.utc)
        start_time = datetime.fromtimestamp(self.start_time, tz=timezone.utc)
        logging.info(
            f"{self.now()} - Start date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End date: {end_time}"
        )
        logging.info(f"{self.now()} - Trading pairs: {self.trading_pairs}")

        coinglass_client = CoinGlassClient(
            host=self.config["timescale_config"]["host"],
            port=self.config["timescale_config"]["port"],
            user=self.config["timescale_config"]["user"],
            password=self.config["timescale_config"]["password"],
            database=self.config["timescale_config"]["database"],
        )
        await coinglass_client.connect()

        for i, trading_pair in enumerate(self.trading_pairs):
            for interval in self.intervals:
                logging.info(
                    f"{self.now()} - Fetching {self.end_point} data for {trading_pair} [{i} from {len(self.trading_pairs)}]"
                )
                try:
                    get_table_name = getattr(
                        coinglass_client, f"get_{self.end_point}_table_name"
                    )
                    table_name = get_table_name(
                        trading_pair=trading_pair,
                        interval=interval,
                        connector_name=self.connector_name,
                    )
                    data = await self.data_feed.get_endpoint(
                        self.end_point,
                        trading_pair,
                        interval,
                        int(start_time.timestamp()),
                        int(end_time.timestamp()),
                        self.connector_name,
                        self.limit,
                    )

                    if not data:
                        logging.info(f"{self.now()} - No new data for {trading_pair}")
                        continue

                    # data = data.values.tolist()
                    append_data = getattr(coinglass_client, f"append_{self.end_point}")
                    data = [tuple(x.values()) for x in data]

                    await append_data(table_name, data)
                    logging.info(
                        f"{self.now()} - Inserted {len(data)} {self.end_point} data for {trading_pair}"
                    )

                except Exception as e:
                    logging.exception(
                        f"{self.now()} - An error occurred during the data load for trading pair {trading_pair}:\n {e}"
                    )
                    continue

        await coinglass_client.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")


if __name__ == "__main__":
    config = {
        "connector_name": "binance_perpetual",
        "quote_asset": "USDT",
        "min_notional_size": 10.0,
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "timescaledb",
    }

    task = CoinGlassDataDownloaderTask("Downloader", timedelta(hours=1), config)
    asyncio.run(task.execute())
