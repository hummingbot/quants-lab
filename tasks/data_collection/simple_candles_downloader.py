import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

from core.data_sources import CLOBDataSource
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class SimpleCandlesDownloader(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config["connector_name"]
        self.days_data_retention = config.get("days_data_retention", 7)
        self.intervals = config.get("intervals", ["1m"])
        self.trading_pairs = config.get("trading_pairs", ["BTC-USDT"])
        self.timescale_client = TimescaleClient(
            host=self.config["timescale_config"].get("db_host", "localhost"),
            port=self.config["timescale_config"].get("db_port", 5432),
            user=self.config["timescale_config"].get("db_user", "admin"),
            password=self.config["timescale_config"].get("db_password", "admin"),
            database=self.config["timescale_config"].get("db_name", "timescaledb")
        )

    async def execute(self):
        clob = CLOBDataSource()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        logging.info(f"{now} - Starting candles downloader for {self.connector_name}")
        end_time = datetime.now(timezone.utc)
        start_time = pd.Timestamp(time.time() - self.days_data_retention * 24 * 60 * 60,unit="s").tz_localize(timezone.utc).timestamp()
        logging.info(f"{now} - Start date: {start_time}, End date: {end_time}")
        logging.info(f"Trading pairs: {self.trading_pairs}")


        await self.timescale_client.connect()

        for i, trading_pair in enumerate(self.trading_pairs):
            for interval in self.intervals:
                logging.info(f"{now} - Fetching candles for {trading_pair} [{i} from {len(self.trading_pairs)}]")
                try:
                    table_name = self.timescale_client.get_ohlc_table_name(self.connector_name, trading_pair, interval)
                    await self.timescale_client.create_candles_table(table_name)
                    last_candle_timestamp = await self.timescale_client.get_last_candle_timestamp(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        interval=interval)
                    start_time = last_candle_timestamp if last_candle_timestamp else start_time
                    try:
                        candles = await clob.get_candles(
                            self.connector_name,
                            trading_pair,
                            interval,
                            int(start_time),
                            int(end_time.timestamp()),
                        )
                    except KeyError as e:
                        logging.error(f"{now} - Error fetching candles for {trading_pair}:\n {e}")
                        continue

                    if candles.data.empty:
                        logging.info(f"{now} - No new candles for {trading_pair}")
                        continue

                    await self.timescale_client.append_candles(table_name=table_name,candles=candles.data.values.tolist())
                    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    cutoff_timestamp = (today_start - timedelta(days=self.days_data_retention)).timestamp()
                    await self.timescale_client.delete_candles(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        interval=interval,
                        timestamp=cutoff_timestamp)
                    await asyncio.sleep(1)
                except Exception as e:
                    logging.exception(
                        f"{now} - An error occurred during the data load for trading pair {trading_pair}:\n {e}")
                    continue

        await self.timescale_client.close()


async def main(config):
    candles_downloader_task = SimpleCandlesDownloader(
        name="Candles Downloader",
        frequency=timedelta(hours=1),
        config=config
    )
    await candles_downloader_task.execute()
    await asyncio.sleep(1)
    await candles_downloader_task.execute()

if __name__ == "__main__":
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    config = {
        "connector_name": "binance_perpetual",
        "intervals": ["15m"],
        "days_data_retention": 7,
        "trading_pairs": ["BTC-USDT"],
        "timescale_config": timescale_config
    }
    asyncio.run(main(config))
