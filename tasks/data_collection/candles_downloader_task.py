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


class CandlesDownloaderTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config["connector_name"]
        self.days_data_retention = config.get("days_data_retention", 7)
        self.intervals = config.get("intervals", ["1m"])
        self.quote_asset = config.get("quote_asset", "USDT")
        self.min_notional_size = Decimal(str(config.get("min_notional_size", 10.0)))
        self.clob = CLOBDataSource()

    async def execute(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        logging.info(
            f"{now} - Starting candles downloader for {self.connector_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        end_time = datetime.now(timezone.utc)
        start_time = pd.Timestamp(time.time() - self.days_data_retention * 24 * 60 * 60,
                                  unit="s").tz_localize(timezone.utc).timestamp()
        logging.info(f"{now} - Start date: {start_time}, End date: {end_time}")
        logging.info(f"{now} - Quote asset: {self.quote_asset}, Min notional size: {self.min_notional_size}")

        timescale_client = TimescaleClient(
            host=self.config["timescale_config"].get("db_host", "localhost"),
            port=self.config["timescale_config"].get("db_port", 5432),
            user=self.config["timescale_config"].get("db_user", "admin"),
            password=self.config["timescale_config"].get("db_password", "admin"),
            database=self.config["timescale_config"].get("db_name", "timescaledb")
        )
        await timescale_client.connect()

        trading_rules = await self.clob.get_trading_rules(self.connector_name)
        trading_pairs = trading_rules.get_all_trading_pairs()
        for i, trading_pair in enumerate(trading_pairs):
            for interval in self.intervals:
                logging.info(f"{now} - Fetching candles for {trading_pair} [{i} from {len(trading_pairs)}]")
                try:
                    table_name = timescale_client.get_ohlc_table_name(self.connector_name, trading_pair, interval)
                    await timescale_client.create_candles_table(table_name)
                    last_candle_timestamp = await timescale_client.get_last_candle_timestamp(
                        connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        interval=interval)
                    start_time = last_candle_timestamp if last_candle_timestamp else start_time
                    candles = await self.clob.get_candles(
                        self.connector_name,
                        trading_pair,
                        interval,
                        int(start_time),
                        int(end_time.timestamp()),
                    )

                    if candles.data.empty:
                        logging.info(f"{now} - No new trades for {trading_pair}")
                        continue

                    await timescale_client.append_candles(table_name=table_name,
                                                          candles=candles.data.values.tolist())
                    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    cutoff_timestamp = (today_start - timedelta(days=self.days_data_retention)).timestamp()
                    await timescale_client.delete_candles(connector_name=self.connector_name, trading_pair=trading_pair,
                                                          interval=interval,
                                                          timestamp=cutoff_timestamp)
                    await asyncio.sleep(1)
                except Exception as e:
                    logging.exception(
                        f"{now} - An error occurred during the data load for trading pair {trading_pair}:\n {e}")
                    continue

        await timescale_client.close()


async def main(config):
    candles_downloader_task = CandlesDownloaderTask(
        name="Candles Downloader",
        frequency=timedelta(hours=1),
        config=config
    )
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
        "quote_asset": "USDT",
        "intervals": ["15m", "1h"],
        "days_data_retention": 30,
        "min_notional_size": 10,
        "timescale_config": timescale_config
    }
    asyncio.run(main(config))
