import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
load_dotenv()



class LocalCacheExportTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config["connector_name"]
        self.output_dir = Path(os.path.join(config.get("root_path", ""), config.get("output_dir", "data/candles")))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        logging.info(f"{now} - Starting candles export for {self.connector_name}")

        timescale_client = TimescaleClient(
            host=self.config["timescale_config"].get("db_host", "localhost"),
            port=self.config["timescale_config"].get("db_port", 5432),
            user=self.config["timescale_config"].get("db_user", "admin"),
            password=self.config["timescale_config"].get("db_password", "admin"),
            database=self.config["timescale_config"].get("db_name", "timescaledb")
        )
        await timescale_client.connect()

        available_candles = await timescale_client.get_available_candles()
        for connector_name, trading_pair, interval in available_candles:
            if connector_name != self.connector_name:
                continue

            logging.info(f"{now} - Exporting candles for {trading_pair} - {interval}")
            try:
                # Get all candles from the database
                candles = await timescale_client.get_all_candles(
                    connector_name=self.connector_name,
                        trading_pair=trading_pair,
                        interval=interval
                    )
                candles_df: pd.DataFrame = candles.data
                candles_df["timestamp"] = candles_df["timestamp"].apply(lambda x: x.timestamp())
                if candles_df.empty:
                    logging.info(f"{now} - No data found for {trading_pair} - {interval}")
                    continue
                filename = f"{self.connector_name}|{trading_pair}|{interval}.parquet"

                # Save to parquet with trading pair and interval in filename
                filepath = self.output_dir / filename

                # Save with compression
                candles_df.to_parquet(
                        filepath,
                        engine='pyarrow',
                        compression='snappy',
                        index=True
                    )
                    
                logging.info(f"{now} - Saved {len(candles_df)} candles to {filepath}")

            except Exception as e:
                logging.exception(f"{now} - Error exporting {trading_pair} - {interval}: {e}")
                continue

        await timescale_client.close()


async def main(config):
    candles_export_task = LocalCacheExportTask(
        name="Candles Exporter",
        frequency=timedelta(hours=1),
        config=config
    )
    await candles_export_task.execute()

if __name__ == "__main__":

    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "binance_perpetual",
        "output_dir": "data/candles",  # Use root path here
        "timescale_config": timescale_config
    }
    asyncio.run(main(config))

