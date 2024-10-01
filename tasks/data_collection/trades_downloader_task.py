import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

from core.data_sources import CLOBDataSource
from core.task_base import BaseTask
from services.timescale_client import TimescaleClient

logging.basicConfig(level=logging.INFO)


class TradesDownloaderTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config['connector_name']
        self.quote_asset = config.get('quote_asset', "USDT")
        self.min_notional_size = Decimal(str(config.get('min_notional_size', 10.0)))
        self.clob = CLOBDataSource()

    async def execute(self):
        logging.info(f"Starting trades downloader for {self.connector_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - self.frequency
        logging.info(f"Start date: {start_time}, End date: {end_time}")
        logging.info(f"Quote asset: {self.quote_asset}, Min notional size: {self.min_notional_size}")

        timescale_client = TimescaleClient(
            host=self.config.get('db_host', "localhost"),
            port=self.config.get('db_port', 5432),
            user=self.config.get('postgres_user', "admin"),
            password=self.config.get('postgres_password', "admin"),
            database=self.config.get('db_name', "timescaledb")
        )
        await timescale_client.connect()
        await timescale_client.create_trades_table()

        trading_rules = await self.clob.get_trading_rules(self.connector_name)
        trading_pairs = trading_rules.filter_by_quote_asset(self.quote_asset) \
            .filter_by_min_notional_size(self.min_notional_size) \
            .get_all_trading_pairs()

        for i, trading_pair in enumerate(trading_pairs):
            logging.info(f"Fetching trades for {trading_pair} [{i} from {len(trading_pairs)}]")
            try:
                last_trade_id = await timescale_client.get_last_trade_id(self.connector_name, trading_pair)
                trades = await self.clob.get_trades(
                    self.connector_name,
                    trading_pair,
                    int(start_time.timestamp()),
                    int(end_time.timestamp()),
                    last_trade_id
                )

                if trades.empty:
                    logging.info(f"No new trades for {trading_pair}")
                    continue

                trades["connector_name"] = self.connector_name
                trades["trading_pair"] = trading_pair

                trades_data = trades[
                    ["id", "connector_name", "trading_pair", "timestamp", "price", "volume",
                     "sell_taker"]].values.tolist()

                await timescale_client.append_trades(trades_data)
                logging.info(f"Inserted {len(trades_data)} trades for {trading_pair}")

            except Exception as e:
                logging.exception(f"An error occurred during the data load for trading pair {trading_pair}:\n {e}")
                continue

        await timescale_client.close()


if __name__ == "__main__":
    config = {
        'connector_name': 'binance_perpetual',
        'quote_asset': 'USDT',
        'min_notional_size': 10.0,
        'db_host': 'localhost',
        'db_port': 5432,
        'db_name': 'timescaledb'
    }
    task = TradesDownloaderTask("Trades Downloader", timedelta(hours=1), config)
    asyncio.run(task.execute())
