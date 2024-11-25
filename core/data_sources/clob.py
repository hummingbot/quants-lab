import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bidict import bidict
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter, get_connector_class
from hummingbot.client.settings import AllConnectorSettings, ConnectorType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig

from core.data_sources.trades_feed.connectors.binance_perpetual import BinancePerpetualTradesFeed
from core.data_structures.candles import Candles
from core.data_structures.trading_rules import TradingRules

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL_MAPPING = {
    '1s': 's',  # seconds
    '1m': 'T',  # minutes
    '3m': '3T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    '1h': 'H',  # hours
    '2h': '2H',
    '4h': '4H',
    '6h': '6H',
    '12h': '12H',
    '1d': 'D',  # days
    '3d': '3D',
    '1w': 'W'  # weeks
}


class CLOBDataSource:
    CONNECTOR_TYPES = [ConnectorType.CLOB_SPOT, ConnectorType.CLOB_PERP, ConnectorType.Exchange, ConnectorType.Derivative]
    EXCLUDED_CONNECTORS = ["vega_perpetual", "hyperliquid_perpetual", "dydx_perpetual", "cube", "ndax",
                           "polkadex", "coinbase_advanced_trade", "kraken", "dydx_v4_perpetual", "hitbtc"]

    def __init__(self):
        logger.info("Initializing ClobDataSource")
        self.candles_factory = CandlesFactory()
        self.trades_feeds = {"binance_perpetual": BinancePerpetualTradesFeed()}
        self.conn_settings = AllConnectorSettings.get_connector_settings()
        self.connectors = {name: self.get_connector(name) for name, settings in self.conn_settings.items()
                           if settings.type in self.CONNECTOR_TYPES and name not in self.EXCLUDED_CONNECTORS and
                           "testnet" not in name}
        self._candles_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    @staticmethod
    def get_connector_config_map(connector_name: str):
        connector_config = AllConnectorSettings.get_connector_config_keys(connector_name)
        return {key: "" for key in connector_config.__fields__.keys() if key != "connector"}

    @property
    def candles_cache(self):
        return {key: Candles(candles_df=value, connector_name=key[0], trading_pair=key[1], interval=key[2])
                for key, value in self._candles_cache.items()}

    def get_candles_from_cache(self,
                               connector_name: str,
                               trading_pair: str,
                               interval: str):
        cache_key = (connector_name, trading_pair, interval)
        if cache_key in self._candles_cache:
            cached_df = self._candles_cache[cache_key]

            return Candles(candles_df=cached_df, connector_name=connector_name,
                           trading_pair=trading_pair, interval=interval)
        else:
            return None



    async def get_candles(self,
                          connector_name: str,
                          trading_pair: str,
                          interval: str,
                          start_time: int,
                          end_time: int,
                          from_trades: bool = False) -> Candles:
        cache_key = (connector_name, trading_pair, interval)

        if cache_key in self._candles_cache:
            cached_df = self._candles_cache[cache_key]
            cached_start_time = int(cached_df.index.min().timestamp())
            cached_end_time = int(cached_df.index.max().timestamp())

            if cached_start_time <= start_time and cached_end_time >= end_time:
                logger.info(
                    f"Using cached data for {connector_name} {trading_pair} {interval} from {start_time} to {end_time}")
                return Candles(candles_df=cached_df[(cached_df.index >= pd.to_datetime(start_time, unit='s')) &
                                                    (cached_df.index <= pd.to_datetime(end_time, unit='s'))],
                               connector_name=connector_name, trading_pair=trading_pair, interval=interval)
            else:
                if start_time < cached_start_time:
                    new_start_time = start_time
                    new_end_time = cached_start_time - 1
                else:
                    new_start_time = cached_end_time + 1
                    new_end_time = end_time
        else:
            new_start_time = start_time
            new_end_time = end_time

        try:
            logger.info(f"Fetching data for {connector_name} {trading_pair} {interval} from {new_start_time} to {new_end_time}")
            if from_trades:
                trades = await self.get_trades(connector_name, trading_pair, new_start_time, new_end_time)
                pandas_interval = self.convert_interval_to_pandas_freq(interval)
                candles_df = trades.resample(pandas_interval).agg({"price": "ohlc", "volume": "sum"}).ffill()
                candles_df.columns = candles_df.columns.droplevel(0)
                candles_df["timestamp"] = pd.to_numeric(candles_df.index) // 1e9
            else:
                candle = self.candles_factory.get_candle(CandlesConfig(
                    connector=connector_name,
                    trading_pair=trading_pair,
                    interval=interval
                ))
                candles_df = await candle.get_historical_candles(HistoricalCandlesConfig(
                    connector_name=connector_name,
                    trading_pair=trading_pair,
                    start_time=new_start_time,
                    end_time=new_end_time,
                    interval=interval
                ))
                candles_df.index = pd.to_datetime(candles_df.timestamp, unit='s')

            if cache_key in self._candles_cache:
                self._candles_cache[cache_key] = pd.concat(
                    [self._candles_cache[cache_key], candles_df]).drop_duplicates(keep='first').sort_index()
            else:
                self._candles_cache[cache_key] = candles_df

            return Candles(candles_df=self._candles_cache[cache_key][
                (self._candles_cache[cache_key].index >= pd.to_datetime(start_time, unit='s')) &
                (self._candles_cache[cache_key].index <= pd.to_datetime(end_time, unit='s'))],
                           connector_name=connector_name, trading_pair=trading_pair, interval=interval)
        except Exception as e:
            logger.error(f"Error fetching candles for {connector_name} {trading_pair} {interval}: {type(e).__name__} - {e}")
            raise

    async def get_candles_last_days(self,
                                    connector_name: str,
                                    trading_pair: str,
                                    interval: str,
                                    days: int,
                                    from_trades: bool = False) -> Candles:
        end_time = int(time.time())
        start_time = end_time - days * 24 * 60 * 60
        return await self.get_candles(connector_name, trading_pair, interval, start_time, end_time, from_trades)

    async def get_candles_batch_last_days(self, connector_name: str, trading_pairs: List, interval: str,
                                          days: int, batch_size: int = 10, sleep_time: float = 2.0):
        number_of_calls = (len(trading_pairs) // batch_size) + 1

        all_candles = []

        for i in range(number_of_calls):
            print(f"Batch {i + 1}/{number_of_calls}")
            start = i * batch_size
            end = (i + 1) * batch_size
            print(f"Start: {start}, End: {end}")
            end = min(end, len(trading_pairs))
            trading_pairs_batch = trading_pairs[start:end]

            tasks = [self.get_candles_last_days(
                connector_name=connector_name,
                trading_pair=trading_pair,
                interval=interval,
                days=days,
            ) for trading_pair in trading_pairs_batch]

            candles = await asyncio.gather(*tasks)
            all_candles.extend(candles)
            if i != number_of_calls - 1:
                logger.info(f"Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
        return all_candles

    def get_connector(self, connector_name: str):
        conn_setting = self.conn_settings.get(connector_name)
        if conn_setting is None:
            logger.error(f"Connector {connector_name} not found")
            raise ValueError(f"Connector {connector_name} not found")

        client_config_map = ClientConfigAdapter(ClientConfigMap())
        init_params = conn_setting.conn_init_parameters(
            trading_pairs=[],
            trading_required=False,
            api_keys=self.get_connector_config_map(connector_name),
            client_config_map=client_config_map,
        )
        connector_class = get_connector_class(connector_name)
        connector = connector_class(**init_params)
        return connector

    # TODO: ADD ORDER BOOK SNAPSHOT METHOD

    async def get_trading_rules(self, connector_name: str):
        connector = self.connectors.get(connector_name)
        await connector._update_trading_rules()
        return TradingRules(list(connector.trading_rules.values()))

    def dump_candles_cache(self, root_path: str = ""):
        candles_path = os.path.join(root_path, "data", "candles")
        os.makedirs(candles_path, exist_ok=True)
        for key, df in self._candles_cache.items():
            candles_path = os.path.join(root_path, "data", "candles")
            filename = os.path.join(candles_path, f"{key[0]}|{key[1]}|{key[2]}.parquet")
            df.to_parquet(
                filename,
                engine='pyarrow',
                compression='snappy',
                index=True
            )

        logger.info("Candles cache dumped")

    def load_candles_cache(self, root_path: str = ""):
        candles_path = os.path.join(root_path, "data", "candles")
        if not os.path.exists(candles_path):
            logger.warning(f"Path {candles_path} does not exist, skipping cache loading.")
            return

        all_files = os.listdir(candles_path)
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                connector_name, trading_pair, interval = file.split(".")[0].split("|")
                candles = pd.read_parquet(os.path.join(candles_path, file))
                candles.index = pd.to_datetime(candles.timestamp, unit='s')
                candles.index.name = None
                columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                for column in columns:
                    candles[column] = pd.to_numeric(candles[column])

                self._candles_cache[(connector_name, trading_pair, interval)] = candles
            except Exception as e:
                logger.error(f"Error loading {file}: {type(e).__name__} - {e}")

    async def get_trades(self, connector_name: str, trading_pair: str, start_time: int, end_time: int,
                         from_id: Optional[int] = None, max_trades_per_call: int = 1_000_000):
        async for chunk in self.trades_feeds[connector_name].get_historical_trades(
                trading_pair, start_time, end_time, from_id, max_trades_per_call
        ):
            yield chunk

    @staticmethod
    def convert_interval_to_pandas_freq(interval: str) -> str:
        """
        Converts a candle interval string to a pandas frequency string.
        """
        return INTERVAL_MAPPING.get(interval, 'T')

    @property
    def interval_to_seconds(self):
        return bidict({
            "1s": 1,
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "3d": 259200,
            "1w": 604800,
            "1M": 2592000
        })