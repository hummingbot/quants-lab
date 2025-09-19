import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
from hummingbot.client.config.config_helpers import get_connector_class
from hummingbot.client.settings import AllConnectorSettings, ConnectorType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig

from core.data_sources.market_feeds.market_feeds_manager import MarketFeedsManager
from core.data_structures.candles import Candles
from core.data_structures.trading_rules import TradingRules
from core.data_paths import data_paths

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
                           "polkadex", "coinbase_advanced_trade", "kraken", "dydx_v4_perpetual", "hitbtc",
                           "hyperliquid", "dexalot", "vertex"]

    def __init__(self):
        logger.info("Initializing ClobDataSource")
        self.candles_factory = CandlesFactory()
        
        # Initialize market feeds manager (lazy loading of feeds)
        self.market_feeds_manager = MarketFeedsManager()
        
        # Cache for loaded feeds (lazy loading)
        self._trades_feeds_cache = {}
        self._oi_feeds_cache = {}
        
        self.conn_settings = AllConnectorSettings.get_connector_settings()
        self.connectors = {name: self.get_connector(name) for name, settings in self.conn_settings.items()
                           if settings.type in self.CONNECTOR_TYPES and name not in self.EXCLUDED_CONNECTORS and
                           "testnet" not in name}
        self._candles_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        
    def _get_trades_feed(self, connector_name: str):
        """Lazy-load trades feed for a specific connector."""
        if connector_name in self._trades_feeds_cache:
            return self._trades_feeds_cache[connector_name]
        
        # Map CLOB connector name to market feeds name
        market_feed_name = self._reverse_map_connector_name(connector_name)
        
        available_feeds = self.market_feeds_manager.available_feeds
        if market_feed_name not in available_feeds or "trades_feed" not in available_feeds[market_feed_name]:
            raise ValueError(f"No trades feed available for {connector_name}")
        
        try:
            trades_feed = self.market_feeds_manager.get_feed(
                connector_name=market_feed_name,
                feed_type="trades_feed"
            )
            self._trades_feeds_cache[connector_name] = trades_feed
            logger.info(f"Loaded trades feed for {connector_name}")
            return trades_feed
        except Exception as e:
            logger.error(f"Failed to load trades feed for {connector_name}: {e}")
            raise
    
    def _get_oi_feed(self, connector_name: str):
        """Lazy-load OI feed for a specific connector."""
        if connector_name in self._oi_feeds_cache:
            return self._oi_feeds_cache[connector_name]
        
        # Map CLOB connector name to market feeds name
        market_feed_name = self._reverse_map_connector_name(connector_name)
        
        available_feeds = self.market_feeds_manager.available_feeds
        if market_feed_name not in available_feeds or "oi_feed" not in available_feeds[market_feed_name]:
            raise ValueError(f"No OI feed available for {connector_name}")
        
        try:
            oi_feed = self.market_feeds_manager.get_feed(
                connector_name=market_feed_name,
                feed_type="oi_feed"
            )
            self._oi_feeds_cache[connector_name] = oi_feed
            logger.info(f"Loaded OI feed for {connector_name}")
            return oi_feed
        except Exception as e:
            logger.error(f"Failed to load OI feed for {connector_name}: {e}")
            raise

    def _reverse_map_connector_name(self, clob_name: str) -> str:
        """Map CLOBDataSource connector names back to market feeds names."""
        # Reverse mapping from CLOB names to market feeds names
        mapping = {
            "binance_perpetual": "binance",  # binance_perpetual in CLOB -> binance in market feeds
        }
        return mapping.get(clob_name, clob_name)

    @staticmethod
    def get_connector_config_map(connector_name: str):
        connector_config = AllConnectorSettings.get_connector_config_keys(connector_name)
        return {key: "" for key in connector_config.__fields__.keys() if key != "connector"}

    @property
    def trades_feeds(self):
        """Property for backward compatibility - returns available trades feeds."""
        result = {}
        for connector_name in self.connectors.keys():
            try:
                feed = self._get_trades_feed(connector_name)
                if feed:
                    result[connector_name] = feed
            except ValueError:
                pass  # No feed available for this connector
        return result
    
    @property
    def oi_feeds(self):
        """Property for backward compatibility - returns available OI feeds."""
        result = {}
        for connector_name in self.connectors.keys():
            try:
                feed = self._get_oi_feed(connector_name)
                if feed:
                    result[connector_name] = feed
            except ValueError:
                pass  # No feed available for this connector
        return result
    
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
                if candles_df is None:
                    return Candles(candles_df=pd.DataFrame(), connector_name=connector_name, trading_pair=trading_pair,
                                   interval=interval)
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

        init_params = conn_setting.conn_init_parameters(
            trading_pairs=[],
            trading_required=False,
            api_keys=self.get_connector_config_map(connector_name),
        )
        connector_class = get_connector_class(connector_name)
        connector = connector_class(**init_params)
        return connector

    # TODO: ADD ORDER BOOK SNAPSHOT METHOD

    async def get_trading_rules(self, connector_name: str):
        connector = self.connectors.get(connector_name)
        await connector._update_trading_rules()
        return TradingRules(list(connector.trading_rules.values()))

    def dump_candles_cache(self):
        # Use centralized data paths
        for key, df in self._candles_cache.items():
            filename = data_paths.get_candles_path(f"{key[0]}|{key[1]}|{key[2]}.parquet")
            df.to_parquet(
                filename,
                engine='pyarrow',
                compression='snappy',
                index=True
            )

        logger.info("Candles cache dumped")

    def load_candles_cache(self):
        # Use centralized data paths
        candles_path = data_paths.candles_dir
        if not candles_path.exists():
            logger.warning(f"Path {candles_path} does not exist, skipping cache loading.")
            return

        all_files = os.listdir(candles_path)
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                connector_name, trading_pair, interval = file.split(".")[0].split("|")
                candles = pd.read_parquet(candles_path / file)
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
                         from_id: Optional[int] = None):
        feed = self._get_trades_feed(connector_name)
        return await feed.get_historical_trades(trading_pair, start_time, end_time, from_id)

    async def get_oi(self, connector_name: str, trading_pair: str, interval: str, start_time: int, end_time: int,
                     limit: int = 500):
        """Get historical open interest data for a trading pair."""
        feed = self._get_oi_feed(connector_name)
        return await feed.get_historical_oi(trading_pair, interval, start_time, end_time, limit)

    async def get_oi_last_days(self, connector_name: str, trading_pair: str, interval: str, days: int, limit: int = 500):
        """Get open interest data for the last N days."""
        end_time = int(time.time())
        start_time = end_time - days * 24 * 60 * 60
        return await self.get_oi(connector_name, trading_pair, interval, start_time, end_time, limit)

    async def get_oi_batch_last_days(self, connector_name: str, trading_pairs: List, interval: str,
                                     days: int, batch_size: int = 5, sleep_time: float = 5.0, limit: int = 500):
        """Get open interest data for multiple trading pairs with batching."""
        number_of_calls = (len(trading_pairs) // batch_size) + 1

        all_oi_data = []

        for i in range(number_of_calls):
            print(f"OI Batch {i + 1}/{number_of_calls}")
            start = i * batch_size
            end = (i + 1) * batch_size
            print(f"Start: {start}, End: {end}")
            end = min(end, len(trading_pairs))
            trading_pairs_batch = trading_pairs[start:end]

            tasks = [self.get_oi_last_days(
                connector_name=connector_name,
                trading_pair=trading_pair,
                interval=interval,
                days=days,
                limit=limit
            ) for trading_pair in trading_pairs_batch]

            oi_data = await asyncio.gather(*tasks)
            all_oi_data.extend(oi_data)
            if i != number_of_calls - 1:
                logger.info(f"Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
        return all_oi_data

    async def filter_oi_supported_pairs(self, connector_name: str, trading_pairs: List, 
                                       interval: str = "1h", max_test_pairs: int = 50, 
                                       batch_size: int = 10) -> List[str]:
        """Filter trading pairs to only those that support OI data."""
        try:
            feed = self._get_oi_feed(connector_name)
            return await feed.filter_supported_pairs(
                trading_pairs, interval, batch_size=batch_size, max_test_pairs=max_test_pairs
            )
        except ValueError:
            logger.error(f"No OI feed available for connector {connector_name}")
            return []

    @staticmethod
    def convert_interval_to_pandas_freq(interval: str) -> str:
        """
        Converts a candle interval string to a pandas frequency string.
        """
        return INTERVAL_MAPPING.get(interval, 'T')

    async def get_funding_rate_history(self, 
                                     symbol: str, 
                                     start_time: Optional[int] = None,
                                     end_time: Optional[int] = None,
                                     limit: int = 1000) -> pd.DataFrame:
        """Get historical funding rates for a symbol"""
        connector = self.connectors.get("binance_perpetual")
        params = {"symbol": symbol, "limit": limit}
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = await connector._api_get(
            path_url="/fapi/v1/fundingRate",
            params=params,
            is_auth_required=False,
            limit_id="REQUEST_WEIGHT"
        )
        
        df = pd.DataFrame(response)
        if not df.empty:
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
            df["markPrice"] = df["markPrice"].astype(float)
            df.set_index("fundingTime", inplace=True)
        return df

    async def get_current_funding_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current funding rate info for a symbol or all symbols"""
        connector = self.connectors.get("binance_perpetual")
        response = await connector._orderbook_ds.get_funding_info(symbol)
        return response

    async def calculate_funding_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate funding rate metrics for a symbol"""
        # Get historical data for last 72 hours
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=72)).timestamp() * 1000)
        
        historical_rates = await self.get_funding_rate_history(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        current_info = await self.get_current_funding_info(symbol)
        
        if historical_rates.empty:
            return {}

        metrics = {
            "symbol": symbol,
            "current_funding_rate": float(current_info[symbol]["lastFundingRate"]),
            "next_funding_time": current_info[symbol]["nextFundingTime"],
            "mark_price": float(current_info[symbol]["markPrice"]),
            "avg_funding_24h": float(historical_rates.tail(8)["fundingRate"].mean()),  # 8 funding intervals = 24h
            "avg_funding_72h": float(historical_rates["fundingRate"].mean()),
            "funding_history": historical_rates.reset_index().to_dict(orient="records"),
            "updated_at": datetime.now()
        }
        
        return metrics
