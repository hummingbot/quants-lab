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
        self._oi_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        
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
        
        # Track what data needs to be fetched
        ranges_to_fetch = []
        
        if cache_key in self._candles_cache:
            cached_df = self._candles_cache[cache_key]
            cached_start_time = int(cached_df.index.min().timestamp())
            cached_end_time = int(cached_df.index.max().timestamp())
            
            # Case 1: All requested data is in cache
            if cached_start_time <= start_time and cached_end_time >= end_time:
                logger.info(
                    f"Using cached data for {connector_name} {trading_pair} {interval} from {start_time} to {end_time}")
                return Candles(candles_df=cached_df[(cached_df.index >= pd.to_datetime(start_time, unit='s')) &
                                                    (cached_df.index <= pd.to_datetime(end_time, unit='s'))],
                               connector_name=connector_name, trading_pair=trading_pair, interval=interval)
            
            # Case 2: Partial overlap - determine what's missing
            if start_time < cached_start_time and end_time > cached_end_time:
                # Need data on both sides
                ranges_to_fetch.append((start_time, cached_start_time - 1))
                ranges_to_fetch.append((cached_end_time + 1, end_time))
            elif start_time < cached_start_time:
                # Need data before cache
                ranges_to_fetch.append((start_time, min(cached_start_time - 1, end_time)))
            elif end_time > cached_end_time:
                # Need data after cache
                ranges_to_fetch.append((max(cached_end_time + 1, start_time), end_time))
            else:
                # Requested range doesn't overlap with cache at all
                ranges_to_fetch.append((start_time, end_time))
        else:
            # No cache, fetch everything
            ranges_to_fetch.append((start_time, end_time))
        
        # Fetch missing data ranges
        for fetch_start, fetch_end in ranges_to_fetch:
            try:
                logger.info(f"Fetching data for {connector_name} {trading_pair} {interval} from {fetch_start} to {fetch_end}")
                
                if from_trades:
                    trades = await self.get_trades(connector_name, trading_pair, fetch_start, fetch_end)
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
                        start_time=fetch_start,
                        end_time=fetch_end,
                        interval=interval
                    ))
                    if candles_df is None or candles_df.empty:
                        continue
                    candles_df.index = pd.to_datetime(candles_df.timestamp, unit='s')
                
                # Update cache with new data
                if cache_key in self._candles_cache:
                    self._candles_cache[cache_key] = pd.concat(
                        [self._candles_cache[cache_key], candles_df]).drop_duplicates(keep='first').sort_index()
                else:
                    self._candles_cache[cache_key] = candles_df
                    
            except Exception as e:
                logger.error(f"Error fetching candles for {connector_name} {trading_pair} {interval} "
                           f"from {fetch_start} to {fetch_end}: {type(e).__name__} - {e}")
                # Continue with partial data if one range fails
                if cache_key not in self._candles_cache:
                    raise
        
        # Return the requested slice from cache
        if cache_key in self._candles_cache:
            result_df = self._candles_cache[cache_key][
                (self._candles_cache[cache_key].index >= pd.to_datetime(start_time, unit='s')) &
                (self._candles_cache[cache_key].index <= pd.to_datetime(end_time, unit='s'))
            ]
            return Candles(candles_df=result_df, connector_name=connector_name, 
                          trading_pair=trading_pair, interval=interval)
        else:
            # No data could be fetched
            return Candles(candles_df=pd.DataFrame(), connector_name=connector_name, 
                          trading_pair=trading_pair, interval=interval)

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
    
    def dump_oi_cache(self):
        # Use centralized data paths for OI
        oi_path = data_paths.oi_dir
        oi_path.mkdir(parents=True, exist_ok=True)
        
        for key, df in self._oi_cache.items():
            filename = oi_path / f"{key[0]}|{key[1]}|{key[2]}.parquet"
            df.to_parquet(
                filename,
                engine='pyarrow',
                compression='snappy',
                index=True
            )
        
        logger.info(f"OI cache dumped - {len(self._oi_cache)} files")

    def load_candles_cache(self, 
                          connector_name: Optional[str] = None,
                          trading_pair: Optional[str] = None, 
                          interval: Optional[str] = None):
        """
        Load candles from cache with optional filtering.
        
        Args:
            connector_name: Optional filter by connector name
            trading_pair: Optional filter by trading pair
            interval: Optional filter by interval
        """
        # Use centralized data paths
        candles_path = data_paths.candles_dir
        if not candles_path.exists():
            logger.warning(f"Path {candles_path} does not exist, skipping cache loading.")
            return

        all_files = os.listdir(candles_path)
        loaded_count = 0
        skipped_count = 0
        
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                file_connector, file_pair, file_interval = file.split(".")[0].split("|")
                
                # Apply filters if provided
                if connector_name and file_connector != connector_name:
                    skipped_count += 1
                    continue
                if trading_pair and file_pair != trading_pair:
                    skipped_count += 1
                    continue
                if interval and file_interval != interval:
                    skipped_count += 1
                    continue
                
                candles = pd.read_parquet(candles_path / file)
                candles.index = pd.to_datetime(candles.timestamp, unit='s')
                candles.index.name = None
                columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                for column in columns:
                    candles[column] = pd.to_numeric(candles[column])

                self._candles_cache[(file_connector, file_pair, file_interval)] = candles
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading {file}: {type(e).__name__} - {e}")
        
        logger.info(f"Loaded {loaded_count} candles cache files, skipped {skipped_count} files due to filters")

    async def get_trades(self, connector_name: str, trading_pair: str, start_time: int, end_time: int,
                         from_id: Optional[int] = None):
        feed = self._get_trades_feed(connector_name)
        return await feed.get_historical_trades(trading_pair, start_time, end_time, from_id)

    async def get_oi(self, connector_name: str, trading_pair: str, interval: str, start_time: int, end_time: int,
                     limit: int = 500):
        """Get historical open interest data for a trading pair with caching."""
        cache_key = (connector_name, trading_pair, interval)
        
        # Check cache first
        if cache_key in self._oi_cache:
            cached_df = self._oi_cache[cache_key]
            if not cached_df.empty:
                # Filter cached data for requested time range
                mask = (cached_df.index >= pd.to_datetime(start_time, unit='s')) & \
                       (cached_df.index <= pd.to_datetime(end_time, unit='s'))
                filtered_df = cached_df[mask]
                
                # If we have some data in the requested range, check if we need more
                if not filtered_df.empty:
                    cached_start = int(cached_df.index.min().timestamp())
                    cached_end = int(cached_df.index.max().timestamp())
                    
                    # If all requested data is in cache, return it
                    if cached_start <= start_time and cached_end >= end_time:
                        logger.info(f"Using cached OI data for {connector_name} {trading_pair} {interval}")
                        return filtered_df
        
        # Fetch from feed if not in cache or need more data
        try:
            feed = self._get_oi_feed(connector_name)
            oi_df = await feed.get_historical_oi(trading_pair, interval, start_time, end_time, limit)
            
            # Update cache
            if not oi_df.empty:
                if cache_key in self._oi_cache:
                    # Merge with existing cache data
                    self._oi_cache[cache_key] = pd.concat(
                        [self._oi_cache[cache_key], oi_df]
                    ).drop_duplicates(keep='first').sort_index()
                else:
                    self._oi_cache[cache_key] = oi_df
                    
                logger.info(f"Cached {len(oi_df)} OI records for {connector_name} {trading_pair} {interval}")
            
            return oi_df
        except Exception as e:
            logger.error(f"Error fetching OI data for {connector_name} {trading_pair}: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame()

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

    async def get_order_book_snapshot(self, connector_name: str, trading_pair: str, depth: int = 10) -> Dict:
        """
        Get order book snapshot for a trading pair.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair to get order book for
            depth: Number of bid/ask levels to return (default: 10)

        Returns:
            Dictionary containing bid and ask data with timestamp
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            # Access the order book data source
            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds

                # Get fresh order book using the data source method
                order_book = await orderbook_ds.get_new_order_book(trading_pair)
                snapshot = order_book.snapshot

                result = {
                    "trading_pair": trading_pair,
                    "bids": snapshot[0].loc[:(depth - 1), ["price", "amount"]].values.tolist(),
                    "asks": snapshot[1].loc[:(depth - 1), ["price", "amount"]].values.tolist(),
                    "timestamp": time.time()
                }

                logger.debug(f"Retrieved order book snapshot for {connector_name}/{trading_pair}")
                return result
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting order book snapshot for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_order_book_price_for_volume(self, connector_name: str, trading_pair: str,
                                             is_buy: bool, volume: float) -> Dict:
        """
        Get the price needed to fill a specific volume.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair
            is_buy: True for buy side, False for sell side
            volume: Volume to query

        Returns:
            Dictionary with result_price and result_volume
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds
                order_book = await orderbook_ds.get_new_order_book(trading_pair)

                result = order_book.get_price_for_volume(is_buy, volume)

                return {
                    "trading_pair": trading_pair,
                    "is_buy": is_buy,
                    "query_volume": volume,
                    "result_price": float(result.result_price) if result.result_price else None,
                    "result_volume": float(result.result_volume) if result.result_volume else None,
                    "timestamp": time.time()
                }
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting price for volume for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_order_book_volume_for_price(self, connector_name: str, trading_pair: str,
                                             is_buy: bool, price: float) -> Dict:
        """
        Get the volume available at a specific price.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair
            is_buy: True for buy side, False for sell side
            price: Price to query

        Returns:
            Dictionary with result_volume and result_price
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds
                order_book = await orderbook_ds.get_new_order_book(trading_pair)

                result = order_book.get_volume_for_price(is_buy, price)

                return {
                    "trading_pair": trading_pair,
                    "is_buy": is_buy,
                    "query_price": price,
                    "result_volume": float(result.result_volume) if result.result_volume else None,
                    "result_price": float(result.result_price) if result.result_price else None,
                    "timestamp": time.time()
                }
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting volume for price for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_order_book_price_for_quote_volume(self, connector_name: str, trading_pair: str,
                                                   is_buy: bool, quote_volume: float) -> Dict:
        """
        Get the price needed to fill a specific quote volume.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair
            is_buy: True for buy side, False for sell side
            quote_volume: Quote volume to query

        Returns:
            Dictionary with result_price and result_volume
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds
                order_book = await orderbook_ds.get_new_order_book(trading_pair)

                result = order_book.get_price_for_quote_volume(is_buy, quote_volume)

                return {
                    "trading_pair": trading_pair,
                    "is_buy": is_buy,
                    "query_quote_volume": quote_volume,
                    "result_price": float(result.result_price) if result.result_price else None,
                    "result_volume": float(result.result_volume) if result.result_volume else None,
                    "timestamp": time.time()
                }
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting price for quote volume for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_order_book_quote_volume_for_price(self, connector_name: str, trading_pair: str,
                                                   is_buy: bool, quote_price: float) -> Dict:
        """
        Get the quote volume available at a specific price.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair
            is_buy: True for buy side, False for sell side
            quote_price: Price to query

        Returns:
            Dictionary with result_quote_volume and crossed_book indicator
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds
                order_book = await orderbook_ds.get_new_order_book(trading_pair)

                result = order_book.get_quote_volume_for_price(is_buy, quote_price)

                # Check if quote crosses the book
                if result.result_volume is None or result.result_price is None:
                    snapshot = order_book.snapshot
                    best_bid = float(snapshot[0].iloc[0]["price"]) if not snapshot[0].empty else None
                    best_ask = float(snapshot[1].iloc[0]["price"]) if not snapshot[1].empty else None
                    mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None

                    crossed_reason = None
                    suggested_price = None

                    if is_buy:
                        if best_ask and quote_price > best_ask:
                            crossed_reason = f"Buy price {quote_price} exceeds best ask {best_ask}"
                            suggested_price = best_ask
                        elif best_bid and quote_price < best_bid:
                            crossed_reason = f"Buy price {quote_price} below best bid {best_bid} - no liquidity available"
                            suggested_price = best_bid
                    else:
                        if best_bid and quote_price < best_bid:
                            crossed_reason = f"Sell price {quote_price} below best bid {best_bid}"
                            suggested_price = best_bid
                        elif best_ask and quote_price > best_ask:
                            crossed_reason = f"Sell price {quote_price} above best ask {best_ask} - no liquidity available"
                            suggested_price = best_ask

                    return {
                        "trading_pair": trading_pair,
                        "is_buy": is_buy,
                        "query_price": quote_price,
                        "result_volume": None,
                        "result_quote_volume": None,
                        "crossed_book": True,
                        "crossed_reason": crossed_reason,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": mid_price,
                        "suggested_price": suggested_price,
                        "timestamp": time.time()
                    }

                return {
                    "trading_pair": trading_pair,
                    "is_buy": is_buy,
                    "query_price": quote_price,
                    "result_quote_volume": float(result.result_volume) if result.result_volume else None,
                    "crossed_book": False,
                    "timestamp": time.time()
                }
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting quote volume for price for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_order_book_vwap(self, connector_name: str, trading_pair: str,
                                 is_buy: bool, volume: float) -> Dict:
        """
        Get the VWAP (Volume Weighted Average Price) for a specific volume.

        Args:
            connector_name: Name of the connector
            trading_pair: Trading pair
            is_buy: True for buy side, False for sell side
            volume: Volume to query

        Returns:
            Dictionary with average_price and result_volume
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            if hasattr(connector, '_orderbook_ds') and connector._orderbook_ds:
                orderbook_ds = connector._orderbook_ds
                order_book = await orderbook_ds.get_new_order_book(trading_pair)

                result = order_book.get_vwap_for_volume(is_buy, volume)

                return {
                    "trading_pair": trading_pair,
                    "is_buy": is_buy,
                    "query_volume": volume,
                    "average_price": float(result.result_price) if result.result_price else None,
                    "result_volume": float(result.result_volume) if result.result_volume else None,
                    "timestamp": time.time()
                }
            else:
                raise ValueError(f"Order book data source not available for {connector_name}")

        except Exception as e:
            logger.error(f"Error getting VWAP for {connector_name}/{trading_pair}: {e}")
            raise

    async def get_prices(self, connector_name: str, trading_pairs: List[str]) -> Dict[str, float]:
        """
        Get current prices for specified trading pairs.

        Args:
            connector_name: Name of the connector
            trading_pairs: List of trading pairs to get prices for

        Returns:
            Dictionary mapping trading pairs to their current prices
        """
        try:
            connector = self.connectors.get(connector_name)
            if not connector:
                raise ValueError(f"Connector {connector_name} not found")

            # Get last traded prices
            prices = await connector.get_last_traded_prices(trading_pairs)

            # Convert Decimal to float for JSON serialization
            result = {pair: float(price) for pair, price in prices.items()}

            logger.debug(f"Retrieved prices for {connector_name}: {len(result)} pairs")
            return result

        except Exception as e:
            logger.error(f"Error getting prices for {connector_name}: {e}")
            raise

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
