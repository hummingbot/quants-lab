import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, TypeVar, Generic, Tuple

import pandas as pd

from .connector_base import ConnectorBase
from ...data_paths import data_paths

ConnectorT = TypeVar('ConnectorT', bound=ConnectorBase)


class TradesFeedBase(ABC, Generic[ConnectorT]):
    # Class variables for rate limiting - to be overridden by subclasses
    RATE_LIMITS: Dict[str, tuple] = {}  # limit_id -> (max_requests, time_window_seconds)
    ENDPOINT_WEIGHTS: Dict[str, int] = {}  # endpoint -> weight
    LIMIT_ID: str = ""  # Main limit ID for this feed
    
    def __init__(self, connector: ConnectorT):
        self.connector = connector
        self._setup_rate_limits()
        self._trades_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        self._cache_dir = data_paths.trades_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_rate_limits(self):
        """Register rate limits from class variables."""
        for limit_id, (max_requests, time_window) in self.RATE_LIMITS.items():
            self.connector.register_rate_limit(limit_id, max_requests, time_window)

    def _get_cache_key(self, trading_pair: str, start_time: int, end_time: int) -> Tuple[str, int, int]:
        """Generate cache key for trades data."""
        return (trading_pair, start_time, end_time)
    
    def _get_cache_filename(self, trading_pair: str, start_time: int, end_time: int) -> Path:
        """Generate cache filename for trades data."""
        connector_name = self.connector.__class__.__name__.lower().replace('base', '').replace('connector', '')
        filename = f"{connector_name}_{trading_pair.replace('-', '')}_{start_time}_{end_time}.parquet"
        return self._cache_dir / filename
    
    def _load_trades_from_cache(self, trading_pair: str, start_time: int, end_time: int) -> Optional[pd.DataFrame]:
        """Load trades from cache if available."""
        cache_key = self._get_cache_key(trading_pair, start_time, end_time)
        
        # Check memory cache first
        if cache_key in self._trades_cache:
            self.connector.logger.info(f"Found trades in memory cache for {trading_pair}")
            return self._trades_cache[cache_key]
        
        # Check disk cache
        cache_file = self._get_cache_filename(trading_pair, start_time, end_time)
        if cache_file.exists():
            try:
                self.connector.logger.info(f"Loading trades from disk cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                
                # Store in memory cache
                self._trades_cache[cache_key] = df
                return df
            except Exception as e:
                self.connector.logger.warning(f"Failed to load trades from cache {cache_file}: {e}")
        
        return None
    
    def _save_trades_to_cache(self, trading_pair: str, start_time: int, end_time: int, trades_df: pd.DataFrame):
        """Save trades to cache."""
        if trades_df.empty:
            return
            
        cache_key = self._get_cache_key(trading_pair, start_time, end_time)
        
        # Save to memory cache
        self._trades_cache[cache_key] = trades_df.copy()
        
        # Save to disk cache
        cache_file = self._get_cache_filename(trading_pair, start_time, end_time)
        try:
            trades_df.to_parquet(
                cache_file,
                engine='pyarrow',
                compression='snappy',
                index=True
            )
            self.connector.logger.info(f"Saved {len(trades_df)} trades to cache: {cache_file}")
        except Exception as e:
            self.connector.logger.warning(f"Failed to save trades to cache {cache_file}: {e}")
    
    def _check_cache_coverage(self, trading_pair: str, start_time: int, end_time: int) -> Tuple[bool, Optional[int], Optional[int]]:
        """Check if we have partial cache coverage and determine what ranges need fetching."""
        cached_df = self._load_trades_from_cache(trading_pair, start_time, end_time)
        
        if cached_df is None or cached_df.empty:
            return False, start_time, end_time
        
        cached_start = int(cached_df.index.min().timestamp())
        cached_end = int(cached_df.index.max().timestamp())
        
        # Check if cache fully covers the requested range
        if cached_start <= start_time and cached_end >= end_time:
            self.connector.logger.info(f"Cache fully covers requested range for {trading_pair}")
            return True, None, None
        
        # Determine what ranges need to be fetched
        fetch_start = None
        fetch_end = None
        
        if start_time < cached_start:
            fetch_start = start_time
            fetch_end = cached_start - 1
        elif end_time > cached_end:
            fetch_start = cached_end + 1
            fetch_end = end_time
        
        return False, fetch_start, fetch_end

    async def get_historical_trades(self, trading_pair: str, start_time: int, end_time: Optional[int] = None,
                                    from_id: Optional[int] = None):
        """Get historical trades for a trading pair within a time range with caching."""
        if not end_time:
            end_time = int(time.time())
        
        # Check cache coverage
        fully_cached, fetch_start, fetch_end = self._check_cache_coverage(trading_pair, start_time, end_time)
        
        if fully_cached:
            # Return cached data filtered to requested range
            cached_df = self._load_trades_from_cache(trading_pair, start_time, end_time)
            return cached_df[
                (cached_df.index >= pd.to_datetime(start_time, unit='s')) &
                (cached_df.index <= pd.to_datetime(end_time, unit='s'))
            ]
        
        # Need to fetch new data
        if fetch_start is not None and fetch_end is not None:
            self.connector.logger.info(f"Fetching missing trades for {trading_pair} from {fetch_start} to {fetch_end}")
            new_trades = await self._get_historical_trades(trading_pair, fetch_start, fetch_end, from_id)
            
            # Save new data to cache
            self._save_trades_to_cache(trading_pair, fetch_start, fetch_end, new_trades)
            
            # Merge with existing cache if available
            existing_cache = self._load_trades_from_cache(trading_pair, start_time, end_time)
            if existing_cache is not None:
                combined_df = pd.concat([existing_cache, new_trades]).drop_duplicates().sort_index()
            else:
                combined_df = new_trades
            
            # Update cache with combined data
            self._save_trades_to_cache(trading_pair, start_time, end_time, combined_df)
            
            # Return filtered data
            return combined_df[
                (combined_df.index >= pd.to_datetime(start_time, unit='s')) &
                (combined_df.index <= pd.to_datetime(end_time, unit='s'))
            ]
        else:
            # Fetch all requested data
            historical_trades = await self._get_historical_trades(trading_pair, start_time, end_time, from_id)
            self._save_trades_to_cache(trading_pair, start_time, end_time, historical_trades)
            return historical_trades

    @abstractmethod
    async def _get_historical_trades(self, trading_pair: str, start_time: int, end_time: int, from_id: Optional[int] = None):
        """Implementation-specific method to fetch historical trades."""
        pass
    
    def clear_cache(self, trading_pair: Optional[str] = None):
        """Clear trades cache for a specific pair or all pairs."""
        if trading_pair:
            # Clear memory cache for specific pair
            keys_to_remove = [key for key in self._trades_cache.keys() if key[0] == trading_pair]
            for key in keys_to_remove:
                del self._trades_cache[key]
            
            # Clear disk cache for specific pair
            pattern = f"*_{trading_pair.replace('-', '')}_*.parquet"
            for cache_file in self._cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    self.connector.logger.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    self.connector.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        else:
            # Clear all cache
            self._trades_cache.clear()
            for cache_file in self._cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.connector.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about current cache state."""
        memory_cache_size = len(self._trades_cache)
        
        disk_cache_files = list(self._cache_dir.glob("*.parquet"))
        disk_cache_size = len(disk_cache_files)
        
        total_disk_size = sum(f.stat().st_size for f in disk_cache_files)
        
        return {
            "memory_cache_entries": memory_cache_size,
            "disk_cache_files": disk_cache_size,
            "total_disk_size_mb": total_disk_size / (1024 * 1024),
            "cache_directory": str(self._cache_dir)
        }
