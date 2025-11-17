import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, TypeVar, Generic, Tuple

import pandas as pd

from .connector_base import ConnectorBase
from ...data_paths import data_paths

ConnectorT = TypeVar('ConnectorT', bound=ConnectorBase)


class OIFeedBase(ABC, Generic[ConnectorT]):
    # Class variables for rate limiting - to be overridden by subclasses
    RATE_LIMITS: Dict[str, tuple] = {}  # limit_id -> (max_requests, time_window_seconds)
    ENDPOINT_WEIGHTS: Dict[str, int] = {}  # endpoint -> weight
    LIMIT_ID: str = ""  # Main limit ID for this feed
    
    def __init__(self, connector: ConnectorT):
        self.connector = connector
        self._setup_rate_limits()
        # Simplified cache key: (trading_pair, interval) like candles
        self._oi_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._cache_dir = data_paths.oi_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_rate_limits(self):
        """Register rate limits from class variables."""
        for limit_id, (max_requests, time_window) in self.RATE_LIMITS.items():
            self.connector.register_rate_limit(limit_id, max_requests, time_window)

    def _get_cache_key(self, trading_pair: str, interval: str) -> Tuple[str, str]:
        """Generate cache key for OI data (matches candles format)."""
        return (trading_pair, interval)
    
    def _get_cache_filename(self, trading_pair: str, interval: str) -> Path:
        """Generate cache filename for OI data (matches candles format)."""
        # Get connector name similar to candles format
        connector_name = self._get_connector_name()
        # Use same format as candles: connector|pair|interval.parquet
        filename = f"{connector_name}|{trading_pair}|{interval}.parquet"
        return self._cache_dir / filename
    
    def _get_connector_name(self) -> str:
        """Get standardized connector name for cache files."""
        # This should be overridden by subclasses if needed
        # Default: derive from class name
        class_name = self.connector.__class__.__name__
        if 'BinancePerpetual' in class_name:
            return 'binance_perpetual'
        return class_name.lower().replace('base', '').replace('connector', '')
    
    def _load_oi_from_cache(self, trading_pair: str, interval: str) -> Optional[pd.DataFrame]:
        """Load OI data from cache if available."""
        cache_key = self._get_cache_key(trading_pair, interval)
        
        # Check memory cache first
        if cache_key in self._oi_cache:
            self.connector.logger.info(f"Found OI data in memory cache for {trading_pair} {interval}")
            return self._oi_cache[cache_key]
        
        # Check disk cache
        cache_file = self._get_cache_filename(trading_pair, interval)
        if cache_file.exists():
            try:
                self.connector.logger.info(f"Loading OI data from disk cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                
                # Store in memory cache
                self._oi_cache[cache_key] = df
                return df
            except Exception as e:
                self.connector.logger.warning(f"Failed to load OI data from cache {cache_file}: {e}")
        
        return None
    
    def _save_oi_to_cache(self, trading_pair: str, interval: str, oi_df: pd.DataFrame):
        """Save OI data to cache."""
        if oi_df.empty:
            return
            
        cache_key = self._get_cache_key(trading_pair, interval)
        
        # Load existing cache and merge if necessary
        existing_df = self._load_oi_from_cache(trading_pair, interval)
        if existing_df is not None and not existing_df.empty:
            # Concatenate and remove duplicates, keeping latest data
            oi_df = pd.concat([existing_df, oi_df]).drop_duplicates().sort_index()
        
        # Save to memory cache
        self._oi_cache[cache_key] = oi_df.copy()
        
        # Save to disk cache
        cache_file = self._get_cache_filename(trading_pair, interval)
        try:
            oi_df.to_parquet(
                cache_file,
                engine='pyarrow',
                compression='snappy',
                index=True
            )
            self.connector.logger.info(f"Saved {len(oi_df)} OI records to cache: {cache_file}")
        except Exception as e:
            self.connector.logger.warning(f"Failed to save OI data to cache {cache_file}: {e}")
    
    def _check_cache_coverage(self, trading_pair: str, start_time: int, end_time: int, interval: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """Check if we have partial cache coverage and determine what ranges need fetching."""
        cached_df = self._load_oi_from_cache(trading_pair, interval)
        
        if cached_df is None or cached_df.empty:
            return False, start_time, end_time
        
        cached_start = int(cached_df.index.min().timestamp())
        cached_end = int(cached_df.index.max().timestamp())
        
        # Check if cache fully covers the requested range
        if cached_start <= start_time and cached_end >= end_time:
            self.connector.logger.info(f"Cache fully covers requested range for {trading_pair} {interval}")
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

    async def get_historical_oi(self, trading_pair: str, interval: str, start_time: int, end_time: Optional[int] = None, limit: int = 500):
        """Get historical open interest data for a trading pair within a time range with caching."""
        if not end_time:
            end_time = int(time.time())
        
        # Check cache coverage
        fully_cached, fetch_start, fetch_end = self._check_cache_coverage(trading_pair, start_time, end_time, interval)
        
        if fully_cached:
            # Return cached data filtered to requested range
            cached_df = self._load_oi_from_cache(trading_pair, interval)
            return cached_df[
                (cached_df.index >= pd.to_datetime(start_time, unit='s')) &
                (cached_df.index <= pd.to_datetime(end_time, unit='s'))
            ]
        
        # Need to fetch new data
        if fetch_start is not None and fetch_end is not None:
            self.connector.logger.info(f"Fetching missing OI data for {trading_pair} {interval} from {fetch_start} to {fetch_end}")
            new_oi = await self._get_historical_oi(trading_pair, interval, fetch_start, fetch_end, limit)
            
            # Save new data to cache (will merge automatically)
            self._save_oi_to_cache(trading_pair, interval, new_oi)
            
            # Load the updated cache
            combined_df = self._load_oi_from_cache(trading_pair, interval)
            
            # Return filtered data
            return combined_df[
                (combined_df.index >= pd.to_datetime(start_time, unit='s')) &
                (combined_df.index <= pd.to_datetime(end_time, unit='s'))
            ]
        else:
            # Fetch all requested data
            historical_oi = await self._get_historical_oi(trading_pair, interval, start_time, end_time, limit)
            self._save_oi_to_cache(trading_pair, interval, historical_oi)
            return historical_oi

    @abstractmethod
    async def _get_historical_oi(self, trading_pair: str, interval: str, start_time: int, end_time: int, limit: int = 500):
        """Implementation-specific method to fetch historical open interest data."""
        pass
    
    def clear_cache(self, trading_pair: Optional[str] = None, interval: Optional[str] = None):
        """Clear OI cache for a specific pair/interval or all."""
        if trading_pair or interval:
            # Clear memory cache for specific criteria
            keys_to_remove = [
                key for key in self._oi_cache.keys() 
                if (not trading_pair or key[0] == trading_pair) and 
                   (not interval or key[1] == interval)
            ]
            for key in keys_to_remove:
                del self._oi_cache[key]
            
            # Clear disk cache for specific criteria - use new format pattern
            connector_name = self._get_connector_name()
            if trading_pair and interval:
                pattern = f"{connector_name}|{trading_pair}|{interval}.parquet"
            elif trading_pair:
                pattern = f"{connector_name}|{trading_pair}|*.parquet"
            elif interval:
                pattern = f"{connector_name}|*|{interval}.parquet"
            else:
                pattern = f"{connector_name}|*.parquet"
            
            for cache_file in self._cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    self.connector.logger.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    self.connector.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        else:
            # Clear all cache
            self._oi_cache.clear()
            for cache_file in self._cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.connector.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about current cache state."""
        memory_cache_size = len(self._oi_cache)
        
        disk_cache_files = list(self._cache_dir.glob("*.parquet"))
        disk_cache_size = len(disk_cache_files)
        
        total_disk_size = sum(f.stat().st_size for f in disk_cache_files)
        
        return {
            "memory_cache_entries": memory_cache_size,
            "disk_cache_files": disk_cache_size,
            "total_disk_size_mb": total_disk_size / (1024 * 1024),
            "cache_directory": str(self._cache_dir)
        }
    
    def get_cache_status(self) -> Dict[str, any]:
        """Get cache status with feed type information."""
        cache_info = self.get_cache_info()
        cache_info["feed_type"] = "OI Feed"
        return cache_info