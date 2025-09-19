import asyncio
from typing import Dict, Optional

import pandas as pd

from ..oi_feed_base import OIFeedBase
from .binance_perpetual_base import BinancePerpetualBase


class BinancePerpetualOIFeed(OIFeedBase[BinancePerpetualBase]):
    # Feed-specific configuration - links to general limits registered by BinancePerpetualBase
    RATE_LIMITS = {}  # No additional limits - uses connector's general limits
    ENDPOINT_WEIGHTS = {
        "open_interest_hist": 0,  # Weight for openInterestHist endpoint (per Binance docs)
    }
    LIMIT_ID = "binance_general"  # Uses the general Binance limit
    
    _endpoints = {
        "open_interest_hist": "/futures/data/openInterestHist"
    }

    def __init__(self, connector: BinancePerpetualBase):
        super().__init__(connector)

    async def _get_historical_oi(self, trading_pair: str, interval: str, start_time: int, end_time: int, limit: int = 500):
        """Get historical open interest data for a specific trading pair."""
        ex_trading_pair = self.connector.get_exchange_trading_pair(trading_pair)
        
        self.connector.logger.info(f"Starting OI collection for {trading_pair} ({ex_trading_pair}) interval {interval}")
        self.connector.logger.info(f"Time range: {start_time} to {end_time} (timestamps in seconds)")
        
        all_oi_data = []
        current_start_time = start_time
        request_count = 0
        max_retries = 3
        
        while current_start_time < end_time:
            request_count += 1
            
            params = {
                "symbol": ex_trading_pair,
                "period": interval,
                "limit": min(limit, 500),  # Binance max limit is 500
                "startTime": int(current_start_time * 1000),  # Convert to milliseconds
                "endTime": int(end_time * 1000)  # Convert to milliseconds
            }
            
            self.connector.logger.info(f"OI Request #{request_count}: Fetching data with params {params}")
            
            retry_count = 0
            oi_data = None
            
            while retry_count < max_retries and oi_data is None:
                try:
                    oi_data = await self._get_historical_oi_request(params)
                    break
                except Exception as e:
                    # Check if it's a 404 error (no OI data available for this symbol)
                    if "404" in str(e) or "Not Found" in str(e):
                        self.connector.logger.info(f"No OI data available for {trading_pair} (404 error) - skipping")
                        return pd.DataFrame()  # Return empty DataFrame
                    
                    retry_count += 1
                    self.connector.logger.warning(f"OI request failed (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        self.connector.logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)

            if oi_data and len(oi_data) > 0:
                all_oi_data.extend(oi_data)
                
                # Update start time for next batch
                last_timestamp = oi_data[-1]["timestamp"]
                current_start_time = int(last_timestamp / 1000) + 1  # Convert back to seconds and add 1
                
                self.connector.logger.info(f"Collected {len(oi_data)} OI records. "
                                         f"Total so far: {len(all_oi_data)}. "
                                         f"Last timestamp: {last_timestamp} "
                                         f"Next start: {current_start_time}")
                
                # If we got fewer records than requested, we've reached the end
                if len(oi_data) < limit:
                    break
            else:
                self.connector.logger.warning("No OI data returned, ending collection")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_oi_data)
        if not df.empty:
            # Convert timestamp from milliseconds to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Convert numeric columns
            numeric_columns = ["sumOpenInterest", "sumOpenInterestValue", "count"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.connector.logger.info(f"Successfully processed {len(df)} OI records for {trading_pair}")
        else:
            self.connector.logger.warning(f"No OI data found for {trading_pair}")
        
        return df

    async def _get_historical_oi_request(self, params: Dict):
        """Make a request for historical open interest data."""
        endpoint = "open_interest_hist"
        weight = self.ENDPOINT_WEIGHTS.get(endpoint, 1)
        return await self.connector.make_request(
            self._endpoints[endpoint], 
            params, 
            limit_id=self.LIMIT_ID, 
            weight=weight
        )
    
    async def test_oi_availability(self, trading_pair: str, interval: str = "1h") -> bool:
        """Test if OI data is available for a trading pair."""
        try:
            ex_trading_pair = self.connector.get_exchange_trading_pair(trading_pair)
            
            # Try to get just one data point
            params = {
                "symbol": ex_trading_pair,
                "period": interval,
                "limit": 1
            }
            
            result = await self._get_historical_oi_request(params)
            return result is not None and len(result) > 0
            
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                return False
            # For other errors, assume it might work later
            self.connector.logger.warning(f"Error testing OI availability for {trading_pair}: {e}")
            return True

    async def filter_supported_pairs(self, trading_pairs: list, interval: str = "1h", 
                                   batch_size: int = 10, max_test_pairs: int = 50) -> list:
        """
        Since all Binance perpetual futures pairs should support OI data, 
        return all trading pairs without filtering.
        """
        self.connector.logger.info(f"All Binance perpetual futures pairs support OI data - returning all {len(trading_pairs)} pairs")
        return trading_pairs

    def get_cache_status(self) -> Dict:
        """Get cache status information for this feed."""
        cache_info = self.get_cache_info()
        cache_info["feed_type"] = "binance_perpetual_oi"
        return cache_info