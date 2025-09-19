import asyncio
from typing import Dict, Optional

import pandas as pd

from ..trades_feed_base import TradesFeedBase
from .binance_perpetual_base import BinancePerpetualBase


class BinancePerpetualTradesFeed(TradesFeedBase[BinancePerpetualBase]):
    # Feed-specific configuration - links to general limits registered by BinancePerpetualBase
    RATE_LIMITS = {}  # No additional limits - uses connector's general limits
    ENDPOINT_WEIGHTS = {
        "historical_agg_trades": 20,  # Weight for aggTrades endpoint
    }
    LIMIT_ID = "binance_general"  # Uses the general Binance limit
    
    _endpoints = {
        "historical_agg_trades": "/fapi/v1/aggTrades"
    }

    def __init__(self, connector: BinancePerpetualBase):
        super().__init__(connector)

    async def _get_historical_trades(self, trading_pair: str, start_time: int, end_time: int, from_id: Optional[int] = None):
        """Get historical trades for a specific trading pair."""
        all_trades_collected = False
        end_ts = int(end_time * 1000)
        start_ts = int(start_time * 1000)
        all_trades = []
        ex_trading_pair = self.connector.get_exchange_trading_pair(trading_pair)
        
        self.connector.logger.info(f"Starting trade collection for {trading_pair} ({ex_trading_pair})")
        self.connector.logger.info(f"Time range: {start_ts} to {end_ts} (timestamps in ms)")
        
        request_count = 0
        max_retries = 3

        while not all_trades_collected:
            request_count += 1
            params = {
                "symbol": ex_trading_pair,
                "limit": 1000,
            }
            if from_id:
                params["fromId"] = from_id
            else:
                params["startTime"] = start_ts

            self.connector.logger.info(f"Request #{request_count}: Fetching trades with params {params}")
            
            retry_count = 0
            trades = None
            
            while retry_count < max_retries and trades is None:
                try:
                    trades = await self._get_historical_trades_request(params)
                    break
                except Exception as e:
                    retry_count += 1
                    self.connector.logger.warning(f"Request failed (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        self.connector.logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)

            if trades:
                last_timestamp = trades[-1]["T"]
                all_trades.extend(trades)
                all_trades_collected = last_timestamp >= end_ts
                from_id = trades[-1]["a"]
                
                self.connector.logger.info(f"Collected {len(trades)} trades. "
                                         f"Total so far: {len(all_trades)}. "
                                         f"Last timestamp: {last_timestamp} "
                                         f"(target: {end_ts})")
            else:
                self.connector.logger.warning("No trades returned, ending collection")
                all_trades_collected = True

        df = pd.DataFrame(all_trades)
        if not df.empty:
            df.rename(columns={"T": "timestamp", "p": "price", "q": "volume", "m": "sell_taker", "a": "id"}, inplace=True)
            df.drop(columns=["f", "l"], inplace=True)
            df["timestamp"] = df["timestamp"] / 1000
            df.index = pd.to_datetime(df["timestamp"], unit="s")
            df["price"] = df["price"].astype(float)
            df["volume"] = df["volume"].astype(float)
        
        return df

    async def _get_historical_trades_request(self, params: Dict):
        """Make a request for historical trades."""
        endpoint = "historical_agg_trades"
        weight = self.ENDPOINT_WEIGHTS.get(endpoint, 1)
        return await self.connector.make_request(
            self._endpoints[endpoint], 
            params, 
            limit_id=self.LIMIT_ID, 
            weight=weight
        )
    
    def get_cache_status(self) -> Dict:
        """Get cache status information for this feed."""
        cache_info = self.get_cache_info()
        cache_info["feed_type"] = "binance_perpetual_trades"
        return cache_info