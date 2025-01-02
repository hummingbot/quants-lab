import asyncio
import logging
from datetime import datetime, timezone
from time import time
from typing import Dict, List, Optional

import aiohttp
from bidict import bidict


class CoinGlassDataFeed:
    _base_url = "https://open-api-v3.coinglass.com"
    _endpoints = {
        "liquidation_aggregated_history": "/api/futures/liquidation/v2/aggregated-history",
        "global_long_short_account_ratio": "/api/futures/globalLongShortAccountRatio/history",
        "aggregated_open_interest_history": "/api/futures/openInterest/ohlc-aggregated-history",
    }
    _exchanges = [
        "Binance",
        "BingX",
        "Bitfinex",
        "Bitget",
        "Bitmex",
        "Bybit",
        "CME",
        "CoinEx",
        "Coinbase",
        "Crypto.com",
        "Deribit",
        "HTX",
        "Hyperliquid",
        "Kraken",
        "OKX",
        "dYdX",
    ]
    _logger = None

    REQUEST_WEIGHT_LIMIT = 2400
    REQUEST_WEIGHT = 25
    ONE_MINUTE = 60  # seconds

    interval_to_seconds = bidict(
        {
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
            "1M": 2592000,
        }
    )

    def __init__(self, api_key: str):
        self._session = aiohttp.ClientSession(headers={"CG-API-KEY": api_key})
        self._request_timestamps = []

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def get_symbol(self, trading_pair: str) -> str:
        base, _ = trading_pair.split("-")
        return base

    def get_pair(self, trading_pair: str) -> str:
        pair = "".join(trading_pair.split("-"))
        return pair

    def get_exchange(self, connector_name: str) -> str:
        _exchanges = [
            "Binance",
            "BingX",
            "Bitfinex",
            "Bitget",
            "Bitmex",
            "Bybit",
            "CME",
            "CoinEx",
            "Coinbase",
            "Crypto.com",
            "Deribit",
            "HTX",
            "Hyperliquid",
            "Kraken",
            "OKX",
            "dYdX",
        ]

        # Normalize the input for matching
        normalized_connector = connector_name.split("_")[0].lower()

        # Search for a match in the exchanges list
        for exchange in _exchanges:
            if exchange.lower() == normalized_connector:
                return exchange

        # Return a default value if no match is found
        raise ValueError("Unsupported exchange")

    async def get_endpoint(
        self,
        endpoint: str,
        trading_pair: str,
        interval: str,
        start_time: int,
        end_time: int,
        connector_name: str,
        limit: Optional[int] = None,
    ):
        if not end_time:
            end_time = int(time())
        data = await self._get_endpoint(
            endpoint,
            trading_pair,
            interval,
            start_time,
            end_time,
            connector_name,
            limit,
        )
        return data

    async def _get_endpoint(
        self,
        endpoint: str,
        trading_pair: str,
        interval: str,
        start_time: int,
        end_time: int,
        connector_name: str,
        limit: Optional[int] = None,
    ) -> List:
        all_data_collected = False
        end_ts = int(end_time)
        start_ts = int(start_time)
        all_data = []
        exchange = self.get_exchange(connector_name)
        if endpoint in ["global_long_short_account_ratio"]:
            symbol = self.get_pair(trading_pair)
        else:
            symbol = self.get_symbol(trading_pair)

        while not all_data_collected:
            await (
                self._enforce_rate_limit()
            )  # Enforce rate limit before making a request

            params = {
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
                "startTime": start_ts,
            }

            if limit:
                params["limit"] = limit

            else:
                limit = 1000

            data = await self._get_endpoint_request(endpoint, params)

            if data:
                last_timestamp = data[-1].get("t") or data[-1].get("time")
                all_data.extend(data)
                self.logger().info(
                    f"Fetched {len(data)} rows data from {datetime.fromtimestamp(data[0].get('t') or data[0].get('time'), tz = timezone.utc)} to {datetime.fromtimestamp(data[-1].get('t') or data[-1].get('time'), tz=timezone.utc)}"
                )
                all_data_collected = (
                    last_timestamp + self.interval_to_seconds[interval] >= end_ts
                )
                start_ts = last_timestamp + self.interval_to_seconds[interval]
            else:
                start_ts = start_ts + self.interval_to_seconds[interval] * limit
                self.logger().info(
                    f"Updated startTime to {datetime.fromtimestamp(start_ts, tz=timezone.utc)}"
                )

        return all_data

    async def _get_endpoint_request(self, endpoint: str, params: Dict) -> List | None:
        try:
            url = f"{self._base_url}{self._endpoints[endpoint]}"
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                self._record_request()  # Record the timestamp of this request
                data = await self._process_cg_response(response)
                return data
        except aiohttp.ClientResponseError as e:
            self.logger().error(f"Error fetching historical data for {params}: {e}")
            if e.status == 429:
                await asyncio.sleep(1)  # Sleep to respect rate limits
            elif e.status == 418:
                await asyncio.sleep(60 * 60 * 2)  # Sleep to respect rate limits
        except Exception as e:
            self.logger().error(f"Error fetching historical data for {params}: {e}")

    async def _enforce_rate_limit(self):
        current_time = time()
        self._request_timestamps = [
            t for t in self._request_timestamps if t > current_time - self.ONE_MINUTE
        ]

        # Calculate the current weight usage
        current_weight_usage = len(self._request_timestamps) * self.REQUEST_WEIGHT

        if current_weight_usage >= self.REQUEST_WEIGHT_LIMIT:
            # Calculate how long to sleep to stay within the rate limit
            sleep_time = self.ONE_MINUTE - (current_time - self._request_timestamps[0])
            self.logger().info(
                f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds."
            )
            await asyncio.sleep(sleep_time)

    async def _process_cg_response(self, response: aiohttp.ClientResponse) -> List:
        response_json = await response.json()
        if response_json["success"]:
            return response_json["data"]
        else:
            raise ValueError(response_json.get("msg"))

    def _record_request(self):
        """Records the timestamp of a request."""
        self._request_timestamps.append(time())
