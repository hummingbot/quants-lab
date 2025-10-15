import asyncio
import logging
from abc import ABC, abstractmethod
from time import time
from typing import Dict, Optional, List
from collections import defaultdict

import aiohttp


class Throttler:
    """Simple throttling system with limit IDs for different endpoints."""
    
    def __init__(self):
        self._limits = {}  # limit_id -> (max_weight, time_window_seconds)
        self._request_history = defaultdict(list)  # limit_id -> [(timestamp, weight)]
        
    def register_limit(self, limit_id: str, max_weight: int, time_window_seconds: int):
        """Register a rate limit for a specific limit ID."""
        self._limits[limit_id] = (max_weight, time_window_seconds)
        
    async def enforce_limit(self, limit_id: str, weight: int = 1):
        """Enforce rate limit for a specific limit ID with optional weight."""
        if limit_id not in self._limits:
            logging.warning(f"No rate limit registered for limit_id: {limit_id}")
            return  # No limit registered
            
        max_weight, time_window = self._limits[limit_id]
        current_time = time()
        
        # Clean old requests outside the time window
        old_count = len(self._request_history[limit_id])
        self._request_history[limit_id] = [
            (t, w) for t, w in self._request_history[limit_id] 
            if t > current_time - time_window
        ]
        cleaned_count = old_count - len(self._request_history[limit_id])
        
        # Calculate current weight usage
        current_weight_usage = sum(w for t, w in self._request_history[limit_id])
        
        logging.info(f"Rate limit check for {limit_id}: "
                    f"current_weight={current_weight_usage}/{max_weight} "
                    f"(adding {weight} weight) "
                    f"requests_in_window={len(self._request_history[limit_id])} "
                    f"cleaned={cleaned_count} old requests")
        
        # Check if adding this request would exceed the limit
        if current_weight_usage + weight > max_weight:
            # Calculate sleep time based on oldest request
            if self._request_history[limit_id]:
                oldest_request_time = min(t for t, w in self._request_history[limit_id])
                sleep_time = time_window - (current_time - oldest_request_time) + 1  # Add 1 second buffer
                if sleep_time > 0:
                    logging.warning(f"Rate limit would be exceeded for {limit_id}. "
                                  f"Current: {current_weight_usage}, adding: {weight}, max: {max_weight}. "
                                  f"Sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Clean again after sleep
                    current_time = time()
                    self._request_history[limit_id] = [
                        (t, w) for t, w in self._request_history[limit_id] 
                        if t > current_time - time_window
                    ]
                
        # Record this request with its weight
        self._request_history[limit_id].append((current_time, weight))
        new_total = sum(w for t, w in self._request_history[limit_id])
        logging.debug(f"Recorded request for {limit_id} with weight {weight}. "
                     f"Total weight in window: {new_total}/{max_weight}")


class ConnectorBase(ABC):
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._session_created = False
        self._logger = None
        self._throttler = Throttler()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session when needed in async context."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._session_created = True
        return self._session
    
    async def close(self):
        """Close the session if it was created by this instance."""
        if self._session and self._session_created:
            await self._session.close()
            self._session = None
            self._session_created = False

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def register_rate_limit(self, limit_id: str, max_requests: int, time_window_seconds: int):
        """Register a rate limit for this connector."""
        self._throttler.register_limit(limit_id, max_requests, time_window_seconds)

    async def enforce_rate_limit(self, limit_id: str, weight: int = 1):
        """Enforce rate limit for a specific limit ID."""
        await self._throttler.enforce_limit(limit_id, weight)

    async def _make_request(self, method: str, url: str, params: Optional[Dict] = None, 
                           headers: Optional[Dict] = None, data: Optional[Dict] = None,
                           limit_id: Optional[str] = None, weight: int = 1) -> Optional[Dict]:
        """Generic method to make HTTP requests with error handling and rate limiting."""
        if limit_id:
            await self.enforce_rate_limit(limit_id, weight)
            
        try:
            session = await self._get_session()
            async with session.request(method, url, params=params, headers=headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            await self._handle_http_error(e, params or {})
        except Exception as e:
            self.logger.error(f"Error making {method} request to {url}: {e}")
        return None

    async def _handle_http_error(self, error: aiohttp.ClientResponseError, params: Dict):
        """Handle HTTP errors with appropriate retry logic."""
        if error.status == 404:
            self.logger.info(f"Resource not found (404) for request {params} - likely no data available for this symbol")
            # Don't treat 404 as a critical error for data endpoints
            return None
        elif error.status == 429:
            self.logger.warning("Rate limit hit (429). Waiting 60 seconds before retry...")
            await asyncio.sleep(60)  # Wait longer for rate limit
        elif error.status == 418:
            self.logger.error("IP banned (418). Waiting 2 hours...")
            await asyncio.sleep(60 * 60 * 2)  # IP ban
        else:
            self.logger.error(f"HTTP error {error.status} for request {params}: {error}")

    @abstractmethod
    def get_exchange_trading_pair(self, trading_pair: str) -> str:
        """Convert standard trading pair format to exchange-specific format."""
        pass