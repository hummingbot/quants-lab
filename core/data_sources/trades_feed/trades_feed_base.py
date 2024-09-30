import time
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp


class TradesFeedBase(ABC):
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session or aiohttp.ClientSession()

    @abstractmethod
    def get_exchange_trading_pair(self, trading_pair: str) -> str:
        ...

    async def get_historical_trades(self, trading_pair: str, start_time: int, end_time: Optional[int] = None,
                                    from_id: Optional[int] = None):
        if not end_time:
            end_time = int(time.time())
        historical_trades = await self._get_historical_trades(trading_pair, start_time, end_time, from_id)
        return historical_trades

    @abstractmethod
    async def _get_historical_trades(self, trading_pair: str, start_time: int, end_time: int, from_id: Optional[int] = None):
        ...
