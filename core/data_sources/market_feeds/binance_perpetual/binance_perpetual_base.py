from typing import Dict

from ..connector_base import ConnectorBase


class BinancePerpetualBase(ConnectorBase):
    _base_url = "https://fapi.binance.com"
    
    # General Binance rate limits
    GENERAL_LIMITS = {
        "binance_general": (2400, 60),  # 2400 weight per minute
        "binance_orders": (300, 60),    # 300 orders per minute  
        "binance_raw_requests": (6000, 60), # 6000 raw requests per minute
    }

    def __init__(self, session=None):
        super().__init__(session)
        self._setup_general_limits()
        
    def _setup_general_limits(self):
        """Register general Binance rate limits."""
        for limit_id, (max_requests, time_window) in self.GENERAL_LIMITS.items():
            self.register_rate_limit(limit_id, max_requests, time_window)

    def get_exchange_trading_pair(self, trading_pair: str) -> str:
        """Convert trading pair format from 'BASE-QUOTE' to 'BASEQUOTE' for Binance."""
        base, quote = trading_pair.split("-")
        return f"{base}{quote}"

    async def make_request(self, endpoint: str, params: Dict, limit_id: str = None, weight: int = 1) -> Dict:
        """Make a request to Binance API with rate limiting."""
        url = f"{self._base_url}{endpoint}"
        return await self._make_request("GET", url, params=params, limit_id=limit_id, weight=weight)