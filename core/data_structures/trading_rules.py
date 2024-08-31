from decimal import Decimal
from typing import List

from core.data_structures.data_structure_base import DataStructureBase
from hummingbot.connector.trading_rule import TradingRule


class TradingRules(DataStructureBase):
    def __init__(self, trading_rules: List[TradingRule]):
        super().__init__(trading_rules)

    def get_all_trading_pairs(self):
        return list(set([tr.trading_pair for tr in self.data]))

    def filter_by_base_asset(self, base_asset: str):
        return TradingRules([tr for tr in self.data if tr.trading_pair.split("-")[0] == base_asset])

    def filter_by_quote_asset(self, quote_asset: str):
        return TradingRules([tr for tr in self.data if tr.trading_pair.split("-")[1] == quote_asset])

    def filter_by_trading_pair(self, trading_pair: str):
        return TradingRules([tr for tr in self.data if tr.trading_pair == trading_pair])

    def filter_by_min_order_size(self, min_order_size: float):
        return TradingRules([tr for tr in self.data if tr.min_order_size <= Decimal(min_order_size)])

    def filter_by_min_notional_size(self, min_notional_size: float):
        return TradingRules([tr for tr in self.data if tr.min_notional_size <= Decimal(min_notional_size)])
