import logging
from datetime import timedelta
from nexus.abc import Strategy
from nexus.event import OrderEvent

class MarketMakerStrategy(Strategy):
    """Simple market making strategy placing bid and ask each tick."""

    def __init__(self, events, assets, strategy_params, backtest_params):
        super().__init__(events, assets, strategy_params, backtest_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.spread = strategy_params.get("spread", 10.0)
        self.order_refresh_time = strategy_params.get("order_refresh_time", 60)
        self.active_orders = {symbol: [] for symbol in self.symbols}

    def run(self, event):
        timestamp, symbol, prices, _ = self.process_event(event)
        mid_price = prices[0]
        quantity = self.lots * self.assets[symbol].lot_amount

        # Cancel expired orders from internal tracking
        remaining = []
        for order in self.active_orders[symbol]:
            if timestamp - order["timestamp"] < timedelta(seconds=self.order_refresh_time):
                remaining.append(order)
        self.active_orders[symbol] = remaining

        bid_price = mid_price - self.spread / 2
        ask_price = mid_price + self.spread / 2

        bid = OrderEvent(symbol, timestamp, "LIMIT", quantity, "BUY", price=bid_price)
        ask = OrderEvent(symbol, timestamp, "LIMIT", quantity, "SELL", price=ask_price)

        self.events.put(bid)
        self.events.put(ask)

        self.active_orders[symbol].append({"timestamp": timestamp, "side": "BUY", "price": bid_price})
        self.active_orders[symbol].append({"timestamp": timestamp, "side": "SELL", "price": ask_price})
