import logging
from nexus.event import FillEvent
from nexus.abc import ExecutionHandler

class SimulatedExecutionHandler(ExecutionHandler):
    """Simulates order execution without latency or slippage."""

    def __init__(self, events, data_handler, assets=None):
        """Create a new simulated execution handler.

        Parameters
        ----------
        events : queue.Queue
            Queue that receives ``FillEvent`` objects.
        data_handler : DataFeed
            Data handler providing latest prices.
        assets : dict, optional
            Mapping of symbols to :class:`Asset` objects. If omitted, a minimal
            default asset with zero spread and commission is used.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ SimulatedExecutionHandler")
        self.events = events
        self.data_handler = data_handler
        self.assets = assets or {}
        # Store pending limit orders
        self.pending_orders = []

    def execute_order(self, event):
        if event.type != 'ORDER':
            return

        if event.order_type == 'LIMIT':
            # Store the order until the price condition is met
            self.pending_orders.append(event)
            return

        # Market orders execute immediately
        self._fill_order(event)

    def _fill_order(self, event):
        asset = self.assets.get(event.symbol)
        if asset is None:
            # Minimal asset placeholder
            from nexus.asset import Asset
            asset = Asset(
                name=event.symbol,
                price=0.0,
                spread=0.0,
                roll_long=0.0,
                roll_short=0.0,
                pip=1.0,
                pip_cost=1.0,
                margin_cost=0.0,
                market='',
                lot_amount=1.0,
                commission=0.0,
                symbol=event.symbol,
                asset_type='forex'
            )
        price = self.get_execution_price(
            event.symbol, event.order_type, event.direction, asset, event.price
        )
        fill_event = FillEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            order_type=event.order_type,
            quantity=event.quantity,
            direction=event.direction,
            price=price,
            commission=self.calculate_commission(event, asset),
            roll=0,
            stop_loss=event.stop_loss
        )
        self.events.put(fill_event)

    def update_market(self, event):
        """Check pending limit orders against the latest market price."""
        if event.type != 'MARKET':
            return

        executed = []
        for order in list(self.pending_orders):
            if order.symbol != event.symbol:
                continue
            price = self.data_handler.get_latest_bar_value(order.symbol, 'Close')
            if price is None:
                continue
            if order.direction == 'BUY' and price <= order.price:
                self._fill_order(order)
                executed.append(order)
            elif order.direction == 'SELL' and price >= order.price:
                self._fill_order(order)
                executed.append(order)

        for order in executed:
            self.pending_orders.remove(order)

    def get_execution_price(self, symbol, order_type, direction, asset, limit_price=None):
        """
        Retrieves the execution price based on the order type and direction, including spread.

        Parameters:
        - symbol (str): The asset symbol.
        - order_type (str): The type of order ('MARKET' or 'LIMIT').
        - direction (str): 'BUY' or 'SELL'
        - asset (Asset): The asset object containing spread information.

        Returns:
        - float: The execution price.
        """
        price = self.data_handler.get_latest_bar_value(symbol, 'Close')
        if price is None:
            raise ValueError(f"Price not available for symbol: {symbol}")

        if order_type == 'MARKET':
            if direction == 'BUY':
                if asset.asset_type == 'forex':
                    execution_price = price
                else: # For BUY orders, pay the ASK price (Mid price + half spread)
                    execution_price = price + (asset.spread / 2)
            elif direction == 'SELL':
                if asset.asset_type == 'forex':
                    execution_price = price - asset.spread
                else: # For SELL orders, receive the BID price (Mid price - half spread)
                    execution_price = price - (asset.spread / 2)
            else:
                raise ValueError(f"Unknown direction: {direction}")
            return execution_price
        elif order_type == 'LIMIT':
            # Limit orders execute at the specified limit price
            return limit_price if limit_price is not None else price
        else:
            raise ValueError(f"Unknown order type: {order_type}")

    def calculate_commission(self, order_event, asset):
        """
        Calculates commission based on asset parameters.

        Parameters:
        - order_event (OrderEvent): The order event.
        - asset (Asset): The asset being traded.

        Returns:
        - float: The commission for the order.
        """
        if asset.asset_type == 'forex':
            # Commission per lot * number of lots
            return asset.commission * (order_event.quantity / asset.lot_amount)
        
        else:
            # Implement commissions for other asset types
            # Example: flat commission per trade
            return asset.commission

class RealExecutionHandler(ExecutionHandler):
    """Executes orders through a broker API."""
    # Implementation depends on the broker's API

    def execute_order(self, event):
        pass

    def update_market(self, event):
        pass
