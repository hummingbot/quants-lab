class Event:
    """Base class for all events."""
    pass

class MarketEvent(Event):
    """Handles the event of receiving new market updates."""

    def __init__(self, symbol, data, data_type):
        self.type = 'MARKET'
        self.symbol = symbol
        self.data = data
        self.data_type = data_type

class SignalEvent(Event):
    """Handles the event of sending a signal from a strategy."""

    def __init__(self, symbol, timestamp, signal_type, quantity,
                 stop_loss=None, limit_price=None):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type  # 'LONG', 'SHORT', or 'EXIT'
        self.quantity = quantity
        self.stop_loss = stop_loss
        # Optional limit price for limit orders. ``None`` implies market order.
        self.limit_price = limit_price
        
class OrderEvent(Event):
    """Handles the event of sending an order to execution."""

    def __init__(self, symbol, timestamp, order_type, quantity, direction, price=None, stop_loss=None):
        self.type = 'ORDER'
        self.symbol = symbol
        self.timestamp = timestamp
        self.order_type = order_type  # 'MARKET' or 'LIMIT'
        self.quantity = quantity
        self.direction = direction  # 'BUY' or 'SELL'
        self.price = price
        self.stop_loss = stop_loss
class FillEvent(Event):
    """Encapsulates the notion of a filled order."""

    def __init__(self, symbol, timestamp, order_type, quantity, direction, price,
                 commission, roll=0.0, stop_loss=None):
        self.type = 'FILL'
        self.symbol = symbol
        self.timestamp = timestamp
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.price = price
        self.commission = commission
        self.roll = roll
        self.stop_loss = stop_loss

