# tests/test_execution_handler.py

import queue
from nexus.execution import SimulatedExecutionHandler
from nexus.event import OrderEvent, FillEvent

class MockDataHandler:
    def __init__(self, price=100.0):
        self.price = price

    def get_latest_bar_value(self, symbol, field):
        if field == 'Close':
            return self.price
        return None

    def set_price(self, price):
        self.price = price

def test_execution_handler():
    events = queue.Queue()
    data_handler = MockDataHandler()
    execution_handler = SimulatedExecutionHandler(events, data_handler)

    # Create a mock order event
    order_event = OrderEvent(
        symbol='TEST',
        timestamp='2020-01-01',
        order_type='MARKET',
        quantity=100,
        direction='BUY',
        price=None
    )

    # Execute the order
    execution_handler.execute_order(order_event)

    # Check if a fill event was generated
    assert not events.empty(), "No event was placed in the queue by execution handler."
    event = events.get()
    assert event.type == 'FILL', "Event type is not 'FILL'."
    assert event.symbol == 'TEST', "Event symbol is incorrect."
    assert event.quantity == 100, "Fill event quantity is incorrect."
    assert event.price == 100.0, "Fill event price is incorrect."


def test_limit_order_execution():
    events = queue.Queue()
    data_handler = MockDataHandler(price=100.0)
    execution_handler = SimulatedExecutionHandler(events, data_handler)

    # Create a buy limit order below current price
    order_event = OrderEvent(
        symbol='TEST',
        timestamp='2020-01-01',
        order_type='LIMIT',
        quantity=10,
        direction='BUY',
        price=90.0,
    )

    execution_handler.execute_order(order_event)

    # Price has not reached limit yet
    from types import SimpleNamespace
    market_event = SimpleNamespace(type='MARKET', symbol='TEST', data={'Close': 100.0})
    execution_handler.update_market(market_event)
    assert events.empty(), "Limit order executed too early"

    # Move price to trigger the limit
    data_handler.set_price(90.0)
    market_event = SimpleNamespace(type='MARKET', symbol='TEST', data={'Close': 90.0})
    execution_handler.update_market(market_event)

    assert not events.empty(), "Limit order did not execute when price hit"
    fill = events.get()
    assert isinstance(fill, FillEvent)
    assert fill.price == 90.0

