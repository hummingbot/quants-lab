# tests/test_portfolio.py

import queue
from nexus.portfolio import NaivePortfolio
from nexus.event import FillEvent

def test_portfolio_update():
    events = queue.Queue()
    symbol_list = ['TEST']
    data_handler = None  # Not needed for this test
    initial_capital = 100000.0

    portfolio = NaivePortfolio(events, data_handler, symbol_list, initial_capital)

    # Simulate a buy fill event
    fill_event_buy = FillEvent(
        symbol='TEST',
        timestamp='2020-01-01',
        order_type='MARKET',
        quantity=1,
        direction='BUY',
        price=100.0,
        commission=1.0
    )
    portfolio.update_fill(fill_event_buy)

    # Check positions and holdings after buy
    assert portfolio.current_positions['TEST'].quantity == 1, "Position not updated correctly after buy."
    expected_cash_after_buy = initial_capital - (1 * 100.0 + 1.0)
    assert portfolio.current_holdings['cash'] == expected_cash_after_buy, "Cash not updated correctly after buy."

    # Simulate a sell fill event
    fill_event_sell = FillEvent(
        symbol='TEST',
        timestamp='2020-01-02',
        order_type='MARKET',
        quantity=1,
        direction='SELL',
        price=110.0,
        commission=1.0
    )
    portfolio.update_fill(fill_event_sell)

    # Check positions and holdings after sell
    assert portfolio.current_positions['TEST'].quantity  == 0, "Position not updated correctly after sell."
    expected_cash_after_sell = expected_cash_after_buy + (1 * 110.0 - 1.0)
    assert portfolio.current_holdings['cash'] == expected_cash_after_sell, "Cash not updated correctly after sell."

    # Check total holdings
    expected_total = portfolio.current_holdings['cash']
    assert portfolio.current_holdings['total'] == expected_total, "Total holdings not updated correctly."
