# tests/test_data_feed.py

import pandas as pd
import queue
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.feed.klines import HistoricKlineDataHandler

def test_data_feed(tmp_path):
    # Create a temporary CSV file
    test_csv = tmp_path / 'TEST.csv'
    test_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=5, freq='D'),
        'Open': [100, 101, 102, 103, 104],
        'High': [101, 102, 103, 104, 105],
        'Low': [99, 100, 101, 102, 103],
        'Close': [100, 101, 102, 103, 104],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })
    test_data.set_index('Date', inplace=True)
    test_data.to_csv(test_csv)

    events = queue.Queue()
    history_files = {'TEST': str(test_csv)}
    symbol_list = ['TEST']

    data_handler = HistoricCSVDataHandler(events, history_files, symbol_list)
    data_handler.get_next_bar()

    # Assert that the data handler has data for the symbol
    assert 'TEST' in data_handler.latest_symbol_data, "Data handler does not contain symbol data."
    assert len(data_handler.latest_symbol_data['TEST']) == 1, "Data handler did not update symbol data correctly."

    # Assert that an event was placed in the queue
    assert not events.empty(), "No event was placed in the queue by data handler."
    event = events.get()
    assert event.type == 'MARKET', "Event type is not 'MARKET'."
    assert event.symbol == 'TEST', "Event symbol is incorrect."


def test_kline_data_feed(tmp_path):
    test_parquet = tmp_path / 'BTCUSDT_1m.parquet'
    test_data = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01', periods=5, freq='min', tz='UTC'),
        'open': [10, 11, 12, 13, 14],
        'high': [11, 12, 13, 14, 15],
        'low': [9, 10, 11, 12, 13],
        'close': [10, 11, 12, 13, 14],
        'volume': [1, 1, 1, 1, 1],
        'quote_vol': [1, 1, 1, 1, 1],
        'trade_count': [1, 1, 1, 1, 1],
        'taker_base_vol': [1, 1, 1, 1, 1],
        'taker_quote_vol': [1, 1, 1, 1, 1],
    })
    # store open_time as int ms to mimic downloader output
    test_data['open_time'] = test_data['open_time'].astype('int64') // 10**6
    test_data.to_parquet(test_parquet)

    events = queue.Queue()
    history_files = {'BTCUSDT': str(test_parquet)}
    handler = HistoricKlineDataHandler(events, history_files)
    handler.get_next_bar()

    assert 'BTCUSDT' in handler.latest_symbol_data
    assert len(handler.latest_symbol_data['BTCUSDT']) == 1
    assert not events.empty()
    event = events.get()
    assert event.type == 'MARKET'
    assert event.symbol == 'BTCUSDT'
