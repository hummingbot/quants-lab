# tests/test_backtest.py
from datetime import datetime
import pandas as pd
import numpy as np
from nexus.backtest import Backtest
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from strategy.ma_cross import MovingAverageCrossStrategy

def test_run_backtest(tmp_path):
    # Create a temporary CSV file for 'TEST' symbol
    test_csv = tmp_path / 'test_data.csv'

    # Set seed for reproducibility
    np.random.seed(69)
    data_length = 200

    # Generate random walk data for more realistic price series
    price_changes = np.random.normal(loc=0, scale=1, size=data_length)
    price_series = 100 + np.cumsum(price_changes)

    # Create new test data with more realistic prices
    test_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=data_length, freq='D'),
        'Open': price_series + np.random.normal(loc=0, scale=0.5, size=data_length),
        'High': price_series + np.random.normal(loc=0.5, scale=0.5, size=data_length),
        'Low': price_series + np.random.normal(loc=-0.5, scale=0.5, size=data_length),
        'Close': price_series + np.random.normal(loc=0, scale=0.5, size=data_length),
        'Volume': 1000 + np.random.normal(loc=0, scale=100, size=data_length).astype(int)
    })

    test_data.set_index('Date', inplace=True)
    test_data.to_csv(test_csv)

    # Backtester parameters
    start_date = datetime(2013, 5, 28, 16, 0)
    end_date = datetime(2013, 12, 31, 18, 0)
    backtest_params = {
        'symbols' : ['TEST'],
        #'history_dir': 'history/zorro',
        'aggregation': 'h',
        'start_date' : start_date,
        'end_date' : end_date,
        'look_back' : 80,
        'initial_capital' : 100000.0,
        'max_long' : -1,
        'max_short' : -1,
    }
    backtest = Backtest(
        #history_dir=history_files,
        backtest_params=backtest_params,
        strategy_class=MovingAverageCrossStrategy,
        strategy_params={}, 
        #symbols=symbol_list,
        #initial_capital=initial_capital,
        data_handler_class=HistoricCSVDataHandler,
        execution_handler_class=SimulatedExecutionHandler,
        portfolio_class=NaivePortfolio,
        reporting=False
    )
    backtest.run()
    # Assert that holdings have been updated
    assert len(backtest.portfolio.all_holdings) > 0, "No holdings were recorded."
    # Assert that the final portfolio value is greater than zero
    final_portfolio_value = backtest.portfolio.all_holdings[-1]['total']
    assert final_portfolio_value > 0, "Final portfolio value is zero or negative."
