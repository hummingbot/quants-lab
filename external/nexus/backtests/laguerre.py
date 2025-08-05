import logging

from nexus.backtest import Backtest
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio
from strategy.laguerre import LaguerrePeakValleyStrategy

backtest_params = {
    "symbols": ["AAPL"],
    "history_dir": "history/yf",
    #'aggregation': 'h',
    "start_date": "2013-01-01",
    "end_date": "2023-01-01",
    "look_back": 80,
    "initial_capital": 100000.0,
    "max_long": -1,
    "max_short": -1,
}

strategy_params = {
    'laguerre_alpha': 0.1,  # Smoothing factor
}

backtest = Backtest(
    backtest_params=backtest_params,
    strategy_class=LaguerrePeakValleyStrategy,
    strategy_params=strategy_params,
    data_handler_class=HistoricCSVDataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    reporting=True,
    log_level=logging.DEBUG,
)

# Run the backtest
backtest.run()