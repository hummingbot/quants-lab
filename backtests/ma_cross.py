import logging

from nexus.backtest import Backtest
from nexus.execution import SimulatedExecutionHandler
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.portfolio import NaivePortfolio
from strategy.ma_cross import MovingAverageCrossStrategy

# Parameters
# symbol_list = ['AAPL']
# history_files = {'AAPL': 'data/AAPL.csv'}
# initial_capital = 100000.0
# start_date = '2013-01-01'
# end_date = '2023-01-01'

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
    "short_window": 10,
    "long_window": 20,
}

backtest = Backtest(
    backtest_params=backtest_params,
    strategy_class=MovingAverageCrossStrategy,
    strategy_params=strategy_params,
    # history_dir=history_files,
    # symbols=symbol_list,
    # initial_capital=initial_capital,
    data_handler_class=HistoricCSVDataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    # start_date=start_date,
    # end_date=end_date,
    reporting=True,
    log_level=logging.DEBUG,
)

backtest.run()
