from nexus.backtest import Backtest
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio
from strategy.counter_trend import CounterTrendStrategy

# Parameters
symbol_list = ['AAPL']
history_files = {'AAPL': 'data/AAPL.csv'}
initial_capital = 100000.0
start_date = '2013-01-01'
end_date = '2023-01-01'

# Strategy parameters
strategy_params = {
    'max_long' : -1,
    'max_short' : -1,
    'laguerre_alpha': 0.1,  # Smoothing factor
}

# Instantiate the backtest
backtest = Backtest(
    history_dir=history_files,
    symbols=symbol_list,
    initial_capital=initial_capital,
    strategy_params=strategy_params,
    strategy_class=CounterTrendStrategy,
    data_handler_class=HistoricCSVDataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    start_date=start_date,
    end_date=end_date,
    reporting=True
)

# Run the backtest
backtest.run()


