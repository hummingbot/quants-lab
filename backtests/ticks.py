from nexus.backtest import Backtest
from nexus.feed.csv_tick import TickCSVDataHandler
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio
from strategy.ma_cross import MovingAverageCrossStrategy

# Parameters
history_files = {
    'RIH0': 'data/RIH0_200103_200103.csv',
}
symbol_list = ['RIH0']
initial_capital = 1.0
start_date = '2020-01-03'
end_date = '2020-01-03'

# Strategy parameters
strategy_params = {
    'short_window': 10,
    'long_window': 20, 
}

# Instantiate the backtest
backtest = Backtest(
    start_date=start_date,
    end_date=end_date,
    history_dir=history_files,
    symbols=symbol_list,
    initial_capital=initial_capital,
    strategy_class=MovingAverageCrossStrategy,
    strategy_params=strategy_params,
    data_handler_class=TickCSVDataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    data_handler_params={
        'aggregation': '1Min'  # Aggregate ticks into 1-minute bars
    },
    reporting=True
)

# Run the backtest
backtest.run()
