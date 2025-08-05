from nexus.backtest import Backtest
from nexus.execution import SimulatedExecutionHandler
from nexus.feed.klines import HistoricKlineDataHandler
from nexus.portfolio import NaivePortfolio
from strategy.market_maker import MarketMakerStrategy

backtest_params = {
    "symbols": ["BTCUSDT"],
    "history_dir": "history/binance-futures",
    "start_date": "2024-01-01",
    "end_date": "2024-01-02",
    "look_back": 10,
    "initial_capital": 10000.0,
}

strategy_params = {
    "spread": 10.0,
    "order_refresh_time": 60,
}

backtest = Backtest(
    backtest_params=backtest_params,
    strategy_class=MarketMakerStrategy,
    strategy_params=strategy_params,
    data_handler_class=HistoricKlineDataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    reporting=True,
    open_in_browser=False
)

backtest.run()
