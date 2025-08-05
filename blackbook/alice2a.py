import logging
import numpy as np

from nexus.abc import Strategy
from nexus.indicators.bandpass import BandPass
from nexus.indicators.fisher import FisherN
from nexus.indicators.indicators import ATR
from nexus.charting import PlotLocation
from nexus.helpers import push, cross_over, cross_under, get_ohlc


class Alice2a(Strategy):
    """Counter trend strategy using a BandPass filter and Fisher transform."""

    def __init__(self, events, assets, strategy_params, backtest_params):
        super().__init__(events, assets, strategy_params, backtest_params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Strategy parameters with sensible defaults
        self.bandpass_period = strategy_params.get("bandpass_period", 30)
        self.bandpass_delta = strategy_params.get("bandpass_delta", 0.1)
        self.fisher_period = strategy_params.get("fisher_period", 500)
        self.atr_period = strategy_params.get("atr_period", 100)
        self.stop_loss_multiplier = strategy_params.get("stop_loss_multiplier", 10)
        self.threshold = strategy_params.get("threshold", 1.0)

        # Per symbol indicator state
        self.bandpass = {}
        self.cycles = {}
        self.fisher = {}
        self.signals = {}
        self.atr = {}

        for symbol in self.symbols:
            self.bandpass[symbol] = BandPass(self.bandpass_period, self.bandpass_delta)
            self.cycles[symbol] = np.zeros(self.look_back)
            self.fisher[symbol] = FisherN(self.fisher_period)
            self.signals[symbol] = np.zeros(self.look_back)
            self.atr[symbol] = ATR(self.atr_period)

        # Indicator data for charting
        for symbol in self.symbols:
            self.indicator_data[symbol]["timestamps"] = []
            self.indicator_data[symbol]["cycles"] = {
                "values": [],
                "plot_location": PlotLocation.NEW,
            }
            self.indicator_data[symbol]["signals"] = {
                "values": [],
                "plot_location": PlotLocation.NEW,
            }

    def run(self, event):
        """Process market events and generate trading signals."""
        timestamp, symbol, prices, indicators = self.process_event(event)

        bandpass = self.bandpass[symbol]
        cycles = self.cycles[symbol]
        fisher = self.fisher[symbol]
        signals = self.signals[symbol]
        atr = self.atr[symbol]

        # Update ATR
        open_p, high, low, close = get_ohlc(event.data)
        atr_value = atr.update(high, low, close)

        # Apply filters
        cycle_val = bandpass.update(prices)
        push(cycles, cycle_val)
        signal_val = fisher.update(cycles)
        push(signals, signal_val)

        # Check for signal crossovers
        thresh_pos = np.array([self.threshold, self.threshold])
        thresh_neg = np.array([-self.threshold, -self.threshold])
        if cross_under(signals, thresh_neg):
            stop_loss = close - self.stop_loss_multiplier * atr_value
            self.enterLong(stop_loss)
        elif cross_over(signals, thresh_pos):
            stop_loss = close + self.stop_loss_multiplier * atr_value
            self.enterShort(stop_loss)

        # Store indicator values
        indicators["timestamps"].append(timestamp)
        indicators["cycles"]["values"].append(cycle_val)
        indicators["signals"]["values"].append(signal_val)


############################### BACKTEST #######################################
if __name__ == "__main__":
    import logging
    from datetime import datetime

    from nexus.backtest import Backtest
    from nexus.feed.t6 import HistoricT6DataHandler
    from nexus.execution import SimulatedExecutionHandler
    from nexus.portfolio import NaivePortfolio

    start_date = datetime(2013, 5, 28, 16, 0)
    end_date = datetime(2013, 12, 31, 18, 0)
    backtest_params = {
        "symbols": ["EUR/USD"],
        "history_dir": "history/zorro",
        "aggregation": "h",
        "start_date": start_date,
        "end_date": end_date,
        "look_back": 80,
        "initial_capital": 100000.0,
        "max_long": -1,
        "max_short": -1,
    }

    strategy_params = {
        "bandpass_period": 30,
        "bandpass_delta": 0.1,
        "fisher_period": 500,
    }

    backtest = Backtest(
        backtest_params=backtest_params,
        strategy_class=Alice2a,
        strategy_params=strategy_params,
        data_handler_class=HistoricT6DataHandler,
        execution_handler_class=SimulatedExecutionHandler,
        portfolio_class=NaivePortfolio,
        reporting=True,
        log_level=logging.DEBUG,
        open_in_browser=False,
    )

    backtest.run()
