import logging

import numpy as np

from nexus.abc import Strategy
from nexus.charting import PlotLocation
from nexus.helpers import cross_over, cross_under, push
from nexus.indicators.indicators import SMA


class MovingAverageCrossStrategy(Strategy):
    """
    A concise Moving Average Cross Strategy.
    """

    def __init__(self, events, assets, strategy_params, backtest_params):
        super().__init__(events, assets, strategy_params, backtest_params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Strategy Parameters with Defaults
        # Use sensible integer defaults for moving average windows
        self.short_window = strategy_params.get("short_window", 50)
        self.long_window = strategy_params.get("long_window", 100)

        # Parameter Validation
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window.")

        # Initialize series for each symbol
        self.sma_short = {}
        self.sma_long = {}

        # Initialize series for each symbol
        for symbol in self.symbols:
            self.sma_short[symbol] = np.zeros(self.look_back)
            self.sma_long[symbol] = np.zeros(self.look_back)
        # self.sma_short = {symbol: np.zeros(self.look_back) for symbol in symbol_list}
        # self.sma_long = {symbol: np.zeros(self.look_back) for symbol in symbol_list}

        # Initialize indicator data storage for each symbol with plot locations
        for symbol in self.symbols:
            self.indicator_data[symbol]["timestamps"] = []
            self.indicator_data[symbol]["sma_short"] = {
                "values": [],
                "plot_location": PlotLocation.MAIN,  # Plot on main chart
            }
            self.indicator_data[symbol]["sma_long"] = {
                "values": [],
                "plot_location": PlotLocation.MAIN,  # Plot on main chart
            }

    def run(self, event):
        """
        Main strategy function, it runs at the end of every bar.
        Processes market events and generates trading signals.
        """
        timestamp, symbol, prices, indicators = self.process_event(event)
        sma_short = self.sma_short[symbol]
        sma_long = self.sma_long[symbol]

        # Calculate moving averages & update the SMA series
        push(sma_short, SMA(prices, self.short_window))
        push(sma_long, SMA(prices, self.long_window))

        # Check for moving average crossover signals
        if cross_over(sma_short, sma_long):
            self.enterLong()
        elif cross_under(sma_short, sma_long):
            self.enterShort()

        # Store the moving averages and timestamp
        indicators["timestamps"].append(timestamp)
        indicators["sma_short"]["values"].append(sma_short[0])
        indicators["sma_long"]["values"].append(sma_long[0])
