import logging

import numpy as np

from nexus.abc import Strategy
from nexus.charting import PlotLocation
from nexus.indicators.laguerre import LaguerreFilter
from nexus.helpers import peak, push, valley


class LaguerrePeakValleyStrategy(Strategy):
    """
    A strategy that uses a Laguerre filter to smooth price data and generates signals based on peaks and valleys.
    """

    def __init__(self, events, assets, strategy_params, backtest_params):
        super().__init__(events, assets, strategy_params, backtest_params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Strategy Parameters with Defaults
        self.laguerre_alpha = strategy_params.get("laguerre_alpha", 0.05)  # Default alpha value

        # Initialize Laguerre filters for each symbol
        self.laguerre_filters = {
            symbol: LaguerreFilter(self.laguerre_alpha) for symbol in self.symbols
        }

        self.trends = {symbol: np.zeros(self.look_back) for symbol in self.symbols}

        # Initialize indicator data storage for each symbol
        for symbol in self.symbols:
            self.indicator_data[symbol]["timestamps"] = []
            self.indicator_data[symbol]["trends"] = {
                "values": [],
                "plot_location": PlotLocation.MAIN,  # Plot on main chart
            }

    def run(self, event):
        """
        Processes market events and generates trading signals based on Laguerre filter peaks and valleys.

        Parameters:
        - event (MarketEvent): The market data event.
        """
        timestamp, symbol, prices, indicators = self.process_event(event)
        laguerre = self.laguerre_filters[symbol]
        trends = self.trends[symbol]

        # Update the Laguerre filter
        push(trends, laguerre.update(prices))

        # Generate signals based on peak and valley
        if valley(trends):
            self.enterLong()
        elif peak(trends):
            self.enterShort()

        # Store the trend value along with timestamp
        indicators["timestamps"].append(timestamp)
        indicators["trends"]["values"].append(trends[0])
