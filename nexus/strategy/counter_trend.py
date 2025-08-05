import numpy as np
from nexus.abc import Strategy
from nexus.indicators.bandpass import BandPass
from nexus.indicators.fisher import FisherN
from nexus.charting import PlotLocation

class CounterTrendStrategy(Strategy):
    """
    A strategy that uses a BandPass filter to identify market cycles for counter-trend trading.
    """
    
    def __init__(self, events, symbol_list, **kwargs):
        super().__init__(events, symbol_list, look_back=500, **kwargs)

        # From last proceed market event
        self.symbol = None 
        self.timestamp = None 
        self.price = None

        # Strategy Parameters with Defaults
        self.bandpass_period = kwargs.get('bandpass_period', 50)  # Default period
        self.bandpass_delta = kwargs.get('bandpass_delta', 0.1)    # Default Delta value
        self.fisher_period = kwargs.get('fisher_period', 500) 

        # Initialize BandPass filters for each symbol
        self.bandpass = {
            symbol: BandPass(period=self.bandpass_period, Delta=self.bandpass_delta) for symbol in self.symbols
        }
        
        self.fisher = {
            symbol: FisherN(period=self.fisher_period) for symbol in self.symbols
        }

        self.cycles = {
            symbol: np.zeros(length=self.fisher_period) for symbol in self.symbols
        }

        # Initialize indicator data storage for each symbol
        for symbol in self.symbols:
            self.indicator_data[symbol]['timestamps'] = []
            self.indicator_data[symbol]['cycles'] = {
                'values': [],
                'plot_location': PlotLocation.NEW  # Plot on a new chart
            }
            self.indicator_data[symbol]['signals'] = {
                'values': [],
                'plot_location': PlotLocation.NEW  # Plot on main chart
            }

    def run(self, event):
        """
        Processes market events and generates trading signals based on BandPass filter peaks and valleys.

        Parameters:
        - event (MarketEvent): The market data event.
        """
        timestamp, symbol, prices, indicators = self.process_event(event)
        bandpass = self.bandpass[symbol]
        cycles = self.cycles[symbol]
        fisher = self.fisher[symbol]

        # Apply the BandPass filter
        cycles.update(bandpass.update(prices))
        signals = fisher.update(cycles) 

        # Store the filtered value along with timestamp
        indicators['timestamps'].append(timestamp)
        indicators['cycles']['values'].append(cycles.last())
        indicators['signals']['values'].append(signals)

        """
        # Create a Series for BandPass trend to analyze peaks and valleys
        trends = Series(3)  # Need at least 3 points to detect a peak or valley
        for trend_value in self.indicator_data[symbol]['bandpass']['values'][-3:]:
            trends.update(trend_value)

        # Generate signals based on peak and valley
        if len(trends) < 3:
            return  # Not enough data to detect peak or valley

        if valley(trends):
            if self.position[symbol] == 'OUT':
                self.enterLong(symbol, datetime)
        elif peak(trends):
            if self.position[symbol] == 'LONG':
                self.enterShort(symbol, datetime)
        """
