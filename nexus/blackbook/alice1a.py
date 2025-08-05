import logging
from datetime import datetime

import numpy as np

from nexus.abc import Strategy
from nexus.indicators.indicators import ATR
from nexus.indicators.laguerre import LaguerreFilter
from nexus.charting import PlotLocation
from nexus.helpers import get_ohlc, peak, push, valley
from nexus.backtest import Backtest
from nexus.feed.t6 import HistoricT6DataHandler
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio

class Alice1a(Strategy):
    """
    A strategy that uses a Laguerre filter to smooth price data and generates signals based on peaks and valleys.
    """
    def __init__(self, events, assets, strategy_params, backtest_params):
        super().__init__(events, assets, strategy_params, backtest_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Strategy Parameters with Defaults
        self.laguerre_alpha = strategy_params.get('laguerre_alpha', 0.05)  # Default alpha value
        self.atr_period = strategy_params.get('atr_period', 14)  # Default ATR period
        self.stop_loss_multiplier = strategy_params.get('stop_loss_multiplier', 10)  # Default stop loss multiplier

        # Initialize Laguerre filters and ATR for each symbol
        self.laguerre = {}
        self.trends = {}
        self.atr = {}

        for symbol in self.symbols:
            self.laguerre[symbol] = LaguerreFilter(self.laguerre_alpha)
            self.trends[symbol] = np.zeros(self.look_back)
            self.atr[symbol] = ATR(self.atr_period)

        # Initialize indicator data storage for each symbol
        for symbol in self.symbols:
            self.indicator_data[symbol]['timestamps'] = []
            self.indicator_data[symbol]['trends'] = {
                'values': [],
                'plot_location': PlotLocation.MAIN  # Plot on main chart
            }

    def run(self, event):
        """
        Processes market events and generates trading signals based on Laguerre filter peaks and valleys.

        Parameters:
        - event (MarketEvent): The market data event.
        """
        timestamp, symbol, prices, indicators = self.process_event(event)
        atr = self.atr[symbol]
        laguerre = self.laguerre[symbol]
        trends = self.trends[symbol]

        # Update ATR
        open, high, low, close = get_ohlc(event.data)
        #self.logger.info(f"<<price,{timestamp},{prices[0]:.8f}>>") 
        atr_value = atr.update(high, low, close)

        # Update the Laguerre filter
        push(trends, laguerre.update(prices))
        #self.logger.info(f"<<laguerre,{timestamp},{trends[0]:.8f}>>") 
        #self.logger.info(f"<<atr,{timestamp},{atr_value:.5f}>>")   
        
        # Generate signals based on peak and valley
        if valley(trends):
            # Calculate stop loss price for long position
            stop_loss = close - self.stop_loss_multiplier * atr_value
            self.enterLong(stop_loss)
        elif peak(trends): 
            # Calculate stop loss price for short position
            stop_loss = close + self.stop_loss_multiplier * atr_value
            self.enterShort(stop_loss)
        
        # Store the trend value along with timestamp
        indicators['timestamps'].append(timestamp)
        indicators['trends']['values'].append(trends[0])

############################### BACKTEST ###########################################

# Backtester parameters
start_date = datetime(2013, 5, 28, 16, 0)
end_date = datetime(2013, 12, 31, 18, 0)
backtest_params = {
    'symbols' : ['EUR/USD'],
    'history_dir': 'history/zorro',
    'aggregation': 'h',
    'start_date' : start_date,
    'end_date' : end_date,
    'look_back' : 80,
    'initial_capital' : 100000.0,
    'max_long' : -1,
    'max_short' : -1,
}

# Strategy parameters
strategy_params = {
    'laguerre_alpha': 0.05,  # Smoothing factor
}

# Instantiate the backtest
backtest = Backtest(
    backtest_params=backtest_params,
    strategy_class=Alice1a,
    strategy_params=strategy_params,
    data_handler_class=HistoricT6DataHandler,
    execution_handler_class=SimulatedExecutionHandler,
    portfolio_class=NaivePortfolio,
    reporting=True,
    log_level=logging.DEBUG,
    open_in_browser=False
)

# Run the backtest
backtest.run()

"""

from misc.trade_cmp import comare_trades
print("Zorro vs Nexus:")
file1 = 'C:/Users/maxpa/Zorro/Log/testtrades.csv'
file2 = 'C:/Projects/nexus/log/Alice1a/trades_log.csv'
comare_trades(file1, file2)
"""

"""
// Zorro’s lite-C is a version of C 
// Trend Trading 
#include <profile.c>

function run()
{
	StartDate = 2013;
	EndDate = 2018; // fixed simulation period
	
	vars Prices = series(price());
	vars Trends = series(Laguerre(Prices,0.05));
	
	Stop = 10*ATR(100);
	MaxLong = MaxShort = -1;
	
	if(valley(Trends))
		enterLong();
	else if(peak(Trends))
		enterShort();
	
	set(LOGFILE); // log all trades
	plotTradeProfile(-50); 
}
"""