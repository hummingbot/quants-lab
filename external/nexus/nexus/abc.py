import logging
import numpy as np
from abc import ABC, abstractmethod
from nexus.event import SignalEvent
from nexus.helpers import push, get_ohlc, get_twap

class DataFeed(ABC):
    @abstractmethod
    def get_next_bar(self):
        pass

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event):
        pass

    def update_market(self, event):
        """Optional hook to process market events (e.g., for limit orders)."""
        pass

class Strategy(ABC):
    def __init__(self, events, assets, strategy_params, backtest_params):
        """
        Initializes the Strategy with an event queue and a list of symbols.

        Parameters:
        - events (queue.Queue): The event queue to communicate signals.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ class Strategy(ABC)")
        self.logger.debug(f"strategy_params: {strategy_params}")  

        self.events = events
        self.assets = assets
        self.symbols = backtest_params.get('symbols',[])
        self.look_back = backtest_params.get('look_back', 80)   # 80 limit by default
        self.max_long = backtest_params.get('max_long', -1)     # Maximum number of open and pending long trades with
        self.max_short = backtest_params.get('max_short', -1)   # Maximum number of open and pending short trades
        self.lots = backtest_params.get('lots', 1) # The smallest possible order size
        self.lot_amount = backtest_params.get('lot_amount', 1)  # The number of contracts per lot
        self.lot_lmit = backtest_params.get('lot_lmit',  1000000000/self.lot_amount) # Maximum number of lots 
  
        self.logger.info("*** Strategy parameters: ***")
        self.logger.info(f"* Assets:   {self.symbols}")
        self.logger.info(f"* look_back: {self.look_back}")
        self.logger.info(f"* max_long:  {self.max_long}")
        self.logger.info(f"* max_short: {self.max_short}")
        self.logger.info(f"* lots: {self.lots}")
        self.logger.info(f"* lot_lmit: {self.lot_lmit}")
        self.logger.info(" ")

        self.indicator_data = {symbol: {} for symbol in self.symbols}

        # Initialize price series for each symbol
        self.prices = {
            symbol: np.zeros(self.look_back) for symbol in self.symbols 
        }

        # Position Tracking: 'OUT', 'LONG', 'SHORT'
        self.position = {symbol: 'OUT' for symbol in self.symbols}
        self.open_long_trades = {symbol: 0 for symbol in self.symbols}   # Track number of open long trades per symbol
        self.open_short_trades = {symbol: 0 for symbol in self.symbols}  # Track number of open short trades per symbol
        
        # Current bar data
        self.current_bar_number = None
        self.timestamp = None
        self.symbol = None
        self.asset = None

    @abstractmethod
    def run(self, event):
        """
        Processes market events and generates trading signals.

        Parameters:
        - event (MarketEvent): The market event to process.
        """
        pass

    def process_event(self, event):
        """
        Processes market events and set internal variables.
        !!!Actially proceed bar data
        Parameters:
        - event (MarketEvent): The market data event.
        """
        if event.type != 'MARKET':
            return  # Only process market events
        self.symbol = event.symbol
        self.asset = self.assets[self.symbol]
        self.timestamp = event.data.name 
        self.current_bar_number = event.data.num
        #price = get_price(event.data)
        twap_price = get_twap(event.data)
        push(self.prices[self.symbol], twap_price)
        open, high, low, close = get_ohlc(event.data)
        self.logger.info(
            f"[{self.current_bar_number+358}: {self.timestamp.day_name()[:3]} {self.timestamp}] {open:.5f}/{high:.5f}/{low:.5f}/{close:.5f}"
        )  # {twap_price:.5f}
        #self.logger.info(f"[{bar_number}: {timestamp}] ({price:.4f})")
        if self.look_back == self.current_bar_number:
            self.logger.info("***** End of lookback period *****\n")
        return self.timestamp, self.symbol, self.prices[self.symbol], self.indicator_data[self.symbol]
    
    def enterLong(self, stop_loss=None, limit_price=None):
        """
        Enters a LONG position if trade limits allow.
        """
        if self.current_bar_number < self.look_back:
            self.logger.info(f"({self.symbol}::L)Skipped (no trading)\n")
            return

        current_open_long = self.open_long_trades.get(self.symbol, 0)
        current_open_short= self.open_short_trades.get(self.symbol, 0)

        quantity = self.lots * self.asset.lot_amount
        if (self.max_short == -1) and (current_open_short > 0):
            self.exitShort()
        
        if (self.max_long == -1) or (current_open_long < self.max_long):
            signal = SignalEvent(self.symbol, self.timestamp, 'LONG', quantity,
                                 stop_loss=stop_loss, limit_price=limit_price)
            self.events.put(signal)
            self.position[self.symbol] = 'LONG'
            self.open_long_trades[self.symbol] += 1
            self.logger.info(f"Entered LONG position for {self.symbol} at {self.timestamp}. Open LONG trades: {self.open_long_trades[self.symbol]}")
        else:
            self.logger.info(f"Max LONG limit reached for {self.symbol}. Cannot enter more LONG positions.")

    def exitLong(self):
        """
        Exits all LONG positions for the symbol.
        """
        current_open_long = self.open_long_trades.get(self.symbol, 0)
        if current_open_long > 0:
            quantity = self.lots * self.asset.lot_amount * current_open_long
            signal = SignalEvent(self.symbol, self.timestamp, 'EXIT', quantity)
            self.events.put(signal)
            self.position[self.symbol] = 'OUT'
            self.open_long_trades[self.symbol] = 0
            self.logger.info(f"Exited all LONG positions for {self.symbol} at {self.timestamp}.")
        else:
            self.logger.info(f"No LONG positions open for {self.symbol}. Cannot exit LONG.")

    def enterShort(self, stop_loss=None, limit_price=None):
        """
        Enters a SHORT position if trade limits allow.
        """
        if self.current_bar_number < self.look_back:
            self.logger.info(f"({self.symbol}::S)Skipped (no trading)\n")
            return

        current_open_long = self.open_long_trades.get(self.symbol, 0)
        current_open_short = self.open_short_trades.get(self.symbol, 0)
        quantity = self.lots * self.asset.lot_amount

        if (self.max_long == -1) and (current_open_long > 0):
            self.exitLong()

        if (self.max_short == -1) or (current_open_short < self.max_short):
            signal = SignalEvent(self.symbol, self.timestamp, 'SHORT', quantity,
                                 stop_loss=stop_loss, limit_price=limit_price)
            self.events.put(signal)
            self.position[self.symbol] = 'SHORT'
            self.open_short_trades[self.symbol] += 1
            self.logger.info(f"Entered SHORT position for {self.symbol} at {self.timestamp}. Open SHORT trades: {self.open_short_trades[self.symbol]}")
        else:
            self.logger.info(f"Max SHORT limit reached for {self.symbol}. Cannot enter more SHORT positions.")

    def exitShort(self):
        """
        Exits all SHORT positions for the symbol.
        """
        current_open_short = self.open_short_trades.get(self.symbol, 0)
        if current_open_short > 0:
            quantity = self.lots * self.asset.lot_amount * current_open_short
            signal = SignalEvent(self.symbol, self.timestamp, 'EXIT', quantity)
            self.events.put(signal)
            self.position[self.symbol] = 'OUT'
            self.open_short_trades[self.symbol] = 0
            self.logger.info(f"Exited all SHORT positions for {self.symbol} at {self.timestamp}.")
        else:
            self.logger.info(f"No SHORT positions open for {self.symbol}. Cannot exit SHORT.")

    def exitPositions(self):
        """
        Exits all positions for the symbol.
        """
        self.exitLong()
        self.exitShort()

class Portfolio(ABC):
    @abstractmethod
    def update_time_index(self, event):
        pass

    @abstractmethod
    def update_signal(self, event):
        pass

    @abstractmethod
    def update_fill(self, event):
        pass
