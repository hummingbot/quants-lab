import logging
import os

import pandas as pd

from nexus.abc import DataFeed
from nexus.event import MarketEvent


class HistoricCSVDataHandler(DataFeed):
    """Reads CSV OHLC data and provides bars sequentially."""

    def __init__(
        self,
        events,
        history_dir,
        symbols=None,
        start_date=None,
        end_date=None,
        aggregation=None,
        cache_dir=None,
    ):
        """Load CSV files for the requested symbols.

        Parameters
        ----------
        events : queue.Queue
            Queue on which ``MarketEvent`` objects will be placed.
        history_dir : str or dict
            If ``str`` it is interpreted as a directory containing ``<symbol>.csv``
            files. If ``dict`` it should map symbols to explicit file paths.
        symbols : list, optional
            List of symbols. Required when ``history_dir`` is a directory.
        start_date, end_date : datetime, optional
            Optional date range to load.
        aggregation : str, optional
            Unused but kept for backwards compatibility.
        cache_dir : str, optional
            Unused but kept for API compatibility.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ HistoricCSVDataHandler")

        self.events = events
        self.start_date = start_date
        self.end_date = end_date

        # Determine symbol -> file mapping
        if isinstance(history_dir, dict):
            self.history_files = history_dir
            if symbols is None:
                symbols = list(history_dir.keys())
        else:
            if symbols is None:
                raise ValueError("symbols must be provided when history_dir is a path")
            self.history_files = {
                sym: os.path.join(history_dir, f"{sym}.csv") for sym in symbols
            }
        self.symbols = symbols

        self.symbol_data = {}
        self.latest_symbol_data = {symbol: [] for symbol in self.symbols}
        self.continue_backtest = True
        self.current_bar = {symbol: 0 for symbol in self.symbols}
        self._open_history_files()

    def _open_history_files(self):
        self.logger.debug("class HistoricCSVDataHandler(): _open_history_files")
        for symbol in self.symbols:
            file_name = self.history_files[symbol]
            self.logger.debug(f"Reading {file_name}")
            try:
                df = pd.read_csv(
                    file_name, header=0, index_col="Date", parse_dates=True
                )
            except FileNotFoundError:
                # Fallback dummy data when file is missing
                self.logger.warning(
                    f"History file {file_name} not found; using default data"
                )
                dates = pd.date_range(
                    self.start_date or "2000-01-01",
                    self.end_date or (self.start_date or "2000-01-10"),
                    freq="D",
                )
                df = pd.DataFrame(
                    {
                        "Open": 100.0,
                        "High": 100.0,
                        "Low": 100.0,
                        "Close": 100.0,
                        "Volume": 0,
                    },
                    index=dates,
                )
            # Apply date range filter if start_date or end_date is specified
            if self.start_date:
                df = df[df.index >= self.start_date]
            if self.end_date:
                df = df[df.index <= self.end_date]
            # Sort by index
            df.sort_index(inplace=True)
            self.symbol_data[symbol] = df
        # Create iterators for each symbol
        self.symbol_iterator = {
            symbol: self.symbol_data[symbol].iterrows() for symbol in self.symbols
        }

    def get_next_bar(self):
        for symbol in self.symbols:
            try:
                index, bar = next(self.symbol_iterator[symbol])
                bar.name = index  # Set the name of the bar to the datetime index
                self.latest_symbol_data[symbol].append(bar)
                self.current_bar[symbol] += 1
                bar.num = self.current_bar[symbol]
                bar["TWAP"] = (
                    bar["Open"] + bar["High"] + bar["Low"] + bar["Close"]
                ) / 4
                event = MarketEvent(symbol, bar, "ohlc")
                self.events.put(event)
            except StopIteration:
                self.continue_backtest = False

    def get_latest_bar_value(self, symbol, field):
        """
        Retrieves the latest value for a given field of a symbol.

        Parameters:
        - symbol (str): The symbol to retrieve data for.
        - field (str): The field to retrieve (e.g., 'Open', 'High').

        Returns:
        - float: The latest value for the specified field.
        """
        try:
            if field == "timestamp":
                return self.latest_symbol_data[symbol][-1].name
            else:
                return self.latest_symbol_data[symbol][-1][field]

        except IndexError:
            print(f"No data for symbol: {symbol}")
            return None
