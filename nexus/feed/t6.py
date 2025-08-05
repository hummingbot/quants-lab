import glob
import logging
import os
import struct
from datetime import datetime, timedelta

import pandas as pd

from nexus.abc import DataFeed
from nexus.event import MarketEvent


class HistoricT6DataHandler(DataFeed):
    """
    Historic T6 data handler with caching mechanism.
    Reads data from T6 files and caches aggregated data.
    """

    def __init__(
        self,
        events,
        symbols,
        history_dir,
        start_date,
        end_date,
        aggregation="h",
        cache_dir=None,
    ):
        """
        Initializes the data handler.

        Parameters:
        - events (queue.Queue): Event queue.
        - symbols (list): List of symbol strings.
        - history_dir (str): Directory containing T6 files.
        - start_date (datetime): Start date of backtest.
        - end_date (datetime): End date of backtest.
        - aggregation (str): Data aggregation level ('h' for hourly, etc.).
        - cache_dir (str): Directory to store cache files.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ class HistoricT6DataHandler()")
        self.events = events
        self.symbols = symbols
        self.history_dir = history_dir
        self.start_date = start_date
        self.end_date = end_date
        self.aggregation = aggregation

        # Use the history directory for caching if cache_dir is not specified
        self.cache_dir = cache_dir if cache_dir else history_dir

        self.symbol_data = {}
        self.latest_symbol_data = {symbol: [] for symbol in self.symbols}
        self.continue_backtest = True
        self.current_bar = {symbol: 0 for symbol in self.symbols}
        self._open_t6_files()

    def _open_t6_files(self):
        self.logger.debug("class HistoricT6DataHandler(): _open_t6_files")
        for symbol in self.symbols:
            # Generate a unique cache file name based on symbol and parameters
            cache_file_name = self._generate_cache_file_name(symbol)
            cache_file_path = os.path.join(self.cache_dir, cache_file_name)

            if os.path.exists(cache_file_path):
                # Load aggregated data from cache
                self.logger.info(
                    f"Loading cached data for {symbol} from {cache_file_path}"
                )
                data = pd.read_feather(cache_file_path)
                data.set_index("Date", inplace=True)
                self.symbol_data[symbol] = data
            else:
                # Read and aggregate data, then cache it
                self.logger.info(f"Processing data for {symbol}")
                data = self._process_symbol_data(symbol)
                if not data.empty:
                    # Save aggregated data to cache
                    self.logger.info(f"Caching data for {symbol} to {cache_file_path}")
                    data.reset_index().to_feather(cache_file_path)
                    self.symbol_data[symbol] = data
                else:
                    self.logger.warning(f"No data found for symbol: {symbol}")
                    self.symbol_data[symbol] = pd.DataFrame()

            self.logger.info(f"{symbol} : {len(self.symbol_data[symbol])} bars loaded")

        # Create iterators for each symbol
        self.symbol_iterator = {
            symbol: self.symbol_data[symbol].iterrows() for symbol in self.symbols
        }

    def _generate_cache_file_name(self, symbol):
        """
        Generates a unique cache file name based on symbol and parameters.

        Parameters:
        - symbol (str): The trading symbol.

        Returns:
        - str: The cache file name.
        """
        # Format dates for the file name
        start_str = self.start_date.strftime("%Y%m%d") if self.start_date else "start"
        end_str = self.end_date.strftime("%Y%m%d") if self.end_date else "end"
        aggregation_str = self.aggregation if self.aggregation else "raw"
        cache_file_name = (
            f"{symbol.replace('/', '')}_{start_str}_{end_str}_{aggregation_str}.feather"
        )
        return cache_file_name

    def _process_symbol_data(self, symbol):
        """
        Reads T6 files for a symbol, aggregates the data, and returns a DataFrame.

        Parameters:
        - symbol (str): The trading symbol.

        Returns:
        - pandas.DataFrame: The aggregated data.
        """
        # Assuming T6 files are named like 'EURUSD_*.t6' and located in history_dir
        file_spec = os.path.join(self.history_dir, f"{symbol.replace('/', '')}_*.t6")
        # Expand the file pattern into a list of file paths
        file_paths = glob.glob(file_spec)

        data_frames = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                data = self._read_t6_file(file_path)
                data_frames.append(data)
            else:
                self.logger.warning(f"File {file_path} does not exist.")

        if data_frames:
            # Concatenate all data frames for the symbol
            data = pd.concat(data_frames)
            # Sort by date
            data.sort_index(inplace=True)
            # Aggregate data if aggregation parameter is specified
            if self.aggregation:
                data = self._aggregate_data(data, self.aggregation)
            # Apply date range filter if start_date or end_date is specified
            if self.start_date:
                data = data[data.index >= self.start_date]
            if self.end_date:
                data = data[data.index <= self.end_date]
            return data
        else:
            return pd.DataFrame()

    def _read_t6_file(self, file_path):
        """
        Reads the binary T6 file and returns a pandas DataFrame.

        Parameters:
        - file_path (str): The path to the T6 binary file.

        Returns:
        - pandas.DataFrame: The data loaded from the T6 file.
        """
        self.logger.debug(f"class HistoricT6DataHandler() _read_t6_file({file_path})")
        records = []
        struct_format = "<d6f"  # Little-endian: 1 double, 6 floats
        record_size = struct.calcsize(struct_format)
        with open(file_path, "rb") as f:
            while True:
                bytes_read = f.read(record_size)
                if not bytes_read or len(bytes_read) < record_size:
                    break
                unpacked_data = struct.unpack(struct_format, bytes_read)
                # Create a dictionary of the unpacked data
                record = {
                    "Date": self._convert_date(unpacked_data[0]),
                    "High": unpacked_data[1],
                    "Low": unpacked_data[2],
                    "Open": unpacked_data[3],
                    "Close": unpacked_data[4],
                    "Val": unpacked_data[5],
                    "Vol": unpacked_data[6],
                }
                records.append(record)
        # Convert the records to a pandas DataFrame
        df = pd.DataFrame.from_records(records)
        # Ensure that 'Date' is of datetime type and set it as index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df

    def _convert_date(self, date_value):
        """
        Converts the DATE value from the T6 file to a Python datetime object.

        Parameters:
        - date_value (float): The date value from the T6 file.

        Returns:
        - datetime.datetime: The corresponding datetime object.
        """
        # Zorro's DATE format is days since 12/30/1899 (Excel serial date format)
        base_date = datetime(1899, 12, 30)
        delta = timedelta(days=date_value)
        return base_date + delta

    def _aggregate_data(self, data, aggregation):
        self.logger.debug("class HistoricT6DataHandler(): _aggregate_data")
        """
        Aggregates the data using the specified frequency.

        Parameters:
        - data (pandas.DataFrame): The DataFrame to aggregate.
        - aggregation (str): The frequency string for aggregation (e.g., '60Min').

        Returns:
        - pandas.DataFrame: The aggregated DataFrame.
        """
        # Define how to aggregate each column
        ohlc_dict_rule1 = {
            "Open": "sum",
            "High": "sum",
            "Low": "sum",
            "Close": "sum",
            "Val": "count",
            "Vol": "sum",
        }
        aggregated_data_1 = (
            data.resample(aggregation, label="right", closed="right")
            .agg(ohlc_dict_rule1)
            .dropna(how="any")
        )
        twap = (
            aggregated_data_1["Open"]
            + aggregated_data_1["High"]
            + aggregated_data_1["Low"]
            + aggregated_data_1["Close"]
        ) / (4 * aggregated_data_1["Val"])

        # Define how to aggregate each column
        ohlc_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Val": "sum",
            "Vol": "sum",
        }
        # Resample the data
        aggregated_data = (
            data.resample(aggregation, label="right", closed="right")
            .agg(ohlc_dict)
            .dropna(how="any")
        )

        # Add TWAP to aggregated data
        aggregated_data["TWAP"] = twap
        return aggregated_data

    def get_next_bar(self):
        """
        Retrieves the next bar for each symbol and places a MarketEvent into the event queue.
        """
        for symbol in self.symbols:
            try:
                index, bar = next(self.symbol_iterator[symbol])
                bar.name = index  # Set the name of the bar to the datetime index
                self.latest_symbol_data[symbol].append(bar)
                self.current_bar[symbol] += 1
                bar.num = self.current_bar[symbol]
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
            self.logger.error(f"No data for symbol: {symbol}")
            return None
