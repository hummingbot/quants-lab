import pandas as pd

from nexus.abc import DataFeed
from nexus.event import MarketEvent


class TickCSVDataHandler(DataFeed):
    """
    TickCSVDataHandler reads tick data from CSV files and provides an interface to obtain the next tick or bar.
    Optionally aggregates tick data into bars of a specified frequency.
    """

    def __init__(
        self,
        events,
        history_files,
        symbol_list,
        start_date=None,
        end_date=None,
        aggregation=None,
        cache_dir=None,
    ):
        self.events = events
        self.history_files = history_files
        self.symbol_list = symbol_list
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.aggregation = aggregation
        self.symbol_data = {}
        self.latest_symbol_data = {symbol: [] for symbol in self.symbol_list}
        self.continue_backtest = True
        self.current_bar = {symbol: 0 for symbol in self.symbol_list}
        self._open_history_files()

    def _open_history_files(self):
        for symbol in self.symbol_list:
            # Read the CSV without parsing dates
            df = pd.read_csv(
                self.history_files[symbol],
                header=0,
                names=["TICKER", "PER", "DATE", "TIME", "LAST", "VOL"],
            )

            # Combine 'DATE' and 'TIME' into a 'datetime' column
            df["datetime"] = pd.to_datetime(
                df["DATE"].astype(str) + " " + df["TIME"], format="%Y%m%d %H:%M:%S"
            )

            # Drop unnecessary columns
            df.drop(["TICKER", "PER", "DATE", "TIME"], axis=1, inplace=True)

            # Reorder columns if necessary
            df = df[["datetime", "LAST", "VOL"]]

            # Apply date range filter if start_date or end_date is specified
            """
            if self.start_date:
                df = df[df['datetime'] >= self.start_date]
            if self.end_date:
                df = df[df['datetime'] <= self.end_date]
            """
            # Sort by 'datetime'
            df.sort_values("datetime", inplace=True)

            if self.aggregation:
                # Rename columns to standard names before aggregation
                df.rename(columns={"LAST": "Price", "VOL": "Volume"}, inplace=True)

                # Set 'datetime' as the index temporarily for resampling
                df.set_index("datetime", inplace=True)

                # Aggregate ticks into OHLCV bars
                bar_df = df.resample(self.aggregation).agg(
                    {"Price": ["first", "max", "min", "last"], "Volume": "sum"}
                )

                # Flatten MultiIndex columns
                bar_df.columns = ["Open", "High", "Low", "Close", "Volume"]

                # Reset index to turn 'datetime' back into a column
                bar_df.reset_index(inplace=True)

                # Drop bars with zero volume
                bar_df = bar_df[bar_df["Volume"] > 0]

                # Set 'datetime' as the index if desired
                bar_df.set_index("datetime", inplace=True)

                self.symbol_data[symbol] = bar_df
            else:
                # Use tick data directly
                # Rename columns to standard names
                df.rename(columns={"LAST": "Price", "VOL": "Volume"}, inplace=True)
                self.symbol_data[symbol] = df

        # Create iterator for the symbol
        self.symbol_iterator = {
            symbol: self.symbol_data[symbol].iterrows() for symbol in self.symbol_list
        }

    def get_next_bar(self):
        for symbol in self.symbol_list:
            try:
                index, bar = next(self.symbol_iterator[symbol])
                bar.name = index  # Set the name of the bar to the datetime index
                self.latest_symbol_data[symbol].append(bar)
                self.current_bar[symbol] += 1
                bar.num = self.current_bar[symbol]
                bar["TWAP"] = (
                    bar["Open"] + bar["High"] + bar["Low"] + bar["Close"]
                ) / 4
                event = MarketEvent(symbol, bar, "tick")
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


class LiveDataFeed(DataFeed):
    """Fetch live data from a broker API."""

    # Implementation depends on the broker's API
    pass
