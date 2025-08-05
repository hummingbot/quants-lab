"""
Binance futures data reader
Reads Binance futures klines data from a Parquet file and returns it as a Pandas DataFrame.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from pyarrow.lib import ArrowInvalid

from nexus.abc import DataFeed
from nexus.event import MarketEvent


def read_klines(path: Path) -> pd.DataFrame:
    """Read kline data from a Parquet file.

    Parameters
    ----------
    path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with kline data.

    Notes
    -----
    If the file is missing or cannot be parsed as parquet, ``FileNotFoundError``
    is raised so callers can fall back to synthetic data.
    """

    try:
        df = pd.read_parquet(path)
    except (FileNotFoundError, ArrowInvalid, OSError, ValueError) as exc:
        raise FileNotFoundError(path) from exc

    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    num_cols = ["open", "high", "low", "close"]
    df[num_cols] = df[num_cols].astype(float)
    return df


class HistoricKlineDataHandler(DataFeed):
    """Reads Binance futures klines data from Parquet files sequentially."""

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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ HistoricKlineDataHandler")

        self.events = events
        #self.start_date = pd.to_datetime(start_date) if start_date else None
        #self.end_date = pd.to_datetime(end_date) if end_date else None
        self.start_date = (
            pd.to_datetime(start_date, utc=True) if start_date else None
        )
        self.end_date = (
            pd.to_datetime(end_date, utc=True) if end_date else None
        )
        if isinstance(history_dir, dict):
            self.history_files = history_dir
            if symbols is None:
                symbols = list(history_dir.keys())
        else:
            if symbols is None:
                raise ValueError("symbols must be provided when history_dir is a path")
            self.history_files = {
                sym: os.path.join(history_dir, f"{sym}_1m.parquet") for sym in symbols
            }
        self.symbols = symbols

        self.symbol_data = {}
        self.latest_symbol_data = {symbol: [] for symbol in self.symbols}
        self.continue_backtest = True
        self.current_bar = {symbol: 0 for symbol in self.symbols}
        self._open_history_files()

    def _open_history_files(self):
        for symbol in self.symbols:
            file_name = self.history_files[symbol]
            self.logger.debug(f"Reading {file_name}")
            try:
                df = read_klines(Path(file_name))
            except FileNotFoundError:
                self.logger.warning(
                    f"History file {file_name} not found; using default data"
                )
                dates = pd.date_range(
                    self.start_date or "2000-01-01",
                    self.end_date or (self.start_date or "2000-01-10"),
                    freq="min",
                    tz="UTC",
                )
                df = pd.DataFrame(
                    {
                        "Open": 100.0,
                        "High": 190.0,
                        "Low": 80.0,
                        "Close": 120.0,
                        "Volume": 10,
                    },
                    index=dates,
                )
                df.index.name = "open_time"
                df.reset_index(inplace=True)
            if self.start_date is not None:
                df = df[df["open_time"] >= self.start_date]
            if self.end_date is not None:
                df = df[df["open_time"] <= self.end_date]
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df.set_index("open_time", inplace=True)
            df.sort_index(inplace=True)
            self.symbol_data[symbol] = df
        self.symbol_iterator = {
            symbol: self.symbol_data[symbol].iterrows() for symbol in self.symbols
        }

    def get_next_bar(self):
        for symbol in self.symbols:
            try:
                index, bar = next(self.symbol_iterator[symbol])
                bar.name = index
                self.latest_symbol_data[symbol].append(bar)
                self.current_bar[symbol] += 1
                bar.num = self.current_bar[symbol]
                bar["TWAP"] = (
                    bar["Open"] + bar["High"] + bar["Low"] + bar["Close"]
                ) / 4
                event = MarketEvent(symbol, bar, "kline")
                self.events.put(event)
            except StopIteration:
                self.continue_backtest = False

    def get_latest_bar_value(self, symbol, field):
        try:
            if field == "timestamp":
                return self.latest_symbol_data[symbol][-1].name
            else:
                return self.latest_symbol_data[symbol][-1][field]
        except IndexError:
            print(f"No data for symbol: {symbol}")
            return None
