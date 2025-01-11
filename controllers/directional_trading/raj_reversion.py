from typing import List

import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
import numpy as np
import pandas as pd


class RAJReversionControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "raj_reversion"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
    )
    candles_trading_pair: str = Field(
        default=None,
    )
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    
    # ALMA Parameters
    close_alma_length: int = Field(
        default=80,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the close ALMA length: ",
            prompt_on_new=True))
    close_alma_offset: float = Field(
        default=0.85,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the close ALMA offset: ",
            prompt_on_new=True))
    close_alma_sigma: int = Field(
        default=16,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the close ALMA sigma: ",
            prompt_on_new=True))
    
    # Pivot Parameters
    pivot_left: int = Field(
        default=7,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the left pivot length: ",
            prompt_on_new=True))
    pivot_right: int = Field(
        default=7,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the right pivot length: ",
            prompt_on_new=True))
    array_percent: float = Field(
        default=86,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the percentile threshold: ",
            prompt_on_new=True))
    
    # ALMA over mean Parameters
    diff_alma_length: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the diff ALMA length: ",
            prompt_on_new=True))
    diff_alma_offset: float = Field(
        default=0.85,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the diff ALMA offset: ",
            prompt_on_new=True))
    diff_alma_sigma: int = Field(
        default=16,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the diff ALMA sigma: ",
            prompt_on_new=True))
    
    percentile_rolling_window: int = Field(
        default=100,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the rolling window for percentile calculation: ",
            prompt_on_new=True))

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class RAJReversionController(DirectionalTradingControllerBase):
    def __init__(self, config: RAJReversionControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(
            self.config.close_alma_length,
            self.config.pivot_left + self.config.pivot_right,
            self.config.diff_alma_length
        ) + 20  # Add buffer for calculations
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def calculate_pivots(self, df: pd.DataFrame, source_column: str, left: int, right: int):
        """
        Calculate pivot highs and lows based on a source column.
        
        Args:
            df (pd.DataFrame): DataFrame containing the price data
            source_column (str): Column name to use for pivot calculations
            left (int): Number of bars to look left
            right (int): Number of bars to look right
        
        Returns:
            tuple: (pivot_highs_idx, pivot_highs, pivot_lows_idx, pivot_lows)
                - pivot_highs_idx: List of datetime indices where pivot highs occur
                - pivot_highs: List of high prices at pivot high points
                - pivot_lows_idx: List of datetime indices where pivot lows occur
                - pivot_lows: List of low prices at pivot low points
        """
        pivot_highs_index = []
        pivot_lows_index = []
        
        # Find pivot points
        for i, value in enumerate(df[source_column]):
            left_index = max(0, i-left)
            right_index = min(len(df), i+right+1)
            range_values = df.iloc[left_index:right_index][source_column]
            
            if value == range_values.max():
                pivot_highs_index.append(i)
            if value == range_values.min():
                pivot_lows_index.append(i)
        
        # Convert indices to datetime and get corresponding prices
        pivot_highs_idx = [df.iloc[i].name for i in pivot_highs_index]
        pivot_lows_idx = [df.iloc[i].name for i in pivot_lows_index]
        pivot_highs = [df.iloc[i][source_column] for i in pivot_highs_index]
        pivot_lows = [df.iloc[i][source_column] for i in pivot_lows_index]
        
        return pivot_highs_idx, pivot_highs, pivot_lows_idx, pivot_lows

    def calculate_alma(self, series: pd.Series, window_size: int = 9, offset: float = 0.85, sigma: float = 6) -> pd.Series:
        """
        Calculate Arnaud Legoux Moving Average (ALMA)
        
        Args:
            series (pd.Series): Input price series
            window_size (int): The window size for the moving average
            offset (float): Controls the smoothing (from 0 to 1)
            sigma (float): Controls the smoothing width
        
        Returns:
            pd.Series: ALMA values
        """
        # Initialize the result series
        result = pd.Series(index=series.index, dtype=float)
        
        # Calculate the offset point
        m = offset * (window_size - 1)
        # Calculate s
        s = window_size / sigma
        
        # Calculate weights
        weights = np.zeros(window_size)
        for i in range(window_size):
            weights[i] = np.exp(-1 * (i - m) ** 2 / (2 * s ** 2))
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate ALMA
        for i in range(window_size - 1, len(series)):
            window = series.iloc[i - window_size + 1:i + 1]
            result.iloc[i] = (window * weights).sum()
        
        return result

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Calculate ALMA
        df["alma"] = self.calculate_alma(
            df["close"],
            window_size=self.config.close_alma_length,
            offset=self.config.close_alma_offset,
            sigma=self.config.close_alma_sigma
        )

        # Calculate source
        df["src"] = np.where(
            ((df["high"] - df["alma"]) > abs(df["low"] - df["alma"])) & 
            ((df["high"] - df["open"]) > abs(df["low"] - df["open"])),
            df["high"],
            np.where(
                ((df["high"] - df["alma"]) < abs(df["low"] - df["alma"])) &
                ((df["high"] - df["open"]) < abs(df["low"] - df["open"])),
                df["low"],
                df["close"]
            )
        )

        # Calculate percentage difference
        df["diff"] = 100 * (df["src"] - df["alma"]) / df["alma"]

        # Calculate ALMA over mean
        df["alma_over_mean"] = self.calculate_alma(
            df["diff"],
            window_size=self.config.diff_alma_length,
            offset=self.config.diff_alma_offset,
            sigma=self.config.diff_alma_sigma
        )

        # Calculate Pivots
        pivot_highs_idx, pivot_highs, pivot_lows_idx, pivot_lows = self.calculate_pivots(
            df=df,
            source_column="diff",
            left=self.config.pivot_left,
            right=self.config.pivot_right
        )

        # Create a Series with pivot values at their respective indices
        pivot_series = pd.Series(index=df.index, dtype=float)
        pivot_series.loc[pivot_highs_idx] = pivot_highs
        pivot_series.loc[pivot_lows_idx] = pivot_lows

        # Calculate rolling percentile (using last 100 periods by default)
        df["pct_rank"] = (
            pivot_series.rolling(window=self.config.percentile_rolling_window, min_periods=1)
            .apply(lambda x: np.percentile(x.dropna(), self.config.array_percent))
            .fillna(method='ffill')
        )

        # Generate signals using the rolling percentile
        df["cross_over_alma_diff"] = (df["diff"] > df["alma_over_mean"]) & (df["diff"].shift(1) < df["alma_over_mean"].shift(1))
        df["cross_under_alma_diff"] = (df["diff"] < df["alma_over_mean"]) & (df["diff"].shift(1) > df["alma_over_mean"].shift(1))
        
        long_condition = df["cross_over_alma_diff"] & (df["diff"] < -df["pct_rank"])
        short_condition = df["cross_under_alma_diff"] & (df["diff"] > df["pct_rank"])

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
