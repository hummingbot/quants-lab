from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class RSIMultiTimeframeControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "rsi_multitimeframe"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
    )
    candles_trading_pair: str = Field(
        default=None,
    )
    timeframe_1: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the first timeframe (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=True))
    timeframe_2: str = Field(
        default="5m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the second timeframe (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=True))
    rsi_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI length: ",
            prompt_on_new=True))
    rsi_overbought: float = Field(
        default=80.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI overbought threshold: ",
            prompt_on_new=True))
    rsi_oversold: float = Field(
        default=20.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI oversold threshold: ",
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


class RSIMultiTimeframeController(DirectionalTradingControllerBase):
    def __init__(self, config: RSIMultiTimeframeControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = self.config.rsi_length * 2  # We double the length to have enough data for calculations
        
        if len(self.config.candles_config) == 0:
            # Initialize candles for both timeframes
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_trading_pair,
                    interval=config.timeframe_1,
                    max_records=self.max_records
                ),
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_trading_pair,
                    interval=config.timeframe_2,
                    max_records=self.max_records
                )
            ]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        # Get data for timeframe 1
        df1 = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.timeframe_1,
            max_records=self.max_records
        )
        
        # Get data for timeframe 2
        df2 = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.timeframe_2,
            max_records=self.max_records
        )

        # Calculate RSI for both timeframes
        df1.ta.rsi(length=self.config.rsi_length, append=True)
        df2.ta.rsi(length=self.config.rsi_length, append=True)

        # Rename RSI columns to distinguish between timeframes
        rsi_col = f"RSI_{self.config.rsi_length}"
        df1.rename(columns={rsi_col: f"{rsi_col}_tf1"}, inplace=True)
        df2.rename(columns={rsi_col: f"{rsi_col}_tf2"}, inplace=True)

        # For timeframe 2, shift the RSI values forward to ensure we're using the last completed candle
        # This prevents look-ahead bias by ensuring we only use completed candles
        df2[f"{rsi_col}_tf2"] = df2[f"{rsi_col}_tf2"].shift(1)

        # Merge the dataframes using asof merge to align timestamps
        # The merge_asof will match the closest past timestamp from df2 for each timestamp in df1
        merged_df = pd.merge_asof(
            df1,
            df2[["timestamp", f"{rsi_col}_tf2"]],
            left_on="timestamp",
            right_on="timestamp",
            direction='backward',
        )

        # Drop rows where we don't have both RSI values
        merged_df = merged_df.dropna(subset=[f"{rsi_col}_tf1", f"{rsi_col}_tf2"])

        # Generate signals based on RSI conditions
        rsi_tf1 = merged_df[f"{rsi_col}_tf1"]
        rsi_tf2 = merged_df[f"{rsi_col}_tf2"]

        # Initialize signal column
        merged_df["signal"] = 0

        # Long signal when both RSIs are below oversold
        long_condition = (rsi_tf1 < self.config.rsi_oversold) & (rsi_tf2 < self.config.rsi_oversold)
        # Short signal when both RSIs are above overbought
        short_condition = (rsi_tf1 > self.config.rsi_overbought) & (rsi_tf2 > self.config.rsi_overbought)

        merged_df.loc[long_condition, "signal"] = 1
        merged_df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = merged_df["signal"].iloc[-1]
        self.processed_data["features"] = merged_df
