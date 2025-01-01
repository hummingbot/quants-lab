from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator

class WhiteRabbitV1ControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "whiterabbit_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    bb_length: int = Field(
        default=100,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands length: ",
            prompt_on_new=True))
    bb_std: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands standard deviation: ",
            prompt_on_new=False))
    bb_long_threshold: float = Field(
        default=0.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands long threshold: ",
            prompt_on_new=True))
    bb_short_threshold: float = Field(
        default=1.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands short threshold: ",
            prompt_on_new=True))
    rsi_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI length: ",
            prompt_on_new=True))
    rsi_overbought: float = Field(
        default=80,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI Overbought: ",
            prompt_on_new=True))
    rsi_oversold: float = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI Oversold: ",
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

class WhiteRabbitV1Controller(DirectionalTradingControllerBase):
    def __init__(self, config: WhiteRabbitV1ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.bb_length, config.rsi_length)
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        # Fetch candles data as DataFrame and ensure it's a copy
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Add indicators
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        df.ta.rsi(length=self.config.rsi_length, overbought=self.config.rsi_overbought, oversold=self.config.rsi_oversold, append=True)

        # Bollinger Bands percentage and RSI
        bbp = df[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
        rsi = df[f"RSI_{self.config.rsi_length}"]

       # Long and short conditions using OR (|) and AND (&) operators
        long_condition = (bbp < self.config.bb_long_threshold) & (rsi < self.config.rsi_oversold)
        short_condition = (bbp > self.config.bb_short_threshold) & (rsi > self.config.rsi_overbought)

        # Adjusted conditions for outside bands
        #outside_lower_band = df["close"] < self.config.bb_long_threshold
        #outside_upper_band = df["close"] > self.config.bb_short_threshold

        # Reversal conditions: Using AND to require both price and volume conditions
        #reverse_to_long = (outside_lower_band & (df["volume"] > df["vol_ma"]))
        #reverse_to_short = (outside_upper_band & (df["volume"] > df["vol_ma"]))

        # Signals initialization
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df





