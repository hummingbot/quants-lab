from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator


class SmugPlugControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "smugplug"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    macd_fast: int = Field(
        default=21,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=42,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    ema_short: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA short period: ",
            prompt_on_new=True))
    ema_medium: int = Field(
        default=29,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA medium period: ",
            prompt_on_new=True))
    ema_long: int = Field(
        default=31,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA long period: ",
            prompt_on_new=True))
    atr_length: int = Field(
        default=11,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR length: ",
            prompt_on_new=True))
    atr_multiplier: float = Field(
        default=1.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR multiplier: ",
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


class SmugPlugController(DirectionalTradingControllerBase):

    def __init__(self, config: SmugPlugControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        # Add indicators and retrieve actual column names
        macd_columns = df.ta.macd(
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            append=True
        )
        atr_columns = df.ta.atr(length=self.config.atr_length, append=True)
        ema_short_columns = df.ta.ema(length=self.config.ema_short, append=True)
        ema_medium_columns = df.ta.ema(length=self.config.ema_medium, append=True)
        ema_long_columns = df.ta.ema(length=self.config.ema_long, append=True)

        # Get actual column names
        macd_hist_col = macd_columns.name  # MACD histogram column
        atr_col = atr_columns.name
        ema_short_col = ema_short_columns.name
        ema_medium_col = ema_medium_columns.name
        ema_long_col = ema_long_columns.name

        # Ensure columns exist
        required_columns = [macd_hist_col, atr_col, ema_short_col, ema_medium_col, ema_long_col]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column {col} not found in DataFrame.")

        # Proceed with calculations
        df["long_atr_support"] = df["close"].shift(1) - df[atr_col] * self.config.atr_multiplier
        df["short_atr_resistance"] = df["close"].shift(1) + df[atr_col] * self.config.atr_multiplier

        short_ema = df[ema_short_col]
        medium_ema = df[ema_medium_col]
        long_ema = df[ema_long_col]
        close = df["close"]
        macdh = df[macd_hist_col]

        long_condition = (
            (short_ema > medium_ema) &
            (medium_ema > long_ema) &
            (close > short_ema) &
            (close > df["long_atr_support"]) &
            (macdh > 0)
        )
        short_condition = (
            (short_ema < medium_ema) &
            (medium_ema < long_ema) &
            (close < short_ema) &
            (close < df["short_atr_resistance"]) &
            (macdh < 0)
        )

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
