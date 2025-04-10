from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, field_validator


class TrendExampleControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "trend_example"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(default="3m")
    # EMAs
    ema_short: int = 8
    ema_medium: int = 29
    ema_long: int = 31

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class TrendExampleController(DirectionalTradingControllerBase):

    def __init__(self, config: TrendExampleControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.ema_short, config.ema_medium, config.ema_long) + 20
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        # Add indicators
        df.ta.ema(length=self.config.ema_short, append=True)
        df.ta.ema(length=self.config.ema_medium, append=True)
        df.ta.ema(length=self.config.ema_long, append=True)

        short_ema = df[f"EMA_{self.config.ema_short}"]
        medium_ema = df[f"EMA_{self.config.ema_medium}"]
        long_ema = df[f"EMA_{self.config.ema_long}"]

        long_condition = (short_ema > medium_ema) & (medium_ema > long_ema) & (short_ema > long_ema)
        short_condition = (short_ema < medium_ema) & (medium_ema < long_ema) & (short_ema < long_ema)

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        self.processed_data.update(df.iloc[-1].to_dict())
        self.processed_data["features"] = df

