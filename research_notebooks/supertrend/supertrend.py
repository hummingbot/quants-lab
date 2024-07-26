from typing import List, Optional

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator


class SuperTrendConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "supertrend_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: Optional[str] = None
    candles_trading_pair: Optional[str] = None
    interval: str = Field(default="3m", client_data=ClientFieldData(
        prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ", prompt_on_new=False))
    length: int = 20
    multiplier: float = 3
    smoothing_window: int = 100
    threshold: float = 0.5

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


class SuperTrend(DirectionalTradingControllerBase):
    def __init__(self, config: SuperTrendConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.length + 10
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
        # Add indicator
        df.ta.supertrend(length=self.config.length, multiplier=self.config.multiplier, append=True)
        df.fillna(0, inplace=True)

        # Generate long and short conditions
        long_condition = df[f"SUPERTd_{self.config.length}_{self.config.multiplier}"] == 1
        short_condition = df[f"SUPERTd_{self.config.length}_{self.config.multiplier}"] == -1

        # Choose side
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        df["smoothed_signal"] = df["signal"].rolling(window=self.config.smoothing_window).mean()
        df["signal"] = 0

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        lower_threshold = -self.config.threshold
        upper_threshold = self.config.threshold
        df.loc[
            (df["smoothed_signal"] > lower_threshold) &
            (df["smoothed_signal"].shift(1) <= lower_threshold),
            "signal"
        ] = 1
        df.loc[
            (df["smoothed_signal"] < upper_threshold) &
            (df["smoothed_signal"].shift(1) >= upper_threshold),
            "signal"
        ] = -1
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
