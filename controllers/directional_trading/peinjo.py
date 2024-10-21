from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from core.features.candles.mean_reversion_channel import MeanReversionChannel, MeanReversionChannelConfig
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class PeinjoControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "peinjo"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    length: int = Field(
        default=200,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the length of the mean reversion channel: ",
            prompt_on_new=True))
    inner_mult: float = Field(
        default=1.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the inner multiplier of the mean reversion channel: ",
            prompt_on_new=True))
    outer_mult: float = Field(
        default=2.415,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the outer multiplier of the mean reversion channel: ",
            prompt_on_new=True))
    source: str = Field(
        default="hlc3",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the source of the mean reversion channel: ",
            prompt_on_new=True))
    filter_type: str = Field(
        default="SuperSmoother",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the filter type of the mean reversion channel: ",
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


class PeinjoController(DirectionalTradingControllerBase):

    def __init__(self, config: PeinjoControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.length + 1500
        self.mean_reversion_channel = MeanReversionChannel(
            MeanReversionChannelConfig(
                length=config.length,
                inner_mult=config.inner_mult,
                outer_mult=config.outer_mult,
                source=config.source,
                filter_type=config.filter_type
            )
        )
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
        candles = self.mean_reversion_channel.calculate(df)
        candles.index = pd.to_datetime(candles.timestamp, unit="s")

        # Compute signals
        candles = self.compute_signal(candles)

        # Drop index
        candles = candles.reset_index(drop=True)

        # Update processed data
        self.processed_data["signal"] = candles["signal"].iloc[-1]
        self.processed_data["features"] = candles

    def compute_signal(self, df):
        # Initialize signals
        df['signal'] = 0
        # Group by date
        grouped = df.groupby(df.index.date)
        for date, group in grouped:
            # Compute open prices for 8, 9, 10, 11 hours for this specific day
            open_prices = group.between_time('08:00', '11:59')['open'].resample('1H').first()
            if not open_prices.empty:
                min_open = open_prices.min()
                max_open = open_prices.max()
                # Create time mask for trading hours (12:00 to 15:00)
                time_mask = (group.index.time >= pd.Timestamp('12:00').time()) \
                    & (group.index.time <= pd.Timestamp('15:00').time())
                # Apply conditions
                buy_condition = (group['low'] <= group['loband2']) & (group['close'] < min_open)
                sell_condition = (group['high'] >= group['upband2']) & (group['close'] > max_open)
                # Combine conditions
                combined_mask = time_mask & (buy_condition | sell_condition)
                # Set signals
                df.loc[group.index[combined_mask & buy_condition], 'signal'] = 1  # Buy signal
                df.loc[group.index[combined_mask & sell_condition], 'signal'] = -1  # Sell signal
        return df
