from decimal import Decimal
from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType, OrderType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig, \
    TrailingStop
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction
from pydantic import Field, validator

from core.features.candles.peak_analyzer import PeakAnalyzer


class XGridTControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "xgridt"
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
    # EMAs
    ema_short: int = 8
    ema_medium: int = 29
    ema_long: int = 31
    donchian_channel_length = 50
    natr_length = 100
    natr_multiplier = 2.0
    tp_default = 0.05

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


class XGridTController(DirectionalTradingControllerBase):

    def __init__(self, config: XGridTControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.ema_short, config.ema_medium, config.ema_long, config.donchian_channel_length,
                               config.natr_length) + 20
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
        df.ta.donchian(lower_length=self.config.donchian_channel_length,
                       upper_length=self.config.donchian_channel_length, append=True)
        df.ta.natr(length=self.config.natr_length, append=True)

        short_ema = df[f"EMA_{self.config.ema_short}"]
        medium_ema = df[f"EMA_{self.config.ema_medium}"]
        long_ema = df[f"EMA_{self.config.ema_long}"]

        long_condition = (short_ema > medium_ema) & (medium_ema > long_ema) & (short_ema > long_ema)
        short_condition = (short_ema < medium_ema) & (medium_ema < long_ema) & (short_ema < long_ema)

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        analyzer = PeakAnalyzer(df)
        peaks = analyzer.get_peaks(prominence_percentage=0.05, distance=100)

        high_peaks = peaks["high_peaks"]
        low_peaks = peaks["low_peaks"]
        df.loc[high_peaks[0], "TP_LONG"] = high_peaks[1]
        df.loc[low_peaks[0], "TP_SHORT"] = low_peaks[1]
        df["TP_LONG"].ffill(inplace=True)
        df["TP_SHORT"].ffill(inplace=True)

        # Apply the function to create the TP_LONG column
        df["TP_LONG"] = df.apply(
            lambda x: x.TP_LONG if pd.notna(x.TP_LONG) and x.TP_LONG > x.high else self.get_unbounded_tp(x,
                                                                                                         self.config.tp_default,
                                                                                                         TradeType.BUY,
                                                                                                         high_peaks,
                                                                                                         low_peaks),
            axis=1)
        df["TP_SHORT"] = df.apply(
            lambda x: x.TP_SHORT if pd.notna(x.TP_SHORT) and x.TP_SHORT < x.low else self.get_unbounded_tp(x,
                                                                                                           self.config.tp_default,
                                                                                                           TradeType.SELL,
                                                                                                           high_peaks,
                                                                                                           low_peaks),
            axis=1)

        df["SL_LONG"] = df[f"DCL_{self.config.donchian_channel_length}_{self.config.donchian_channel_length}"]
        df["SL_SHORT"] = df[f"DCU_{self.config.donchian_channel_length}_{self.config.donchian_channel_length}"]
        df["LIMIT_LONG"] = df[f"DCL_{self.config.donchian_channel_length}_{self.config.donchian_channel_length}"] * (
                1 + self.config.natr_multiplier * df[f"NATR_{self.config.natr_length}"])
        df["LIMIT_SHORT"] = df[f"DCU_{self.config.donchian_channel_length}_{self.config.donchian_channel_length}"] * (
                1 - self.config.natr_multiplier * df[f"NATR_{self.config.natr_length}"])
        # Update processed data
        self.processed_data.update(df.iloc[-1].to_dict())
        self.processed_data["features"] = df

    @staticmethod
    def get_unbounded_tp(row, tp_default, side, high_peaks, low_peaks, criteria="latest"):
        timestamp = row.name
        close = row["close"]
        if side == TradeType.BUY:
            previous_peaks_higher_than_price = [price_peak for price_timestamp, price_peak in
                                                zip(high_peaks[0], high_peaks[1]) if
                                                price_timestamp < timestamp and price_peak > close]
            if previous_peaks_higher_than_price:
                if criteria == "latest":
                    return previous_peaks_higher_than_price[-1]
                elif criteria == "closest":
                    return min(previous_peaks_higher_than_price, key=lambda x: abs(x - row["close"]))
            else:
                return close * (1 + tp_default)
        else:
            previous_peaks_lower_than_price = [price_peak for price_timestamp, price_peak in
                                               zip(low_peaks[0], low_peaks[1]) if
                                               price_timestamp < timestamp and price_peak < close]
            if previous_peaks_lower_than_price:
                if criteria == "latest":
                    return previous_peaks_lower_than_price[-1]
                elif criteria == "closest":
                    return min(previous_peaks_lower_than_price, key=lambda x: abs(x - row["close"]))
            else:
                return close * (1 - tp_default)

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal):
        tp_price = self.processed_data["TP_LONG"] if trade_type == TradeType.BUY else self.processed_data["TP_SHORT"]
        sl_price = self.processed_data["SL_LONG"] if trade_type == TradeType.BUY else self.processed_data["SL_SHORT"]
        limit_price = self.processed_data["LIMIT_LONG"] if trade_type == TradeType.BUY else self.processed_data[
            "LIMIT_SHORT"]
        tp_pct = abs(Decimal(tp_price) - price) / price
        sl_pct = abs(Decimal(sl_price) - price) / price
        ts_ap = tp_pct * Decimal("0.5")
        ts_td = ts_ap * Decimal("0.2")
        triple_barrier_config = TripleBarrierConfig(
            stop_loss=sl_pct,
            take_profit=tp_pct,
            time_limit=self.config.time_limit,
            trailing_stop=TrailingStop(
                activation_price=ts_ap,
                trailing_delta=ts_td
            ),
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.MARKET,
        )
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=trade_type,
            entry_price=price,
            amount=amount,
            triple_barrier_config=triple_barrier_config,
            leverage=self.config.leverage,
        )

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        signal = self.processed_data["signal"]
        if signal == 1:
            active_short_executors = self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.side == TradeType.SELL and x.is_active
            )
            return [StopExecutorAction(controller_id=self.config.id,
                                       executor_id=executor.id) for executor in active_short_executors]
        elif signal == -1:
            active_long_executors = self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.side == TradeType.BUY and x.is_active
            )
            return [StopExecutorAction(controller_id=self.config.id,
                                       executor_id=executor.id) for executor in active_long_executors]
        return []
