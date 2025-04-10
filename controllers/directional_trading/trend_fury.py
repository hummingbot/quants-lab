from typing import List

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction
from pydantic import Field, validator, field_validator
from pydantic_core.core_schema import ValidationInfo

from core.features.candles.trend_fury import TrendFury, TrendFuryConfig


class TrendFuryControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "trend_fury"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = None
    candles_trading_pair: str = None
    interval: str = "3m"
    window: int = 50
    vwap_window: int = 50
    use_returns: bool = False
    use_volume_weighting: bool = False
    volume_normalization_window: int = 50
    cum_diff_quantile_threshold: float = 0.5
    reversal_sensitivity: float = 0.3
    use_vwap_filter: bool = False
    use_slope_filter: bool = False
    slope_quantile_threshold: float = 0.4

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


class TrendFuryController(DirectionalTradingControllerBase):
    def __init__(self, config: TrendFuryControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(self.config.window, self.config.vwap_window, self.config.volume_normalization_window)
        self.trend_fury = TrendFury(
            TrendFuryConfig(window=config.window, vwap_window=config.vwap_window, use_returns=config.use_returns,
                            use_volume_weighting=config.use_volume_weighting,
                            volume_normalization_window=config.volume_normalization_window,
                            cum_diff_quantile_threshold=config.cum_diff_quantile_threshold,
                            reversal_sensitivity=config.reversal_sensitivity,
                            use_vwap_filter=config.use_vwap_filter,
                            use_slope_filter=config.use_slope_filter,
                            slope_quantile_threshold=config.slope_quantile_threshold))
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

        df = self.trend_fury.calculate(df)

        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        stop_actions = []
        signal = self.processed_data["signal"]
        if signal == 1:
            short_executors = self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.is_active and x.side == TradeType.SELL)
            stop_actions.extend([StopExecutorAction(controller_id=self.config.id, executor_id=executor.id)
                                 for executor in short_executors])
        elif signal == -1:
            long_executors = self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.is_active and x.side == TradeType.BUY)
            stop_actions.extend([StopExecutorAction(controller_id=self.config.id, executor_id=executor.id)
                                 for executor in long_executors])
        return stop_actions
