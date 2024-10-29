from typing import List

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction
from pydantic import Field, validator

from core.features.candles.trend_fury import TrendFury, TrendFuryConfig


class TrendFuryControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "trend_fury"
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
    window: int = Field(
        default=50,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the rolling window size for regression: ",
            prompt_on_new=True))
    vwap_window: int = Field(
        default=50,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the rolling window size for VWAP: ",
            prompt_on_new=True))
    use_returns: bool = Field(
        default=False,
        client_data=ClientFieldData(
            prompt=lambda mi: "Use returns instead of prices? (True/False): ",
            prompt_on_new=True))
    use_volume_weighting: bool = Field(
        default=False,
        client_data=ClientFieldData(
            prompt=lambda mi: "Use volume-weighted regression? (True/False): ",
            prompt_on_new=True))
    volume_normalization_window: int = Field(
        default=50,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the window size for volume normalization: ",
            prompt_on_new=True))
    cum_diff_quantile_threshold: float = Field(
        default=0.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the threshold for significant slope changes (0-1): ",
            prompt_on_new=True))
    reversal_sensitivity: float = Field(
        default=0.3,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the sensitivity for detecting reversals (0-1): ",
            prompt_on_new=True))
    use_vwap_filter: bool = Field(
        default=False,
        client_data=ClientFieldData(
            prompt=lambda mi: "Use VWAP based signal filtering? (True/False): ",
            prompt_on_new=True))
    use_slope_filter: bool = Field(
        default=False,
        client_data=ClientFieldData(
            prompt=lambda mi: "Use slope based signal filtering? (True/False): ",
            prompt_on_new=True))
    slope_quantile_threshold: float = Field(
        default=0.4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the threshold for slope quantile (0-1): ",
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
