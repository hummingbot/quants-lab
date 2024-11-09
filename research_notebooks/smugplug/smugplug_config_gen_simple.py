from decimal import Decimal

from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.smug_plug_v1 import SmugPlugControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class SmugPlugConfigGenerator(BaseStrategyConfigGenerator):
    trading_pair = None

    async def generate_config(self, trial: trial) -> BacktestingConfig:
        connector_name = "binance_perpetual"
        trading_pair = self.trading_pair
        interval = trial.suggest_categorical("interval", ["3m", "5m", "1h"])

        # Generate specific SmugPlug metrics
        macd_fast = trial.suggest_int("macd_fast", 10, 30, step=10)
        macd_slow = trial.suggest_int("macd_slow", 30, 60, step=10)
        macd_signal = trial.suggest_int("macd_signal", 9, 18, step=3)
        ema_short = trial.suggest_int("ema_short", 8, 16, step=8)
        ema_medium = trial.suggest_int("ema_medium", 20, 30, step=2)
        ema_long = trial.suggest_int("ema_long", 30, 40, step=2)
        atr_length = trial.suggest_int("atr_length", 8, 16, step=2)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 2.0, step=0.5)

        # Triple barrier metrics
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.01)
        trailing_stop_activation = trial.suggest_float("trailing_stop_activation", 0.01, 0.05, step=0.01)
        trailing_stop_delta = trial.suggest_float("trailing_stop_delta", 0.006, 0.012, step=0.002)
        trailing_stop = TrailingStop(
            activation_price=Decimal(trailing_stop_activation),
            trailing_delta=Decimal(trailing_stop_delta)
        )
        # Id Generation
        controller_id = (f"smugplug_{connector_name}_{interval}_{trading_pair}_"
                         f"macd_{macd_fast}_{macd_slow}_{macd_signal}_"
                         f"ema_{ema_short}_{ema_medium}_{ema_long}_"
                         f"atr_{atr_length}_{atr_multiplier}"
                         f"sl{round(100 * stop_loss, 1)}_"
                         f"ts{round(100 * trailing_stop_activation, 1)}-"
                         f"{round(100 * trailing_stop_delta, 1)}")

        config = SmugPlugControllerConfig(
            id=controller_id,
            total_amount_quote=Decimal("500"),
            connector_name=connector_name,
            trading_pair=trading_pair,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
            candles_trading_pair=trading_pair,
            interval=interval,
            stop_loss=Decimal(stop_loss),
            trailing_stop=trailing_stop,
            time_limit=60 * 60 * 2,
            cooldown_time=60,
        )
        return BacktestingConfig(config=config, start=self.start, end=self.end)
