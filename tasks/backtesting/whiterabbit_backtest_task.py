import asyncio
import datetime
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict
from decimal import Decimal

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop

from controllers.directional_trading.macd_bb_v1 import MACDBBV1ControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class WhiteRabbitConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for WhiteRabbit optimization.
    """

    async def generate_config(self, trial) -> BacktestingConfig:
        # Suggest hyperparameters using the trial object
        bb_length = trial.suggest_int("bb_length", 10, 200, step=5)
        bb_std = trial.suggest_float("bb_std", 0.5, 3.5, step=0.25)
        bb_long_threshold = trial.suggest_float("bb_long_threshold", 0.0, 0.2, step=0.02)
        bb_short_threshold = trial.suggest_float("bb_short_threshold", 0.8, 1.0, step=0.02)
        rsi_length = trial.suggest_int("rsi_length", 7, 21, step=2)
        rsi_overbought = trial.suggest_float("rsi_overbought", 80, 99, step=1)
        rsi_oversold = trial.suggest_float("rsi_oversold", 1, 20, step=1)
        total_amount_quote = 100
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)
        take_profit = trial.suggest_float("take_profit", 0.04, 0.05, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.01)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.3, step=0.1)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15

        # Create the strategy configuration
        config = WhiteRabbitV1ControllerConfig(
            connector_name=self.config["connector_name"],
            trading_pair=self.config["trading_pair"],
            bb_length=bb_length,
            bb_std=bb_std,
            bb_long_threshold=bb_long_threshold,
            bb_short_threshold=bb_short_threshold,
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            total_amount_quote=Decimal(total_amount_quote),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class WhiteRabbitBacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config
        self.root_path = self.config.get('root_path', "")

    async def execute(self):
        optimizer = StrategyOptimizer(root_path=self.root_path, resolution=self.resolution, load_cached_data=True)
        selected_pairs = ['BTC-USDT', 'DOGE-USDT', 'ETH-USDT', 'SUI-USDT', 'SOL-USDT', 'XRP-USDT']
        connector_name = "binance_perpetual"
        for trading_pair in selected_pairs:
            end_date = time.time() - self.config["end_time_buffer_hours"]
            start_date = end_date - self.config["lookback_days"] * 24 * 60 * 60
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = WhiteRabbitConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"),
                end_date=pd.to_datetime(end_date, unit="s"),
                config={"connector_name": self.config["connector_name"], "trading_pair": trading_pair}
            )
            logger.info(f"Fetching candles for {connector_name} {trading_pair} {start_date} {end_date}")
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"white_rabbit_v1_task_{today_str}",
                                     config_generator=config_generator, n_trials=50)


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "binance_perpetual",
        "total_amount": 100,
        "lookback_days": 7,
        "end_time_buffer_hours": 6,
        "resolution": "1m",
    }

    task = WhiteRabbitBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
