import asyncio
import datetime
import logging
import os
from datetime import timedelta
from dotenv import load_dotenv
from decimal import Decimal

from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop

from controllers.directional_trading.trend_example import TrendExampleControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BaseStrategyConfigGenerator, BacktestingConfig
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class TrendExampleConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for MACD and Bollinger Bands optimization.
    """

    async def generate_config(self, trial) -> BacktestingConfig:
        # Suggest hyperparameters using the trial object
        interval = trial.suggest_categorical("interval", ["1m"])
        ema_short = trial.suggest_int("ema_short", 9, 30)
        ema_medium = trial.suggest_int("ema_medium", ema_short, 70)
        ema_long = trial.suggest_int("slow_ma", ema_medium, 200)
        total_amount_quote = 1000
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)
        take_profit = trial.suggest_float("take_profit", 0.04, 0.05, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.01)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.3, step=0.1)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15

        # Create the strategy configuration
        config = TrendExampleControllerConfig(
            connector_name="binance_perpetual",
            trading_pair="1000BONK-USDT",
            interval=interval,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            total_amount_quote=Decimal(total_amount_quote),
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price),
                trailing_delta=Decimal(trailing_stop_trailing_delta),
            ),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)

class TrendExampleBacktestingTask(BaseTask):
    async def execute(self):
        resolution = self.config["resolution"]
        kwargs = {
            "root_path": self.config["root_path"],
            "db_host": self.config["optuna_config"]["host"],
            "db_port": self.config["optuna_config"]["port"],
            "db_user": self.config["optuna_config"]["user"],
            "db_pass": self.config["optuna_config"]["password"],
            "database_name": self.config["optuna_config"]["database"],
        }
        storage_name = StrategyOptimizer.get_storage_name(
            engine=self.config.get("engine", "sqlite"),
            **kwargs)
        for trading_pair in self.config.get("selected_pairs"):
            connector_name = self.config.get("connector_name")
            optimizer = StrategyOptimizer(
                storage_name=storage_name,
                resolution=resolution,
                root_path=self.config["root_path"],
                custom_backtester=DirectionalTradingBacktesting()
            )

            optimizer.load_candles_cache_by_connector_pair(connector_name, trading_pair, self.config["root_path"])

            start_date = datetime.datetime(2024, 11, 1)
            end_date = datetime.datetime(2024, 11, 16)
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = TrendExampleConfigGenerator(start_date=start_date, end_date=end_date)
            config_generator.trading_pair = trading_pair
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"{self.config['study_name']}_{today_str}",
                                     config_generator=config_generator, n_trials=50)


async def main():
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "resolution": "1m",
        "optuna_config": optuna_config,
        "connector_name": "binance_perpetual",
        "selected_pairs": ["1000BONK-USDT"],
        "engine": "sqlite",
        "study_name": "Trend Example Docker",
    }

    task = TrendExampleBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
