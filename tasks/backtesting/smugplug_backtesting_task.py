import asyncio
import datetime
import logging
import os
from datetime import timedelta
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting

from core.backtesting.optimizer import StrategyOptimizer
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask
from research_notebooks.smugplug.smugplug_config_gen_simple import SmugPlugConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class BacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config
        self.root_path = self.config.get('root_path', "")

    def generate_top_markets_report(self, status_db_df: pd.DataFrame):
        df = status_db_df.copy()
        df.sort_values("volume_usd", ascending=False, inplace=True)
        screener_top_markets = df.head(self.screener_config["max_top_markets"])
        return screener_top_markets[["connector_name", "trading_pair", "from_timestamp", "to_timestamp"]]

    async def execute(self):
        ts_client = TimescaleClient(
            host=self.config.get("TIMESCALE_HOST"),
            port=self.config.get("TIMESCALE_PORT"),
        )
        await ts_client.connect()

        logger.info("Generating top markets report")
        metrics_df = await ts_client.get_db_status_df()
        top_markets_df = self.generate_top_markets_report(metrics_df)

        resolution = self.resolution
        optimizer = StrategyOptimizer(engine="postgres",
                                      root_path=self.root_path,
                                      resolution=resolution,
                                      db_client=ts_client,
                                      db_host=self.config.get("OPTUNA_HOST"),
                                      db_port=self.config.get("OPTUNA_DOCKER_PORT"),
                                      db_user=self.config.get("OPTUNA_USER"),
                                      db_pass=self.config.get("OPTUNA_PASSWORD")
                                      )
        logger.info("Optimizing strategy for top markets: {}".format(top_markets_df.shape[0]))
        for index, row in top_markets_df.iterrows():
            connector_name = row["connector_name"]
            trading_pair = row["trading_pair"]
            start_date = pd.Timestamp(row["from_timestamp"].timestamp(), unit="s")
            end_date = pd.Timestamp(row["to_timestamp"].timestamp(), unit="s")
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = SmugPlugConfigGenerator(start_date=start_date, end_date=end_date,
                                                       backtester=DirectionalTradingBacktesting())
            config_generator.trading_pair = trading_pair
            logger.info(f"Fetching candles for {connector_name} {trading_pair} {start_date} {end_date}")
            candles = await optimizer._db_client.get_candles(connector_name, trading_pair,
                                                             resolution, start_date.timestamp(), end_date.timestamp())
            if len(candles.data) == 0:
                continue
            start_time = candles.data["timestamp"].min()
            end_time = candles.data["timestamp"].max()
            config_generator.backtester.backtesting_data_provider.candles_feeds[
                f"{connector_name}_{trading_pair}_{resolution}"] = candles.data
            config_generator.start = int(start_time)
            config_generator.end = int(end_time)

            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"smugplug_test_task_{today_str}",
                                     config_generator=config_generator, n_trials=50)


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "total_amount": 500,
        "activation_bounds": 0.002,
        "max_executors_per_side": 1,
        "cooldown_time": 0,
        "leverage": 20,
        "time_limit": 86400,  # 60 * 60 * 24
        "bb_lengths": [50, 100, 200],
        "bb_stds": [1.0, 2.0, 3.0],
        "intervals": ["1m"],
        "volume_threshold": 0.5,
        "volatility_threshold": 0.5,
        "ts_delta_multiplier": 0.2,
        "max_top_markets": 50,
        "max_dca_amount_ratio": 5,
        "backtesting_resolution": "1m",
        "min_distance_between_orders": 0.01,
        "max_ts_sl_ratio": 0.5,
        "lookback_days": 7,
        "resolution": "1m",
        "TIMESCALE_HOST": os.getenv("TIMESCALE_HOST", "localhost"),
        "TIMESCALE_PORT": 5432,
        "OPTUNA_HOST": os.getenv("OPTUNA_HOST", "localhost"),
        "OPTUNA_DOCKER_PORT": 5433,
        "OPTUNA_USER": "admin",
        "OPTUNA_PASSWORD": "admin",
    }

    task = BacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
