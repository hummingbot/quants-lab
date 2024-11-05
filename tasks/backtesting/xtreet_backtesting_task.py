import asyncio
import datetime
import logging
import os
from datetime import timedelta
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

from core.backtesting.optimizer import StrategyOptimizer
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting
from research_notebooks.xtreet_bb.xtreet_config_gen_simple import XtreetConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class BacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config
        self.root_path = self.config.get('root_path', "")

    def generate_top_markets_report(self, metrics_df: pd.DataFrame):
        metrics_df.sort_values(by=['trading_pair', 'to_timestamp'], ascending=[True, False])
        screener_report = metrics_df.drop_duplicates(subset='trading_pair', keep='first')
        screener_report.sort_values("mean_natr", ascending=False, inplace=True)
        natr_percentile = screener_report['mean_natr'].astype(float).quantile(
            self.screener_config["volatility_threshold"])
        volume_percentile = screener_report['average_volume_per_hour'].astype(float).quantile(
            self.screener_config["volume_threshold"])
        screener_top_markets = screener_report[
            (screener_report['mean_natr'] > natr_percentile) &
            (screener_report['average_volume_per_hour'] > volume_percentile)
            ].sort_values(by="average_volume_per_hour").head(self.screener_config["max_top_markets"])
        return screener_top_markets[["connector_name", "trading_pair", "from_timestamp", "to_timestamp"]]

    async def execute(self):
        ts_client = TimescaleClient(
            host=self.config.get("TIMESCALE_HOST", "localhost"),
            port=self.config.get("TIMESCALE_PORT", 5432),
            user=self.config.get("TIMESCALE_USER", "admin"),
            password=self.config.get("TIMESCALE_PASSWORD", "admin"),
            database="timescaledb"
        )
        await ts_client.connect()

        logger.info("Generating top markets report")
        metrics_df = await ts_client.get_metrics_df()
        top_markets_df = self.generate_top_markets_report(metrics_df)

        resolution = self.resolution
        optimizer = StrategyOptimizer(engine="postgres",
                                      root_path=self.root_path,
                                      resolution=resolution,
                                      db_client=ts_client,
                                      db_host=self.config.get("OPTUNA_HOST", "localhost"),
                                      db_port=self.config.get("OPTUNA_DOCKER_PORT", 5433),
                                      db_user=self.config.get("OPTUNA_USER", "admin"),
                                      db_pass=self.config.get("OPTUNA_PASSWORD", "admin"))
        logger.info("Optimizing strategy for top markets: {}".format(top_markets_df.shape[0]))
        for index, row in top_markets_df.iterrows():
            connector_name = row["connector_name"]
            trading_pair = row["trading_pair"]
            start_date = pd.Timestamp(row["from_timestamp"].timestamp(), unit="s")
            end_date = pd.Timestamp(row["to_timestamp"].timestamp(), unit="s")
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = XtreetConfigGenerator(start_date=start_date, end_date=end_date,
                                                     backtester=XtreetBacktesting())
            config_generator.trading_pair = trading_pair
            logger.info(f"Fetching candles for {connector_name} {trading_pair} {start_date} {end_date}")
            candles = await optimizer._db_client.get_candles(connector_name, trading_pair,
                                                             resolution, start_date.timestamp(), end_date.timestamp())
            start_time = candles.data["timestamp"].min()
            end_time = candles.data["timestamp"].max()
            config_generator.backtester.backtesting_data_provider.candles_feeds[
                f"{connector_name}_{trading_pair}_{resolution}"] = candles.data
            config_generator.start = start_time
            config_generator.end = end_time

            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"xtreet_bb_task_{today_str}",
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
        "bb_stds": [1.0, 1.4, 1.8, 2.0, 3.0],
        "intervals": ["1m", "5m", "15m"],
        "volume_threshold": 0.5,
        "volatility_threshold": 0.5,
        "ts_delta_multiplier": 0.2,
        "max_top_markets": 20,
        "max_dca_amount_ratio": 5,
        "backtesting_resolution": "1s",
        "min_distance_between_orders": 0.01,
        "max_ts_sl_ratio": 0.5,
        "lookback_days": 7,
        "resolution": "1s",
        "TIMESCALE_HOST": "localhost",
        "TIMESCALE_PORT": 5432,
        "TIMESCALE_USER": "admin",
        "TIMESCALE_PASSWORD": "admin",
        "OPTUNA_HOST": "localhost",
        "OPTUNA_DOCKER_PORT": 5433,
        "OPTUNA_USER": "admin",
        "OPTUNA_PASSWORD": "admin",
    }

    task = BacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
