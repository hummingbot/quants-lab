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


class XtreetBacktestingTask(BaseTask):
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
            host=self.config["timescale_config"]["host"],
            port=self.config["timescale_config"]["port"],
            user=self.config["timescale_config"]["user"],
            password=self.config["timescale_config"]["password"],
            database=self.config["timescale_config"]["database"]
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
                                      db_host=self.config["optuna_config"]["host"],
                                      db_port=self.config["optuna_config"]["port"],
                                      db_user=self.config["optuna_config"]["user"],
                                      db_pass=self.config["optuna_config"]["password"],
                                      database_name=self.config["optuna_config"]["database"],
                                      )
        selected_pairs = [
            # Cluster 1
            '1000SHIB-USDT', 'WLD-USDT',
            # Cluster 2
            'ACT-USDT', '1000BONK-USDT',
            # Cluster 3
            'DOGE-USDT', 'AGLD-USDT',
            # Cluster 4
            'SUI-USDT', '1000SATS-USDT',
            # Cluster 5
            'MOODENG-USDT', 'NEIRO-USDT',
            # Cluster 6
            'HBAR-USDT', 'ENA-USDT',
            # Cluster 7
            'HMSTR-USDT', 'TROY-USDT',
            # Cluster 8
            '1000PEPE-USDT', '1000X-USDT',
            # Cluster 9
            'PNUT-USDT', 'SOL-USDT',
            # Cluster 10
            'XRP-USDT', 'SWELL-USDT'
        ]
        logger.info("Optimizing strategy for top markets: {}".format(top_markets_df.shape[0]))
        for index, row in top_markets_df[top_markets_df["trading_pair"].isin(selected_pairs)].iterrows():
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
            await optimizer.optimize(study_name=f"data_server_xtreet_bb_task_{today_str}",
                                     config_generator=config_generator, n_trials=50)


async def main():
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
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
        "timescale_config": timescale_config,
        "optuna_config": optuna_config
    }

    task = XtreetBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
