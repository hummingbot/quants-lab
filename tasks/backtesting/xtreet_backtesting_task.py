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
from core.utils import load_dict_from_yaml
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting
from research_notebooks.xtreet_bb.xtreet_config_gen_simple import XtreetConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class BacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config["config"]
        self.root_path = config.get('root_path', "")

    def generate_top_markets_report(self, metrics_df: pd.DataFrame):
        metrics_df.sort_values(by=['trading_pair', 'to_timestamp'], ascending=[True, False])
        screener_report = metrics_df.drop_duplicates(subset='trading_pair', keep='first')
        screener_report.sort_values("mean_natr", ascending=False, inplace=True)
        natr_percentile = screener_report['mean_natr'].astype(float).quantile(
            self.screener_config["screener_params"]["volatility_threshold"])
        volume_percentile = screener_report['average_volume_per_hour'].astype(float).quantile(
            self.screener_config["screener_params"]["volume_threshold"])
        screener_top_markets = screener_report[
            (screener_report['mean_natr'] > natr_percentile) &
            (screener_report['average_volume_per_hour'] > volume_percentile)
            ].sort_values(by="average_volume_per_hour").head(self.screener_config["screener_params"]["max_top_markets"])
        return screener_top_markets[["connector_name", "trading_pair", "from_timestamp", "to_timestamp"]]

    async def execute(self):
        ts_client = TimescaleClient(
            host=os.getenv("TIMESCALE_HOST", "localhost"),
            port=5432,
            user=os.getenv("POSTGRES_USER", "admin"),
            password=os.getenv("POSTGRES_PASSWORD", "admin"),
            database="timescaledb"
        )
        await ts_client.connect()

        logger.info("Generating top markets report")
        metrics_df = await ts_client.get_metrics_df()
        top_markets_df = self.generate_top_markets_report(metrics_df)

        resolution = self.resolution
        optimizer = StrategyOptimizer(resolution=resolution, db_client=ts_client)

        logger.info("Optimizing strategy")
        for index, row in top_markets_df.iterrows():
            connector_name = row["connector_name"]
            trading_pair = row["trading_pair"]
            start_date = pd.Timestamp(row["from_timestamp"].timestamp(), unit="s")
            end_date = pd.Timestamp(row["to_timestamp"].timestamp(), unit="s")
            config_generator = XtreetConfigGenerator(start_date=start_date, end_date=end_date,
                                                     backtester=XtreetBacktesting())
            config_generator.trading_pair = trading_pair
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


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = {
        'config': load_dict_from_yaml(file_name="binance_config.yml", folder=f"{root_path}/config"),
        'resolution': "1s",
    }
    task = BacktestingTask("Backtesting", timedelta(hours=24), config)
    asyncio.run(task.execute())
