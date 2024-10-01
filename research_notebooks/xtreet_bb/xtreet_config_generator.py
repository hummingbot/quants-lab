import asyncio
import datetime
import logging
import os
from typing import Dict, List

import pandas as pd

from controllers.directional_trading.xtreet_bb import XtreetBBControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator
from core.data_sources.clob import CLOBDataSource
from core.features.candles.volatility import VolatilityConfig
from core.features.candles.volume import VolumeConfig
from research_notebooks.xtreet_bb.utils import generate_config, generate_screener_report
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting
from services.timescale_client import TimescaleClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XtreetConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime, backtester=XtreetBacktesting()):
        super().__init__(start_date, end_date, backtester)
        self.report = None
        self.trading_pairs = None
        self.candles = None
        self.screener_config = None
        logging.info(os.getenv("POSTGRES_HOST", "localhost"))
        self.client = TimescaleClient(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            user=os.getenv("POSTGRES_USER", "admin"),
            password=os.getenv("POSTGRES_PASSWORD", "admin"),
            database=os.getenv("POSTGRES_DB", "timescaledb")
        )
        self.clob = CLOBDataSource()

    async def generate_config(self, trial) -> BacktestingConfig:
        pass

    async def generate_top_markets_report(self, screener_config: Dict) -> pd.DataFrame:
        self.screener_config = screener_config
        # Screener parameters
        CONNECTOR_NAME = screener_config["screener_params"]["connector_name"]
        INTERVAL = screener_config["screener_params"]["interval"]
        DAYS = screener_config["screener_params"]["days_to_download"]
        VOLUME_THRESHOLD = screener_config["screener_params"]["volume_threshold"]
        VOLATILITY_THRESHOLD = screener_config["screener_params"]["volatility_threshold"]
        MAX_TOP_MARKETS = screener_config["screener_params"]["max_top_markets"]

        VOLATILITY_WINDOW = screener_config["screener_params"]["volatility_window"]
        VOLUME_FAST_WINDOW = screener_config["screener_params"]["volume_fast_window"]

        MAX_PRICE_STEP = screener_config["screener_params"]["max_price_step"]

        trading_rules = await self.clob.get_trading_rules(CONNECTOR_NAME)
        await self.client.connect()
        available_pairs = await self.client.get_available_pairs()
        trading_pairs = [pair[1] for pair in available_pairs if pair[0] == CONNECTOR_NAME]
        candles_tasks = [self.client.get_candles_last_days(CONNECTOR_NAME, trading_pair, INTERVAL, DAYS) for
                         trading_pair in
                         trading_pairs]
        candles = await asyncio.gather(*candles_tasks)
        screener_report = generate_screener_report(
            candles=candles,
            trading_rules=trading_rules,
            volatility_config=VolatilityConfig(window=VOLATILITY_WINDOW),
            volume_config=VolumeConfig(short_window=VOLUME_FAST_WINDOW, long_window=VOLUME_FAST_WINDOW))

        screener_report.sort_values("mean_natr", ascending=False, inplace=True)

        # Calculate the 20th percentile (0.2 quantile) for both columns
        natr_percentile = screener_report['mean_natr'].quantile(VOLATILITY_THRESHOLD)
        volume_percentile = screener_report['average_volume_per_hour'].quantile(VOLUME_THRESHOLD)

        # Filter the DataFrame to get observations where mean_natr is greater than its 20th percentile
        # and average_volume_per_hour is greater than its 20th percentile
        screener_top_markets = screener_report[
            (screener_report['mean_natr'] > natr_percentile) &
            (screener_report['average_volume_per_hour'] > volume_percentile) &
            (screener_report["price_step_pct"] < MAX_PRICE_STEP)
            ].sort_values(by="average_volume_per_hour").head(MAX_TOP_MARKETS)

        # Display the filtered DataFrame
        self.screener_top_markets = screener_top_markets
        self.candles = candles

    def generate_custom_configs(self) -> List[BacktestingConfig]:
        strategy_configs = generate_config(
            connector_name=self.screener_config["screener_params"]["connector_name"],
            interval=self.screener_config["screener_params"]["interval"],
            screener_top_markets=self.screener_top_markets,
            candles=self.candles,
            total_amount=self.screener_config["config_generation"]["total_amount"],
            max_executors_per_side=self.screener_config["config_generation"]["max_executors_per_side"],
            cooldown_time=self.screener_config["config_generation"]["cooldown_time"],
            leverage=self.screener_config["config_generation"]["leverage"],
            time_limit=self.screener_config["config_generation"]["time_limit"],
            bb_lengths=self.screener_config["config_generation"]["bb_lengths"],
            bb_stds=self.screener_config["config_generation"]["bb_stds"],
            sl_std_multiplier=self.screener_config["config_generation"]["sl_std_multiplier"],
            min_distance_between_orders=self.screener_config["config_generation"]["min_distance_between_orders"],
            max_ts_sl_ratio=self.screener_config["config_generation"]["max_ts_sl_ratio"],
            ts_delta_multiplier=self.screener_config["config_generation"]["ts_delta_multiplier"],
        )
        return [BacktestingConfig(config=XtreetBBControllerConfig(**config),
                                  start=self.start,
                                  end=self.end,
                                  ) for config in strategy_configs]
