import logging
import os
from typing import Dict, Optional

import pandas as pd

from core.data_structures.backtesting_result import BacktestingResult
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    def __init__(self, load_cached_data: bool = True, root_path: str = "", custom_backtester: Optional[BacktestingEngineBase] = None):
        self._bt_engine = custom_backtester if custom_backtester is not None else BacktestingEngineBase()
        self.root_path = root_path
        if load_cached_data:
            self._load_candles_cache(root_path)

    def _load_candles_cache(self, root_path: str):
        all_files = os.listdir(os.path.join(root_path, "data", "candles"))
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                connector_name, trading_pair, interval = file.split(".")[0].split("|")
                candles = pd.read_parquet(os.path.join(root_path, "data", "candles", file))
                candles.index = pd.to_datetime(candles.timestamp, unit='s')
                candles.index.name = None
                columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                for column in columns:
                    candles[column] = pd.to_numeric(candles[column])
                self._bt_engine.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{interval}"] = candles
                # TODO: evaluate start and end time for each feed
                start_time = candles["timestamp"].min()
                end_time = candles["timestamp"].max()
                self._bt_engine.backtesting_data_provider.start_time = start_time
                self._bt_engine.backtesting_data_provider.end_time = end_time
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    def load_candles_cache_by_connector_pair(self, connector_name: str, trading_pair: str, root_path: str = ""):
            all_files = os.listdir(os.path.join(root_path, "data", "candles"))
            for file in all_files:
                if file == ".gitignore":
                    continue
                try:
                    if connector_name in file and trading_pair in file:
                        connector_name, trading_pair, interval = file.split(".")[0].split("|")
                        candles = pd.read_parquet(os.path.join(root_path, "data", "candles", file))
                        candles.index = pd.to_datetime(candles.timestamp, unit='s')
                        candles.index.name = None
                        columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                                   'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                        for column in columns:
                            candles[column] = pd.to_numeric(candles[column])
                        self._bt_engine.backtesting_data_provider.candles_feeds[
                            f"{connector_name}_{trading_pair}_{interval}"] = candles
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

    def get_controller_config_instance_from_dict(self, config: Dict):
        return BacktestingEngineBase.get_controller_config_instance_from_dict(
            config_data=config,
            controllers_module="controllers",
        )

    async def run_backtesting(self, config: ControllerConfigBase, start: int,
                              end: int, backtesting_resolution: str, trade_cost: float = 0.0006) -> BacktestingResult:
        bt_result = await self._bt_engine.run_backtesting(config, start, end, backtesting_resolution, trade_cost)
        return BacktestingResult(bt_result, config)

    async def backtest_controller_from_yml(self,
                                           config_file: str,
                                           controllers_conf_dir_path: str,
                                           start: int,
                                           end: int,
                                           backtesting_resolution: str = "1m",
                                           trade_cost: float = 0.0006,
                                           backtester: Optional[BacktestingEngineBase] = None):
        config = self._bt_engine.get_controller_config_instance_from_yml(config_file, controllers_conf_dir_path)
        return await self.run_backtesting(config, start, end, backtesting_resolution, trade_cost, backtester)
