import logging
import os
from typing import Optional

import pandas as pd

from core.data_structures.backtesting_result import BacktestingResult
from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting, MarketMakingBacktesting
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    def __init__(self, load_cached_data: bool = True, root_path: str = ""):
        self._mm_bt = MarketMakingBacktesting()
        self._dt_bt = DirectionalTradingBacktesting()
        if load_cached_data:
            self._load_candles_cache(root_path)

    def _load_candles_cache(self, root_path: str):
        all_files = os.listdir(os.path.join(root_path, "data", "candles"))
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                connector_name, trading_pair, interval = file.split(".")[0].split("|")
                candles = pd.read_csv(os.path.join(root_path, "data", "candles", file))
                candles.index = pd.to_datetime(candles.timestamp, unit='s')
                candles.index.name = None
                self._dt_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{interval}"] = candles
                self._mm_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{interval}"] = candles
                # TODO: evaluate start and end time for each feed
                start_time = candles["timestamp"].min()
                end_time = candles["timestamp"].max()
                self._dt_bt.backtesting_data_provider.start_time = start_time
                self._dt_bt.backtesting_data_provider.end_time = end_time
                self._mm_bt.backtesting_data_provider.start_time = start_time
                self._mm_bt.backtesting_data_provider.end_time = end_time
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    def get_controller_config_instance_from_yml(self, config_file: str,
                                                controllers_conf_dir_path: str) -> ControllerConfigBase:
        return self._dt_bt.get_controller_config_instance_from_yml(
            config_path=config_file,
            controllers_conf_dir_path=controllers_conf_dir_path,
            controllers_module="controllers")

    async def run_backtesting(self, config: ControllerConfigBase, start: int,
                              end: int, backtesting_resolution: str, trade_cost: float = 0.0006,
                              backtester: Optional[BacktestingEngineBase] = None) -> BacktestingResult:
        if config.controller_type == "market_making":
            backtester = self._mm_bt
        elif config.controller_type == "directional_trading":
            backtester = self._dt_bt
        else:
            if backtester is None:
                raise Exception("Backtester not specified")
        bt_result = await backtester.run_backtesting(config, start, end, backtesting_resolution, trade_cost)
        return BacktestingResult(bt_result, config)

    async def backtest_controller_from_yml(self,
                                           config_file: str,
                                           controllers_conf_dir_path: str,
                                           start: int,
                                           end: int,
                                           backtesting_resolution: str = "1m",
                                           trade_cost: float = 0.0006,
                                           backtester: Optional[BacktestingEngineBase] = None):
        config = self.get_controller_config_instance_from_yml(config_file, controllers_conf_dir_path)
        return await self.run_backtesting(config, start, end, backtesting_resolution, trade_cost, backtester)
