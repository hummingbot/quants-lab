from typing import Dict

from hummingbot.strategy_v2.controllers import ControllerConfigBase

from data_structures.data_handler_base import DataStructureBase


class BacktestingResult(DataStructureBase):
    def __init__(self, backtesting_result: Dict, controller_config: ControllerConfigBase):
        super().__init__(backtesting_result)
        self.processed_data = backtesting_result["processed_data"]["features"]
        self.results = backtesting_result["results"]
        self.executors = backtesting_result["executors"]
        self.controller_config = controller_config
