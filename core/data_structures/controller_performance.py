from typing import Dict, List, Optional

import pandas as pd

from core.backtesting import BacktestingEngine
from core.data_structures.data_structure_base import DataStructureBase
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class ControllerPerformance(DataStructureBase):
    def __init__(self, data: pd.DataFrame, controller_config: Dict, root_path: str = "",
                 load_cache_data: bool = False, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.backtesting_engine = BacktestingEngine(root_path=root_path, load_cached_data=load_cache_data)
        self.controller_config = self.backtesting_engine.get_controller_config_instance_from_dict(controller_config)
        self.start_time = data["timestamp"].min()
        self.end_time = data["timestamp"].max()
        self.backtesting_result = None

    def get_executor_info(self) -> List[ExecutorInfo]:
        return [ExecutorInfo(**controller) for controller in self.data.to_dict(orient="records")]

    async def get_backtesting_result_same_period(self,
                                                 buffer_time: int = 60,
                                                 backtesting_resolution: str = "1m",
                                                 backtester: Optional[BacktestingEngineBase] = None,
                                                 trade_cost: float = 0.0007):
        start_time = int(self.start_time - buffer_time)
        end_time = int(self.end_time + buffer_time)
        backtesting_result = await self.backtesting_engine.run_backtesting(
            config=self.controller_config,
            trade_cost=trade_cost,
            start=start_time,
            end=end_time,
            backtesting_resolution=backtesting_resolution,
            backtester=backtester,
        )
        self.backtesting_result = backtesting_result
        return backtesting_result

    def live_vs_backtesting_figure(self):
        if not self.backtesting_result:
            raise Exception("Backtesting result not available. Run get_backtesting_result_same_period first.")
        fig = self.backtesting_result.get_backtesting_figure()
        fig = self.backtesting_result._add_executors_trace(fig, self.get_executor_info(), line_style=None)
        fig.add_trace(self.backtesting_result._get_pnl_trace(self.get_executor_info(), line_style=None), row=2, col=1)
        return fig

    def live_vs_backtesting_performance_summary(self):
        if not self.backtesting_result:
            raise Exception("Backtesting result not available. Run get_backtesting_result_same_period first.")
        bt_summary = self.backtesting_result.get_results_summary()
        live_results = self.backtesting_engine._dt_bt.summarize_results(
            self.get_executor_info(),
            total_amount_quote=self.controller_config.total_amount_quote)
        live_summary = self.backtesting_result.get_results_summary(live_results)
        return f"Backtesting Results: {bt_summary} | Live Results: {live_summary}"
