import pandas as pd

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase


class GridStrikeBT(BacktestingEngineBase):
    async def update_processed_data(self, row: pd.Series):
        await self.controller.update_processed_data()
