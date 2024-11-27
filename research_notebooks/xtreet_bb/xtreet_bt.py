import pandas as pd
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase



class XtreetBacktesting(BacktestingEngineBase):
    def prepare_market_data(self) -> pd.DataFrame:
        df = super().prepare_market_data()
        df["signal"] = 0
        long_condition = df["close"] <= df[f"BBL_{self.controller.config.bb_length}_{self.controller.config.bb_std}"]
        short_condition = df["close"] >= df[f"BBU_{self.controller.config.bb_length}_{self.controller.config.bb_std}"]
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1
        return df
