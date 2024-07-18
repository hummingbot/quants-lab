import pandas as pd

from data_handler.data_handler_base import DataHandlerBase


class OHLC(DataHandlerBase):
    def __init__(self, candles_df: pd.DataFrame):
        super().__init__(candles_df)
