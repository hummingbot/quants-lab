import pandas as pd
import plotly.graph_objects as go

from core import theme
from core.data_structures.data_structure_base import DataStructureBase


class Candles(DataStructureBase):
    def __init__(self, candles_df: pd.DataFrame, connector_name: str, trading_pair: str, interval: str):
        super().__init__(candles_df)
        self.connector_name = connector_name
        self.trading_pair = trading_pair
        self.interval = interval

    @property
    def max_timestamp(self):
        return self.data["timestamp"].max()

    @property
    def min_timestamp(self):
        return self.data["timestamp"].min()

    def fig(self, type: str = "candles", height=600, width=1200):
        if type == 'candles':
            return self.candles_fig(height, width)
        elif type == 'returns':
            return self.returns_distribution_fig(height, width)
        else:
            raise ValueError(f"Unknown type {type}")

    def plot(self, type: str = "candles", height=600, width=1200):
        fig = self.fig(type, height, width)
        fig.show()

    def candles_trace(self):
        return go.Candlestick(x=self.data.index,
                              open=self.data['open'],
                              high=self.data['high'],
                              low=self.data['low'],
                              close=self.data['close'],
                              name="Candlesticks",
                              increasing_line_color='#2ECC71', decreasing_line_color='#E74C3C')

    def candles_fig(self, height=600, width=1200):
        fig = go.Figure(data=self.candles_trace())
        fig.update_layout(title=f"{self.connector_name}: {self.trading_pair} ({self.interval})",
                          **theme.get_default_layout(height=height, width=width))
        return fig

    def returns_distribution_fig(self, height=600, width=1200, nbins=50):
        returns = self.data['close'].pct_change().dropna()
        fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=nbins)])
        fig.update_layout(title=f"{self.connector_name}: {self.trading_pair} ({self.interval})",
                          **theme.get_default_layout(height=height, width=width))
        return fig
