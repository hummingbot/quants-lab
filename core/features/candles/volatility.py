"""
Volatility analysis feature using multiple indicators.
"""
import pandas_ta as ta  # noqa: F401
from typing import Optional, TYPE_CHECKING
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class VolatilityConfig(FeatureConfig):
    name: str = "volatility"
    window: int = 100


class Volatility(FeatureBase[VolatilityConfig]):
    """
    Calculate volatility using multiple measures:
    - Standard deviation of returns
    - Normalized ATR
    - Bollinger Band width
    """

    def calculate(self, data):
        """Calculate volatility indicators on data."""
        window = self.config.window
        df = data.copy()

        # Standard deviation of returns
        df["volatility"] = df["close"].pct_change().rolling(window=window).std()

        # Normalized ATR
        df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=window) / 100

        # Bollinger Bands width
        bbands = ta.bbands(df["close"], length=window)
        df["bb_width"] = bbands[f"BBB_{window}_2.0"]

        return df

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles data."""
        df = self.calculate(candles.data)

        return Feature(
            feature_name="volatility",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={
                'mean_volatility': float(df['volatility'].mean()),
                'max_volatility': float(df['volatility'].max()),
                'min_volatility': float(df['volatility'].min()),
                'mean_natr': float(df['natr'].mean()),
                'max_natr': float(df['natr'].max()),
                'min_natr': float(df['natr'].min()),
                'mean_bb_width': float(df['bb_width'].mean()),
                'max_bb_width': float(df['bb_width'].max()),
                'min_bb_width': float(df['bb_width'].min()),
                'current_volatility': float(df['volatility'].iloc[-1]),
                'current_natr': float(df['natr'].iloc[-1]),
                'current_bb_width': float(df['bb_width'].iloc[-1]),
            },
            info={
                'window': self.config.window,
                'description': 'Multi-indicator volatility analysis',
                'interval': candles.interval
            }
        )

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add volatility indicator as a subplot."""
        df = self.calculate(candles.data)

        # Plot NATR (normalized ATR)
        trace = go.Scatter(
            x=df.index,
            y=df['natr'],
            mode='lines',
            name=f'NATR {self.config.window}',
            line=dict(color='orange', width=2),
            showlegend=True
        )

        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

        return fig
