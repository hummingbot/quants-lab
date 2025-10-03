"""
Linear regression-based trend analysis feature.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, TYPE_CHECKING
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class TrendConfig(FeatureConfig):
    name: str = "trend"
    short_window: int = 50
    long_window: int = 200


class Trend(FeatureBase[TrendConfig]):
    """
    Calculate trend using linear regression slopes on moving averages.

    Generates:
    - Short and long-term slopes
    - Trend score (normalized difference)
    """

    def calculate(self, data):
        """Calculate trend indicators on data."""
        short_window = self.config.short_window
        long_window = self.config.long_window

        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for trend calculation.")

        df = data.copy()
        trend_score, short_slope, long_slope = self.calculate_trending_score(
            df['close'], short_window, long_window
        )
        df['trend_score'] = trend_score
        df['short_slope'] = short_slope
        df['long_slope'] = long_slope
        return df

    def calculate_trending_score(self, series: pd.Series, short_window: int, long_window: int):
        """Calculate trending scores using linear regression slopes."""
        short_mavg = series.rolling(window=short_window, min_periods=1).mean()
        long_mavg = series.rolling(window=long_window, min_periods=1).mean()

        short_slope = short_mavg.rolling(window=short_window, min_periods=1).apply(
            self.calculate_slope, raw=True
        )
        long_slope = long_mavg.rolling(window=long_window, min_periods=1).apply(
            self.calculate_slope, raw=True
        )

        trending_score = (short_slope - long_slope) / (np.abs(short_slope) + np.abs(long_slope) + 1e-8)
        return trending_score, short_slope, long_slope

    @staticmethod
    def calculate_slope(values: np.ndarray) -> float:
        """Calculate linear regression slope."""
        if len(values) < 2:
            return 0.0
        X = np.arange(len(values)).reshape(-1, 1)
        y = values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles data."""
        df = self.calculate(candles.data)

        return Feature(
            feature_name="trend",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={
                'trend_score': float(df['trend_score'].iloc[-1]),
                'short_slope': float(df['short_slope'].iloc[-1]),
                'long_slope': float(df['long_slope'].iloc[-1]),
            },
            info={
                'short_window': self.config.short_window,
                'long_window': self.config.long_window,
                'description': 'Linear regression trend analysis',
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles", min_score: float = 0.5) -> Optional[Signal]:
        """Create trend following signal."""
        df = self.calculate(candles.data)
        trend_score = float(df['trend_score'].iloc[-1])

        if abs(trend_score) > min_score:
            return Signal(
                signal_name=f"trend_{self.config.short_window}_{self.config.long_window}",
                trading_pair=candles.trading_pair,
                category='tf',  # trend following
                value=float(np.clip(trend_score, -1, 1))  # Ensure -1 to 1
            )
        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add trend score as a subplot."""
        df = self.calculate(candles.data)

        trace = go.Scatter(
            x=df.index,
            y=df['trend_score'],
            mode='lines',
            name='Trend Score',
            line=dict(color='purple', width=2),
            showlegend=True
        )

        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

        # Add zero line
        if row is not None:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)

        return fig
