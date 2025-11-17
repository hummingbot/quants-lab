"""
Volume analysis feature with buy/sell pressure and market regime detection.
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class VolumeConfig(FeatureConfig):
    name: str = "volume"
    short_term_window: int = 5
    long_term_window: int = 100


class Volume(FeatureBase[VolumeConfig]):
    """
    Comprehensive volume analysis including:
    - Buy/sell pressure
    - Volume surges
    - Market regime detection (accumulation/distribution)
    """

    def calculate(self, data: pd.DataFrame):
        """Calculate comprehensive volume indicators."""
        required_columns = ['volume', 'close', 'taker_buy_base_volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column for volume calculation.")

        df = data.copy()

        # Calculate base volume metrics
        df["volume_usd"] = df["volume"] * df["close"]
        df["buy_volume_usd"] = df["taker_buy_base_volume"] * df["close"]
        df["sell_volume_usd"] = df["volume_usd"] - df["buy_volume_usd"]

        # Calculate buy percentage (0 to 1, where 0.5 is neutral)
        df["buy_percentage"] = df["buy_volume_usd"] / df["volume_usd"]
        df["buy_percentage"] = df["buy_percentage"].fillna(0.5)

        # Simple buy/sell imbalance (-1 to 1, where 0 is neutral)
        df["buy_sell_imbalance"] = (df["buy_percentage"] - 0.5) * 2

        # Short-term buy pressure (smoothed over recent periods)
        df["buy_pressure_short_term"] = df["buy_percentage"].rolling(
            window=self.config.short_term_window, min_periods=1
        ).mean()

        # Long-term buy pressure (smoothed over extended periods)
        df["buy_pressure_long_term"] = df["buy_percentage"].rolling(
            window=self.config.long_term_window, min_periods=1
        ).mean()

        # Buy pressure divergence: short-term vs long-term trend
        df["buy_pressure_divergence"] = df["buy_pressure_short_term"] - df["buy_pressure_long_term"]

        # Buy momentum (rate of change in short-term pressure)
        df["buy_momentum"] = df["buy_pressure_short_term"].diff()

        # Volume moving averages
        df["volume_ma_short_term"] = df["volume_usd"].rolling(
            window=self.config.short_term_window, min_periods=1
        ).mean()
        df["volume_ma_long_term"] = df["volume_usd"].rolling(
            window=self.config.long_term_window, min_periods=1
        ).mean()

        # Volume surge (current vs long-term average)
        df["volume_surge"] = df["volume_usd"] / df["volume_ma_long_term"]
        df["volume_surge"] = df["volume_surge"].replace([np.inf, -np.inf], 1).fillna(1)

        # Combined buy signal: volume surge * buy pressure divergence
        df["volume_buy_signal"] = df["volume_surge"] * df["buy_pressure_divergence"]

        # Trend consistency (lower std = more consistent)
        df["trend_consistency"] = 1 - df["buy_percentage"].rolling(
            window=self.config.short_term_window, min_periods=1
        ).std()
        df["trend_consistency"] = df["trend_consistency"].fillna(0.5)

        # Market regime scores
        df["accumulation_score"] = (
            df["buy_pressure_long_term"] * df["volume_surge"] * df["trend_consistency"]
        )
        df["distribution_score"] = (
            (1 - df["buy_pressure_long_term"]) * df["volume_surge"] * df["trend_consistency"]
        )

        return df

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles data."""
        df = self.calculate(candles.data)

        return Feature(
            feature_name="volume",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={
                'buy_pressure_short_term': float(df['buy_pressure_short_term'].iloc[-1]),
                'buy_pressure_long_term': float(df['buy_pressure_long_term'].iloc[-1]),
                'buy_pressure_divergence': float(df['buy_pressure_divergence'].iloc[-1]),
                'volume_surge': float(df['volume_surge'].iloc[-1]),
                'volume_buy_signal': float(df['volume_buy_signal'].iloc[-1]),
                'accumulation_score': float(df['accumulation_score'].iloc[-1]),
                'distribution_score': float(df['distribution_score'].iloc[-1]),
                'buy_sell_imbalance': float(df['buy_sell_imbalance'].iloc[-1]),
            },
            info={
                'short_term_window': self.config.short_term_window,
                'long_term_window': self.config.long_term_window,
                'description': 'Buy/sell pressure and market regime analysis (short-term vs long-term trends)',
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles", min_score: float = 0.6) -> Optional[Signal]:
        """Create signal based on accumulation/distribution scores."""
        df = self.calculate(candles.data)

        accumulation = float(df['accumulation_score'].iloc[-1])
        distribution = float(df['distribution_score'].iloc[-1])

        # Determine signal based on dominant regime
        if accumulation > distribution and accumulation > min_score:
            # Accumulation regime - long signal
            signal_value = min(accumulation, 1.0)
            return Signal(
                signal_name=f"volume_{self.config.short_term_window}_{self.config.long_term_window}",
                trading_pair=candles.trading_pair,
                category='tf',  # Can also be used as confirmation
                value=signal_value
            )
        elif distribution > accumulation and distribution > min_score:
            # Distribution regime - short signal
            signal_value = -min(distribution, 1.0)
            return Signal(
                signal_name=f"volume_{self.config.short_term_window}_{self.config.long_term_window}",
                trading_pair=candles.trading_pair,
                category='tf',
                value=signal_value
            )

        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add volume and buy pressure visualization."""
        df = self.calculate(candles.data)

        # Add volume bars
        colors = ['red' if bp < 0.5 else 'green' for bp in df['buy_percentage']]

        trace = go.Bar(
            x=df.index,
            y=df['volume_usd'],
            name='Volume',
            marker_color=colors,
            showlegend=True
        )

        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

        return fig
