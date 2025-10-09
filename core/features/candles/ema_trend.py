"""
EMA-based trend following feature calculator.
"""
import pandas as pd
import pandas_ta as ta  # noqa: F401
from typing import Optional, TYPE_CHECKING, List
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class EMATrendConfig(FeatureConfig):
    """Configuration for EMA trend feature"""
    name: str = "ema_trend"
    ema_lengths: List[int] = [20, 200, 500]
    rolling_window: int = 3000


class EMATrend(FeatureBase[EMATrendConfig]):
    """
    EMA-based trend following feature calculator.

    Calculates multiple EMAs and generates trend signals based on crossovers.
    """

    def calculate(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA trend features and add them to the candles dataframe.

        Args:
            candles: OHLCV dataframe with pandas_ta available

        Returns:
            DataFrame with added EMA columns and trend indicators
        """
        if 'close' not in candles.columns:
            raise ValueError("Candles dataframe must contain 'close' column")

        df = candles.copy()

        # Calculate EMAs dynamically
        for length in self.config.ema_lengths:
            df.ta.ema(length=length, append=True)

        # Generate EMA crossover signals
        df['signal'] = 0

        # Long signal: EMA_short > EMA_mid and EMA_short > EMA_long
        long_condition = (
            (df[f'EMA_{self.config.ema_lengths[0]}'] > df[f'EMA_{self.config.ema_lengths[1]}']) &
            (df[f'EMA_{self.config.ema_lengths[0]}'] > df[f'EMA_{self.config.ema_lengths[2]}'])
        )

        # Short signal: EMA_short < EMA_mid and EMA_short < EMA_long
        short_condition = (
            (df[f'EMA_{self.config.ema_lengths[0]}'] < df[f'EMA_{self.config.ema_lengths[1]}']) &
            (df[f'EMA_{self.config.ema_lengths[0]}'] < df[f'EMA_{self.config.ema_lengths[2]}'])
        )

        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1

        # Calculate signal intensity based on EMA divergence
        ema_long_avg = (df[f'EMA_{self.config.ema_lengths[1]}'] + df[f'EMA_{self.config.ema_lengths[2]}']) / 2
        ema_divergence = df[f'EMA_{self.config.ema_lengths[0]}'] - ema_long_avg

        # Normalize intensity using rolling percentiles
        df['ema_divergence'] = ema_divergence
        df['ema_divergence_pct'] = ema_divergence.rolling(self.config.rolling_window).rank(pct=True)

        # Calculate intensity: 0 (neutral) to 1 (maximum strength)
        df['signal_intensity'] = 0.0

        # For long signals: intensity based on how much EMA_short is above the long EMAs
        long_mask = df['signal'] == 1
        df.loc[long_mask, 'signal_intensity'] = df.loc[long_mask, 'ema_divergence_pct']

        # For short signals: intensity based on how much EMA_short is below the long EMAs
        short_mask = df['signal'] == -1
        df.loc[short_mask, 'signal_intensity'] = 1 - df.loc[short_mask, 'ema_divergence_pct']

        # Calculate price range for potential grid levels
        df['price_range'] = df['high'].rolling(self.config.rolling_window).max() - df['low'].rolling(self.config.rolling_window).min()
        df['range_pct'] = df['price_range'] / df['close']

        return df

    def create_feature(self, candles: "Candles") -> Feature:
        """
        Create a single Feature object containing all EMA trend data.

        Args:
            candles: Candles object containing OHLCV data and metadata

        Returns:
            Feature object with all EMA trend values
        """
        df = self.calculate(candles.data)

        # Build value dict with all related data
        value = {
            f'ema_{length}': float(df[f'EMA_{length}'].iloc[-1])
            for length in self.config.ema_lengths
        }
        value.update({
            'direction': int(df['signal'].iloc[-1]),
            'intensity': float(df['signal_intensity'].iloc[-1]),
            'divergence': float(df['ema_divergence'].iloc[-1]),
            'range_pct': float(df['range_pct'].iloc[-1]),
            'price': float(df['close'].iloc[-1])
        })

        return Feature(
            feature_name="ema_trend",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value=value,
            info={
                'ema_lengths': self.config.ema_lengths,
                'rolling_window': self.config.rolling_window,
                'description': f"EMA trend analysis using {self.config.ema_lengths}",
                'interval': candles.interval
            }
        )

    def create_signal(
        self,
        candles: "Candles",
        min_intensity: float = 0.7,
        min_range_pct: float = 0.01
    ) -> Optional[Signal]:
        """
        Create a trading signal if conditions are met.

        Args:
            candles: Candles object containing OHLCV data and metadata
            min_intensity: Minimum intensity threshold (default: 0.7)
            min_range_pct: Minimum range percentage (default: 0.01)

        Returns:
            Signal object or None if conditions not met
        """
        df = self.calculate(candles.data)

        signal_direction = int(df['signal'].iloc[-1])
        intensity = float(df['signal_intensity'].iloc[-1])
        range_pct = float(df['range_pct'].iloc[-1])

        # Check if signal meets criteria
        if abs(signal_direction) == 1 and intensity > min_intensity and range_pct > min_range_pct:
            # Map signal direction to -1 to 1 scale
            signal_value = signal_direction * intensity  # -1 to 1 based on direction and intensity

            return Signal(
                signal_name=f"ema_trend_{self.config.ema_lengths[0]}_{self.config.ema_lengths[1]}_{self.config.ema_lengths[2]}",
                trading_pair=candles.trading_pair,
                category='tf',  # trend following
                value=signal_value
            )

        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """
        Add EMA lines to the candlestick chart.

        Args:
            fig: Plotly figure to add traces to
            candles: Candles object with calculated features
            row: Subplot row number (if using subplots)
            **kwargs: Additional plotting parameters

        Returns:
            Modified figure with EMA lines
        """
        df = self.calculate(candles.data)

        # Define colors for EMAs
        colors = ['blue', 'orange', 'purple', 'green', 'red']

        # Add EMA lines
        for idx, length in enumerate(self.config.ema_lengths):
            color = colors[idx % len(colors)]
            trace = go.Scatter(
                x=df.index,
                y=df[f'EMA_{length}'],
                mode='lines',
                name=f'EMA {length}',
                line=dict(color=color, width=2),
                showlegend=True
            )

            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

        return fig

    def __str__(self):
        """String representation showing feature configuration."""
        return f"EMATrend(lengths={self.config.ema_lengths}, window={self.config.rolling_window})"
