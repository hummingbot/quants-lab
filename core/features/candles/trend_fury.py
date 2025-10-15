"""
Advanced trend detection using rolling regression with volume weighting and reversal detection.
"""
from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class TrendFuryConfig(FeatureConfig):
    name: str = "trend_fury"
    window: int = 50
    vwap_window: int = 50
    use_returns: bool = False
    use_ema: bool = False
    use_volume_weighting: bool = False
    volume_normalization_window: int = 50
    cum_diff_quantile_threshold: float = 0.5
    reversal_sensitivity: float = 0.3
    slope_quantile_threshold: float = 0.4
    use_vwap_filter: bool = False
    use_slope_filter: bool = False


class TrendFury(FeatureBase[TrendFuryConfig]):
    """
    Advanced trend detection using:
    - Rolling regression slopes (optionally volume-weighted)
    - Cumulative slope differences with reversal detection
    - VWAP for trend confirmation
    - Quantile-based thresholds
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend fury indicators."""
        required_columns = ['close', 'quote_asset_volume', 'taker_buy_quote_volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column for trend fury calculation.")

        df = data.copy()

        # Prepare price series
        if self.config.use_returns:
            df['price_series'] = np.log(df['close'] / df['close'].shift(1))
        elif self.config.use_ema:
            df['price_series'] = df['close'].ewm(span=self.config.window, adjust=False).mean()
        else:
            df['price_series'] = df['close']

        # Calculate taker sell quote volume
        df['taker_sell_quote_volume'] = df['quote_asset_volume'] - df['taker_buy_quote_volume']

        # Compute volume weights
        if self.config.use_volume_weighting:
            average_rolling_volume = df['quote_asset_volume'].rolling(
                self.config.volume_normalization_window, min_periods=1
            ).mean()
            df['volume_weight'] = df['quote_asset_volume'] / average_rolling_volume
        else:
            df['volume_weight'] = 1.0

        # Calculate rolling regression slopes
        df['slope'] = df['price_series'].rolling(
            window=self.config.window, min_periods=self.config.window
        ).apply(
            lambda x: self.calculate_slope(
                x,
                weights=df['volume_weight'].loc[x.index] if self.config.use_volume_weighting else None
            ),
            raw=False
        )

        # Calculate slope differences and cumulative slope diff with reversals
        df['slope_diff'] = df['slope'].diff()
        df['cumulative_slope_diff'] = self.cumsum_reset_on_reversal(
            df['slope_diff'], reversal_threshold=self.config.reversal_sensitivity
        )

        # Volume metrics
        df['taker_buy_volume_ratio'] = df['taker_buy_quote_volume'] / df['quote_asset_volume']

        # VWAP calculations
        df['cum_volume'] = df['quote_asset_volume'].cumsum()
        df['cum_volume_price'] = (df['close'] * df['quote_asset_volume']).cumsum()
        df['vwap'] = df['cum_volume_price'] / df['cum_volume']

        df["rolling_cum_volume"] = df["quote_asset_volume"].rolling(window=self.config.vwap_window).sum()
        df["rolling_cum_volume_price"] = (df['close'] * df['quote_asset_volume']).rolling(
            window=self.config.vwap_window
        ).sum()
        df["rolling_vwap"] = df["rolling_cum_volume_price"] / df["rolling_cum_volume"]

        # Quantile thresholds
        positive_slope_quantile_threshold = df[df["slope"] > 0]["slope"].quantile(
            self.config.slope_quantile_threshold
        )
        negative_slope_quantile_threshold = df[df["slope"] < 0]["slope"].quantile(
            self.config.slope_quantile_threshold
        )
        pos_cum_slope_diff_thresh = df[df["cumulative_slope_diff"] > 0]["cumulative_slope_diff"].quantile(
            self.config.cum_diff_quantile_threshold
        )
        neg_cum_slope_diff_thresh = df[df["cumulative_slope_diff"] < 0]["cumulative_slope_diff"].quantile(
            1 - self.config.cum_diff_quantile_threshold
        )

        # Generate signals
        df['signal'] = 0
        df.loc[
            (df["cumulative_slope_diff"] == 0) &
            (df['cumulative_slope_diff'] - df["cumulative_slope_diff"].shift(1) > pos_cum_slope_diff_thresh) &
            (df['close'] < df['rolling_vwap'] if self.config.use_vwap_filter else True) &
            (df['slope'] < negative_slope_quantile_threshold if self.config.use_slope_filter else True),
            'signal'
        ] = 1

        df.loc[
            (df["cumulative_slope_diff"] == 0) &
            (df['cumulative_slope_diff'] - df["cumulative_slope_diff"].shift(1) < neg_cum_slope_diff_thresh) &
            (df['close'] > df['rolling_vwap']) &
            (df['slope'] > positive_slope_quantile_threshold if self.config.use_slope_filter else True),
            'signal'
        ] = -1

        return df

    @staticmethod
    def calculate_slope(values: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """Calculate slope using linear regression (optionally weighted)."""
        if len(values) < 2:
            return 0.0

        X = np.arange(len(values)).reshape(-1, 1)
        y = values.values.reshape(-1, 1)

        if weights is not None:
            weights = weights.values.flatten()
            model = LinearRegression().fit(X, y, sample_weight=weights)
        else:
            model = LinearRegression().fit(X, y)

        return model.coef_[0][0]

    @staticmethod
    def cumsum_reset_on_reversal(series, reversal_threshold=0.3):
        """Calculate cumulative sum with reset on trend reversal."""
        cumsum = 0
        max_cumsum = 0
        min_cumsum = 0
        output = []
        trend = 0  # 0=no trend, 1=uptrend, -1=downtrend

        for i, change in enumerate(series):
            if np.isnan(change):
                output.append(np.nan)
                continue

            cumsum += series.iloc[i]

            if trend == 0:
                if cumsum > 0:
                    trend = 1
                    max_cumsum = cumsum
                elif cumsum < 0:
                    trend = -1
                    min_cumsum = cumsum
            elif trend > 0:
                max_cumsum = max(max_cumsum, cumsum)
                if cumsum <= max_cumsum * (1 - reversal_threshold):
                    cumsum = 0
                    max_cumsum = 0
                    min_cumsum = 0
                    trend = -1
            elif trend < 0:
                min_cumsum = min(min_cumsum, cumsum)
                if cumsum >= min_cumsum * (1 - reversal_threshold):
                    cumsum = 0
                    max_cumsum = 0
                    min_cumsum = 0
                    trend = 1

            output.append(cumsum)

        return pd.Series(output, index=series.index)

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles."""
        df = self.calculate(candles.data)

        return Feature(
            feature_name="trend_fury",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={
                'slope': float(df['slope'].iloc[-1]),
                'slope_diff': float(df['slope_diff'].iloc[-1]),
                'cumulative_slope_diff': float(df['cumulative_slope_diff'].iloc[-1]),
                'rolling_vwap': float(df['rolling_vwap'].iloc[-1]),
                'taker_buy_ratio': float(df['taker_buy_volume_ratio'].iloc[-1]),
                'signal': int(df['signal'].iloc[-1]),
            },
            info={
                'window': self.config.window,
                'vwap_window': self.config.vwap_window,
                'use_volume_weighting': self.config.use_volume_weighting,
                'reversal_sensitivity': self.config.reversal_sensitivity,
                'description': 'Advanced trend detection with volume weighting',
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles") -> Optional[Signal]:
        """Create signal based on trend fury conditions."""
        df = self.calculate(candles.data)
        signal_value = int(df['signal'].iloc[-1])

        if signal_value != 0:
            return Signal(
                signal_name=f"trend_fury_{self.config.window}",
                trading_pair=candles.trading_pair,
                category='tf',  # trend following
                value=float(signal_value)  # 1 or -1
            )
        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add VWAP and signal markers to the chart."""
        df = self.calculate(candles.data)

        # Add rolling VWAP
        trace = go.Scatter(
            x=df.index,
            y=df['rolling_vwap'],
            mode='lines',
            name=f'VWAP {self.config.vwap_window}',
            line=dict(color='orange', width=2, dash='dash'),
            showlegend=True
        )

        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

        # Add buy/sell signals as markers
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]

        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                showlegend=True
            ))

        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                showlegend=True
            ))

        return fig
