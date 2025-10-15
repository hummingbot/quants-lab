"""
Mean reversion channel using advanced smoothing filters (SuperSmoother, Gaussian, Butterworth, etc.).
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, TYPE_CHECKING
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class MeanReversionChannelConfig(FeatureConfig):
    name: str = "mean_reversion_channel"
    length: int = 200
    inner_mult: float = 1.0
    outer_mult: float = 2.415
    source: str = "hlc3"
    filter_type: str = "SuperSmoother"


class MeanReversionChannel(FeatureBase[MeanReversionChannelConfig]):
    """
    Mean reversion channel with advanced smoothing:
    - SuperSmoother, Gaussian, Butterworth, BandStop filters
    - Inner and outer bands for mean reversion zones
    - Condition-based signal detection
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion channel indicators."""
        length = self.config.length
        inner_mult = self.config.inner_mult
        outer_mult = self.config.outer_mult
        source = self.config.source
        filter_type = self.config.filter_type

        df = data.copy()

        # Calculate source
        if source == "hlc3":
            df["source"] = (df["high"] + df["low"] + df["close"]) / 3
        else:
            df["source"] = df[source]

        # Calculate mean line
        if filter_type == "SuperSmoother":
            df["meanline"] = self.supersmoother(df["source"], length)
        else:
            df["meanline"] = self.sak_smoothing(df["source"], length, filter_type)

        # Calculate mean range
        df["tr"] = ta.true_range(df["high"], df["low"], df["close"])
        df["meanrange"] = self.supersmoother(df["tr"].dropna(), length)

        # Calculate bands
        df["upband1"] = df["meanline"] + (df["meanrange"] * inner_mult * np.pi)
        df["loband1"] = df["meanline"] - (df["meanrange"] * inner_mult * np.pi)
        df["upband2"] = df["meanline"] + (df["meanrange"] * outer_mult * np.pi)
        df["loband2"] = df["meanline"] - (df["meanrange"] * outer_mult * np.pi)

        # Calculate condition
        df["condition"] = self.calculate_condition(df)

        return df

    def supersmoother(self, src, length):
        a1 = np.exp(-np.sqrt(2) * np.pi / length)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / length)
        c3 = -a1 * a1
        c2 = b1
        c1 = 1 - c2 - c3

        def smooth(x):
            ss = pd.Series(index=x.index, dtype=float)
            ss.iloc[0] = x.iloc[0]
            ss.iloc[1] = x.iloc[1]
            for i in range(2, len(x)):
                ss.iloc[i] = c1 * x.iloc[i] + c2 * ss.iloc[i-1] + c3 * ss.iloc[i-2]
            return ss

        return src.to_frame().apply(smooth).squeeze()

    def sak_smoothing(self, src, length, filter_type):
        import numpy as np

        c0 = 1.0
        c1 = 0.0
        b0 = 1.0
        b1 = 0.0
        b2 = 0.0
        a1 = 0.0
        a2 = 0.0
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        cycle = 2 * np.pi / length

        if filter_type == "Ehlers EMA":
            alpha = (np.cos(cycle) + np.sin(cycle) - 1) / np.cos(cycle)
            b0 = alpha
            a1 = 1 - alpha
        elif filter_type == "Gaussian":
            beta = 2.415 * (1 - np.cos(cycle))
            alpha = -beta + np.sqrt((beta * beta) + (2 * beta))
            c0 = alpha * alpha
            a1 = 2 * (1 - alpha)
            a2 = -(1 - alpha) * (1 - alpha)
        elif filter_type == "Butterworth":
            beta = 2.415 * (1 - np.cos(cycle))
            alpha = -beta + np.sqrt((beta * beta) + (2 * beta))
            c0 = alpha * alpha / 4
            b1 = 2
            b2 = 1
            a1 = 2 * (1 - alpha)
            a2 = -(1 - alpha) * (1 - alpha)
        elif filter_type == "BandStop":
            beta = np.cos(cycle)
            gamma = 1 / np.cos(cycle * 2 * 0.1)  # delta default to 0.1
            alpha = gamma - np.sqrt((gamma * gamma) - 1)
            c0 = (1 + alpha) / 2
            b1 = -2 * beta
            b2 = 1
            a1 = beta * (1 + alpha)
            a2 = -alpha
        elif filter_type == "SMA":
            c1 = 1 / length
            b0 = 1 / length
            a1 = 1
        elif filter_type == "EMA":
            alpha = 2 / (length + 1)
            b0 = alpha
            a1 = 1 - alpha
        elif filter_type == "RMA":
            alpha = 1 / length
            b0 = alpha
            a1 = 1 - alpha

        def smooth(x):
            input_1 = x.shift(1).fillna(x)
            input_2 = x.shift(2).fillna(x)
            output_1 = x.shift(1).fillna(0)
            output_2 = x.shift(2).fillna(0)
            input_length = x.shift(length).fillna(x)
            return (c0 * ((b0 * x) + (b1 * input_1) + (b2 * input_2))) + (a1 * output_1) + (a2 * output_2) - (c1 * input_length)

        return src.rolling(window=length).apply(smooth)

    def calculate_condition(self, df):
        """Calculate mean reversion conditions based on price position relative to bands."""
        conditions = []
        for _, row in df.iterrows():
            if row["close"] > row["meanline"]:
                upband2_1 = row["upband2"] + (row["meanrange"] * 0.5 * 4)
                upband2_9 = row["upband2"] + (row["meanrange"] * 0.5 * -4)
                if row["high"] >= upband2_9 and row["high"] < row["upband2"]:
                    conditions.append(1)
                elif row["high"] >= row["upband2"] and row["high"] < upband2_1:
                    conditions.append(2)
                elif row["high"] >= upband2_1:
                    conditions.append(3)
                elif row["close"] <= row["meanline"] + row["meanrange"]:
                    conditions.append(4)
                else:
                    conditions.append(5)
            elif row["close"] < row["meanline"]:
                loband2_1 = row["loband2"] - (row["meanrange"] * 0.5 * 4)
                loband2_9 = row["loband2"] - (row["meanrange"] * 0.5 * -4)
                if row["low"] <= loband2_9 and row["low"] > row["loband2"]:
                    conditions.append(-1)
                elif row["low"] <= row["loband2"] and row["low"] > loband2_1:
                    conditions.append(-2)
                elif row["low"] <= loband2_1:
                    conditions.append(-3)
                elif row["close"] >= row["meanline"] + row["meanrange"]:
                    conditions.append(-4)
                else:
                    conditions.append(-5)
            else:
                conditions.append(0)
        return pd.Series(conditions, index=df.index)

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles."""
        df = self.calculate(candles.data)

        return Feature(
            feature_name="mean_reversion_channel",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={
                'meanline': float(df['meanline'].iloc[-1]),
                'upband1': float(df['upband1'].iloc[-1]),
                'loband1': float(df['loband1'].iloc[-1]),
                'upband2': float(df['upband2'].iloc[-1]),
                'loband2': float(df['loband2'].iloc[-1]),
                'meanrange': float(df['meanrange'].iloc[-1]),
                'condition': int(df['condition'].iloc[-1]),
            },
            info={
                'length': self.config.length,
                'inner_mult': self.config.inner_mult,
                'outer_mult': self.config.outer_mult,
                'filter_type': self.config.filter_type,
                'source': self.config.source,
                'description': f'Mean reversion channel with {self.config.filter_type} smoothing',
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles") -> Optional[Signal]:
        """Create signal based on mean reversion conditions."""
        df = self.calculate(candles.data)
        condition = int(df['condition'].iloc[-1])

        # Conditions 1-3 are bearish (price above outer band), -1 to -3 are bullish (price below outer band)
        if condition in [1, 2, 3]:
            # Price at or above outer upper band - mean reversion SHORT signal
            signal_value = -min(abs(condition) / 3.0, 1.0)  # Normalize to -1 to 0
            return Signal(
                signal_name=f"mean_reversion_{self.config.length}_{self.config.filter_type}",
                trading_pair=candles.trading_pair,
                category='mr',  # mean reversion
                value=signal_value
            )
        elif condition in [-1, -2, -3]:
            # Price at or below outer lower band - mean reversion LONG signal
            signal_value = min(abs(condition) / 3.0, 1.0)  # Normalize to 0 to 1
            return Signal(
                signal_name=f"mean_reversion_{self.config.length}_{self.config.filter_type}",
                trading_pair=candles.trading_pair,
                category='mr',
                value=signal_value
            )

        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add mean reversion channel bands to the chart."""
        df = self.calculate(candles.data)

        # Add meanline
        trace_mean = go.Scatter(
            x=df.index,
            y=df['meanline'],
            mode='lines',
            name=f'Mean ({self.config.filter_type})',
            line=dict(color='blue', width=2),
            showlegend=True
        )

        # Add inner bands
        trace_upper1 = go.Scatter(
            x=df.index,
            y=df['upband1'],
            mode='lines',
            name='Inner Upper Band',
            line=dict(color='orange', width=1, dash='dash'),
            showlegend=True
        )

        trace_lower1 = go.Scatter(
            x=df.index,
            y=df['loband1'],
            mode='lines',
            name='Inner Lower Band',
            line=dict(color='orange', width=1, dash='dash'),
            showlegend=True
        )

        # Add outer bands
        trace_upper2 = go.Scatter(
            x=df.index,
            y=df['upband2'],
            mode='lines',
            name='Outer Upper Band',
            line=dict(color='red', width=1, dash='dot'),
            showlegend=True
        )

        trace_lower2 = go.Scatter(
            x=df.index,
            y=df['loband2'],
            mode='lines',
            name='Outer Lower Band',
            line=dict(color='red', width=1, dash='dot'),
            showlegend=True
        )

        traces = [trace_mean, trace_upper1, trace_lower1, trace_upper2, trace_lower2]

        for trace in traces:
            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

        return fig
