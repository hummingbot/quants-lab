"""
Support and resistance level detection using peak clustering and hierarchical analysis.
"""
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class PeakAnalyzerConfig(FeatureConfig):
    name: str = "peak_analyzer"
    prominence_percentage: float = 0.01
    distance: int = 5
    num_clusters: int = 3
    close_price_filter: bool = True
    window_size: int = 100
    calculation_interval: int = 50


class PeakAnalyzer(FeatureBase[PeakAnalyzerConfig]):
    """
    Advanced support/resistance detection using:
    - Peak detection with prominence filtering
    - Hierarchical clustering for level identification
    - Time-windowed analysis
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate peak clusters and support/resistance levels."""
        df = data.copy()

        # Get the latest cluster data
        clusters = self.get_peaks_and_clusters(df)

        # Add the latest cluster levels to dataframe
        if clusters:
            latest_cluster = clusters[-1]
            df['high_clusters'] = str(latest_cluster['high_clusters'])
            df['low_clusters'] = str(latest_cluster['low_clusters'])

            # Add nearest support/resistance
            close_price = df['close'].iloc[-1]
            high_clusters = latest_cluster['high_clusters']
            low_clusters = latest_cluster['low_clusters']

            # Find nearest resistance (above price)
            resistances = [c for c in high_clusters if c > close_price]
            df['nearest_resistance'] = min(resistances) if resistances else np.nan

            # Find nearest support (below price)
            supports = [c for c in low_clusters if c < close_price]
            df['nearest_support'] = max(supports) if supports else np.nan
        else:
            df['high_clusters'] = '[]'
            df['low_clusters'] = '[]'
            df['nearest_resistance'] = np.nan
            df['nearest_support'] = np.nan

        return df

    def get_peaks_and_clusters(self, candles: pd.DataFrame) -> List[Dict]:
        """Get peaks and cluster them into support/resistance levels."""
        candles_length = len(candles)
        if candles_length < self.config.window_size:
            return []

        intervals = candles_length // self.config.calculation_interval
        clusters = []
        prominence_nominal = self._calculate_prominence(candles, self.config.prominence_percentage)

        for i in range(intervals):
            end_idx = (i + 1) * self.config.calculation_interval
            start_idx = end_idx - self.config.window_size if end_idx - self.config.window_size >= 0 else 0
            df = candles.iloc[start_idx:end_idx].copy()
            start_time = df.index.min()
            end_time = df.index.max()
            high_peaks, low_peaks = self._find_price_peaks(df, prominence_nominal, self.config.distance)
            close_price = df['close'].iloc[-1]
            high_peak_prices = df['high'].iloc[high_peaks]
            low_peak_prices = df['low'].iloc[low_peaks]
            high_peaks_index = df.iloc[high_peaks].index
            low_peaks_index = df.iloc[low_peaks].index

            if self.config.close_price_filter:
                filtered_high_peaks = high_peak_prices[high_peak_prices > close_price]
                filtered_low_peaks = low_peak_prices[low_peak_prices < close_price]

                if len(filtered_high_peaks) == 0 and i > 0:
                    high_clusters = []
                else:
                    high_clusters, _ = self._hierarchical_clustering(filtered_high_peaks, self.config.num_clusters)
                if len(filtered_low_peaks) == 0 and i > 0:
                    low_clusters = []
                else:
                    low_clusters, _ = self._hierarchical_clustering(filtered_low_peaks, self.config.num_clusters)
            else:
                filtered_high_peaks = high_peak_prices
                filtered_low_peaks = low_peak_prices
                high_clusters, _ = self._hierarchical_clustering(filtered_high_peaks, self.config.num_clusters) if len(filtered_high_peaks) > self.config.num_clusters else ([], [])
                low_clusters, _ = self._hierarchical_clustering(filtered_low_peaks, self.config.num_clusters) if len(filtered_low_peaks) > self.config.num_clusters else ([], [])

            clusters.append({
                'start_time': start_time,
                'end_time': end_time,
                'high_peaks_index': high_peaks_index,
                'low_peaks_index': low_peaks_index,
                'high_clusters': [cluster for cluster in high_clusters if not pd.isna(cluster)],
                'low_clusters': [cluster for cluster in low_clusters if not pd.isna(cluster)]
            })

        return clusters

    def get_peaks(self, candles: pd.DataFrame):
        """Get all peaks without clustering."""
        prominence_nominal = self._calculate_prominence(candles, self.config.prominence_percentage)
        high_peaks, low_peaks = self._find_price_peaks(candles, prominence_nominal, self.config.distance)
        high_peak_prices = candles['high'].iloc[high_peaks]
        low_peak_prices = candles['low'].iloc[low_peaks]
        high_peaks_index = candles.iloc[high_peaks].index
        low_peaks_index = candles.iloc[low_peaks].index
        return {
            "high_peaks": [high_peaks_index, high_peak_prices],
            "low_peaks": [low_peaks_index, low_peak_prices],
        }

    def _calculate_prominence(self, candles: pd.DataFrame, prominence_percentage: float) -> float:
        """Calculate prominence threshold based on price range."""
        price_range = candles['high'].max() - candles['low'].min()
        return price_range * prominence_percentage

    def _find_price_peaks(self, candles: pd.DataFrame, prominence_nominal: float, distance: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find high and low peaks using scipy find_peaks."""
        high_peaks, _ = find_peaks(candles['high'], prominence=prominence_nominal, distance=distance)
        low_peaks, _ = find_peaks(-candles['low'], prominence=prominence_nominal, distance=distance)
        return high_peaks, low_peaks

    @staticmethod
    def _hierarchical_clustering(peaks: pd.Series, num_clusters: int = 3) -> Tuple[List[float], np.ndarray]:
        """Cluster peaks using hierarchical clustering."""
        if len(peaks) == 0:
            return [], np.array([])
        if len(peaks) < num_clusters:
            return peaks.tolist(), np.arange(len(peaks))

        Z = linkage(peaks.values.reshape(-1, 1), method='ward')
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        centroids = [peaks[labels == k].mean() for k in range(1, num_clusters + 1)]
        return centroids, labels

    def create_feature(self, candles: "Candles") -> Feature:
        """Create Feature object from candles."""
        df = self.calculate(candles.data)
        clusters = self.get_peaks_and_clusters(candles.data)

        # Get latest cluster data
        latest_cluster = clusters[-1] if clusters else None

        value = {
            'high_clusters': latest_cluster['high_clusters'] if latest_cluster else [],
            'low_clusters': latest_cluster['low_clusters'] if latest_cluster else [],
            'nearest_resistance': float(df['nearest_resistance'].iloc[-1]) if not pd.isna(df['nearest_resistance'].iloc[-1]) else None,
            'nearest_support': float(df['nearest_support'].iloc[-1]) if not pd.isna(df['nearest_support'].iloc[-1]) else None,
            'num_resistance_levels': len(latest_cluster['high_clusters']) if latest_cluster else 0,
            'num_support_levels': len(latest_cluster['low_clusters']) if latest_cluster else 0,
        }

        return Feature(
            feature_name="peak_analyzer",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value=value,
            info={
                'prominence_percentage': self.config.prominence_percentage,
                'distance': self.config.distance,
                'num_clusters': self.config.num_clusters,
                'window_size': self.config.window_size,
                'calculation_interval': self.config.calculation_interval,
                'description': 'Support/resistance detection via peak clustering',
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles", proximity_threshold: float = 0.02) -> Optional[Signal]:
        """Create signal when price is near support/resistance levels."""
        df = self.calculate(candles.data)
        close_price = df['close'].iloc[-1]

        nearest_resistance = df['nearest_resistance'].iloc[-1]
        nearest_support = df['nearest_support'].iloc[-1]

        # Calculate proximity to levels
        if not pd.isna(nearest_resistance):
            resistance_distance = (nearest_resistance - close_price) / close_price
            if 0 < resistance_distance < proximity_threshold:
                # Near resistance - potential reversal SHORT
                signal_value = -min(1.0, (proximity_threshold - resistance_distance) / proximity_threshold)
                return Signal(
                    signal_name=f"peak_analyzer_{self.config.num_clusters}",
                    trading_pair=candles.trading_pair,
                    category='mr',  # mean reversion at S/R levels
                    value=signal_value
                )

        if not pd.isna(nearest_support):
            support_distance = (close_price - nearest_support) / close_price
            if 0 < support_distance < proximity_threshold:
                # Near support - potential reversal LONG
                signal_value = min(1.0, (proximity_threshold - support_distance) / proximity_threshold)
                return Signal(
                    signal_name=f"peak_analyzer_{self.config.num_clusters}",
                    trading_pair=candles.trading_pair,
                    category='mr',
                    value=signal_value
                )

        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """Add support/resistance levels and peaks to chart."""
        clusters = self.get_peaks_and_clusters(candles.data)

        for i, cluster in enumerate(clusters):
            # Plot high peaks
            fig.add_trace(go.Scatter(
                x=cluster['high_peaks_index'],
                y=candles.data['high'].loc[cluster['high_peaks_index']],
                mode='markers',
                marker=dict(size=7, color='yellow'),
                name=f'High Peaks {cluster["start_time"]}',
                showlegend=False
            ))

            # Plot low peaks
            fig.add_trace(go.Scatter(
                x=cluster['low_peaks_index'],
                y=candles.data['low'].loc[cluster['low_peaks_index']],
                mode='markers',
                marker=dict(size=7, color='yellow'),
                name=f'Low Peaks {cluster["start_time"]}',
                showlegend=False
            ))

            # Plot resistance levels (high clusters)
            for level in cluster['high_clusters']:
                x_start = cluster['end_time']
                x_end = clusters[i+1]['end_time'] if i < len(clusters)-1 else candles.data.index[-1]
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    y0=level,
                    x1=x_end,
                    y1=level,
                    line=dict(color='orange', width=2),
                )

            # Plot support levels (low clusters)
            for level in cluster['low_clusters']:
                x_start = cluster['end_time']
                x_end = clusters[i+1]['end_time'] if i < len(clusters)-1 else candles.data.index[-1]
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    y0=level,
                    x1=x_end,
                    y1=level,
                    line=dict(color='blue', width=2, dash='dash'),
                )

        return fig
