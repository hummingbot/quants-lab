from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks


class PeakAnalyzer:
    def __init__(self, candles: pd.DataFrame):
        self.candles = candles

    def get_peaks_and_clusters(self,
                               prominence_percentage: float = 0.01,
                               distance: int = 5,
                               num_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
        prominence_nominal = self._calculate_prominence(prominence_percentage)
        high_peaks, low_peaks = self._find_price_peaks(prominence_nominal, distance)
        high_peak_prices = self.candles['high'].iloc[high_peaks]
        low_peak_prices = self.candles['low'].iloc[low_peaks]
        high_clusters, _ = self._hierarchical_clustering(high_peak_prices, num_clusters)
        low_clusters, _ = self._hierarchical_clustering(low_peak_prices, num_clusters)
        return high_peaks, low_peaks, high_clusters, low_clusters

    def plot_candles_with_peaks_and_clusters(self, prominence_percentage: float = 0.01, distance: int = 5,
                                             num_clusters: int = 3):
        high_peaks, low_peaks, high_clusters, low_clusters = self.get_peaks_and_clusters(prominence_percentage,
                                                                                         distance, num_clusters)
        self.plot_price_chart_with_clusters(high_peaks, low_peaks, high_clusters, low_clusters)

    def _calculate_prominence(self, prominence_percentage: float) -> float:
        price_range = self.candles['high'].max() - self.candles['low'].min()
        return price_range * prominence_percentage

    def _find_price_peaks(self, prominence_nominal: float, distance: int) -> Tuple[np.ndarray, np.ndarray]:
        high_peaks, _ = find_peaks(self.candles['high'], prominence=prominence_nominal, distance=distance)
        low_peaks, _ = find_peaks(-self.candles['low'], prominence=prominence_nominal, distance=distance)
        return high_peaks, low_peaks

    @staticmethod
    def _hierarchical_clustering(peaks: pd.Series, num_clusters: int = 3) -> Tuple[List[float], np.ndarray]:
        Z = linkage(peaks.values.reshape(-1, 1), method='ward')
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        centroids = [peaks[labels == k].mean() for k in range(1, num_clusters + 1)]
        return sorted(centroids, reverse=True), labels

    def plot_price_chart_with_clusters(self, high_peaks: np.ndarray, low_peaks: np.ndarray, high_clusters: List[float],
                                       low_clusters: List[float]):
        fig = go.Figure(data=[go.Candlestick(
            x=self.candles.index,
            open=self.candles['open'],
            high=self.candles['high'],
            low=self.candles['low'],
            close=self.candles['close'],
            name='OHLC Data'
        )])
        fig.add_trace(go.Scatter(
            x=self.candles.index[high_peaks],
            y=self.candles['high'].iloc[high_peaks],
            mode='markers',
            marker=dict(size=7, color='red'),
            name='High Peaks'
        ))
        fig.add_trace(go.Scatter(
            x=self.candles.index[low_peaks],
            y=self.candles['low'].iloc[low_peaks],
            mode='markers',
            marker=dict(size=7, color='green'),
            name='Low Peaks'
        ))
        for level in high_clusters:
            fig.add_hline(y=level, line=dict(color='orange', width=2), annotation_text=f"High Cluster: {level:.2f}")
        for level in low_clusters:
            fig.add_hline(y=level, line=dict(color='blue', width=2, dash='dash'),
                          annotation_text=f"Low Cluster: {level:.2f}")
        fig.update_layout(xaxis_rangeslider_visible=False, height=800)
        fig.show()
