from typing import Dict, List, Tuple

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
                               num_clusters: int = 3,
                               close_price_filter: bool = True,
                               window_size: int = 100,
                               calculation_interval: int = 50) -> List[Dict]:
        candles_length = len(self.candles)
        if candles_length < window_size:
            raise ValueError(f"Candles length is less than window size: {candles_length} < {window_size}")
        intervals = candles_length // calculation_interval
        clusters = []
        prominence_nominal = self._calculate_prominence(self.candles, prominence_percentage)
        for i in range(intervals):
            end_idx = (i + 1) * calculation_interval
            start_idx = end_idx - window_size if end_idx - window_size >= 0 else 0
            df = self.candles.iloc[start_idx:end_idx].copy()
            start_time = df.index.min()
            end_time = df.index.max()
            high_peaks, low_peaks = self._find_price_peaks(df, prominence_nominal, distance)
            close_price = df['close'].iloc[-1]
            high_peak_prices = df['high'].iloc[high_peaks]
            low_peak_prices = df['low'].iloc[low_peaks]
            high_peaks_index = df.iloc[high_peaks].index
            low_peaks_index = df.iloc[low_peaks].index
            if close_price_filter:
                filtered_high_peaks = high_peak_prices[high_peak_prices > close_price]
                filtered_low_peaks = low_peak_prices[low_peak_prices < close_price]
                
                # Find last valid clusters recursively
                if len(filtered_high_peaks) == 0 and i > 0:
                    high_clusters = []
                else:
                    high_clusters, _ = self._hierarchical_clustering(filtered_high_peaks, num_clusters)
                if len(filtered_low_peaks) == 0 and i > 0:
                    low_clusters = []
                else:
                    low_clusters, _ = self._hierarchical_clustering(filtered_low_peaks, num_clusters)
            else:
                filtered_high_peaks = high_peak_prices
                filtered_low_peaks = low_peak_prices
                high_clusters, _ = self._hierarchical_clustering(filtered_high_peaks, num_clusters) if len(filtered_high_peaks) > num_clusters else [], []
                low_clusters, _ = self._hierarchical_clustering(filtered_low_peaks, num_clusters) if len(filtered_low_peaks) > num_clusters else [], []

            clusters.append(
                {
                    'start_time': start_time,
                    'end_time': end_time,
                    'high_peaks_index': high_peaks_index,
                    'low_peaks_index': low_peaks_index,
                    'high_clusters': [cluster for cluster in high_clusters if not pd.isna(cluster)],
                    'low_clusters': [cluster for cluster in low_clusters if not pd.isna(cluster)]
                }
            )
        return clusters

    def get_peaks(self, prominence_percentage: float = 0.01, distance: int = 5):
        prominence_nominal = self._calculate_prominence(self.candles, prominence_percentage)
        high_peaks, low_peaks = self._find_price_peaks(self.candles, prominence_nominal, distance)
        high_peak_prices = self.candles['high'].iloc[high_peaks]
        low_peak_prices = self.candles['low'].iloc[low_peaks]
        high_peaks_index = self.candles.iloc[high_peaks].index
        low_peaks_index = self.candles.iloc[low_peaks].index
        return {
            "high_peaks": [high_peaks_index, high_peak_prices],
            "low_peaks": [low_peaks_index, low_peak_prices],
        }
    

    def _calculate_prominence(self, candles: pd.DataFrame, prominence_percentage: float) -> float:
        price_range = candles['high'].max() - candles['low'].min()
        return price_range * prominence_percentage

    def _find_price_peaks(self, candles: pd.DataFrame, prominence_nominal: float, distance: int) -> Tuple[np.ndarray, np.ndarray]:
        high_peaks, _ = find_peaks(candles['high'], prominence=prominence_nominal, distance=distance)
        low_peaks, _ = find_peaks(-candles['low'], prominence=prominence_nominal, distance=distance)
        return high_peaks, low_peaks

    @staticmethod
    def _hierarchical_clustering(peaks: pd.Series, num_clusters: int = 3) -> Tuple[List[float], np.ndarray]:
        if len(peaks) == 0:
            return [], np.array([])
        if len(peaks) < num_clusters:
            # If we have fewer peaks than requested clusters, return each peak as its own cluster
            return peaks.tolist(), np.arange(len(peaks))
        
        Z = linkage(peaks.values.reshape(-1, 1), method='ward')
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        centroids = [peaks[labels == k].mean() for k in range(1, num_clusters + 1)]
        return centroids, labels
    
    def add_clusters_to_candles_fig(self, fig: go.Figure, clusters: List[Dict]):
        for i, cluster in enumerate(clusters):
            # Plot high peaks for current cluster
            fig.add_trace(go.Scatter(
                x=cluster['high_peaks_index'],
                y=self.candles['high'].loc[cluster['high_peaks_index']],
                mode='markers',
                marker=dict(size=7, color='yellow'),
                name=f'High Peaks {cluster["start_time"]}',
                showlegend=False
            ))
            
            # Plot low peaks for current cluster
            fig.add_trace(go.Scatter(
                x=cluster['low_peaks_index'],
                y=self.candles['low'].loc[cluster['low_peaks_index']],
                mode='markers',
                marker=dict(size=7, color='yellow'),
                name=f'Low Peaks {cluster["start_time"]}',
                showlegend=False
            ))

            # Plot horizontal lines for high clusters
            for level in cluster['high_clusters']:
                x_start = cluster['end_time']
                x_end = clusters[i+1]['end_time'] if i < len(clusters)-1 else self.candles.index[-1]
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    y0=level,
                    x1=x_end,
                    y1=level,
                    line=dict(color='orange', width=2),
                )
                # fig.add_annotation(
                #     x=x_start,
                #     y=level,
                #     text=f"High: {level:.2f}",
                #     showarrow=False,
                #     yshift=10
                # )

            # Plot horizontal lines for low clusters
            for level in cluster['low_clusters']:
                x_start = cluster['end_time']
                x_end = clusters[i+1]['end_time'] if i < len(clusters)-1 else self.candles.index[-1]
                fig.add_shape(
                    type="line",
                    x0=x_start,
                    y0=level,
                    x1=x_end,
                    y1=level,
                    line=dict(color='blue', width=2, dash='dash'),
                )
                # fig.add_annotation(
                #     x=x_start,
                #     y=level,
                #     text=f"Low: {level:.2f}",
                #     showarrow=False,
                #     yshift=-10
                # )

    def _get_last_valid_clusters(self, clusters: List[Dict], current_index: int, cluster_type: str, close_price: float) -> List[float]:
        """Recursively look back through clusters to find the last valid cluster values."""
        for idx in range(current_index - 1, -1, -1):
            if cluster_type == "high_cluster":
                high_clusters = clusters[idx][cluster_type]
            if len(clusters[idx][cluster_type]) > 0:
                if all(value > close_price for value in clusters[idx][cluster_type]):
                    return clusters[idx][cluster_type]
                else:
                    # get the low clusters and use the values to set the high clusters symmetrically
                    low_clusters = clusters[idx]['low_clusters']
                    high_clusters = [close_price * 2 - value for value in low_clusters]
                    return high_clusters
        # If no valid clusters found, return empty list or handle as needed
        return []
